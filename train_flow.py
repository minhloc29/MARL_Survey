from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import mate
from flow import (
    CameraObsEncoder,
    IndividualCritic,
    MateObserver,
    OneStepPolicy,
    ReplayBuffer,
    VelocityField,
    advantage_weighted_flow_bc_loss,
    avg_mixer,
    distillation_and_qmax_loss,
    flow_bc_loss,
    sample_joint_flow,
    split_per_agent,
)


# --------------------------------------------------------------------------
# Actions: the env Box is low=[-5,-2.5], high=[5,2.5]. Networks output
# normalized actions in [-1,1] (OneStepPolicy has a final Tanh; the flow
# samples in [-1,1] space too). We convert only at env-step time.
# --------------------------------------------------------------------------
ACTION_LOW = np.asarray([-5.0, -2.5], dtype=np.float64)
ACTION_HIGH = np.asarray([5.0, 2.5], dtype=np.float64)


def denormalize(a_norm: np.ndarray) -> np.ndarray:
    """Map a normalized [-1,1] action to the env Box."""
    return ACTION_LOW + 0.5 * (a_norm + 1.0) * (ACTION_HIGH - ACTION_LOW)


class RunningMeanStd:
    """Online mean/std via Welford's algorithm; rescales rewards at train time.

    Rewards are stored raw in the replay buffer; we normalize them to unit
    scale when building TD targets. This keeps the buffer stationary even as
    the reward magnitude drifts during training.
    """

    def __init__(self, eps: float = 1e-4):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        for v in values.ravel():
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(float(np.sqrt(self.M2 / self.n)), self.eps)

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "M2": self.M2, "eps": self.eps}

    @classmethod
    def from_dict(cls, d: dict) -> "RunningMeanStd":
        obj = cls(eps=d["eps"])
        obj.n = d["n"]
        obj.mean = d["mean"]
        obj.M2 = d["M2"]
        return obj

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        """Standardize rewards to ~zero mean / unit std before building TD targets.

        Centering is important: an uncentered denominator only rescales, it does
        not remove the systematic offset, which otherwise biases bootstrapped
        value targets and makes them scale-dependent as reward magnitude drifts.
        """
        return torch.clamp((x - self.mean) / self.std, -clip, clip)


# --------------------------------------------------------------------------
# Shared observation encoder helper.
#
# Parsing is pure-numpy Python work (the CPU bottleneck), while the CamObsEncoder
# forward is torch/GPU. We split the two so a batch that must be encoded more
# than once per update (the shared encoder is stepped in-place between stages,
# forcing a re-forward) is only ever parsed once.
# --------------------------------------------------------------------------
def parse_obs(observer, joint_obs_np):
    """(B, C, obs_dim) -> list of B per-camera parse results (numpy only)."""
    return [observer.parse(joint_obs_np[b]) for b in range(joint_obs_np.shape[0])]


def encode_parsed(camera_encoder, parsed_batch, device):
    """Torch-encode an already-parsed batch -> list of C tensors, each (B, out_dim)."""
    B = len(parsed_batch)
    C = len(parsed_batch[0])
    cam_embs = []
    for c in range(C):
        self_states = np.stack([parsed_batch[b][c]["self_state"] for b in range(B)])
        entities = {}
        masks = {}
        kinds = list(parsed_batch[0][c]["entities"])
        for k in kinds:
            shape = (B,) + parsed_batch[0][c]["entities"][k][0].shape
            entities[k] = np.zeros(shape, dtype=np.float64)
            masks[k] = np.zeros((B,) + parsed_batch[0][c]["entities"][k][1].shape, dtype=bool)
        for b in range(B):
            items = parsed_batch[b][c]["entities"]
            for k in kinds:
                entities[k][b] = items[k][0]
                masks[k][b] = items[k][1]
        self_state_t = torch.as_tensor(self_states, dtype=torch.float32, device=device)
        entities_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in entities.items()
        }
        masks_t = {k: torch.as_tensor(v, dtype=torch.bool, device=device) for k, v in masks.items()}
        cam_embs.append(camera_encoder(self_state_t, entities_t, masks_t))
    return cam_embs


def encode_obs(observer, camera_encoder, joint_obs_np, device):
    """Parse + encode: (B, C, obs_dim) -> list of C (B, out_dim) tensors."""
    return encode_parsed(camera_encoder, parse_obs(observer, joint_obs_np), device)


def polyak_update(target, source, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


@torch.no_grad()
def select_normalized_actions(policies, obs_embs, device):
    """Sample normalized [-1,1] per-agent actions for a rollout step."""
    return torch.stack(
        [p(o, z_i=None) for p, o in zip(policies, obs_embs)], dim=1
    )  # (B, C, act_dim)


# --------------------------------------------------------------------------
# Model construction / checkpointing.
# --------------------------------------------------------------------------
def build_models(num_cameras: int, act_dim: int, embed_dim: int, device):
    """Construct all networks for the current config.

    Returns:
        (models_dict, obs_dim_out, joint_act_dim). models_dict contains
        camera_encoder, velocity_field, critics, target_critics, policies.
    """
    entity_dims = {"targets": 5, "obstacles": 4, "teammates": 7}
    camera_encoder = CameraObsEncoder(
        self_state_dim=9, entity_dims=entity_dims, embed_dim=embed_dim
    ).to(device)
    obs_dim_out = camera_encoder.out_dim
    joint_act_dim = num_cameras * act_dim

    velocity_field = VelocityField(
        joint_obs_dim=num_cameras * obs_dim_out,
        joint_act_dim=joint_act_dim,
    ).to(device)

    critics = nn.ModuleList(
        [IndividualCritic(obs_dim_out, act_dim).to(device) for _ in range(num_cameras)]
    )
    target_critics = nn.ModuleList(
        [IndividualCritic(obs_dim_out, act_dim).to(device) for _ in range(num_cameras)]
    )
    for t, s in zip(target_critics, critics):
        t.load_state_dict(s.state_dict())

    policies = nn.ModuleList(
        [OneStepPolicy(obs_dim_out, noise_dim=8, act_dim=act_dim).to(device)
         for _ in range(num_cameras)]
    )

    models = {
        "camera_encoder": camera_encoder,
        "velocity_field": velocity_field,
        "critics": critics,
        "target_critics": target_critics,
        "policies": policies,
    }
    return models, obs_dim_out, joint_act_dim


def checkpoint_config_dict(num_cameras, num_targets, num_obstacles,
                           act_dim, embed_dim, obs_dim_out):
    return {
        "num_cameras": num_cameras,
        "num_targets": num_targets,
        "num_obstacles": num_obstacles,
        "act_dim": act_dim,
        "embed_dim": embed_dim,
        "obs_dim_out": obs_dim_out,
    }


def save_checkpoint(path: str, models: dict, optimizers: dict, reward_norm,
                    step: int, ep_count: int, config: dict, replay=None,
                    episode_stats=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "config": config,
            "models": {k: v.state_dict() for k, v in models.items()},
            "optimizers": (
                {k: opt.state_dict() for k, opt in optimizers.items()}
                if optimizers is not None else None
            ),
            "reward_norm": reward_norm.to_dict() if reward_norm is not None else None,
            "step": step,
            "ep_count": ep_count,
            "replay": replay.to_dict() if replay is not None else None,
            "episode_stats": episode_stats or None,
        },
        path,
    )


def load_checkpoint(path: str, device):
    """Load a checkpoint, rebuilding the models to match its config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    models, _, _ = build_models(
        num_cameras=cfg["num_cameras"],
        act_dim=cfg["act_dim"],
        embed_dim=cfg["embed_dim"],
        device=device,
    )
    for name, net in models.items():
        net.load_state_dict(ckpt["models"][name])
    return ckpt, models, cfg


def make_env(config_name: str):
    base_env = gym.make("MultiAgentTracking-v0", config=config_name)
    return mate.MultiCamera.make(base_env, target_agent=mate.GreedyTargetAgent())


# --------------------------------------------------------------------------
# Common subcommand helpers.
# --------------------------------------------------------------------------
def add_model_args(parser):
    parser.add_argument("--config", default="MATE-4v8-9.yaml")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument("--device", default="cpu")


def add_train_args(parser):
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--adv-beta", type=float, default=1.0,
                        help="Advantage temperature for advantage-weighted flow-BC.")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=100000)
    parser.add_argument("--outdir", default="ckpts")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb-api", default="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",
                        help="If set, log training curves to Weights & Biases (a wandb API key).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                    help="CPU threads this process may use (torch intra-op "
                         "parallelism). Set explicitly when running several "
                         "training processes concurrently on one machine.")

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="train_flow",
        description="MAC-Flow (flow-matching CTDE) training CLI for MATE.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train and save checkpoints")
    add_model_args(p_train)
    add_train_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    add_model_args(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.set_defaults(func=cmd_evaluate)

    p_render = sub.add_parser("render", help="render one rollout from a checkpoint")
    add_model_args(p_render)
    p_render.add_argument("--checkpoint", required=True)
    p_render.add_argument("--max-steps", type=int, default=2000)
    p_render.set_defaults(func=cmd_render)

    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Subcommands.
# --------------------------------------------------------------------------
def cmd_train(args):
    torch.set_num_threads(args.num_workers)
    torch.set_num_interop_threads(max(1, args.num_workers // 2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # autotune fixed input shapes on GPU
    device = torch.device(args.device)

    # Optional W&B dashboard. Off by default to avoid forcing a login.
    wandb_run = None
    if args.wandb_api:
        import wandb
        wandb.login(key=args.wandb_api)
        wandb_run = wandb.init(
            project="mate-flow", config=vars(args), reinit=True
        )

    env = make_env(args.config)
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    num_obstacles = env.unwrapped.num_obstacles
    obs_dim = np.asarray(env.reset()[0]).shape[-1]

    models, obs_dim_out, joint_act_dim = build_models(
        num_cameras, args.act_dim, args.embed_dim, device
    )
    camera_encoder = models["camera_encoder"]
    velocity_field = models["velocity_field"]
    critics = models["critics"]
    target_critics = models["target_critics"]
    policies = models["policies"]

    config = checkpoint_config_dict(
        num_cameras, num_targets, num_obstacles, args.act_dim,
        args.embed_dim, obs_dim_out,
    )

 
    encoder_params = list(camera_encoder.parameters())
    flow_opt = torch.optim.Adam(
        encoder_params + list(velocity_field.parameters()), lr=args.lr
    )
    critic_opt = torch.optim.Adam(list(critics.parameters()), lr=args.lr)
    policy_opt = torch.optim.Adam(list(policies.parameters()), lr=args.lr)
    optimizers = {"flow": flow_opt, "critic": critic_opt, "policy": policy_opt}
    replay = ReplayBuffer(args.replay_capacity)
    reward_norm = RunningMeanStd()
    stats = None  # episode log-stats restored on resume (if saved)

    start_step = 0
    ep_count = 0
    if args.resume:
        ckpt, _, _ = load_checkpoint(args.resume, device)
        for name, net in models.items():
            net.load_state_dict(ckpt["models"][name])
        for name, opt in optimizers.items():
            opt.load_state_dict(ckpt["optimizers"][name])
        reward_norm = RunningMeanStd.from_dict(ckpt["reward_norm"])
        start_step = ckpt["step"]
        ep_count = ckpt["ep_count"]
        # Restore the replay buffer (replay-based training genuinely continues
        # only if the buffer it samples from survives the restart) and the
        # episode/coverage log stats so the resumed run reports real values
        # instead of re-living a blank warmup.
        if ckpt.get("replay") is not None:
            replay = ReplayBuffer.from_dict(ckpt["replay"])
        stats = ckpt.get("episode_stats") or {} or None
        print(f"Resumed from {args.resume} at step {start_step} "
              f"(buffer {len(replay)})")

    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    # ---- Collect loop ----
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_len = 0
    episode_coverage = 0.0  # summed per-step coverage within the current episode
    # Completed-episode log stats, persisted across resumes so a restarted run
    # reports real values instead of a blank 0.000. The mid-episode accumulators
    # above are intentionally NOT persisted (resume restarts mid-episode cleanly).
    last_ep_reward = 0.0
    last_ep_coverage = 0.0
    episode_stats = {"last_ep_reward": last_ep_reward, "last_ep_coverage": last_ep_coverage}
    if stats:
        last_ep_reward = stats.get("last_ep_reward", 0.0)
        last_ep_coverage = stats.get("last_ep_coverage", 0.0)
        episode_stats.update(stats)
    running = {"flow": 0.0, "critic": 0.0, "distill": 0.0,
               "qmax": 0.0, "bc": 0.0, "adv_ess": 0.0}

    for step in range(start_step, args.steps):
        obs_np = np.asarray(obs, dtype=np.float64)[None]  # (1, C, obs_dim)
        with torch.no_grad():
            embs = encode_obs(observer, camera_encoder, obs_np, device)
        # Exploration: epsilon-greedy over a RANDOM action vs the policy.
        if np.random.rand() < 0.1:
            a_norm = np.random.uniform(-1, 1, size=(num_cameras, args.act_dim))
        else:
            with torch.no_grad():
                a_norm = select_normalized_actions(
                    policies, [e[:1] for e in embs], device
                )[0].cpu().numpy()  # (C, act_dim)

        a_env = denormalize(a_norm)
        obs2, reward, done, truncated, info = env.step(a_env)

        replay.add(obs_np[0], a_norm, reward, np.asarray(obs2, dtype=np.float64),
                   done or truncated)
        reward_norm.update(np.asarray(reward))

        # Per-step coverage: mean across cameras from MATE's per-camera info.
        cam_infos = info.get("infos") if isinstance(info, dict) else None
        if cam_infos:
            step_coverage = float(np.mean([c.get("coverage_rate", 0.0) for c in cam_infos]))
        else:
            step_coverage = 0.0

        episode_reward += float(reward)
        episode_coverage += step_coverage
        episode_len += 1
        obs = obs2
        if done or truncated:
            last_ep_reward = episode_reward
            last_ep_coverage = episode_coverage / max(1, episode_len)
            episode_stats["last_ep_reward"] = last_ep_reward
            episode_stats["last_ep_coverage"] = last_ep_coverage
            episode_reward = 0.0
            episode_coverage = 0.0
            episode_len = 0
            obs, _ = env.reset()
            ep_count += 1

        # ---- Update loop ----
        if len(replay) >= args.warmup:
            batch = replay.sample(args.batch_size, device=device)
            o_np = batch["obs"].cpu().numpy()    # (B, C, obs_dim)
            no_np = batch["next_obs"].cpu().numpy()

            # Parse each numpy batch ONCE (pure-Python, the CPU bottleneck), then
            # re-encode the same parse across the training stages. Stage 3 needs a
            # re-forward of o_np after the shared encoder is stepped in-place, but
            # re-encodING must not re-parse.
            parsed_o = parse_obs(observer, o_np)
            parsed_no = parse_obs(observer, no_np)
            embs = encode_parsed(camera_encoder, parsed_o, device)
            n_embs = encode_parsed(camera_encoder, parsed_no, device)

            o_joint = torch.cat(embs, dim=-1)     # (B, C*D)
            a_joint = batch["action"].reshape(batch["action"].size(0), -1)

            # ---- Stage 2: critics (TD regression only; own optimizer) ----
            with torch.no_grad():
                # Bootstrap on the deterministic ONE-STEP policy's action at s',
                # not a random draw from the flow. This matches what is actually
                # deployed (the actor executes one-step policies), so the value
                # target is the return of the real policy rather than of a random
                # flow sample. A random next-action inflates the target (a
                # positive-bias bootstrap) and contributes to the critic
                # overestimation seen as Q_tot pinning to a constant +3.
                a_next_slices = split_per_agent(
                    torch.stack(
                        [p(no) for p, no in zip(policies, n_embs)], dim=-1
                    ),
                    num_cameras, args.act_dim,
                )
                q_next_list = torch.stack(
                    [tc(no, a) for tc, no, a in zip(target_critics, n_embs, a_next_slices)],
                    dim=-1,
                )
                td_target = (
                    reward_norm.normalize(batch["reward"])
                    + args.gamma * (1.0 - batch["done"]) * avg_mixer(q_next_list)
                )
            a_slices = split_per_agent(a_joint, num_cameras, args.act_dim)
            q_list = torch.stack(
                [c(o, a) for c, o, a in zip(critics, embs, a_slices)], dim=-1
            )  # Q of the STORED joint action
            q_tot = avg_mixer(q_list)
            loss_critic = nn.functional.mse_loss(q_tot, td_target)

            with torch.no_grad():
                # Q of what the one-step policies would do RIGHT NOW.
                a_policy = [p(o) for p, o in zip(policies, embs)]
                q_policy_list = torch.stack(
                    [c(o, a) for c, o, a in zip(critics, embs, a_policy)], dim=-1
                )
                q_policy_tot = avg_mixer(q_policy_list)
                # Advantage: how much better the stored action is than the
                # current policy's own action (a true baseline comparison).
                advantage = (q_tot - q_policy_tot).detach()

            critic_opt.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_opt.step()

            # ---- Stage 1: advantage-weighted flow BC (flow + encoder) ----
            loss_flow, adv_w_ess = advantage_weighted_flow_bc_loss(
                velocity_field, o_joint, a_joint, advantage, beta=args.adv_beta
            )
            flow_opt.zero_grad()
            loss_flow.backward(retain_graph=True)
            flow_opt.step()

            # ---- Stage 3: distill joint flow -> one-step policies ----
            # Freeze critics so the -Q_tot actor term cannot push gradients
            # into the critic weights (only into policy + shared encoder).
            # NOTE: re-encode WITH gradients, NOT under no_grad. The shared
            # camera_encoder was updated in-place by flow_opt.step() above, so
            # the earlier `embs` graph is stale — re-running the encoder here
            # produces a fresh graph whose backward pushes gradient into both
            # the policies AND the shared encoder. (Wrapping it in no_grad
            # would silently drop all encoder learning in this stage.
            # No inplace crash occurs because this fresh graph is built after
            # flow_opt.step() and nothing mutates the encoder after it.)
            embs_v2 = encode_parsed(camera_encoder, parsed_o, device)
            with torch.no_grad():
                o_joint_v2 = torch.cat(embs_v2, dim=-1)
                a_joint_sample = sample_joint_flow(
                    velocity_field, o_joint_v2, act_dim=joint_act_dim, steps=args.flow_steps
                )
                joint_slices = split_per_agent(a_joint_sample, num_cameras, args.act_dim)
            for p in critics.parameters():
                p.requires_grad_(False)
            loss_qmax, loss_distill_raw = distillation_and_qmax_loss(
                policies, critics, avg_mixer, embs_v2, joint_slices
            )
            # Fused Eq. 9 loss for the backward pass: -Q_tot + alpha * distill.
            loss_distill = loss_qmax + args.alpha * loss_distill_raw
            policy_opt.zero_grad()
            loss_distill.backward()
            policy_opt.step()
            for p in critics.parameters():
                p.requires_grad_(True)

            polyak_update(target_critics, critics, args.tau)

            running["flow"] += loss_flow.item()
            running["critic"] += loss_critic.item()
            running["distill"] += loss_distill.item()
            running["qmax"] += loss_qmax.item()
            running["bc"] += loss_distill_raw.item()
            running["adv_ess"] += adv_w_ess

        # ---- Logging / checkpointing ----
        step_1 = step + 1
        if step_1 % args.log_every == 0:
            n = args.log_every
            print(
                f"step {step_1}/{args.steps} | ep {ep_count} | last_ep_rew {last_ep_reward:.3f}"
                f" | last_ep_cov {last_ep_coverage:.3f}"
                f" | flow {running['flow']/n:.4f} | critic {running['critic']/n:.4f}"
                f" | distill {running['distill']/n:.4f}"
                f" | qmax {running['qmax']/n:.4f} | bc {running['bc']/n:.4f}"
                f" | adv_ess {running['adv_ess']/n:.4f} | buffer {len(replay)}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode_reward": last_ep_reward,
                        "coverage_rate": last_ep_coverage,
                        "loss/flow": running["flow"] / n,
                        "loss/critic": running["critic"] / n,
                        "loss/distill": running["distill"] / n,
                        "loss/qmax": running["qmax"] / n,
                        "loss/bc": running["bc"] / n,
                        "adv_weight_ess": running["adv_ess"] / n,
                        "buffer_size": len(replay),
                        "episodes": ep_count,
                    },
                    step=step_1,
                )
            running = {"flow": 0.0, "critic": 0.0, "distill": 0.0,
               "qmax": 0.0, "bc": 0.0, "adv_ess": 0.0}

        if step_1 % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.outdir, f"step_{step_1}.pt"),
                models, optimizers, reward_norm, step_1, ep_count, config,
                replay=replay, episode_stats=episode_stats,
            )
            print(f"  saved {os.path.join(args.outdir, 'step_' + str(step_1) + '.pt')}", flush=True)

    save_checkpoint(
        os.path.join(args.outdir, "final.pt"),
        models, optimizers, reward_norm, args.steps, ep_count, config,
        replay=replay, episode_stats=episode_stats,
    )
    if wandb_run is not None:
        wandb_run.finish()
    print("Training complete; final checkpoint saved.")


@torch.no_grad()
def _rollout(models, observer, env, device, max_steps, num_cameras,
             render: bool = False):
    """Play one episode with the learned one-step policies (no exploration)."""
    camera_encoder = models["camera_encoder"]
    policies = models["policies"]

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        obs_np = np.asarray(obs, dtype=np.float64)[None]
        embs = encode_obs(observer, camera_encoder, obs_np, device)
        a_norm = select_normalized_actions(policies, [e[:1] for e in embs], device)[0]
        a_env = denormalize(a_norm.cpu().numpy())
        obs, reward, done, truncated, _ = env.step(a_env)
        total_reward += float(reward)
        steps += 1
        if render:
            run = env.render()
            if not run:
                break
        if done or truncated:
            break
    return total_reward, steps


def cmd_evaluate(args):
    device = torch.device(args.device)
    env = make_env(args.config)
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    num_obstacles = env.unwrapped.num_obstacles
    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    ckpt, models, _ = load_checkpoint(args.checkpoint, device)

    rewards, lengths = [], []
    for _ in range(args.episodes):
        r, ln = _rollout(models, observer, env, device,
                         max_steps=10_000_000, num_cameras=num_cameras, render=False)
        rewards.append(r)
        lengths.append(ln)
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f"Evaluated {args.checkpoint} over {args.episodes} episodes")
    print(f"  mean reward: {mean_r:.3f} +/- {std_r:.3f}")
    print(f"  mean length: {float(np.mean(lengths)):.1f} steps")


def cmd_render(args):
    device = torch.device(args.device)
    env = make_env(args.config)
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    num_obstacles = env.unwrapped.num_obstacles
    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    ckpt, models, _ = load_checkpoint(args.checkpoint, device)
    r, ln = _rollout(models, observer, env, device,
                     max_steps=args.max_steps, num_cameras=num_cameras, render=True)
    print(f"Render finished: reward {r:.3f}, {ln} steps")


def main(argv=None):
    args = parse_arguments(argv)
    args.func(args)


if __name__ == "__main__":
    main()
