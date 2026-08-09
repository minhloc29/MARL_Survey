"""CLI for the MAC-Flow (flow-matching CTDE) algorithm on MATE's camera team.

Drives MATE's camera team with per-agent LOW-LEVEL continuous actions (a 2-dim
Box per camera), collected into a replay buffer, training three stages jointly:

  Stage 1  joint flow policy       (advantage-weighted flow-matching BC)
  Stage 2  per-agent IGM critics   (TD with flow-sampled next actions)
  Stage 3  one-step policies       (distilled from the flow)

Subcommands:

    train     train and periodically save checkpoints
    evaluate  load a checkpoint, run the learned one-step policies, report stats
    render    load a checkpoint, play one rollout with pygame rendering

Examples:

    python train_flow.py train --steps 5000 --save-every 1000 --outdir ckpts
    python train_flow.py train --steps 5000 --resume ckpts/step_3000.pt
    python train_flow.py evaluate --checkpoint ckpts/step_5000.pt --episodes 20
    python train_flow.py render   --checkpoint ckpts/step_5000.pt
"""

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


def normalize(a_env: np.ndarray) -> np.ndarray:
    """Map an env Box action to normalized [-1,1]."""
    return 2.0 * (a_env - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0


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
        return torch.clamp(x / self.std, -clip, clip)


# --------------------------------------------------------------------------
# Shared observation encoder helper.
# --------------------------------------------------------------------------
def encode_obs(observer, camera_encoder, joint_obs_np, device):
    """Encode a batch of joint observations into per-camera embeddings.

    Args:
        observer: MateObserver.
        camera_encoder: shared CameraObsEncoder.
        joint_obs_np: (B, num_cameras, obs_dim) float64 np.
        device: torch device.

    Returns:
        list of num_cameras tensors, each (B, camera_encoder.out_dim).
    """
    B, C, _ = joint_obs_np.shape
    cam_embs = []
    for c in range(C):
        self_states = np.stack(
            [observer.parse(joint_obs_np[b])[c]["self_state"] for b in range(B)]
        )  # (B, 9)
        # Build padded entity dicts + masks per camera c, per kind.
        entities = {}
        masks = {}
        kinds = None
        for b in range(B):
            items = observer.parse(joint_obs_np[b])[c]["entities"]
            if kinds is None:
                kinds = list(items)
                for k in kinds:
                    entities[k] = np.zeros((B,) + items[k][0].shape, dtype=np.float64)
                    masks[k] = np.zeros((B,) + items[k][1].shape, dtype=bool)
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


def save_checkpoint(path: str, models: dict, optimizer, reward_norm,
                    step: int, ep_count: int, config: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "config": config,
            "models": {k: v.state_dict() for k, v in models.items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "reward_norm": reward_norm.to_dict() if reward_norm is not None else None,
            "step": step,
            "ep_count": ep_count,
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--adv-beta", type=float, default=1.0,
                        help="Advantage temperature for advantage-weighted flow-BC.")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--outdir", default="ckpts")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument("--seed", type=int, default=0)


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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

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

    params = (
        list(camera_encoder.parameters())
        + list(velocity_field.parameters())
        + list(critics.parameters())
        + list(policies.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)
    replay = ReplayBuffer(args.replay_capacity)
    reward_norm = RunningMeanStd()

    start_step = 0
    ep_count = 0
    if args.resume:
        ckpt, _, _ = load_checkpoint(args.resume, device)
        for name, net in models.items():
            net.load_state_dict(ckpt["models"][name])
        optimizer.load_state_dict(ckpt["optimizer"])
        reward_norm = RunningMeanStd.from_dict(ckpt["reward_norm"])
        start_step = ckpt["step"]
        ep_count = ckpt["ep_count"]
        print(f"Resumed from {args.resume} at step {start_step}")

    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    # ---- Collect loop ----
    obs, _ = env.reset()
    episode_reward = 0.0
    running = {"flow": 0.0, "critic": 0.0, "distill": 0.0}

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
        obs2, reward, done, truncated, _ = env.step(a_env)

        replay.add(obs_np[0], a_norm, reward, np.asarray(obs2, dtype=np.float64),
                   done or truncated)
        reward_norm.update(np.asarray(reward))

        episode_reward += float(reward)
        obs = obs2
        if done or truncated:
            obs, _ = env.reset()
            ep_count += 1

        # ---- Update loop ----
        if len(replay) >= args.warmup:
            batch = replay.sample(args.batch_size, device=device)
            o_np = batch["obs"].cpu().numpy()    # (B, C, obs_dim)
            no_np = batch["next_obs"].cpu().numpy()

            embs = encode_obs(observer, camera_encoder, o_np, device)
            n_embs = encode_obs(observer, camera_encoder, no_np, device)

            o_joint = torch.cat(embs, dim=-1)     # (B, C*D)
            a_joint = batch["action"].reshape(batch["action"].size(0), -1)

            # Stage 2: critics via TD target with flow-sampled next actions.
            with torch.no_grad():
                a_next_joint = sample_joint_flow(
                    velocity_field, torch.cat(n_embs, dim=-1),
                    act_dim=joint_act_dim, steps=args.flow_steps,
                )
                a_next_slices = split_per_agent(a_next_joint, num_cameras, args.act_dim)
                q_next_list = torch.stack(
                    [tc(no, a) for tc, no, a in zip(target_critics, n_embs, a_next_slices)],
                    dim=-1,
                )
                target = (
                    reward_norm.normalize(batch["reward"])
                    + args.gamma * (1.0 - batch["done"]) * avg_mixer(q_next_list)
                )
            a_slices = split_per_agent(a_joint, num_cameras, args.act_dim)
            q_list = torch.stack(
                [c(o, a) for c, o, a in zip(critics, embs, a_slices)], dim=-1
            )
            q_tot = avg_mixer(q_list)
            loss_critic = nn.functional.mse_loss(q_tot, target)

            # Advantage estimate (TD residual) for the stored joint action.
            # Detach q_tot so the flow-BC weighting does not backpropagate
            # through the critic.
            advantage = (target - q_tot).detach()

            # Stage 1: advantage-weighted flow BC (implicit policy improvement).
            loss_flow = advantage_weighted_flow_bc_loss(
                velocity_field, o_joint, a_joint, advantage, beta=args.adv_beta
            )

            # Stage 3: distill joint flow -> one-step policies.
            with torch.no_grad():
                a_joint_sample = sample_joint_flow(
                    velocity_field, o_joint, act_dim=joint_act_dim, steps=args.flow_steps
                )
                joint_slices = split_per_agent(a_joint_sample, num_cameras, args.act_dim)
            loss_distill = distillation_and_qmax_loss(
                policies, critics, avg_mixer, embs, joint_slices, alpha=args.alpha
            )

            loss = loss_flow + loss_critic + loss_distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            polyak_update(target_critics, critics, args.tau)

            running["flow"] += loss_flow.item()
            running["critic"] += loss_critic.item()
            running["distill"] += loss_distill.item()

        # ---- Logging / checkpointing ----
        step_1 = step + 1
        if step_1 % args.log_every == 0:
            n = args.log_every
            print(
                f"step {step_1}/{args.steps} | ep {ep_count} | rew {episode_reward:.3f}"
                f" | flow {running['flow']/n:.4f} | critic {running['critic']/n:.4f}"
                f" | distill {running['distill']/n:.4f} | buffer {len(replay)}",
                flush=True,
            )
            running = {"flow": 0.0, "critic": 0.0, "distill": 0.0}
            episode_reward = 0.0

        if step_1 % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.outdir, f"step_{step_1}.pt"),
                models, optimizer, reward_norm, step_1, ep_count, config,
            )
            print(f"  saved {os.path.join(args.outdir, 'step_' + str(step_1) + '.pt')}", flush=True)

    save_checkpoint(
        os.path.join(args.outdir, "final.pt"),
        models, optimizer, reward_norm, args.steps, ep_count, config,
    )
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
