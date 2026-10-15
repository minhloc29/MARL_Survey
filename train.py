from argparse import ArgumentParser
import os
import gymnasium as gym
import gym_agent as ga
import mate
import numpy as np
from agents import GreedyCameraAgent
from mate.agents import GreedyTargetAgent
from wrapper import NetWrapper, AgentAsActionWrapper
from agents.base import GoalsBaseAgent
from gym_agent.core.polices import ActorCriticPolicy
from net import Net, FeatureExtractor
from common_net.attentions import MHAConfig
import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(
        self,
        history_len: int = 8,
        camera_dim: int = 8,
        target_dim: int = 3,
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(num_heads=1),
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            history_len=history_len,
            camera_dim=camera_dim,
            target_dim=target_dim,
            n_linear_attn_blocks=n_linear_attn_blocks,
            n_attn_blocks=n_attn_blocks,
            embed_dim=embed_dim,
            mha_config=mha_config,
        )   # B x C x D

        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        features: torch.Tensor = self.feature_extractor(x)
        avg_features = features.mean(dim=-2)  # B x D
        value = self.linear(avg_features)
        return value


# --------------------------------------------------------------------------
# MAPPO networks: shared FeatureExtractor backbone + a decentralized goal
# head (actor) and a centralized global-state value (critic).
# --------------------------------------------------------------------------
class MAPPOActor(nn.Module):
    """Decentralized goal-mask actor head over the shared backbone features.

    Input: (B, C, embed_dim) shared features. Output: (B, C, T) goal logits,
    consumed by gym_agent's MultiBinary distribution.
    """

    def __init__(self, embed_dim: int = 64, num_targets: int = 8):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_targets)

    def forward(self, features: torch.Tensor):
        return self.head(features)  # (B, C, T)


class GlobalCritic(nn.Module):
    """Centralized value: concatenate all camera features -> single scalar.

    The value is shared for GAE over the whole cooperative team.
    """

    def __init__(self, embed_dim: int = 64, num_cameras: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * num_cameras, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor):
        B = features.size(0)
        flat = features.reshape(B, -1)  # (B, C*D)
        return self.mlp(flat).squeeze(-1)  # (B,)


def make_actor_net():
    return Net()


def make_critic_net():
    return Critic()


def _mappo_feature_extractor(embed_dim=64, num_cameras=4):
    """Shared backbone for MAPPO: FeatureExtractor -> (B, C, embed_dim)."""
    return FeatureExtractor(
        history_len=8,
        camera_dim=8,
        target_dim=3,
        n_linear_attn_blocks=0,
        n_attn_blocks=3,
        embed_dim=embed_dim,
        mha_config=MHAConfig(num_heads=1),
    )


def make_mappo_actor_net(embed_dim=64, num_targets=8):
    return MAPPOActor(embed_dim=embed_dim, num_targets=num_targets)


def make_mappo_critic_net(embed_dim=64, num_cameras=4):
    return GlobalCritic(embed_dim=embed_dim, num_cameras=num_cameras)


def make_env(**kwargs):
    base_env = gym.make("MultiAgentTracking-v0",
                        config="MATE-4v8-9.yaml", **kwargs)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    return NetWrapper(
        AgentAsActionWrapper(
            env,
            [GoalsBaseAgent(executor=GreedyCameraAgent())
             for _ in range(env.unwrapped.num_cameras)],
        ),
        reward_type="coverage_rate",
    )


def make_maddpq_env(config_name: str = "MATE-4v8-9.yaml"):
    """Raw continuous-action env for MADDPG (no goal-mask wrapper)."""
    base_env = gym.make("MultiAgentTracking-v0", config=config_name)
    return mate.wrappers.CoverageReward(
        mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())
    )


def make_discrete_env(config_name: str = "MATE-4v8-9.yaml", levels: int = 5):
    """Discrete-action env for QMIX/QPLEX (levels*levels actions per camera).

    ``DiscreteCamera`` must wrap the raw base ``MultiAgentTracking`` env
    (``assert_base_environment`` rejects single-team wrappers like
    ``MultiCamera``), *then* ``MultiCamera.make`` replays the wrapper chain and
    attaches the GreedyTargetAgent so targets are handled internally.
    """
    base_env = gym.make("MultiAgentTracking-v0", config=config_name)
    base_env = mate.DiscreteCamera(base_env, levels=levels)
    return mate.wrappers.CoverageReward(
        mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())
    )


# QMIX/QPLEX discrete camera action count (levels=5 -> 25 per camera).
_VD_LEVELS = 5
_VD_NUM_ACTIONS = _VD_LEVELS * _VD_LEVELS


# MADDPG continuous camera action Box (per agency, 2-dim).
_MADDPQ_ACTION_LOW = np.asarray([-5.0, -2.5], dtype=np.float64)
_MADDPQ_ACTION_HIGH = np.asarray([5.0, 2.5], dtype=np.float64)


def _denormalize_maddpq(a_norm: np.ndarray) -> np.ndarray:
    """Map a normalized [-1,1] action to the MADDPG env Box."""
    return _MADDPQ_ACTION_LOW + 0.5 * (
        a_norm + 1.0
    ) * (_MADDPQ_ACTION_HIGH - _MADDPQ_ACTION_LOW)


def add_common_args(parser):
    parser.add_argument("--algo", choices=["ppo", "mappo", "maddpq", "qmix", "qplex", "rode", "coma", "mat", "dcg", "i2c", "dm2", "facmac", "iql"], default="mappo",
                        help="Training algorithm to run.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run training on: 'auto', 'cuda', 'mps' (Apple Silicon), or 'cpu'.")


def run_ppo(args):
    agent = ga.PPO(
        env=make_env,
        policy=ActorCriticPolicy(
            actor_factory_func=make_actor_net,
            critic_factory_func=make_critic_net,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr
        ),
        config=ga.PPOConfig(
            num_envs=args.num_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            log_score_type='mean'
        )
    )

    agent.fit(
        total_timesteps=args.total_timesteps,
        save_every=args.save_every,
        wandb_api=args.wandb_api,
    )


def run_mappo(args):
    # MAPPO reuses gym_agent's on-policy PPO machinery (GAE + clipping), with
    # the architecture reshaped to CTDE: a shared FeatureExtractor backbone,
    # a decentralized goal-mask actor head, and a centralized global-state
    # critic used for shared advantages.
    num_cameras = 4
    num_targets = 8
    embed_dim = 64
    agent = ga.PPO(
        env=make_env,
        policy=ActorCriticPolicy(
            feature_extractor_func=lambda: _mappo_feature_extractor(
                embed_dim=embed_dim, num_cameras=num_cameras
            ),
            actor_factory_func=lambda: make_mappo_actor_net(
                embed_dim=embed_dim, num_targets=num_targets
            ),
            critic_factory_func=lambda: make_mappo_critic_net(
                embed_dim=embed_dim, num_cameras=num_cameras
            ),
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr
        ),
        config=ga.PPOConfig(
            num_envs=args.num_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            log_score_type='mean'
        )
    )

    agent.fit(
        total_timesteps=args.total_timesteps,
        save_every=args.save_every,
        wandb_api=args.wandb_api,
    )


def run_maddpq(args):
    """Standalone MADDPG: deterministic actors + centralized critic, off-policy.

    Uses the raw continuous camera action space (Box per camera), reusing
    flow/'s encoders, replay buffer, and polyak-update pattern.
    """
    steps = args.total_timesteps if args.steps is None else args.steps
    args.steps = steps
    from flow import CameraObsEncoder, MateObserver, ReplayBuffer, DeterministicActor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = make_maddpq_env()
    # Initialize wandb ONCE, before the collection loop. Only wandb_run.log()
    # belongs inside the loop — re-initing on every gradient step would spam
    # the project with a new run per update (slow network calls + dashboard
    # garbage). The API key is read from the environment, never hardcoded.
    wandb_run = None
    if args.wandb_api:
        import wandb

        wandb.login(key=args.wandb_api)
        wandb_run = wandb.init(
            project="mate-maddpq",
            name="maddpq-run",
            config={
                "algorithm": "MADDPG",
                "warmup": args.warmup,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "tau": args.tau,
            },
        )
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    num_obstacles = env.unwrapped.num_obstacles
    obs_dim = np.asarray(env.reset()[0]).shape[-1]
    act_dim = 2  # per-camera [dorientation, dviewing_angle]

    embed_dim = args.embed_dim
    entity_dims = {"targets": 5, "obstacles": 4, "teammates": 7}
    camera_encoder = CameraObsEncoder(
        self_state_dim=9, entity_dims=entity_dims, embed_dim=embed_dim
    ).to(device)
    obs_emb_dim = camera_encoder.out_dim
    joint_act_dim = num_cameras * act_dim

    actor = DeterministicActor(obs_emb_dim, act_dim).to(device)
    target_actor = DeterministicActor(obs_emb_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())

    class CentralizedCritic(nn.Module):
        def __init__(self, obs_emb_dim, joint_act_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_emb_dim * num_cameras + joint_act_dim, hidden),
                nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, obs_joint, joint_action):
            return self.net(torch.cat([obs_joint, joint_action], dim=-1)).squeeze(-1)

    critic = CentralizedCritic(obs_emb_dim, joint_act_dim).to(device)
    target_critic = CentralizedCritic(obs_emb_dim, joint_act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(
        list(camera_encoder.parameters()) + list(actor.parameters()), lr=args.lr)
    critic_opt = torch.optim.Adam(list(critic.parameters()), lr=args.lr)

    replay = ReplayBuffer(args.replay_capacity)
    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    def encode(joint_obs_np):
        # joint_obs_np: (B, C, obs_dim) -> list of (B, obs_emb_dim)
        B = joint_obs_np.shape[0]
        parsed = [observer.parse(joint_obs_np[b]) for b in range(B)]
        embs = []
        for c in range(num_cameras):
            ss = np.stack([parsed[b][c]["self_state"] for b in range(B)])
            ents, msks = {}, {}
            kinds = list(parsed[0][c]["entities"])
            for k in kinds:
                ents[k] = np.stack([parsed[b][c]["entities"][k][0]
                                   for b in range(B)])
                msks[k] = np.stack([parsed[b][c]["entities"][k][1]
                                   for b in range(B)])
            embs.append(camera_encoder(
                torch.as_tensor(ss, dtype=torch.float32, device=device),
                {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                 for k, v in ents.items()},
                {k: torch.as_tensor(v, dtype=torch.bool, device=device)
                 for k, v in msks.items()},
            ))
        return torch.cat(embs, dim=-1)  # (B, C*obs_emb_dim)

    # ---- Collect loop ----
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_len = 0
    last_ep_reward = 0.0
    episode_coverage = 0.0
    last_ep_coverage = 0.0

    for step in range(args.steps):
        obs_np = np.asarray(obs, dtype=np.float64)[None]  # (1, C, obs_dim)
        with torch.no_grad():
            o_joint = encode(obs_np)                       # (1, C*D)
            per_cam = [o_joint.view(1, num_cameras, -1)[0, c]
                       for c in range(num_cameras)]  # list of (D,)
            a_norm = torch.stack([actor(o.unsqueeze(0))
                                 # (C, act_dim)
                                  for o in per_cam], dim=0).squeeze(1)
            a_norm = a_norm.cpu().numpy()
        # Exploration noise
        a_norm = a_norm + \
            np.random.normal(0, args.noise_scale, size=a_norm.shape)
        a_norm = np.clip(a_norm, -1, 1)
        a_env = _denormalize_maddpq(a_norm)

        obs2, reward, done, truncated, info = env.step(a_env)
        replay.add(obs_np[0], a_norm, reward, np.asarray(
            obs2, dtype=np.float64), done or truncated)

        cam_infos = info.get("infos") if isinstance(info, dict) else None
        step_coverage = float(
            np.mean([c.get("coverage_rate", 0.0) for c in cam_infos])) if cam_infos else 0.0

        episode_reward += float(reward)
        episode_coverage += step_coverage
        episode_len += 1
        obs = obs2
        if done or truncated:
            last_ep_reward = episode_reward
            last_ep_coverage = episode_coverage / max(1, episode_len)
            episode_reward = 0.0
            episode_coverage = 0.0
            episode_len = 0
            obs, _ = env.reset()

        # ---- Update ----
        if len(replay) >= args.warmup:
            batch = replay.sample(args.batch_size, device=device)
            o_np = batch["obs"].cpu().numpy()
            no_np = batch["next_obs"].cpu().numpy()

            o_joint = encode(o_np)          # (B, C*D)
            no_joint = encode(no_np)
            a_joint = batch["action"].reshape(
                batch["action"].size(0), -1)  # (B, C*2)

            with torch.no_grad():
                # target next actions
                per_cam_no = [no_joint.view(no_joint.size(
                    0), num_cameras, -1)[:, c] for c in range(num_cameras)]
                a_next = torch.cat([target_actor(o)
                                   for o in per_cam_no], dim=-1)
                q_next = target_critic(no_joint, a_next)
                target = batch["reward"] + args.gamma * \
                    (1.0 - batch["done"]) * q_next

            q = critic(o_joint, a_joint)
            loss_critic = nn.functional.mse_loss(q, target)

            # Actor loss first: maximize Q(o, [mu_1..mu_C]) through the CURRENT
            # critic weights + shared encoder.
            per_cam = [o_joint.view(o_joint.size(0), num_cameras, -1)[:, c]
                       for c in range(num_cameras)]
            joint_mu = torch.cat([actor(o) for o in per_cam], dim=-1)
            # Freeze the critic so the actor loss cannot leak gradients into it
            # (matches the pattern used for the actor/distillation loss in flow.py).
            for p in critic.parameters():
                p.requires_grad_(False)
            loss_actor = -critic(o_joint, joint_mu).mean()

            # Actor before critic keeps the critic untouched while its graph is
            # alive. But actor_opt shares the camera_encoder: its step mutates
            # the encoder in-place, so the earlier `o_joint`/`loss_critic` graph
            # becomes stale. Re-encode freshly for the critic loss to build a
            # valid backward graph.
            actor_opt.zero_grad()
            loss_actor.backward()
            actor_opt.step()
            for p in critic.parameters():
                p.requires_grad_(True)

            o_joint2 = encode(o_np)
            a_joint2 = batch["action"].reshape(batch["action"].size(0), -1)
            q2 = critic(o_joint2, a_joint2)
            loss_critic = nn.functional.mse_loss(q2, target)

            critic_opt.zero_grad()
            loss_critic.backward()
            critic_opt.step()

            # polyak
            for t, s in zip(target_actor.parameters(), actor.parameters()):
                t.data.copy_(args.tau * s.data + (1 - args.tau) * t.data)
            for t, s in zip(target_critic.parameters(), critic.parameters()):
                t.data.copy_(args.tau * s.data + (1 - args.tau) * t.data)

            if wandb_run is not None:
                wandb_run.log({"maddpq/critic_loss": loss_critic.item(),
                               "maddpq/actor_loss": loss_actor.item(),
                               "maddpq/coverage_rate": last_ep_coverage})

        if (step + 1) % args.log_every == 0:
            print(f"step {step+1}/{args.steps} | last_ep_rew {last_ep_reward:.3f} "
                  f"| last_ep_cov {last_ep_coverage:.3f} | buffer {len(replay)}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()


def run_vd(args, mixer_kind):
    """Shared QMIX/QPLEX runner: discrete value-decomposition off-policy.

    Both algorithms use per-agent discrete Q-networks over the shared
    ``CameraObsEncoder`` backbone, an epsilon-greedy joint policy, a replay
    buffer, and a target Q/mixer network. They differ only in the mixer used to
    combine the per-agent ``Q_i`` into a joint ``Q_tot`` (QMIX: monotonic
    hypernetwork; QPLEX: duplex-dueling transformation).
    """
    steps = args.total_timesteps if args.steps is None else args.steps
    args.steps = steps
    from flow import CameraObsEncoder, MateObserver, ReplayBuffer
    from flow import AgentQNetwork, QplexMixer, encode_joint
    from flow.nets import MonotonicMixer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = make_discrete_env(levels=_VD_LEVELS)
    num_cameras = env.unwrapped.num_cameras
    num_targets = env.unwrapped.num_targets
    num_obstacles = env.unwrapped.num_obstacles
    obs_dim = np.asarray(env.reset()[0]).shape[-1]
    n_actions = _VD_NUM_ACTIONS

    embed_dim = args.embed_dim
    entity_dims = {"targets": 5, "obstacles": 4, "teammates": 7}
    camera_encoder = CameraObsEncoder(
        self_state_dim=9, entity_dims=entity_dims, embed_dim=embed_dim
    ).to(device)
    obs_emb_dim = camera_encoder.out_dim
    state_dim = num_cameras * obs_emb_dim

    # Per-agent Q heads over the shared backbone.
    agent_q = nn.ModuleList(
        [AgentQNetwork(obs_emb_dim, n_actions).to(device)
         for _ in range(num_cameras)]
    )
    target_agent_q = nn.ModuleList(
        [AgentQNetwork(obs_emb_dim, n_actions).to(device)
         for _ in range(num_cameras)]
    )

    if mixer_kind == "qmix":
        mixer = MonotonicMixer(num_cameras, state_dim).to(device)
        target_mixer = MonotonicMixer(num_cameras, state_dim).to(device)

        def mix_q(q_sel, g_state):
            return mixer(q_sel, g_state)

        def target_mix_q(q_sel, g_state):
            return target_mixer(q_sel, g_state)
    else:  # qplex
        mixer = QplexMixer(num_cameras, state_dim, n_actions).to(device)
        target_mixer = QplexMixer(num_cameras, state_dim, n_actions).to(device)

        def mix_q(q_sel, g_state):
            raise RuntimeError(
                "QPLEX mixing requires full Q-values and action indices")

        def target_mix_q(q_sel, g_state):
            raise RuntimeError(
                "QPLEX mixing requires full Q-values and action indices")

    # Hard-copy online -> target.
    def hard_update():
        for t, s in zip(target_agent_q.parameters(), agent_q.parameters()):
            t.data.copy_(s.data)
        for t, s in zip(target_mixer.parameters(), mixer.parameters()):
            t.data.copy_(s.data)

    opt = torch.optim.Adam(
        list(camera_encoder.parameters()) + list(agent_q.parameters())
        + list(mixer.parameters()),
        lr=args.lr,
    )

    replay = ReplayBuffer(args.replay_capacity)
    observer = MateObserver(num_cameras, num_targets, num_obstacles)

    def joint_q_values(emb):
        # emb (B, C, obs_emb_dim) -> q_values (B, C, n_actions)
        return torch.stack(
            [agent_q[c](emb[:, c]) for c in range(num_cameras)], dim=1
        )

    def target_joint_q_values(emb):
        return torch.stack(
            [target_agent_q[c](emb[:, c]) for c in range(num_cameras)], dim=1
        )

    def select_actions(emb, eps):
        B = emb.size(0)
        qv = joint_q_values(emb).detach()  # (B, C, n_actions)
        greedy = qv.argmax(dim=-1)  # (B, C)
        rand = torch.randint(0, n_actions, greedy.shape, device=emb.device)
        use_rand = torch.rand(greedy.shape, device=emb.device) < eps
        return torch.where(use_rand, rand, greedy).long()

    # ---- Collect loop ----
    obs, _ = env.reset()  # (C, obs_dim) camera-only joint obs from MultiCamera
    obs_np0 = np.asarray(obs, dtype=np.float64)[None]  # (1, C, obs_dim)
    episode_reward = 0.0
    episode_len = 0
    last_ep_reward = 0.0
    episode_coverage = 0.0
    last_ep_coverage = 0.0

    hard_update()  # sync targets at start

    wandb_run = None
    if args.wandb_api:
        import wandb
        wandb.login(key=args.wandb_api)
        wandb_run = wandb.init(project=f"mate-{mixer_kind}", name=mixer_kind,
                               config=vars(args), reinit=True)

    for step in range(1, args.steps + 1):
        eps = max(args.eps_end, args.eps_start
                  - (step - 1) * (args.eps_start - args.eps_end) / args.eps_decay)
        with torch.no_grad():
            emb = encode_joint(observer, camera_encoder,
                               obs_np0, num_cameras, device)
            a_joint = select_actions(emb, eps)[0]  # (C,) ints
        a_env = a_joint.cpu().numpy()

        obs2, reward, done, truncated, info = env.step(a_env)
        replay.add(obs_np0[0], a_joint.cpu().numpy().astype(np.float64)[:, None],
                   float(reward), np.asarray(obs2, dtype=np.float64), done or truncated)

        cam_infos = info.get("infos") if isinstance(info, dict) else None
        step_coverage = float(
            np.mean([c.get("coverage_rate", 0.0) for c in cam_infos])) if cam_infos else 0.0

        episode_reward += float(reward)
        episode_coverage += step_coverage
        episode_len += 1
        obs = obs2
        obs_np0 = np.asarray(obs, dtype=np.float64)[None]
        if done or truncated:
            last_ep_reward = episode_reward
            last_ep_coverage = episode_coverage / max(1, episode_len)
            episode_reward = 0.0
            episode_coverage = 0.0
            episode_len = 0
            obs, _ = env.reset()
            obs_np0 = np.asarray(obs, dtype=np.float64)[None]

        # ---- Update ----
        if len(replay) >= args.warmup:
            batch = replay.sample(args.batch_size, device=device)
            o_np = batch["obs"].cpu().numpy()
            no_np = batch["next_obs"].cpu().numpy()
            a = batch["action"].long().squeeze(-1)  # (B, C)

            emb = encode_joint(observer, camera_encoder,
                               o_np, num_cameras, device)
            no_emb = encode_joint(observer, camera_encoder,
                                  no_np, num_cameras, device)
            g_state = emb.view(emb.size(0), -1)
            no_g_state = no_emb.view(no_emb.size(0), -1)

            qv = joint_q_values(emb)  # (B, C, n_actions)
            q_sel = torch.gather(
                qv, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)  # (B, C)
            if mixer_kind == "qmix":
                q_tot = mix_q(q_sel, g_state)  # (B,)
            else:
                q_tot = mixer.joint_q_tot(qv, a, g_state)  # (B,)

            with torch.no_grad():
                # Double-DQN: pick next greedy joint action with the ONLINE net,
                # evaluate its value with the TARGET net.
                no_qv_online = joint_q_values(no_emb)  # (B, C, n_actions)
                a_next = no_qv_online.argmax(dim=-1)  # (B, C)
                no_qv_target = target_joint_q_values(
                    no_emb)  # (B, C, n_actions)
                q_sel_target = torch.gather(
                    no_qv_target, dim=-1, index=a_next.unsqueeze(-1)
                ).squeeze(-1)  # (B, C)
                if mixer_kind == "qmix":
                    q_next_tot = target_mix_q(q_sel_target, no_g_state)  # (B,)
                else:
                    q_next_tot = target_mixer.joint_q_tot(
                        no_qv_target, a_next, no_g_state
                    )  # (B,)
                target_val = batch["reward"] + args.gamma * \
                    (1.0 - batch["done"]) * q_next_tot

            loss = nn.functional.mse_loss(q_tot, target_val)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.target_update_every == 0:
                hard_update()

            if wandb_run is not None:
                wandb_run.log({f"{mixer_kind}/loss": loss.item(),
                               f"{mixer_kind}/eps": eps,
                               f"{mixer_kind}/coverage_rate": last_ep_coverage})

        if step % args.log_every == 0:
            print(f"step {step}/{args.steps} | last_ep_rew {last_ep_reward:.3f} "
                  f"| eps {eps:.3f} | buffer {len(replay)}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()


def run_qmix(args):
    run_vd(args, "qmix")


def run_qplex(args):
    run_vd(args, "qplex")


def run_dm2(args):
    """DM²: on-policy IPPO + GAIL distribution matching (discrete camera team)."""
    from mate.algorithms.dm2 import DM2Config, DM2Trainer

    data_paths = None
    if args.dm2_gail_data_paths:
        data_paths = [p.strip() for p in args.dm2_gail_data_paths.split(",") if p.strip()]

    rt = args.dm2_rew_type
    # GAIL reward shaping needs either external demos or the self-imitation flag.
    if rt in ("gail", "mixed"):
        if not data_paths and not args.dm2_gail_sil:
            if args.dm2_update_gail:
                raise SystemExit(
                    "--algo dm2 with rew-type gail/mixed requires either "
                    "--dm2-gail-data-paths or --dm2-gail-sil")
    else:
        args.dm2_update_gail = False

    cfg = DM2Config(
        config=args.dm2_config, levels=args.dm2_levels,
        episode_limit=args.dm2_episode_limit,
        episodes=args.dm2_episodes if args.dm2_episodes is not None
        else args.total_timesteps // args.dm2_episode_limit + 1,
        total_env_steps=args.total_timesteps,
        hidden=args.dm2_hidden, rew_type=rt,
        update_gail=args.dm2_update_gail, gail_rew_coef=args.dm2_gail_rew_coef,
        gail_epoch=args.dm2_gail_epoch, gail_batch_size=args.dm2_gail_batch_size,
        gail_state_discrim=args.dm2_gail_state_discrim,
        gail_sil=args.dm2_gail_sil, gail_buffer_size=args.dm2_gail_buffer_size,
        gail_data_paths=data_paths, gail_exp_eps=args.dm2_gail_exp_eps,
        gail_exp_use_same_data_idxs=args.dm2_gail_use_same_idxs,
        train_after=args.dm2_train_after, ppo_epoch=args.dm2_ppo_epoch,
        clip_param=args.dm2_clip_param, lr=args.dm2_lr,
        gamma=args.dm2_gamma, gae_lambda=args.dm2_gae_lambda,
        device=args.device, seed=args.seed, outdir=args.dm2_outdir,
        wandb_api=args.wandb_api,
    )
    DM2Trainer(cfg).train()


def run_facmac(args):
    """FACMAC: off-policy factored centralised policy gradients (discrete)."""
    from mate.algorithms.facmac import FACMACConfig, FACMACTrainer

    cfg = FACMACConfig(
        config=args.facmac_config, levels=args.facmac_levels,
        episode_limit=args.facmac_episode_limit,
        episodes=args.facmac_episodes if args.facmac_episodes is not None
        else args.total_timesteps // args.facmac_episode_limit + 1,
        total_env_steps=args.total_timesteps,
        hidden=args.facmac_hidden, lr=args.facmac_lr, critic_lr=args.facmac_critic_lr,
        gamma=args.facmac_gamma, td_lambda=args.facmac_td_lambda,
        batch_size=args.facmac_batch_size, buffer_size=args.facmac_buffer_size,
        gumbel_tau=args.facmac_gumbel_tau,
        target_update_interval=args.facmac_target_update_interval,
        epsilon_start=args.facmac_epsilon_start, epsilon_finish=args.facmac_epsilon_finish,
        device=args.device, seed=args.seed, outdir=args.facmac_outdir,
        wandb_api=args.wandb_api,
    )
    FACMACTrainer(cfg).train()


def main():
    parser = ArgumentParser()
    add_common_args(parser)

    parser.add_argument("--actor-lr", type=float, default=5e-5,
                        help="Learning rate for the agent's actor/optimizer.")
    parser.add_argument("--critic-lr", type=float, default=1e-4,
                        help="Learning rate for the agent's critic/optimizer.")

    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments to use for training.")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps to run in each environment per iteration.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training the agent.")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Total number of timesteps to train the agent.")
    parser.add_argument("--save-every", type=int, default=10_000,
                        help="Save the agent every N timesteps.")

    # MADDPG-specific
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-api", default=None,
                        help="W&B API key (or set $WANDB_API_KEY). If not set, no W&B logging.")

    # QMIX/QPLEX (value-decomposition) specific
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Minimum (annealed-to) epsilon.")
    parser.add_argument("--epsilon-decay", type=int, default=20_000,
                        help="Number of steps over which epsilon anneals start->end.")
    parser.add_argument("--target-update-every", type=int, default=200,
                        help="Hard-copy online -> target networks every N gradient steps.")
    parser.add_argument("--rode-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--rode-levels", type=int, default=3)
    parser.add_argument("--rode-episode-limit", type=int, default=64)
    parser.add_argument("--rode-episodes", type=int, default=None)
    parser.add_argument("--rode-batch-size", type=int, default=8)
    parser.add_argument("--rode-warmup-episodes", type=int, default=8)
    parser.add_argument("--rode-n-roles", type=int, default=3)
    parser.add_argument("--rode-role-interval", type=int, default=5)
    parser.add_argument("--rode-outdir", default="ckpts/rode")
    parser.add_argument("--coma-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--coma-levels", type=int, default=3)
    parser.add_argument("--coma-episode-limit", type=int, default=64)
    parser.add_argument("--coma-episodes", type=int, default=None)
    parser.add_argument("--coma-update-every", type=int, default=8)
    parser.add_argument("--coma-batch-size", type=int, default=8)
    parser.add_argument("--coma-outdir", default="ckpts/coma")
    parser.add_argument("--mat-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--mat-levels", type=int, default=3)
    parser.add_argument("--mat-episode-limit", type=int, default=64)
    parser.add_argument("--mat-updates", type=int, default=None)
    parser.add_argument("--mat-rollout-steps", type=int, default=128)
    parser.add_argument("--mat-ppo-epochs", type=int, default=4)
    parser.add_argument("--mat-minibatch-size", type=int, default=64)
    parser.add_argument("--mat-d-model", type=int, default=128)
    parser.add_argument("--mat-n-heads", type=int, default=4)
    parser.add_argument("--mat-n-layers", type=int, default=2)
    parser.add_argument("--mat-outdir", default="ckpts/mat")
    parser.add_argument("--dcg-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--dcg-levels", type=int, default=3)
    parser.add_argument("--dcg-episode-limit", type=int, default=64)
    parser.add_argument("--dcg-episodes", type=int, default=None)
    parser.add_argument("--dcg-update-every", type=int, default=8)
    parser.add_argument("--dcg-batch-size", type=int, default=8)
    parser.add_argument("--dcg-hidden", type=int, default=64)
    parser.add_argument("--dcg-cg-edges", default="full")
    parser.add_argument("--dcg-payoff-rank", type=int, default=0)
    parser.add_argument("--dcg-duelling", action="store_true")
    parser.add_argument("--dcg-msg-iterations", type=int, default=8)
    parser.add_argument("--dcg-lr", type=float, default=5e-4)
    parser.add_argument("--dcg-outdir", default="ckpts/dcg")
    parser.add_argument("--i2c-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--i2c-levels", type=int, default=3)
    parser.add_argument("--i2c-episode-limit", type=int, default=64)
    parser.add_argument("--i2c-episodes", type=int, default=None)
    parser.add_argument("--i2c-update-every", type=int, default=8)
    parser.add_argument("--i2c-batch-size", type=int, default=32)
    parser.add_argument("--i2c-hidden", type=int, default=128)
    parser.add_argument("--i2c-msg-dim", type=int, default=64)
    parser.add_argument("--i2c-tau", type=float, default=1.0)
    parser.add_argument("--i2c-blz-coef", type=float, default=0.01)
    parser.add_argument("--i2c-epsilon", type=float, default=0.05)
    parser.add_argument("--i2c-actor-lr", type=float, default=1e-4)
    parser.add_argument("--i2c-critic-lr", type=float, default=1e-3)
    parser.add_argument("--i2c-outdir", default="ckpts/i2c")
    parser.add_argument("--dm2-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--dm2-levels", type=int, default=3)
    parser.add_argument("--dm2-episode-limit", type=int, default=64)
    parser.add_argument("--dm2-episodes", type=int, default=None)
    parser.add_argument("--dm2-hidden", type=int, default=64)
    parser.add_argument("--dm2-rew-type", type=str, default="mixed",
                        choices=["env", "gail", "mixed"])
    parser.add_argument("--dm2-update-gail", action="store_true")
    parser.add_argument("--dm2-gail-rew-coef", type=float, default=1.0)
    parser.add_argument("--dm2-gail-epoch", type=int, default=3)
    parser.add_argument("--dm2-gail-batch-size", type=int, default=8)
    parser.add_argument("--dm2-gail-state-discrim", action="store_true")
    parser.add_argument("--dm2-gail-sil", action="store_true")
    parser.add_argument("--dm2-gail-buffer-size", type=int, default=100)
    parser.add_argument("--dm2-gail-data-paths", type=str, default=None,
                        help="Comma-separated expert demo dirs for GAIL.")
    parser.add_argument("--dm2-gail-exp-eps", type=int, default=1000)
    parser.add_argument("--dm2-gail-use-same-idxs", action="store_true")
    parser.add_argument("--dm2-train-after", type=int, default=16)
    parser.add_argument("--dm2-ppo-epoch", type=int, default=15)
    parser.add_argument("--dm2-clip-param", type=float, default=0.2)
    parser.add_argument("--dm2-lr", type=float, default=3e-4)
    parser.add_argument("--dm2-gamma", type=float, default=0.99)
    parser.add_argument("--dm2-gae-lambda", type=float, default=0.95)
    parser.add_argument("--dm2-outdir", default="ckpts/dm2")
    parser.add_argument("--facmac-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--facmac-levels", type=int, default=3)
    parser.add_argument("--facmac-episode-limit", type=int, default=64)
    parser.add_argument("--facmac-episodes", type=int, default=None)
    parser.add_argument("--facmac-hidden", type=int, default=64)
    parser.add_argument("--facmac-lr", type=float, default=2.5e-3)
    parser.add_argument("--facmac-critic-lr", type=float, default=5e-4)
    parser.add_argument("--facmac-gamma", type=float, default=0.99)
    parser.add_argument("--facmac-td-lambda", type=float, default=0.8)
    parser.add_argument("--facmac-batch-size", type=int, default=32)
    parser.add_argument("--facmac-buffer-size", type=int, default=5000)
    parser.add_argument("--facmac-gumbel-tau", type=float, default=1.0)
    parser.add_argument("--facmac-target-update-interval", type=int, default=200)
    parser.add_argument("--facmac-epsilon-start", type=float, default=0.5)
    parser.add_argument("--facmac-epsilon-finish", type=float, default=0.05)
    parser.add_argument("--facmac-outdir", default="ckpts/facmac")
    parser.add_argument("--iql-config", default="MATE-4v8-9.yaml")
    parser.add_argument("--iql-levels", type=int, default=3)
    parser.add_argument("--iql-episode-limit", type=int, default=64)
    parser.add_argument("--iql-episodes", type=int, default=None)
    parser.add_argument("--iql-update-every", type=int, default=8)
    parser.add_argument("--iql-batch-size", type=int, default=8)
    parser.add_argument("--iql-replay-capacity", type=int, default=128)
    parser.add_argument("--iql-hidden", type=int, default=128)
    parser.add_argument("--iql-lr", type=float, default=1e-3)
    parser.add_argument("--iql-gamma", type=float, default=0.99)
    parser.add_argument("--iql-target-update-every", type=int, default=10)
    parser.add_argument("--iql-epsilon-start", type=float, default=1.0)
    parser.add_argument("--iql-epsilon-end", type=float, default=0.05)
    parser.add_argument("--iql-epsilon-decay", type=int, default=500)
    parser.add_argument("--iql-outdir", default="ckpts/iql")

    args = parser.parse_args()
    if args.wandb_api is None:
        args.wandb_api = os.environ.get("WANDB_API_KEY")

    dispatcher = {
        "ppo": run_ppo,
        "mappo": run_mappo,
        "maddpq": run_maddpq,
        "qmix": run_qmix,
        "qplex": run_qplex,
    }
    from mate.algorithms.rode import run_rode
    from mate.algorithms.coma import run_coma
    from mate.algorithms.mat import run_mat
    from mate.algorithms.dcg import run_dcg
    from mate.algorithms.i2c import run_i2c
    from mate.algorithms.iql import run_iql
    dispatcher["rode"] = run_rode
    dispatcher["coma"] = run_coma
    dispatcher["mat"] = run_mat
    dispatcher["dcg"] = run_dcg
    dispatcher["i2c"] = run_i2c
    dispatcher["iql"] = run_iql
    dispatcher["dm2"] = run_dm2
    dispatcher["facmac"] = run_facmac
    dispatcher[args.algo](args)


if __name__ == "__main__":
    main()
