"""MAMBA learner using the upstream RSSM, imagination and PPO objectives."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .core.action import Actor
from .core.config import DreamerConfig
from .core.critic import AugmentedCritic
from .core.loss import actor_loss, actor_rollout, model_loss, value_loss
from .core.loss_utils import advantage
from .core.model import DreamerModel
from .environment import MAMBAEnvironment
from .replay import SequenceReplay


@dataclass
class MAMBAConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    sequence_length: int = 16
    batch_size: int = 8
    replay_capacity: int = 100_000  # transitions, allocated only as collected
    warmup: int = 256
    updates_per_episode: int = 1
    horizon: int = 15
    ppo_epochs: int = 5
    hidden: int = 256
    categoricals: int = 32
    classes: int = 32
    model_lr: float = 2e-4
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    lambda_: float = 0.95
    entropy: float = 0.001
    entropy_decay: float = 0.99998
    grad_clip: float = 100.0
    device: str = "cpu"
    seed: int = 0
    num_envs: int = 1
    log_every: int = 10  # episodes
    save_every: int = 100  # episodes
    outdir: str = "ckpts/mamba"
    resume: str | None = None
    wandb_api: str | None = None

    def __post_init__(self):
        positive = ("episodes", "batch_size", "updates_per_episode", "ppo_epochs", "hidden",
                    "categoricals", "log_every", "save_every", "grad_clip")
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("Counts, dimensions and gradient clip must be positive")
        if self.hidden % 8 or self.classes < 2 or self.levels < 2:
            raise ValueError("hidden must be divisible by 8; classes and levels must be >= 2")
        if not 2 <= self.sequence_length <= self.episode_limit <= self.replay_capacity:
            raise ValueError("Require 2 <= sequence_length <= episode_limit <= replay_capacity")
        if self.horizon < 2 or not 0 <= self.warmup <= self.replay_capacity:
            raise ValueError("Require horizon >= 2 and 0 <= warmup <= replay_capacity")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lambda_ <= 1:
            raise ValueError("gamma and lambda must be in [0, 1]")
        if min(self.model_lr, self.actor_lr, self.critic_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.entropy < 0 or not 0 <= self.entropy_decay <= 1:
            raise ValueError("Require entropy >= 0 and entropy_decay in [0, 1]")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class MAMBATrainer:
    def __init__(self, config: MAMBAConfig):
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.env = MAMBAEnvironment(config.config, config.levels, config.episode_limit)
        cfg = self.model_config = DreamerConfig()
        for key in ("HIDDEN", "MODEL_HIDDEN", "EMBED", "DETERMINISTIC", "VALUE_HIDDEN",
                    "PCONT_HIDDEN", "ACTION_HIDDEN", "REWARD_HIDDEN"):
            setattr(cfg, key, config.hidden)
        cfg.N_CATEGORICALS, cfg.N_CLASSES = config.categoricals, config.classes
        cfg.STOCHASTIC = config.categoricals * config.classes
        cfg.FEAT = cfg.STOCHASTIC + cfg.DETERMINISTIC
        cfg.IN_DIM, cfg.ACTION_SIZE = self.env.obs_dim, self.env.n_actions
        cfg.HORIZON, cfg.GAMMA, cfg.DISCOUNT_LAMBDA = config.horizon, config.gamma, config.lambda_
        self.model = DreamerModel(cfg).to(self.device)
        self.actor = Actor(cfg.FEAT, cfg.ACTION_SIZE, cfg.ACTION_HIDDEN, cfg.ACTION_LAYERS).to(self.device)
        self.critic = AugmentedCritic(cfg.FEAT, cfg.HIDDEN).to(self.device)
        for module in (self.model, self.critic):
            for parameter in module.parameters():
                if parameter.ndim >= 2:
                    torch.nn.init.xavier_uniform_(parameter)
        for parameter in self.actor.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.orthogonal_(parameter)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = SequenceReplay(config.replay_capacity, config.sequence_length, self.env.n_actions, config.seed)
        self.entropy = config.entropy
        self.episode = self.steps = self.updates = 0
        self.model.eval()
        self.rollout = None
        if config.num_envs > 1:
            from mate.algorithms.batched_rollout import BatchedRollout
            self.rollout = BatchedRollout(
                lambda: _make_mamba_vec_env(config), config.num_envs)
        if config.resume:
            self.load(config.resume)

    @torch.no_grad()
    def act(self, obs, previous_action=None, state=None, greedy=False):
        observation = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        state = self.model(observation, previous_action, state)
        action, logits = self.actor(state.get_features())
        if greedy:
            action = torch.nn.functional.one_hot(logits.argmax(-1), self.env.n_actions).float()
        return action[0].argmax(-1).cpu().numpy(), action, state

    def collect_episode(self, index, *, training=True):
        obs, _ = self.env.reset(seed=self.config.seed + index)
        observations, actions, rewards, terminals, coverage = [obs], [], [], [], []
        previous_action = state = None
        for _ in range(self.config.episode_limit):
            action, previous_action, state = self.act(obs, previous_action, state, greedy=not training)
            if training and len(self.replay) < self.config.warmup:
                action = np.random.randint(self.env.n_actions, size=self.env.n_agents)
                previous_action = torch.nn.functional.one_hot(
                    torch.as_tensor(action, device=self.device), self.env.n_actions
                ).float().unsqueeze(0)
            obs, reward, terminated, truncated, info = self.env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated)  # time limits are not terminal MDP states
            infos = info.get("infos", []) if isinstance(info, dict) else []
            values = [float(item["coverage_rate"]) for item in infos if "coverage_rate" in item]
            if values:
                coverage.append(float(np.mean(values)))
            if terminated or truncated:
                break
        if training:
            self.replay.add(observations, actions, rewards, terminals)
            self.steps += len(actions)
        return {"reward": float(sum(rewards)), "length": len(actions),
                "coverage": float(np.mean(coverage)) if coverage else 0.0}

    @torch.no_grad()
    def batched_act(self, observations, prev_acts, states, training=True):
        """Actions for all slots from their obs + per-slot RSSM state + previous
        action (parent process). Mirrors ``collect_episode``'s ``act`` and the
        warmup random override; returns updated prev_acts and states."""
        actions, new_prev_acts, new_states = [], [], []
        for i, obs in enumerate(observations):
            if training and len(self.replay) < self.config.warmup:
                action = np.random.randint(self.env.n_actions, size=self.env.n_agents)
                prev_action = torch.nn.functional.one_hot(
                    torch.as_tensor(action, device=self.device), self.env.n_actions
                ).float().unsqueeze(0)
                state = states[i]
            else:
                action, diff_action, state = self.act(
                    obs, prev_acts[i], states[i], greedy=not training)
                prev_action = diff_action
            actions.append(action)
            new_prev_acts.append(prev_action)
            new_states.append(state)
        return np.stack(actions), new_prev_acts, new_states

    def collect_batch(self, first_ep, training=True):
        """Concurrently collect ``num_envs`` episodes in parallel workers.

        Episodes complete independently per slot (mirroring the single-env
        path); each finished episode is added to replay via ``SequenceReplay.add``
        exactly as ``collect_episode`` does (T+1 observations, terminal on the
        last transition only)."""
        num = self.rollout.num_envs
        obs_list = self.rollout.reset(
            [self.config.seed + first_ep + i for i in range(num)])
        prev_acts = [None] * num
        states = [None] * num
        # per-slot current-episode buffers
        ep_obs = [[obs_list[i]] for i in range(num)]
        ep_actions = [[] for _ in range(num)]
        ep_rewards = [[] for _ in range(num)]
        ep_terminals = [[] for _ in range(num)]
        ep_rewards_sum = np.zeros(num)
        ep_length = np.zeros(num, dtype=np.int64)
        coverages = [[] for _ in range(num)]
        n_episodes = [0] * num
        done = [False] * num
        for _ in range(self.config.episode_limit):
            actions, prev_acts, states = self.batched_act(
                obs_list, prev_acts, states, training=training)
            step = self.rollout.step(actions)
            for i in range(num):
                ep_actions[i].append(actions[i])
                ep_rewards[i].append(float(step.rewards[i]))
                ep_terminals[i].append(bool(step.done[i]))
                ep_rewards_sum[i] += float(step.rewards[i])
                ep_length[i] += 1
                infos = step.infos[i].get("infos", []) if isinstance(step.infos[i], dict) else []
                values = [float(item["coverage_rate"]) for item in infos if "coverage_rate" in item]
                if values:
                    coverages[i].append(float(np.mean(values)))
                if step.done[i]:
                    # terminal obs for the finished episode, then finalize it
                    ep = ep_obs[i] + [step.obs[i]]
                    if training:
                        # enforce a single terminal boundary at the final step
                        term = np.asarray(ep_terminals[i], dtype=np.float32)
                        self.replay.add(ep, ep_actions[i], ep_rewards[i], term)
                        n_episodes[i] += 1
                    # start the continuation episode from the post-reset obs
                    ep_obs[i] = [step.obs[i]]
                    ep_actions[i] = []
                    ep_rewards[i] = []
                    ep_terminals[i] = []
                    prev_acts[i] = None
                    states[i] = None
                else:
                    ep_obs[i].append(step.obs[i])
                done[i] = done[i] or step.done[i]
            obs_list = step.obs
            if all(done):
                break
        if training:
            self.steps += int(ep_length.sum())
        reward = float(np.mean([r for r in ep_rewards_sum]))
        length = float(np.mean([l for l in ep_length]))
        cov = float(np.mean([np.mean(c) if c else 0.0 for c in coverages]))
        return {"reward": reward, "length": length, "coverage": cov}

    def _optimize(self, optimizer, module, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("MAMBA produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_clip, error_if_nonfinite=True)
        optimizer.step()

    def update(self):
        batch = self.replay.sample(self.config.batch_size, self.device)
        self.model.train()
        loss = model_loss(self.model_config, self.model, batch["observation"], batch["action"],
                          None, batch["reward"], batch["done"], batch["fake"], batch["last"])
        self._optimize(self.model_optimizer, self.model, loss)
        self.model.eval()
        # Policies and values learn from imagined transitions only. Detach the
        # rollout before PPO so policy steps cannot mutate a live model graph.
        with torch.no_grad():
            actions, avail, old_policy, features, returns = actor_rollout(
                batch["observation"], batch["action"], batch["last"], self.model,
                self.actor, self.target_critic, self.model_config)
            advantages = advantage(returns - self.critic(features))
        actor_losses, critic_losses = [], []
        for _ in range(self.config.ppo_epochs):
            for indices in torch.randperm(len(actions), device=self.device).split(2000):
                policy_loss = actor_loss(features[indices], actions[indices], avail,
                                         old_policy[indices], advantages[indices], self.actor, self.entropy)
                self._optimize(self.actor_optimizer, self.actor, policy_loss)
                critic_loss = value_loss(self.critic, features[indices], returns[indices])
                self._optimize(self.critic_optimizer, self.critic, critic_loss)
                actor_losses.append(policy_loss.item())
                critic_losses.append(critic_loss.item())
                self.entropy *= self.config.entropy_decay
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.updates += 1
        return {"model_loss": loss.item(), "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)), "imagined_return": returns.mean().item()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Credentials and machine-specific resume paths do not belong in a checkpoint.
        config = asdict(self.config)
        config.pop("wandb_api")
        config["resume"] = None
        payload = {"format_version": 1, "config": config, "episode": self.episode,
                   "steps": self.steps, "updates": self.updates, "entropy": self.entropy,
                   "env_spec": (self.env.n_agents, self.env.obs_dim, self.env.n_actions)}
        for name in ("model", "actor", "critic", "target_critic", "model_optimizer", "actor_optimizer", "critic_optimizer"):
            payload[name] = getattr(self, name).state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("Unsupported MAMBA checkpoint version")
        if tuple(payload["env_spec"]) != (self.env.n_agents, self.env.obs_dim, self.env.n_actions):
            raise ValueError("Checkpoint camera, observation or action dimensions do not match")
        for key in ("hidden", "categoricals", "classes", "levels", "config"):
            if payload["config"][key] != getattr(self.config, key):
                raise ValueError(f"Checkpoint setting {key!r} does not match")
        for name in ("model", "actor", "critic", "target_critic", "model_optimizer", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.episode, self.steps, self.updates, self.entropy = (
            payload[key] for key in ("episode", "steps", "updates", "entropy"))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = dict(payload["config"], device=device, resume=str(path))
        return cls(MAMBAConfig(**config))

    def train(self):
        run = None
        try:
            if self.config.wandb_api:
                import wandb
                wandb.login(key=self.config.wandb_api)
                config = asdict(self.config)
                config.pop("wandb_api")
                run = wandb.init(project="mate-mamba", config=config)
            if self.config.resume:
                print("Loaded learner checkpoint; collecting fresh replay before updates.", flush=True)
            end = self.episode + self.config.episodes
            while self.episode < end:
                if self.rollout is not None:
                    metrics = self.collect_batch(self.episode)
                    self.episode += self.rollout.num_envs
                else:
                    metrics = self.collect_episode(self.episode)
                    self.episode += 1
                if len(self.replay) >= self.config.warmup and self.replay.ready:
                    for _ in range(self.config.updates_per_episode):
                        metrics.update(self.update())
                if self.episode % self.config.log_every == 0:
                    print(f"mamba episode {self.episode} | steps {self.steps} | updates {self.updates} | "
                          + " | ".join(f"{key} {value:.4f}" for key, value in metrics.items()), flush=True)
                if run is not None:
                    run.log(metrics, step=self.steps)
                if self.episode % self.config.save_every == 0:
                    self.save(Path(self.config.outdir) / f"episode_{self.episode}.pt")
            self.save(Path(self.config.outdir) / "latest.pt")
            if self.updates == 0:
                print("No learner updates yet: increase episodes or reduce warmup/sequence length.", flush=True)
        finally:
            self.env.close()
            if self.rollout is not None:
                self.rollout.close()
            if run is not None:
                run.finish()


def _make_mamba_vec_env(config) -> "MateCameraVecEnv":
    """Top-level, cloudpickle-safe env factory for the batched rollout worker."""
    from mate.algorithms.vec_adapter import MateCameraVecEnv
    env = MAMBAEnvironment(config.config, config.levels, config.episode_limit)
    return MateCameraVecEnv(env, continuous=False)


def add_mamba_args(parser):
    defaults = MAMBAConfig()
    fields = ("config", "levels", "episode_limit", "episodes", "sequence_length", "batch_size",
              "replay_capacity", "warmup", "updates_per_episode", "horizon", "ppo_epochs", "hidden",
              "categoricals", "classes", "model_lr", "actor_lr", "critic_lr", "gamma", "lambda_",
              "entropy", "entropy_decay", "grad_clip", "log_every", "save_every", "outdir",
              "num_envs")
    group = parser.add_argument_group("MAMBA")
    for name in fields:
        default = getattr(defaults, name)
        flag = "--mamba-" + name.rstrip("_").replace("_", "-")
        group.add_argument(flag, dest="mamba_" + name, type=type(default), default=default)
    group.add_argument("--mamba-resume", default=None)


def run_mamba(args):
    kwargs = {key.removeprefix("mamba_"): value for key, value in vars(args).items() if key.startswith("mamba_")}
    kwargs.update(device=args.device, seed=args.seed, wandb_api=args.wandb_api)
    MAMBATrainer(MAMBAConfig(**kwargs)).train()
