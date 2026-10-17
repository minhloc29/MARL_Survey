"""MAMBPO with local continuous actors, central critics and one shared model."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from mate.algorithms.camera_environment import ContinuousCameraEnvironment
from mate.algorithms.model_based_replay import TransitionReplay, mixed_batch
from .networks import MAMBPOAgent, WorldModel


@dataclass
class MAMBPOConfig:
    config: str = "MATE-4v8-9.yaml"
    episodes: int = 1000
    episode_limit: int = 64
    actor_hidden: int = 128
    critic_hidden: int = 256
    model_hidden: int = 200
    ensemble_size: int = 10
    batch_size: int = 64
    model_batch_size: int = 128
    model_steps: int = 20
    model_update_every: int = 250
    update_every: int = 10
    policy_updates: int = 10
    actor_delay: int = 2
    replay_capacity: int = 100_000
    model_capacity: int = 10_000
    warmup: int = 500
    rollout_batch_size: int = 256
    rollout_length: int = 1
    real_ratio: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    model_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.1
    autotune: bool = True
    target_entropy: float = -2.0
    grad_clip: float = 0.5
    device: str = "cpu"
    seed: int = 0
    log_every: int = 10
    save_every: int = 100
    outdir: str = "ckpts/mambpo"
    resume: str | None = None
    wandb_api: str | None = None

    def __post_init__(self):
        for key in ("episodes", "episode_limit", "actor_hidden", "critic_hidden", "model_hidden",
                    "ensemble_size", "batch_size", "model_batch_size", "model_steps", "model_update_every",
                    "update_every", "policy_updates", "actor_delay", "model_capacity", "rollout_batch_size",
                    "rollout_length", "log_every", "save_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.replay_capacity < 5 or not 0 <= self.warmup <= self.replay_capacity:
            raise ValueError("Require replay_capacity >= 5 and warmup within capacity")
        for key in ("gamma", "tau", "real_ratio"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        for key in ("actor_lr", "critic_lr", "model_lr", "alpha", "grad_clip"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class MAMBPOTrainer:
    def __init__(self, config: MAMBPOConfig):
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.env = ContinuousCameraEnvironment(config.config, config.episode_limit)
        self.agents = torch.nn.ModuleList([
            MAMBPOAgent(self.env.n_agents, self.env.obs_dim, self.env.action_dim,
                        config.actor_hidden, config.critic_hidden, config.alpha)
            for _ in range(self.env.n_agents)
        ]).to(self.device)
        self.model = WorldModel(self.env.n_agents, self.env.obs_dim, self.env.action_dim,
                                config.model_hidden, config.ensemble_size).to(self.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.model_lr, amsgrad=True)
        self.optimizers = []
        for agent in self.agents:
            opts = {"actor": torch.optim.Adam(agent.actor.parameters(), lr=config.actor_lr, weight_decay=1e-4),
                    "critics": torch.optim.Adam(agent.critics.parameters(), lr=config.critic_lr)}
            if config.autotune:
                opts["alpha"] = torch.optim.Adam([agent.log_alpha], lr=config.actor_lr)
            else:
                agent.log_alpha.requires_grad_(False)
            self.optimizers.append(opts)
        self.real_replay = TransitionReplay(config.replay_capacity, config.seed, continuous_actions=True)
        self.model_replay = TransitionReplay(config.model_capacity, config.seed + 1, continuous_actions=True)
        self.model_train = TransitionReplay(config.replay_capacity, config.seed + 2, continuous_actions=True)
        self.model_validation = TransitionReplay(max(1, config.replay_capacity // 5), config.seed + 3, continuous_actions=True)
        self.steps = self.episode = self.updates = self.actor_updates = self.model_updates = 0
        self.last_model_step = -1
        self.needs_refit = True
        if config.resume:
            try:
                self.load(config.resume)
            except Exception:
                self.env.close()
                raise

    @property
    def ready(self):
        return (len(self.real_replay) >= self.config.warmup
                and len(self.model_train) >= 2 and len(self.model_validation) > 0)

    @torch.no_grad()
    def act(self, observations, greedy=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if obs.shape != (self.env.n_agents, self.env.obs_dim):
            raise ValueError("Expected one observation per camera")
        # Each actor sees only its own observation, including at evaluation.
        return torch.stack([agent.actor(obs[i:i + 1], greedy=greedy)[0][0]
                            for i, agent in enumerate(self.agents)]).cpu().numpy()

    def collect_episode(self, index, training=True, learn=False):
        obs, _ = self.env.reset(seed=self.config.seed + index)
        reward_sum, coverage, metrics = 0.0, [], {}
        for step in range(self.config.episode_limit):
            if training and len(self.real_replay) < self.config.warmup:
                actions = self.rng.uniform(-1, 1, (self.env.n_agents, self.env.action_dim)).astype(np.float32)
            else:
                actions = self.act(obs, greedy=not training)
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            if training:
                transition = (obs, actions, np.full(self.env.n_agents, reward), next_obs, terminated)
                self.real_replay.add(*transition)
                destination = self.model_validation if self.steps % 5 == 0 else self.model_train
                destination.add(*transition)
                self.steps += 1
                if learn and self.ready and self.steps % self.config.update_every == 0:
                    metrics.update(self.update())
            obs = next_obs
            reward_sum += reward
            infos = info.get("infos", []) if isinstance(info, dict) else []
            values = [float(item["coverage_rate"]) for item in infos if "coverage_rate" in item]
            if values:
                coverage.append(float(np.mean(values)))
            if terminated or truncated:
                break
        return dict(metrics, reward=reward_sum, length=step + 1,
                    coverage=float(np.mean(coverage)) if coverage else 0.)

    @staticmethod
    def _optimize(optimizer, parameters, loss, clip):
        if not torch.isfinite(loss):
            raise FloatingPointError("MAMBPO produced a non-finite loss")
        parameters = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()

    def fit_model(self):
        losses = []
        for _ in range(self.config.model_steps):
            batch = self.model_train.sample(self.config.model_batch_size, self.device)
            loss = self.model.objective(batch)
            losses.append(self._optimize(self.model_optimizer, self.model.parameters(), loss, 100.0))
        validation = self.model_validation.latest(self.config.model_batch_size, self.device)
        error = self.model.validation_error(validation)
        if not torch.isfinite(error).all():
            raise FloatingPointError("Non-finite world-model validation error")
        self.model_updates += 1
        self.last_model_step, self.needs_refit = self.steps, False
        return {"model_loss": float(np.mean(losses)), "model_validation_mse": error.mean().item()}

    @torch.no_grad()
    def rollout_model(self, starts=None):
        if starts is None:
            starts = self.real_replay.sample(self.config.rollout_batch_size, self.device)["obs"]
        obs = starts.clone()
        self.model_replay.clear()
        generated = 0
        for _ in range(self.config.rollout_length):
            actions = torch.stack([agent.actor(obs[:, i])[0] for i, agent in enumerate(self.agents)], dim=1)
            next_obs, rewards, terminated = self.model.predict(obs, actions)
            if not all(torch.isfinite(x).all() for x in (next_obs, rewards, terminated)):
                raise FloatingPointError("Non-finite MAMBPO model rollout")
            self.model_replay.extend(dict(obs=obs, actions=actions, rewards=rewards,
                                          next_obs=next_obs, terminated=terminated))
            generated += len(obs)
            obs = next_obs[terminated == 0]
            if not len(obs):
                break
        return generated

    def update_critics(self, batch):
        cfg = self.config
        with torch.no_grad():
            samples = [agent.target_actor(batch["next_obs"][:, i]) for i, agent in enumerate(self.agents)]
            next_actions = torch.stack([item[0] for item in samples], dim=1)
            next_log_probs = torch.stack([item[1] for item in samples], dim=1)
        losses = []
        for i, (agent, opts) in enumerate(zip(self.agents, self.optimizers)):
            with torch.no_grad():
                next_q = agent.q_values(batch["next_obs"], next_actions, target=True).min(0).values
                next_value = next_q - agent.log_alpha.exp() * next_log_probs[:, i]
                target = batch["rewards"][:, i] + cfg.gamma * (1 - batch["terminated"]) * next_value
            loss = (agent.q_values(batch["obs"], batch["actions"]) - target.unsqueeze(0)).square().mean()
            losses.append(self._optimize(opts["critics"], agent.critics.parameters(), loss, cfg.grad_clip))
        return float(np.mean(losses))

    def update_actors(self, batch):
        losses = []
        for i, (agent, opts) in enumerate(zip(self.agents, self.optimizers)):
            action, log_prob = agent.actor(batch["obs"][:, i])
            actions = batch["actions"].detach().clone()
            actions[:, i] = action  # hold all other agents' replay actions fixed
            agent.critics.requires_grad_(False)
            try:
                q = agent.q_values(batch["obs"], actions).min(0).values
                loss = (agent.log_alpha.exp().detach() * log_prob - q).mean()
                losses.append(self._optimize(opts["actor"], agent.actor.parameters(), loss, self.config.grad_clip))
            finally:
                agent.critics.requires_grad_(True)
            if self.config.autotune:
                loss_alpha = -(agent.log_alpha * (log_prob.detach() + self.config.target_entropy)).mean()
                self._optimize(opts["alpha"], [agent.log_alpha], loss_alpha, self.config.grad_clip)
        with torch.no_grad():
            for agent in self.agents:
                for online, target in ((agent.actor, agent.target_actor), (agent.critics, agent.target_critics)):
                    for source, destination in zip(online.parameters(), target.parameters()):
                        destination.lerp_(source, self.config.tau)
        self.actor_updates += 1
        return float(np.mean(losses))

    def update(self):
        if not self.ready:
            raise ValueError("Collect warmup data and disjoint model-training/validation samples first")
        metrics = {}
        if self.needs_refit or self.steps - self.last_model_step >= self.config.model_update_every:
            metrics.update(self.fit_model())
        metrics["model_transitions"] = self.rollout_model()
        critics, actors = [], []
        for _ in range(self.config.policy_updates):
            batch = mixed_batch(self.real_replay, self.model_replay, self.config.batch_size,
                                self.config.real_ratio, self.device)
            critics.append(self.update_critics(batch))
            self.updates += 1
            if self.updates % self.config.actor_delay == 0:
                actors.append(self.update_actors(batch))
        metrics["critic_loss"] = float(np.mean(critics))
        if actors:
            metrics["actor_loss"] = float(np.mean(actors))
        metrics["alpha"] = float(np.mean([agent.log_alpha.detach().exp().item() for agent in self.agents]))
        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        config.pop("wandb_api")
        config["resume"] = None
        payload = {"format_version": 1, "config": config, "agents": self.agents.state_dict(),
                   "model": self.model.state_dict(), "model_optimizer": self.model_optimizer.state_dict(),
                   "optimizers": [{key: opt.state_dict() for key, opt in opts.items()} for opts in self.optimizers],
                   "env_spec": (self.env.n_agents, self.env.obs_dim, self.env.action_dim),
                   **{key: getattr(self, key) for key in ("episode", "steps", "updates", "actor_updates",
                                                        "model_updates", "last_model_step")}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path):
        payload = torch.load(path, weights_only=True, map_location=self.device)
        if payload["format_version"] != 1 or tuple(payload["env_spec"]) != (
                self.env.n_agents, self.env.obs_dim, self.env.action_dim):
            raise ValueError("Checkpoint version or environment dimensions do not match")
        for key in ("config", "actor_hidden", "critic_hidden", "model_hidden", "ensemble_size", "autotune"):
            if payload["config"][key] != getattr(self.config, key):
                raise ValueError(f"Checkpoint setting {key!r} does not match")
        self.agents.load_state_dict(payload["agents"])
        self.model.load_state_dict(payload["model"])
        self.model_optimizer.load_state_dict(payload["model_optimizer"])
        for opts, state in zip(self.optimizers, payload["optimizers"], strict=True):
            for key, optimizer in opts.items():
                optimizer.load_state_dict(state[key])
        for key in ("episode", "steps", "updates", "actor_updates", "model_updates", "last_model_step"):
            setattr(self, key, payload[key])
        self.needs_refit = True

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        return cls(MAMBPOConfig(**dict(payload["config"], device=device, resume=str(path))))

    def train(self):
        run = None
        try:
            if self.config.wandb_api:
                import wandb
                wandb.login(key=self.config.wandb_api)
                config = asdict(self.config)
                config.pop("wandb_api")
                run = wandb.init(project="mate-mambpo", config=config)
            if self.config.resume:
                print("Loaded MAMBPO learner; collecting fresh replay and refitting before model rollouts.", flush=True)
            for index in range(self.episode, self.episode + self.config.episodes):
                metrics = self.collect_episode(index, learn=True)
                self.episode = index + 1
                if self.episode % self.config.log_every == 0:
                    print(f"mambpo episode {self.episode} | steps {self.steps} | critic updates {self.updates} "
                          f"| actor updates {self.actor_updates} | model fits {self.model_updates} | "
                          + " | ".join(f"{key} {value:.4f}" for key, value in metrics.items()), flush=True)
                if run is not None:
                    run.log(metrics, step=self.steps)
                if self.episode % self.config.save_every == 0:
                    self.save(Path(self.config.outdir) / f"episode_{self.episode}.pt")
            self.save(Path(self.config.outdir) / "latest.pt")
            if self.updates == 0:
                print("No learner updates yet: increase episodes or reduce warmup/update interval.", flush=True)
        finally:
            self.env.close()
            if run is not None:
                run.finish()


def add_mambpo_args(parser):
    group = parser.add_argument_group("MAMBPO")
    for name, default in asdict(MAMBPOConfig()).items():
        if name in ("device", "seed", "wandb_api"):
            continue
        options = {"action": argparse.BooleanOptionalAction} if isinstance(default, bool) else {
            "type": type(default) if default is not None else str}
        group.add_argument("--mambpo-" + name.replace("_", "-"), dest="mambpo_" + name,
                           default=default, **options)


def run_mambpo(args):
    kwargs = {key.removeprefix("mambpo_"): value for key, value in vars(args).items() if key.startswith("mambpo_")}
    kwargs.update(device=args.device, seed=args.seed, wandb_api=args.wandb_api)
    MAMBPOTrainer(MAMBPOConfig(**kwargs)).train()
