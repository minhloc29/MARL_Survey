"""AORPO camera-team learner with adaptive opponent-wise model rollouts."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mate.algorithms.camera_environment import DiscreteCameraEnvironment
from .networks import AORPOAgent, adaptive_lengths
from .replay import TransitionReplay, mixed_batch


@dataclass
class AORPOConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episodes: int = 1000
    episode_limit: int = 64
    hidden: int = 64
    model_hidden: int = 128
    ensemble_size: int = 5
    batch_size: int = 64
    model_batch_size: int = 128
    model_samples: int = 512
    model_epochs: int = 5
    opponent_epochs: int = 5
    policy_updates: int = 5
    replay_capacity: int = 100_000
    model_capacity: int = 10_000
    warmup: int = 256
    rollout_batch_size: int = 64
    rollout_min: int = 1
    rollout_horizon: int = 5
    rollout_start: int = 20
    rollout_end: int = 100
    real_ratio: float = 0.5
    lr: float = 3e-4
    model_lr: float = 1e-3
    opponent_lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.01
    reward_scale: float = 1.0
    target_entropy_ratio: float = 0.7
    opponent_entropy: float = 0.1
    device: str = "cpu"
    seed: int = 0
    num_envs: int = 1
    log_every: int = 10
    save_every: int = 100
    outdir: str = "ckpts/aorpo"
    resume: str | None = None
    wandb_api: str | None = None

    def __post_init__(self):
        for key in ("episodes", "episode_limit", "hidden", "model_hidden", "ensemble_size",
                    "batch_size", "model_batch_size", "model_samples", "model_epochs",
                    "opponent_epochs", "policy_updates", "replay_capacity", "model_capacity",
                    "rollout_batch_size", "log_every", "save_every"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.levels < 2 or self.model_samples < 2 or not 0 <= self.warmup <= self.replay_capacity:
            raise ValueError("Require levels >= 2, model_samples >= 2 and warmup within replay capacity")
        if self.replay_capacity < 5:
            raise ValueError("Replay capacity must be >= 5 for disjoint train/validation data")
        if not 1 <= self.rollout_min <= self.rollout_horizon or not 0 <= self.rollout_start < self.rollout_end:
            raise ValueError("Invalid rollout length or episode schedule")
        for key in ("real_ratio", "gamma", "tau", "target_entropy_ratio"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if min(self.lr, self.model_lr, self.opponent_lr) <= 0 or self.opponent_entropy < 0:
            raise ValueError("Learning rates must be positive and opponent entropy weight nonnegative")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def bellman_target(reward, terminated, next_value, gamma):
    return reward + gamma * (1.0 - terminated) * next_value


class AORPOTrainer:
    def __init__(self, config: AORPOConfig):
        self.config = config
        self.device = resolve_device(config.device)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.env = DiscreteCameraEnvironment(config.config, config.levels, config.episode_limit)
        self.agents = torch.nn.ModuleList([
            AORPOAgent(i, self.env.n_agents, self.env.obs_dim, self.env.n_actions,
                       config.hidden, config.model_hidden, config.ensemble_size)
            for i in range(self.env.n_agents)
        ]).to(self.device)
        self.optimizers = []
        for agent in self.agents:
            opts = {
                "actor": torch.optim.Adam(agent.actor.parameters(), lr=config.lr),
                "critics": torch.optim.Adam(agent.critics.parameters(), lr=config.lr),
                "dynamics": torch.optim.Adam(agent.dynamics.parameters(), lr=config.model_lr),
                "alpha": torch.optim.Adam([agent.log_alpha], lr=config.lr),
            }
            if agent.other_indices:
                opts["opponents"] = torch.optim.Adam(agent.opponents.parameters(), lr=config.opponent_lr)
            self.optimizers.append(opts)
        self.real_replay = TransitionReplay(config.replay_capacity, config.seed)
        # Split each incoming real transition exactly once, before any fitting.
        # SAC may use all real transitions, while model validation stays held out.
        self.model_train = TransitionReplay(config.replay_capacity, config.seed + 1)
        self.model_validation = TransitionReplay(max(1, config.replay_capacity // 5), config.seed + 2)
        self.model_replays = [TransitionReplay(config.model_capacity, config.seed + 3 + i)
                              for i in range(self.env.n_agents)]
        self.steps = self.episode = self.updates = 0
        self.rollout = None
        if config.num_envs > 1:
            from mate.algorithms.batched_rollout import BatchedRollout
            self.rollout = BatchedRollout(
                lambda: _make_aorpo_vec_env(config), config.num_envs)
        if config.resume:
            try:
                self.load(config.resume)
            except Exception:
                self.env.close()
                raise

    @property
    def horizon(self):
        cfg = self.config
        fraction = np.clip((self.episode - cfg.rollout_start) / (cfg.rollout_end - cfg.rollout_start), 0, 1)
        return cfg.rollout_min + int(fraction * (cfg.rollout_horizon - cfg.rollout_min))

    @property
    def ready(self):
        return (len(self.real_replay) >= self.config.warmup
                and len(self.model_train) >= 2 and len(self.model_validation) >= 1)

    @torch.no_grad()
    def act(self, observation, greedy=False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.shape != (self.env.n_agents, self.env.obs_dim):
            raise ValueError("Expected a joint camera observation")
        return np.asarray([agent.policy(obs.unsqueeze(0), stochastic=not greedy)[0]
                           .argmax(-1).item() for agent in self.agents], dtype=np.int64)

    def collect_episode(self, index, training=True):
        obs, _ = self.env.reset(seed=self.config.seed + index)
        reward_sum, coverage = 0.0, []
        for step in range(self.config.episode_limit):
            if training and len(self.real_replay) < self.config.warmup:
                actions = self.rng.integers(self.env.n_actions, size=self.env.n_agents)
            else:
                actions = self.act(obs, greedy=not training)
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            if training:
                row = (obs, actions, np.full(self.env.n_agents, reward), next_obs, terminated)
                self.real_replay.add(*row)
                destination = self.model_validation if self.steps % 5 == 0 else self.model_train
                destination.add(*row)
                self.steps += 1
            reward_sum += reward
            infos = info.get("infos", []) if isinstance(info, dict) else []
            values = [float(item["coverage_rate"]) for item in infos if "coverage_rate" in item]
            if values:
                coverage.append(float(np.mean(values)))
            obs = next_obs
            if terminated or truncated:
                break
        return {"reward": reward_sum, "length": step + 1,
                "coverage": float(np.mean(coverage)) if coverage else 0.0}

    @torch.no_grad()
    def batched_act(self, observations, training=True):
        """Actions for all slots from their obs (parent process). Mirrors
        ``collect_episode``'s per-agent shared policy + warmup random override."""
        actions = []
        for obs in observations:
            if training and len(self.real_replay) < self.config.warmup:
                actions.append(self.rng.integers(
                    self.env.n_actions, size=self.env.n_agents))
            else:
                actions.append(self.act(obs, greedy=not training))
        return np.stack(actions)

    def collect_batch(self, first_ep, training=True):
        """Concurrently collect ``num_envs`` episodes in parallel workers."""
        num = self.rollout.num_envs
        obs_list = self.rollout.reset(
            [self.config.seed + first_ep + i for i in range(num)])
        reward_sums = np.zeros(num)
        lengths = np.zeros(num, dtype=np.int64)
        coverages = [[] for _ in range(num)]
        for _ in range(self.config.episode_limit):
            actions = self.batched_act(obs_list, training=training)
            step = self.rollout.step(actions)
            for i in range(num):
                next_obs = step.obs[i] if not step.done[i] else (
                    step.infos[i].get("final_observation", step.obs[i]))
                reward = float(step.rewards[i])
                terminated = bool(step.done[i])
                if training:
                    row = (obs_list[i], actions[i],
                           np.full(self.env.n_agents, reward), next_obs, terminated)
                    self.real_replay.add(*row)
                    destination = self.model_validation if self.steps % 5 == 0 else self.model_train
                    destination.add(*row)
                    self.steps += 1
                reward_sums[i] += reward
                lengths[i] += 1
                infos = step.infos[i].get("infos", []) if isinstance(step.infos[i], dict) else []
                values = [float(item["coverage_rate"]) for item in infos if "coverage_rate" in item]
                if values:
                    coverages[i].append(float(np.mean(values)))
            obs_list = step.obs
            if all(step.done):
                break
        avg_reward = float(reward_sums.mean())
        avg_cover = float(np.mean([np.mean(c) if c else 0.0 for c in coverages]))
        avg_len = float(np.mean(lengths))
        return {"reward": avg_reward, "length": avg_len, "coverage": avg_cover}

    @staticmethod
    def _optimize(optimizer, parameters, loss, clip):
        parameters = list(parameters)
        if not torch.isfinite(loss):
            raise FloatingPointError("AORPO produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()

    def fit_models(self):
        cfg = self.config
        train = self.model_train.latest(cfg.model_samples, self.device)
        validation = self.model_validation.latest(cfg.model_samples, self.device)
        losses, opponent_losses, errors = [], [], []
        for agent, opts in zip(self.agents, self.optimizers):
            agent.dynamics.fit_stats(agent.dynamics.inputs(train["obs"], train["actions"]))
            for _ in range(cfg.model_epochs):
                indices = torch.randint(len(train["obs"]), (cfg.model_batch_size,), device=self.device)
                batch = {key: value[indices] for key, value in train.items()}
                loss = agent.dynamics.objective(batch)
                losses.append(self._optimize(opts["dynamics"], agent.dynamics.parameters(), loss, 200.0))
            if agent.other_indices:
                for _ in range(cfg.opponent_epochs):
                    indices = torch.randint(len(train["obs"]), (cfg.model_batch_size,), device=self.device)
                    terms = []
                    for j in agent.other_indices:
                        logits = agent.opponents[str(j)](train["obs"][indices, j])
                        entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean()
                        terms.append(F.cross_entropy(logits, train["actions"][indices, j]) - cfg.opponent_entropy * entropy)
                    loss = torch.stack(terms).mean()
                    opponent_losses.append(self._optimize(opts["opponents"], agent.opponents.parameters(), loss, 1.0))
            with torch.no_grad():
                rates = [(agent.opponents[str(j)](validation["obs"][:, j]).argmax(-1)
                          == validation["actions"][:, j]).float().mean() for j in agent.other_indices]
                if rates:
                    agent.match_rates.copy_(torch.stack(rates))
                prediction, _, _ = agent.dynamics(agent.dynamics.inputs(validation["obs"], validation["actions"]))
                target = torch.cat([validation["next_obs"].flatten(1), validation["rewards"]], -1)
                errors.append((prediction - target).square().mean().item())
        return {"model_loss": float(np.mean(losses)), "model_validation_mse": float(np.mean(errors)),
                "opponent_loss": float(np.mean(opponent_losses)) if opponent_losses else 0.0,
                "opponent_match_rate": float(torch.cat([a.match_rates for a in self.agents]).mean())
                if self.env.n_agents > 1 else 1.0}

    @torch.no_grad()
    def rollout_agent(self, index, start_obs, horizon=None):
        """Query a real agent policy only after its modeled-action budget ends.

        Calls are in-process policy queries on simulated observations; they do
        not advance MATE. Every ego owns a separate dynamics model and replay.
        """
        horizon = self.horizon if horizon is None else horizon
        agent = self.agents[index]
        lengths = adaptive_lengths(agent.match_rates, horizon)
        replay = self.model_replays[index]
        replay.clear()  # discard stale predictions after every model refit
        obs = start_obs.clone()
        generated = 0
        for depth in range(horizon):
            joint = [None] * self.env.n_agents
            own, _, predicted_joint, _ = agent.policy(obs)
            joint[index] = own.argmax(-1)
            for opponent_index, j in enumerate(agent.other_indices):
                if depth < lengths[opponent_index].item():
                    joint[j] = predicted_joint[:, j].argmax(-1)
                else:
                    joint[j] = self.agents[j].policy(obs)[0].argmax(-1)
                    agent.query_counts[opponent_index] += len(obs)
            actions = torch.stack(joint, dim=1)
            next_obs, rewards, terminated = agent.dynamics.predict(obs, actions)
            if not all(torch.isfinite(x).all() for x in (next_obs, rewards, terminated)):
                raise FloatingPointError("Non-finite AORPO model rollout")
            replay.extend(dict(obs=obs, actions=actions, rewards=rewards,
                               next_obs=next_obs, terminated=terminated))
            generated += len(obs)
            obs = next_obs[terminated == 0]
            if len(obs) == 0:
                break
        return generated

    def update_policy(self, index, batch):
        cfg, agent, opts = self.config, self.agents[index], self.optimizers[index]
        obs, next_obs = batch["obs"], batch["next_obs"]
        # The opponent models receive supervised updates only.
        _, log_prob, joint, logits = agent.policy(obs)
        target_entropy = cfg.target_entropy_ratio * np.log(self.env.n_actions)
        alpha_loss = -(agent.log_alpha * (log_prob.detach() + target_entropy)).mean()
        self._optimize(opts["alpha"], [agent.log_alpha], alpha_loss, 1.0)
        alpha = agent.log_alpha.exp().detach()
        agent.critics.requires_grad_(False)
        try:
            actor_loss = (alpha * log_prob - agent.q_values(obs, joint).min(0).values).mean()
            actor_loss = actor_loss + 1e-3 * logits.square().mean()
            actor_value = self._optimize(opts["actor"], agent.actor.parameters(), actor_loss, 1.0)
        finally:
            agent.critics.requires_grad_(True)
        with torch.no_grad():
            _, next_log_prob, next_joint, _ = agent.policy(next_obs)
            next_value = agent.q_values(next_obs, next_joint, target=True).min(0).values - alpha * next_log_prob
            target = bellman_target(cfg.reward_scale * batch["rewards"][:, index],
                                    batch["terminated"], next_value, cfg.gamma)
        replay_actions = F.one_hot(batch["actions"], self.env.n_actions).float()
        critics_loss = (agent.q_values(obs, replay_actions) - target.unsqueeze(0)).square().mean()
        critic_value = self._optimize(opts["critics"], agent.critics.parameters(), critics_loss, 40.0)
        with torch.no_grad():
            for target_p, online_p in zip(agent.target_critics.parameters(), agent.critics.parameters()):
                target_p.lerp_(online_p, cfg.tau)
        return {"actor_loss": actor_value, "critic_loss": critic_value, "alpha": alpha.item()}

    def update(self):
        if not self.ready:
            raise ValueError("AORPO needs warmup data and disjoint model train/validation samples")
        metrics = self.fit_models()
        starts = self.real_replay.sample(self.config.rollout_batch_size, self.device)["obs"]
        generated = sum(self.rollout_agent(i, starts) for i in range(self.env.n_agents))
        policy_metrics = []
        for _ in range(self.config.policy_updates):
            for index in range(self.env.n_agents):
                batch = mixed_batch(self.real_replay, self.model_replays[index], self.config.batch_size,
                                    self.config.real_ratio, self.device)
                policy_metrics.append(self.update_policy(index, batch))
        metrics.update({key: float(np.mean([row[key] for row in policy_metrics])) for key in policy_metrics[0]})
        metrics.update(horizon=self.horizon, model_transitions=generated,
                       policy_queries=sum(a.query_counts.sum().item() for a in self.agents))
        self.updates += 1
        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        config.pop("wandb_api")
        config["resume"] = None
        payload = {"format_version": 1, "config": config, "agents": self.agents.state_dict(),
                   "optimizers": [{key: opt.state_dict() for key, opt in opts.items()} for opts in self.optimizers],
                   "episode": self.episode, "steps": self.steps, "updates": self.updates,
                   "env_spec": (self.env.n_agents, self.env.obs_dim, self.env.n_actions)}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path):
        payload = torch.load(path, weights_only=True, map_location=self.device)
        if payload["format_version"] != 1 or tuple(payload["env_spec"]) != (
                self.env.n_agents, self.env.obs_dim, self.env.n_actions):
            raise ValueError("Checkpoint version or environment dimensions do not match")
        for key in ("config", "levels", "hidden", "model_hidden", "ensemble_size"):
            if payload["config"][key] != getattr(self.config, key):
                raise ValueError(f"Checkpoint setting {key!r} does not match")
        self.agents.load_state_dict(payload["agents"])
        for opts, state in zip(self.optimizers, payload["optimizers"], strict=True):
            for key, optimizer in opts.items():
                optimizer.load_state_dict(state[key])
        self.episode, self.steps, self.updates = (payload[key] for key in ("episode", "steps", "updates"))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        return cls(AORPOConfig(**dict(payload["config"], device=device, resume=str(path))))

    def train(self):
        run = None
        try:
            if self.config.wandb_api:
                import wandb
                wandb.login(key=self.config.wandb_api)
                config = asdict(self.config)
                config.pop("wandb_api")
                run = wandb.init(project="mate-aorpo", config=config)
            if self.config.resume:
                print("Loaded AORPO learner; collecting fresh replay before updates.", flush=True)
            end = self.episode + self.config.episodes
            while self.episode < end:
                if self.rollout is not None:
                    metrics = self.collect_batch(self.episode)
                    self.episode += self.rollout.num_envs
                else:
                    metrics = self.collect_episode(self.episode)
                    self.episode += 1
                if self.ready:
                    metrics.update(self.update())
                if self.episode % self.config.log_every == 0:
                    print(f"aorpo episode {self.episode} | steps {self.steps} | updates {self.updates} | "
                          + " | ".join(f"{key} {value:.4f}" for key, value in metrics.items()), flush=True)
                if run is not None:
                    run.log(metrics, step=self.steps)
                if self.episode % self.config.save_every == 0:
                    self.save(Path(self.config.outdir) / f"episode_{self.episode}.pt")
            self.save(Path(self.config.outdir) / "latest.pt")
            if not self.updates:
                print("No learner updates yet: collect more episodes or reduce warmup.", flush=True)
        finally:
            self.env.close()
            if self.rollout is not None:
                self.rollout.close()
            if run is not None:
                run.finish()


def _make_aorpo_vec_env(config) -> "MateCameraVecEnv":
    """Top-level, cloudpickle-safe env factory for the batched rollout worker."""
    from mate.algorithms.vec_adapter import MateCameraVecEnv
    env = DiscreteCameraEnvironment(config.config, config.levels, config.episode_limit)
    return MateCameraVecEnv(env, continuous=False)


def add_aorpo_args(parser):
    group = parser.add_argument_group("AORPO")
    for name, default in asdict(AORPOConfig()).items():
        if name in ("device", "seed", "wandb_api"):
            continue
        group.add_argument("--aorpo-" + name.replace("_", "-"), dest="aorpo_" + name,
                           type=type(default) if default is not None else str, default=default)


def run_aorpo(args):
    kwargs = {key.removeprefix("aorpo_"): value for key, value in vars(args).items() if key.startswith("aorpo_")}
    kwargs.update(device=args.device, seed=args.seed, wandb_api=args.wandb_api)
    AORPOTrainer(AORPOConfig(**kwargs)).train()
