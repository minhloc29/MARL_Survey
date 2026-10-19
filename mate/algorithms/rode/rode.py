"""A compact, modern PyTorch port of RODE for MATE."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from flow.nets import MonotonicMixer
from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms.rode.replay import RODEEpisodeBuilder, RODEEpisodeReplay
from mate.algorithms._wandb import WandbLogger


class RODENetwork(nn.Module):
    """Shared recurrent agent with role-conditioned action values."""

    def __init__(self, obs_shape: int, n_actions: int, n_roles: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_roles = n_roles
        self.hidden_size = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_shape, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.role_head = nn.Linear(hidden, n_roles)
        self.role_actions = nn.Parameter(
            torch.randn(n_roles, hidden, n_actions) * 0.02)
        self.action_bias = nn.Parameter(torch.zeros(n_roles, n_actions))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        batch, steps, agents, _ = obs.shape
        if hidden is None:
            hidden = torch.zeros(
                batch, agents, self.hidden_size, device=obs.device)
        outputs = []
        role_logits = []
        for step in range(steps):
            encoded = self.encoder(obs[:, step])
            hidden = self.gru(encoded.reshape(batch * agents, -1),
                              hidden.reshape(batch * agents, -1))
            hidden = hidden.view(batch, agents, -1)
            role_logits.append(self.role_head(hidden))
            outputs.append(torch.einsum("bah,rhk->brak", hidden,
                           self.role_actions) + self.action_bias.view(1, self.n_roles, 1, -1))
        return torch.stack(outputs, dim=1), torch.stack(role_logits, dim=1)


@dataclass
class RODEConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    total_env_steps: Optional[int] = None
    warmup_episodes: int = 8
    batch_size: int = 8
    replay_capacity: int = 500
    hidden: int = 128
    n_roles: int = 3
    role_interval: int = 5
    lr: float = 3e-4
    gamma: float = 0.99
    target_update_every: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    device: str = "cpu"
    seed: int = 0
    num_envs: int = 1
    outdir: str = "ckpts/rode"
    wandb_api: Optional[str] = None


class RODETrainer:
    """Train RODE on MATE's discrete camera task."""

    def __init__(self, config: RODEConfig) -> None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        self.env = make_mate_rode_env(
            config=config.config,
            levels=config.levels,
            episode_limit=config.episode_limit,
            render_mode="rgb_array",
        )
        self.online = RODENetwork(
            self.env.obs_shape, self.env.n_actions, config.n_roles, config.hidden
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.mixer = MonotonicMixer(
            self.env.n_agents, self.env.state_shape, config.hidden).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.online.parameters()) + list(self.mixer.parameters()), lr=config.lr
        )
        self.replay = RODEEpisodeReplay(config.replay_capacity, self.device)
        self.wandb = WandbLogger(
            "mate-rode", "rode", dict(config.__dict__), config.wandb_api)
        self._env_steps = 0
        self._batch_episodes = []
        self.rollout = None
        if config.num_envs > 1:
            from mate.algorithms.batched_rollout import BatchedRollout
            self.rollout = BatchedRollout(
                lambda: _make_rode_vec_env(config), config.num_envs)

    def batched_act(self, payloads, hiddens, roles, ep_eps, training=True):
        """Actions for all slots from their obs + per-slot GRU hidden + roles.

        Mirrors ``collect_episode``: role re-assignment every ``role_interval``
        steps, role-conditioned argmax actions, and the epsilon override."""
        n_agents = self.env.n_agents
        actions = []
        new_hiddens = []
        for i, payload in enumerate(payloads):
            obs_tensor = torch.as_tensor(
                payload["obs"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values, role_logits = self.online(obs_tensor, hiddens[i])
                new_hiddens.append(self._next_hidden(obs_tensor, hiddens[i]))
                if self._step_counts[i] % self.config.role_interval == 0:
                    roles[i] = role_logits[0, 0].argmax(-1).cpu().numpy()
                action_values = q_values[0, 0, roles[i], np.arange(n_agents)]
                act = action_values.argmax(-1).cpu().numpy()
            if np.random.random() < ep_eps:
                act = np.random.randint(
                    self.env.n_actions, size=n_agents)
            actions.append(act)
            self._step_counts[i] += 1
        return np.stack(actions), new_hiddens

    def collect_batch(self, first_ep):
        """Concurrently collect ``num_envs`` episodes in parallel workers."""
        num = self.rollout.num_envs
        payloads = self.rollout.reset(
            [self.config.seed + first_ep + i for i in range(num)])
        builders = [
            RODEEpisodeBuilder(self.env.n_agents, self.env.obs_shape, self.env.n_actions)
            for _ in range(num)
        ]
        for b, p in zip(builders, payloads):
            b.start(p["obs"], p["state"], p["avail_actions"])
        hiddens = [None] * num
        roles = [np.zeros(self.env.n_agents, dtype=np.int64) for _ in range(num)]
        self._step_counts = [0] * num
        total_rewards = np.zeros(num, dtype=np.float64)
        n_steps = np.zeros(num, dtype=np.int64)
        ep_eps = max(
            self.config.epsilon_end,
            self.config.epsilon_start *
            (1.0 - first_ep / max(1, self.config.epsilon_decay)),
        )
        for _ in range(self.config.episode_limit):
            actions, hiddens = self.batched_act(payloads, hiddens, roles, ep_eps)
            step = self.rollout.step(actions)
            for i in range(num):
                n_steps[i] += 1
                self._env_steps += 1
                terminal = step.infos[i].get("final_observation")
                fin = terminal if step.done[i] else step.obs[i]
                builders[i].add(
                    actions[i], float(step.rewards[i]), bool(step.done[i]),
                    fin["obs"], fin["state"], fin["avail_actions"], roles[i],
                )
                total_rewards[i] += float(step.rewards[i])
                if step.done[i]:
                    self._record_episode(builders[i].finish())
                    # Auto-reset continuation: re-seed builder, reset hidden/roles/counter.
                    builders[i].start(step.obs[i]["obs"], step.obs[i]["state"],
                                      step.obs[i]["avail_actions"])
                    hiddens[i] = None
                    roles[i] = np.zeros(self.env.n_agents, dtype=np.int64)
                    self._step_counts[i] = 0
            payloads = step.obs
            if all(step.done):
                break
        mean = float(total_rewards.mean())
        avg_steps = float(np.mean(np.maximum(n_steps, 1)))
        return self._batch_episodes, mean, mean / max(avg_steps, 1e-8)

    def _record_episode(self, episode):
        """Hook: accumulate finished episodes for the outer train() replay flow."""
        self._batch_episodes.append(episode)

    def collect_episode(self, episode_index: int):
        obs, state = self.env.reset(seed=self.config.seed + episode_index)
        builder = RODEEpisodeBuilder(
            self.env.n_agents, self.env.obs_shape, self.env.n_actions)
        builder.start(obs, state, self.env.get_avail_actions())
        hidden = None
        roles = np.zeros(self.env.n_agents, dtype=np.int64)
        epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start *
            (1.0 - episode_index / max(1, self.config.epsilon_decay)),
        )
        total_reward = 0.0
        n_steps = 0
        for _ in range(self.config.episode_limit):
            n_steps += 1
            self._env_steps += 1
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values, role_logits = self.online(obs_tensor, hidden)
                hidden = self._next_hidden(obs_tensor, hidden)
                if builder._rewards.__len__() % self.config.role_interval == 0:
                    roles = role_logits[0, 0].argmax(-1).cpu().numpy()
                action_values = q_values[0, 0, roles,
                                         np.arange(self.env.n_agents)]
                actions = action_values.argmax(-1).cpu().numpy()
            if np.random.random() < epsilon:
                actions = np.random.randint(
                    self.env.n_actions, size=self.env.n_agents)
            next_obs_reward, done, _ = self.env.step(actions)
            next_obs = self.env.get_obs()
            next_state = self.env.get_state()
            builder.add(actions, next_obs_reward, done, next_obs,
                        next_state, self.env.get_avail_actions(), roles)
            obs, state = next_obs, next_state
            total_reward += next_obs_reward
            if done:
                break
        return builder.finish(), total_reward, total_reward / max(1, n_steps)

    def _next_hidden(self, obs: torch.Tensor, hidden: torch.Tensor | None):
        batch, _, agents, _ = obs.shape
        if hidden is None:
            hidden = torch.zeros(
                batch, agents, self.config.hidden, device=self.device)
        encoded = self.online.encoder(obs[:, 0])
        return self.online.gru(encoded.reshape(batch * agents, -1), hidden.reshape(batch * agents, -1)).view(batch, agents, -1)

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        q_by_role, role_logits = self.online(batch["obs"])
        with torch.no_grad():
            target_q_by_role, _ = self.target(batch["obs"])
        roles = batch["roles"]
        q = torch.gather(q_by_role[:, :-1], 2, roles.unsqueeze(
            2).unsqueeze(-1).expand(-1, -1, 1, -1, self.env.n_actions)).squeeze(2)
        chosen = torch.gather(
            q, -1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        next_roles = torch.argmax(role_logits[:, 1:].detach(), dim=-1)
        target_next = torch.gather(target_q_by_role[:, 1:], 2, next_roles.unsqueeze(
            2).unsqueeze(-1).expand(-1, -1, 1, -1, self.env.n_actions)).squeeze(2).max(-1).values
        batch_size, time_steps, _ = chosen.shape
        q_total = self.mixer(
            chosen.reshape(-1, self.env.n_agents),
            batch["state"][:, :-1].reshape(-1, self.env.state_shape),
        ).reshape(batch_size, time_steps)
        with torch.no_grad():
            target_total = self.target_mixer(
                target_next.reshape(-1, self.env.n_agents),
                batch["state"][:, 1:].reshape(-1, self.env.state_shape),
            ).reshape(batch_size, time_steps)
            target = batch["rewards"].squeeze(-1) + self.config.gamma * (
                ~batch["terminated"].squeeze(-1)).float() * target_total
        mask = batch["filled"].squeeze(-1).float()
        loss = ((q_total - target) ** 2 * mask).sum() / \
            mask.sum().clamp_min(1.0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.online.parameters()) + list(self.mixer.parameters()), 10.0)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def train(self) -> None:
        episode_index = 0
        while episode_index < self.config.episodes:
            # Honor a shared env-step budget (default 10M policy actions); each
            # episode consumes 1..episode_limit env steps.
            if (self.config.total_env_steps is not None
                    and self._env_steps >= self.config.total_env_steps):
                break
            if self.rollout is not None:
                # Batched path: collect num_envs episodes concurrently, then run
                # each finished episode through the same replay/update flow.
                self._batch_episodes = []
                _, reward, coverage_rate = self.collect_batch(episode_index)
                step_span = self.rollout.num_envs
                for idx, episode in enumerate(self._batch_episodes):
                    self.replay.add(episode)
                    ep_idx = episode_index + idx
                    loss = None
                    if len(self.replay) >= max(self.config.warmup_episodes, self.config.batch_size):
                        loss = self.update(self.replay.sample(self.config.batch_size))
                    if (ep_idx + 1) % 10 == 0:
                        print(
                            f"episode {ep_idx + 1}/{self.config.episodes} | reward {reward:.3f} | loss {loss}", flush=True)
                        self.wandb.log({"reward": reward, "coverage_rate": coverage_rate,
                                        "loss": loss if loss is not None else float("nan")},
                                       step=ep_idx + 1)
                    if (ep_idx + 1) % self.config.target_update_every == 0:
                        self.target.load_state_dict(self.online.state_dict())
                        self.target_mixer.load_state_dict(self.mixer.state_dict())
                episode_index += step_span
            else:
                episode, reward, coverage_rate = self.collect_episode(episode_index)
                self.replay.add(episode)
                loss = None
                if len(self.replay) >= max(self.config.warmup_episodes, self.config.batch_size):
                    loss = self.update(self.replay.sample(self.config.batch_size))
                if (episode_index + 1) % 10 == 0:
                    print(
                        f"episode {episode_index + 1}/{self.config.episodes} | reward {reward:.3f} | loss {loss}", flush=True)
                    self.wandb.log({"reward": reward, "coverage_rate": coverage_rate,
                                    "loss": loss if loss is not None else float("nan")},
                                   step=episode_index + 1)
                if (episode_index + 1) % self.config.target_update_every == 0:
                    self.target.load_state_dict(self.online.state_dict())
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
                episode_index += 1
        Path(self.config.outdir).mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict(
        ), "mixer": self.mixer.state_dict()}, Path(self.config.outdir) / "latest.pt")
        self.wandb.finish()
        self.env.close()
        if self.rollout is not None:
            self.rollout.close()


def _make_rode_vec_env(config) -> "MateRodeVecEnv":
    """Top-level, cloudpickle-safe env factory for the batched rollout worker."""
    from mate.algorithms.vec_adapter import MateRodeVecEnv
    env = make_mate_rode_env(
        config=config.config,
        levels=config.levels,
        episode_limit=config.episode_limit,
    )
    return MateRodeVecEnv(env)


def run_rode(args: Any) -> None:
    # Shared env-step budget governs the run; `--rode-episodes` is an explicit
    # override / hard ceiling on top of it.
    total_env_steps = getattr(args, "total_timesteps", 10_000_000)
    episodes = args.rode_episodes
    if episodes is None:
        episodes = total_env_steps // args.rode_episode_limit + 1
    RODETrainer(RODEConfig(
        config=args.rode_config,
        levels=args.rode_levels,
        episode_limit=args.rode_episode_limit,
        episodes=episodes,
        total_env_steps=total_env_steps,
        batch_size=args.rode_batch_size,
        warmup_episodes=args.rode_warmup_episodes,
        n_roles=args.rode_n_roles,
        role_interval=args.rode_role_interval,
        device=args.device,
        seed=args.seed,
        num_envs=getattr(args, "rode_num_envs", 1),
        outdir=args.rode_outdir,
        wandb_api=args.wandb_api,
    )).train()
