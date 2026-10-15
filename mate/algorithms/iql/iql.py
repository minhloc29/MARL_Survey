"""MATE-native Independent Q-Learning (IQL) for discrete cooperative camera control.

Each agent is a DQN learning from its own observations alone, ignoring teammates.
There is no mixer and no shared state -- agents only bootstrap their own Q-mix over
their own next-action maximum (independent TD learning).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms.rode.replay import RODEEpisodeBuilder, RODEEpisodeReplay
from mate.algorithms._wandb import WandbLogger


class IQLNetwork(nn.Module):
    """Shared feed-forward Q-network with a one-hot agent-id embedding.

    One copy of weights serves every agent; the embedding lets each agent
    specialize its own Q-function within the shared network.
    """

    def __init__(self, obs_shape: int, n_agents: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_embedding = nn.Embedding(n_agents, hidden)
        self.network = nn.Sequential(
            nn.Linear(obs_shape + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, steps, agents, _ = obs.shape
        identity = self.agent_embedding(
            torch.arange(agents, device=obs.device)).view(1, 1, agents, -1)
        return self.network(
            torch.cat([obs, identity.expand(batch, steps, -1, -1)], dim=-1))


@dataclass
class IQLConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    total_env_steps: Optional[int] = None
    update_every: int = 8
    batch_size: int = 8
    replay_capacity: int = 128
    hidden: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    target_update_every: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    device: str = "cpu"
    seed: int = 0
    outdir: str = "ckpts/iql"
    wandb_api: Optional[str] = None


class IQLTrainer:
    """Collect episodes, store them, and run independent-per-agent TD updates."""

    def __init__(self, config: IQLConfig) -> None:
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
        self.online = IQLNetwork(
            self.env.obs_shape, self.env.n_agents, self.env.n_actions, config.hidden
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = RODEEpisodeReplay(config.replay_capacity, self.device)
        self.wandb = WandbLogger("mate-iql", "iql", dict(config.__dict__), config.wandb_api)
        self._env_steps = 0

    def collect_episode(self, episode_index: int):
        obs, state = self.env.reset(seed=self.config.seed + episode_index)
        builder = RODEEpisodeBuilder(
            self.env.n_agents, self.env.obs_shape, self.env.n_actions)
        builder.start(obs, state, self.env.get_avail_actions())
        total_reward = 0.0
        n_steps = 0
        for _ in range(self.config.episode_limit):
            n_steps += 1
            self._env_steps += 1
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online(obs_tensor)[0, 0]
                actions = q_values.argmax(-1)  # greedy; eps handled below
            if np.random.random() < self._epsilon(episode_index):
                actions = torch.randint(
                    self.env.n_actions, (self.env.n_agents,), device=self.device)
            action_array = actions.cpu().numpy()
            reward, done, _ = self.env.step(action_array)
            next_obs = self.env.get_obs()
            next_state = self.env.get_state()
            builder.add(action_array, reward, done, next_obs,
                        next_state, self.env.get_avail_actions())
            obs, state = next_obs, next_state
            total_reward += reward
            if done:
                break
        return builder.finish(), total_reward, total_reward / max(1, n_steps)

    def _epsilon(self, episode_index: int) -> float:
        return max(
            self.config.epsilon_end,
            self.config.epsilon_start *
            (1.0 - episode_index / max(1, self.config.epsilon_decay)),
        )

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        obs = batch["obs"][:, :-1]
        next_obs = batch["obs"][:, 1:]
        actions = batch["actions"]
        # rewards/terminated/filled are (b, t, 1); broadcast against (b, t, agents).
        rewards = batch["rewards"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()

        q = self.online(obs)
        q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target(next_obs).max(-1).values  # independent max over own a'
        targets = rewards + self.config.gamma * (1.0 - terminated) * next_q

        loss = ((q_taken - targets) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def train(self) -> None:
        last_loss = None
        episode_index = 0
        while episode_index < self.config.episodes:
            if (self.config.total_env_steps is not None
                    and self._env_steps >= self.config.total_env_steps):
                break
            episode, reward, coverage_rate = self.collect_episode(episode_index)
            self.replay.add(episode)
            if (episode_index + 1) % self.config.update_every == 0 and len(self.replay) >= self.config.batch_size:
                last_loss = self.update(self.replay.sample(self.config.batch_size))
            if (episode_index + 1) % 10 == 0:
                print(
                    f"episode {episode_index + 1}/{self.config.episodes} | reward {reward:.3f} | loss {last_loss}", flush=True)
                self.wandb.log({"reward": reward,
                                "coverage_rate": coverage_rate,
                                "loss": last_loss if last_loss is not None else float("nan")},
                               step=episode_index + 1)
            if (episode_index + 1) % self.config.target_update_every == 0:
                self.target.load_state_dict(self.online.state_dict())
            episode_index += 1
        Path(self.config.outdir).mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict()}, Path(self.config.outdir) / "latest.pt")
        self.wandb.finish()
        self.env.close()


def run_iql(args: Any) -> None:
    total_env_steps = getattr(args, "total_timesteps", 10_000_000)
    episodes = args.iql_episodes
    if episodes is None:
        episodes = total_env_steps // args.iql_episode_limit + 1
    IQLTrainer(IQLConfig(
        config=args.iql_config,
        levels=args.iql_levels,
        episode_limit=args.iql_episode_limit,
        episodes=episodes,
        total_env_steps=total_env_steps,
        update_every=args.iql_update_every,
        batch_size=args.iql_batch_size,
        replay_capacity=args.iql_replay_capacity,
        hidden=args.iql_hidden,
        learning_rate=args.iql_lr,
        gamma=args.iql_gamma,
        target_update_every=args.iql_target_update_every,
        epsilon_start=args.iql_epsilon_start,
        epsilon_end=args.iql_epsilon_end,
        epsilon_decay=args.iql_epsilon_decay,
        device=args.device,
        seed=args.seed,
        outdir=args.iql_outdir,
        wandb_api=args.wandb_api,
    )).train()
