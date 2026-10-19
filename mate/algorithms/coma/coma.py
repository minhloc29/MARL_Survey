"""MATE-native COMA trainer for discrete cooperative camera control."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms.rode.replay import RODEEpisodeBuilder, RODEEpisodeReplay
from mate.algorithms._wandb import WandbLogger


class COMAActor(nn.Module):
    """Shared categorical actor with a learned camera identity embedding."""

    def __init__(self, obs_shape: int, n_agents: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.agent_embedding = nn.Embedding(n_agents, hidden)
        self.network = nn.Sequential(
            nn.Linear(obs_shape + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, steps, agents, _ = obs.shape
        agent_ids = torch.arange(agents, device=obs.device)
        identity = self.agent_embedding(agent_ids).view(1, 1, agents, -1)
        return self.network(torch.cat([obs, identity.expand(batch, steps, -1, -1)], dim=-1))


class COMACritic(nn.Module):
    """Centralized critic returning Q-values for each evaluated agent."""

    def __init__(self, state_shape: int, n_agents: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_embedding = nn.Embedding(n_agents, hidden)
        self.network = nn.Sequential(
            nn.Linear(state_shape + n_agents * n_actions + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = state.shape
        one_hot = torch.nn.functional.one_hot(
            actions.long(), self.n_actions).float()
        joint_action = one_hot.reshape(batch, steps, -1)
        agent_ids = torch.arange(self.n_agents, device=state.device)
        identity = self.agent_embedding(
            agent_ids).view(1, 1, self.n_agents, -1)
        critic_input = torch.cat(
            [state.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
             joint_action.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
             identity.expand(batch, steps, -1, -1)],
            dim=-1,
        )
        return self.network(critic_input)


@dataclass
class COMAConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    total_env_steps: Optional[int] = None
    update_every: int = 8
    batch_size: int = 8
    replay_capacity: int = 128
    hidden: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    target_update_every: int = 10
    entropy_coef: float = 0.01
    epsilon: float = 0.05
    device: str = "cpu"
    seed: int = 0
    num_envs: int = 1
    outdir: str = "ckpts/coma"
    wandb_api: Optional[str] = None


class COMATrainer:
    """Collect on-policy episodes and optimize COMA actor/critic updates."""

    def __init__(self, config: COMAConfig) -> None:
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
        self.rollout = None
        if config.num_envs > 1:
            from mate.algorithms.batched_rollout import BatchedRollout
            self.rollout = BatchedRollout(
                lambda: _make_coma_vec_env(config), config.num_envs)
        self.actor = COMAActor(
            self.env.obs_shape, self.env.n_agents, self.env.n_actions, config.hidden
        ).to(self.device)
        self.critic = COMACritic(
            self.env.state_shape, self.env.n_agents, self.env.n_actions, config.hidden
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr)
        self.replay = RODEEpisodeReplay(config.replay_capacity, self.device)
        self.wandb = WandbLogger("mate-coma", "coma", dict(config.__dict__), config.wandb_api)
        self._env_steps = 0
        self._batch_episodes = []

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
                logits = self.actor(obs_tensor)[0, 0]
                actions = Categorical(logits=logits).sample()
            if self.config.epsilon > 0 and np.random.random() < self.config.epsilon:
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

    def batched_act(self, payloads, training):
        """Compute actions for all slots from their current obs (parent process).

        Mirrors the single-env action logic of ``collect_episode`` (shared
        ``COMAActor`` forward + Categorical sample + optional epsilon greed).
        """
        actions = []
        for payload in payloads:
            obs_tensor = torch.as_tensor(
                payload["obs"], dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.actor(obs_tensor)[0, 0]
                act = Categorical(logits=logits).sample()
            if self.config.epsilon > 0 and np.random.random() < self.config.epsilon:
                act = torch.randint(
                    self.env.n_actions, (self.env.n_agents,), device=self.device)
            actions.append(act.cpu().numpy())
        return np.stack(actions)

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
        total_rewards = np.zeros(num, dtype=np.float64)
        n_steps = np.zeros(num, dtype=np.int64)
        for _ in range(self.config.episode_limit):
            actions = self.batched_act(payloads, training=True)
            step = self.rollout.step(actions)
            for i in range(num):
                n_steps[i] += 1
                self._env_steps += 1
                terminal = step.infos[i].get("final_observation")
                fin = terminal if step.done[i] else step.obs[i]
                builders[i].add(
                    actions[i], float(step.rewards[i]), bool(step.done[i]),
                    fin["obs"], fin["state"], fin["avail_actions"],
                )
                total_rewards[i] += float(step.rewards[i])
                if step.done[i]:
                    self._record_episode(builders[i].finish())
                    # Auto-reset continuation episode: re-seed the builder with the
                    # post-reset obs so the next add() records the fresh episode's
                    # first transition (finish() already clears the arrays).
                    builders[i].start(step.obs[i]["obs"], step.obs[i]["state"],
                                      step.obs[i]["avail_actions"])
            payloads = step.obs
            if all(step.done):
                break
        mean = float(total_rewards.mean())
        avg_steps = float(np.mean(np.maximum(n_steps, 1)))
        return self._batch_episodes, mean, mean / max(avg_steps, 1e-8)

    def _record_episode(self, episode):
        """Hook: what to do with a finished episode in the batched path.

        For COMA, episodes accumulate in ``self.replay`` over the outer train
        loop (same as the single-env path, where ``train()`` calls
        ``self.replay.add(episode)``).
        """
        self._batch_episodes.append(episode)

    def update(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        obs = batch["obs"][:, :-1]
        next_obs = batch["obs"][:, 1:]
        state = batch["state"][:, :-1]
        next_state = batch["state"][:, 1:]
        actions = batch["actions"]
        rewards = batch["rewards"].squeeze(-1)
        terminated = batch["terminated"].squeeze(-1).float()
        mask = batch["filled"].squeeze(-1).float()

        logits = self.actor(obs)
        probabilities = torch.softmax(logits, dim=-1)
        q_values = self.critic(state, actions)
        q_taken = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        baseline = (probabilities * q_values.detach()).sum(-1)
        advantages = (q_taken.detach() - baseline)
        log_prob = Categorical(logits=logits).log_prob(actions)
        actor_loss = -((advantages * log_prob).sum(-1) *
                       mask).sum() / mask.sum().clamp_min(1.0)
        entropy = Categorical(logits=logits).entropy().sum(-1)
        actor_loss = actor_loss - self.config.entropy_coef * \
            (entropy * mask).sum() / mask.sum().clamp_min(1.0)

        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_actions = Categorical(logits=next_logits).sample()
            next_q = self.target_critic(next_state, next_actions)
            next_q_taken = next_q.gather(-1,
                                         next_actions.unsqueeze(-1)).squeeze(-1)
            targets = rewards.unsqueeze(-1) + self.config.gamma * (
                1.0 - terminated.unsqueeze(-1)) * next_q_taken
        critic_loss = (((q_taken - targets) ** 2) *
                       mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())

    def train(self) -> None:
        last_losses = (None, None)
        episode_index = 0
        while episode_index < self.config.episodes:
            if (self.config.total_env_steps is not None
                    and self._env_steps >= self.config.total_env_steps):
                break
            if self.rollout is not None:
                # Batched path: collect `num_envs` episodes concurrently, then
                # feed each finished episode through the same replay/update flow.
                self._batch_episodes = []
                _, reward, coverage_rate = self.collect_batch(episode_index)
                step_span = self.rollout.num_envs
                for idx, episode in enumerate(self._batch_episodes):
                    self.replay.add(episode)
                    ep_idx = episode_index + idx
                    if (ep_idx + 1) % self.config.update_every == 0 and len(self.replay) >= self.config.batch_size:
                        last_losses = self.update(
                            self.replay.sample(self.config.batch_size))
                    if (ep_idx + 1) % 10 == 0:
                        actor_l, critic_l = last_losses
                        print(
                            f"episode {ep_idx + 1}/{self.config.episodes} | reward {reward:.3f} | actor {actor_l} | critic {critic_l}", flush=True)
                        self.wandb.log({"reward": reward,
                                        "coverage_rate": coverage_rate,
                                        "actor_loss": actor_l if actor_l is not None else float("nan"),
                                        "critic_loss": critic_l if critic_l is not None else float("nan")},
                                       step=ep_idx + 1)
                    if (ep_idx + 1) % self.config.target_update_every == 0:
                        self.target_critic.load_state_dict(self.critic.state_dict())
                episode_index += step_span
            else:
                episode, reward, coverage_rate = self.collect_episode(episode_index)
            self.replay.add(episode)
            if (episode_index + 1) % self.config.update_every == 0 and len(self.replay) >= self.config.batch_size:
                last_losses = self.update(
                    self.replay.sample(self.config.batch_size))
            if (episode_index + 1) % 10 == 0:
                actor_l, critic_l = last_losses
                print(
                    f"episode {episode_index + 1}/{self.config.episodes} | reward {reward:.3f} | actor {actor_l} | critic {critic_l}", flush=True)
                self.wandb.log({"reward": reward,
                                "coverage_rate": coverage_rate,
                                "actor_loss": actor_l if actor_l is not None else float("nan"),
                                "critic_loss": critic_l if critic_l is not None else float("nan")},
                               step=episode_index + 1)
            if (episode_index + 1) % self.config.target_update_every == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())
            episode_index += 1
        Path(self.config.outdir).mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(
        ), "critic": self.critic.state_dict()}, Path(self.config.outdir) / "latest.pt")
        self.wandb.finish()
        self.env.close()
        if self.rollout is not None:
            self.rollout.close()


def _make_coma_vec_env(config) -> "MateRodeVecEnv":
    """Top-level, cloudpickle-safe env factory for the batched rollout worker."""
    from mate.algorithms.vec_adapter import MateRodeVecEnv
    env = make_mate_rode_env(
        config=config.config,
        levels=config.levels,
        episode_limit=config.episode_limit,
    )
    return MateRodeVecEnv(env)


def run_coma(args: Any) -> None:
    total_env_steps = getattr(args, "total_timesteps", 10_000_000)
    episodes = args.coma_episodes
    if episodes is None:
        episodes = total_env_steps // args.coma_episode_limit + 1
    COMATrainer(COMAConfig(
        config=args.coma_config,
        levels=args.coma_levels,
        episode_limit=args.coma_episode_limit,
        episodes=episodes,
        total_env_steps=total_env_steps,
        update_every=args.coma_update_every,
        batch_size=args.coma_batch_size,
        device=args.device,
        seed=args.seed,
        num_envs=getattr(args, "coma_num_envs", 1),
        outdir=args.coma_outdir,
        wandb_api=args.wandb_api,
    )).train()
