"""A compact autoregressive Multi-Agent Transformer PPO trainer for MATE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms._wandb import WandbLogger


class MATPolicy(nn.Module):
    """Transformer encoder plus causal decoder for a cooperative joint action."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.d_model = d_model
        self.obs_encoder = nn.Linear(obs_dim, d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.Tanh())
        self.agent_embedding = nn.Embedding(n_agents, d_model)
        self.action_embedding = nn.Embedding(n_actions + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers)
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations shaped ``[batch, time, agents, obs_dim]``."""
        batch, time, agents, _ = obs.shape
        agent_ids = torch.arange(agents, device=obs.device)
        x = self.obs_encoder(
            obs) + self.agent_embedding(agent_ids).view(1, 1, agents, -1)
        return self.encoder(x.reshape(batch * time, agents, self.d_model))

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, agents, _ = obs.shape
        memory = self.encode(obs)
        if actions is None:
            actions = torch.full(
                (batch, time, agents), self.n_actions,
                dtype=torch.long, device=obs.device
            )
        previous = torch.cat(
            [torch.full_like(actions[..., :1], self.n_actions), actions[..., :-1]], dim=-1
        )
        agent_ids = torch.arange(agents, device=obs.device)
        target = self.action_embedding(
            previous) + self.agent_embedding(agent_ids).view(1, 1, agents, -1)
        target = target.reshape(batch * time, agents, self.d_model)
        causal_mask = torch.triu(
            torch.ones(agents, agents, dtype=torch.bool, device=obs.device), diagonal=1
        )
        decoded = self.decoder(target, memory, tgt_mask=causal_mask)
        logits = self.policy_head(decoded).reshape(
            batch, time, agents, self.n_actions)
        pooled = memory.mean(dim=1).reshape(batch, time, self.d_model)
        values = self.value_head(
            torch.cat([pooled, self.state_encoder(state)], dim=-1)).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = torch.zeros(obs.size(0), obs.size(
            1), self.n_agents, dtype=torch.long, device=obs.device)
        log_probs = []
        for agent in range(self.n_agents):
            logits, _ = self(obs, state, actions)
            distribution = Categorical(logits=logits[:, :, agent])
            action = distribution.sample()
            actions[:, :, agent] = action
            log_probs.append(distribution.log_prob(action))
        logits, values = self(obs, state, actions)
        joint_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        return actions, joint_log_prob, values

    def evaluate(self, obs: torch.Tensor, state: torch.Tensor, actions: torch.Tensor):
        logits, values = self(obs, state, actions)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy, values


@dataclass
class MATConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    updates: int = 1000
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "cpu"
    seed: int = 0
    outdir: str = "ckpts/mat"
    wandb_api: Optional[str] = None


class MATTrainer:
    """On-policy MAT/PPO trainer using a MATE camera-team environment."""

    def __init__(self, config: MATConfig) -> None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        self.env = make_mate_rode_env(
            config=config.config, levels=config.levels,
            episode_limit=config.episode_limit, render_mode="rgb_array"
        )
        self.policy = MATPolicy(
            self.env.obs_shape, self.env.state_shape, self.env.n_agents,
            self.env.n_actions, config.d_model, config.n_heads, config.n_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr)
        self.wandb = WandbLogger("mate-mat", "mat", dict(config.__dict__), config.wandb_api)

    def collect_rollout(self):
        steps = self.config.rollout_steps
        agents, obs_dim, state_dim = self.env.n_agents, self.env.obs_shape, self.env.state_shape
        obs_buffer = np.zeros((steps + 1, agents, obs_dim), dtype=np.float32)
        state_buffer = np.zeros((steps + 1, state_dim), dtype=np.float32)
        action_buffer = np.zeros((steps, agents), dtype=np.int64)
        log_prob_buffer = np.zeros(steps, dtype=np.float32)
        value_buffer = np.zeros(steps + 1, dtype=np.float32)
        reward_buffer = np.zeros(steps, dtype=np.float32)
        done_buffer = np.zeros(steps, dtype=np.float32)
        obs, state = self.env.reset(
            seed=self.config.seed + np.random.randint(1_000_000))
        episode_reward = 0.0
        for step in range(steps):
            obs_buffer[step], state_buffer[step] = obs, state
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                actions, log_prob, value = self.policy.sample(
                    obs_tensor, state_tensor)
            action = actions[0, 0].cpu().numpy()
            reward, done, _ = self.env.step(action)
            next_obs, next_state = self.env.get_obs(), self.env.get_state()
            action_buffer[step] = action
            log_prob_buffer[step] = log_prob.item()
            value_buffer[step] = value.item()
            reward_buffer[step] = reward
            done_buffer[step] = float(done)
            episode_reward += reward
            obs, state = next_obs, next_state
            if done:
                obs, state = self.env.reset(
                    seed=self.config.seed + np.random.randint(1_000_000))
        obs_buffer[-1], state_buffer[-1] = obs, state
        with torch.no_grad():
            _, _, final_value = self.policy.sample(
                torch.as_tensor(obs[None, None],
                                dtype=torch.float32, device=self.device),
                torch.as_tensor(state[None, None],
                                dtype=torch.float32, device=self.device),
            )
        value_buffer[-1] = final_value.item()
        advantages = np.zeros(steps, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(steps)):
            continuation = 1.0 - done_buffer[step]
            delta = reward_buffer[step] + self.config.gamma * \
                value_buffer[step + 1] * continuation - value_buffer[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * continuation * gae
            advantages[step] = gae
        returns = advantages + value_buffer[:-1]
        return {
            "obs": obs_buffer[:-1], "state": state_buffer[:-1],
            "actions": action_buffer, "old_log_prob": log_prob_buffer,
            "advantages": advantages, "returns": returns,
        }, episode_reward

    def update(self, rollout: dict[str, np.ndarray]) -> tuple[float, float]:
        batch_size = rollout["actions"].shape[0]
        obs = torch.as_tensor(
            rollout["obs"], dtype=torch.float32, device=self.device).unsqueeze(1)
        state = torch.as_tensor(
            rollout["state"], dtype=torch.float32, device=self.device).unsqueeze(1)
        actions = torch.as_tensor(
            rollout["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        old_log_prob = torch.as_tensor(
            rollout["old_log_prob"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(
            rollout["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(
            rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)
        policy_total = value_total = 0.0
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                indices = order[start:start + self.config.minibatch_size]
                new_log_prob, entropy, values = self.policy.evaluate(
                    obs[indices], state[indices], actions[indices])
                ratio = torch.exp(new_log_prob.squeeze(1) -
                                  old_log_prob[indices])
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip, 1.0 + self.config.clip)
                policy_loss = - \
                    torch.minimum(
                        ratio * advantages[indices], clipped * advantages[indices]).mean()
                value_loss = torch.nn.functional.mse_loss(
                    values.squeeze(1), returns[indices])
                loss = policy_loss + self.config.value_coef * value_loss - \
                    self.config.entropy_coef * entropy.squeeze(1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                self.optimizer.step()
                policy_total += float(policy_loss.detach())
                value_total += float(value_loss.detach())
        return policy_total, value_total

    def train(self) -> None:
        for update in range(self.config.updates):
            rollout, reward = self.collect_rollout()
            policy_loss, value_loss = self.update(rollout)
            if (update + 1) % 10 == 0:
                print(
                    f"update {update + 1}/{self.config.updates} | reward {reward:.3f} | policy {policy_loss:.4f} | value {value_loss:.4f}", flush=True)
                self.wandb.log({"reward": reward, "policy_loss": policy_loss,
                                "value_loss": value_loss}, step=update + 1)
        output = Path(self.config.outdir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(),
                   "config": self.config.__dict__}, output / "latest.pt")
        self.wandb.finish()
        self.env.close()


def run_mat(args: Any) -> None:
    MATTrainer(MATConfig(
        config=args.mat_config, levels=args.mat_levels,
        episode_limit=args.mat_episode_limit, updates=args.mat_updates,
        rollout_steps=args.mat_rollout_steps, ppo_epochs=args.mat_ppo_epochs,
        minibatch_size=args.mat_minibatch_size, d_model=args.mat_d_model,
        n_heads=args.mat_n_heads, n_layers=args.mat_n_layers,
        device=args.device, seed=args.seed, outdir=args.mat_outdir,
        wandb_api=args.wandb_api,
    )).train()
