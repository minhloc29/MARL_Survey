"""MATE-native adaptation of I2C (Individually Inferred Communication).

I2C ("Learning Individually Inferred Communication for Multi-Agent
Cooperation", Ding, Huang & Lu -- NeurIPS 2020) is a MADDPG-style actor-critic
augmented with *individualized communication*: each agent learns which other
agents to share observations with and encodes the selected observations into a
message used to condition its own policy.

I2C was released as a TensorFlow 1 / MADDPG / MPE program.  This module is a
leaner PyTorch reimplementation of its mechanism, adapted to MATE's camera-team
setting where cameras are independent sensors.  Concretely:

* **Communication target = other cameras' observations.**  A learnable
  communication selector (the ``C`` head) decides, per camera, which other
  cameras' observation embeddings to attend to.
* **Message encoder (``M``)** -- an LSTM reads the selected other-camera
  observations and emits one fixed-size message per camera.
* **Policy (``P``)** -- each camera's action distribution conditions on its own
  observation plus its message.
* **Critic (``Q``)** -- a centralized MADDPG-style critic over the joint
  observation+action.

The distinctive I2C "blz" regularizer (pulling the policy toward a Boltzmann
distribution of Q over each agent's actions) is included.  The two-phase
KL-percentile prior-network training for ``C`` is left as future work; here the
selector is learned end-to-end through policy gradients (straight-through
Gumbel selection).

Only numpy + torch are required, matching the rest of MATE.
"""

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


def _mlp(input_dim: int, hidden: int, output: int, n_layers: int = 2) -> nn.Sequential:
    """Fully-connected ReLU network (no output activation)."""
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(n_layers - 1, 0)):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        dim = hidden
    layers.append(nn.Linear(dim, output))
    return nn.Sequential(*layers)


class I2CPolicy(nn.Module):
    """Communication selector ``C`` + message encoder ``M`` + policy ``P``.

    Shared across cameras (homogeneous) and vectorised over the batch.
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        n_actions: int,
        hidden: int = 128,
        msg_dim: int = 64,
        n_msg_others: Optional[int] = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        # Max number of "other cameras" whose obs may be packed into a message.
        self.n_msg_others = n_agents - 1 if n_msg_others is None else n_msg_others
        self.tau = tau

        self.agent_embedding = nn.Embedding(n_agents, hidden)
        self.obs_encoder = _mlp(obs_dim, hidden, hidden, n_layers=2)   # -> (N, hidden)
        # C: for each camera, decide to communicate with (any of) the other cameras.
        # We score each pair (sender i, receiver j) from their obs embeddings.
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        # M: LSTM reads the (padded, selected) other-camera embeddings.
        self.message_lstm = nn.LSTM(hidden, msg_dim, batch_first=True)
        # P: own obs embedding + agent identity + message -> action logits.
        self.policy = _mlp(hidden + hidden + msg_dim, hidden, n_actions, n_layers=2)

    # ------------------------------------------------------------- message

    def select_and_encode(self, obs_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce per-camera messages from a joint observation encoding.

        Args:
            obs_emb: per-camera observation embeddings ``(B, N, hidden)``.

        Returns:
            ``(messages, gates)`` where ``messages`` is ``(B, N, msg_dim)`` and
            ``gates`` is ``(B, N, N)`` soft communication weights
            (``gates[:, i, j]`` = how much camera i attends to camera j; the
            self-diagonal is zero).
        """
        batch, n, h = obs_emb.shape
        q = self.query(obs_emb)          # (B, N, h)
        k = self.key(obs_emb)            # (B, N, h)
        diag = torch.eye(n, dtype=torch.bool, device=obs_emb.device)  # self-diagonal
        # Pairwise attention logits; self-diagonal masked out (out-of-place so
        # autograd survives).
        logits = torch.bmm(q, k.transpose(-2, -1)) / (h ** 0.5)      # (B, N, N)
        logits = logits.masked_fill(diag, -float("inf"))
        gates = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=False)
        gates = gates.masked_fill(diag, 0.0)                          # (B, N, N)

        # For each camera i, the message is an LSTM over the other cameras'
        # observation embeddings, weighted by the communication gates.
        others = torch.tensor(
            [[j for j in range(n) if j != i] for i in range(n)],
            dtype=torch.long, device=obs_emb.device,
        )                                                             # (N, N-1)
        seq = obs_emb[:, others]          # (B, N, N-1, h)  -- gather others' emb
        i_idx = torch.arange(n, device=obs_emb.device).view(n, 1)
        w = gates[:, i_idx, others]       # (B, N, N-1)  -- gate(i -> each other)
        seq = seq * w.unsqueeze(-1)

        flat = seq.reshape(batch * n, n - 1, h)
        lstm_out, _ = self.message_lstm(flat)
        messages = lstm_out[:, -1, :].reshape(batch, n, -1)          # (B, N, msg_dim)
        return messages, gates

    # -------------------------------------------------------------- policy

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits from observations ``(B, N, O)``.

        Returns:
            ``(logits (B, N, A), gates (B, N, N))``.
        """
        batch, n, _ = obs.shape
        emb = self.obs_encoder(obs)                                  # (B, N, hidden)
        messages, gates = self.select_and_encode(emb)
        agent_ids = torch.arange(n, device=obs.device)
        identity = self.agent_embedding(agent_ids).view(1, n, -1).expand(batch, -1, -1)
        policy_in = torch.cat([emb, identity, messages], dim=-1)
        logits = self.policy(policy_in)
        return logits, gates


class I2CCritic(nn.Module):
    """Centralized MADDPG-style critic over the joint observation + action."""

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_encoder = _mlp(obs_dim, hidden, hidden, n_layers=2)
        self.net = _mlp(n_agents * (hidden + n_actions), hidden, n_agents, n_layers=2)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``obs`` (B, N, O), ``actions`` (B, N) -> per-agent Q ``(B, N)``."""
        batch, n, _ = obs.shape
        emb = self.obs_encoder(obs)                                   # (B, N, hidden)
        one_hot = torch.nn.functional.one_hot(
            actions.long(), self.n_actions).float()                    # (B, N, A)
        joint = torch.cat([emb, one_hot], dim=-1).reshape(batch, -1)  # (B, N*(H+A))
        return self.net(joint)                                        # (B, N)


# ============================================================================
# Trainer
# ============================================================================


@dataclass
class I2CConfig:
    """Configuration for the MATE-native I2C trainer."""

    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    update_every: int = 8          # train on the buffer after every N episodes
    batch_size: int = 32
    replay_capacity: int = 1000
    hidden: int = 128              # network width
    msg_dim: int = 64              # message hidden dim (M head)
    tau: float = 1.0               # gumbel temperature for the C selector
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    target_update_every: int = 10  # syncs per ... train steps (polyak applied each update)
    polyak: float = 0.995
    entropy_coef: float = 0.01
    blz_coef: float = 0.01         # I2C "blz" KL regularizer weight
    blz_beta: float = 8.0          # temperature for the Boltzmann-of-Q target
    epsilon: float = 0.05          # exploration floor during collection
    grad_norm_clip: float = 10.0
    device: str = "cpu"
    seed: int = 0
    outdir: str = "ckpts/i2c"
    wandb_api: Optional[str] = None


class I2CTrainer:
    """Off-policy MADDPG-style I2C trainer on the MATE camera team."""

    def __init__(self, config: I2CConfig) -> None:
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
        self.actor = I2CPolicy(
            self.env.obs_shape, self.env.n_agents, self.env.n_actions,
            hidden=config.hidden, msg_dim=config.msg_dim, tau=config.tau,
        ).to(self.device)
        self.critic = I2CCritic(
            self.env.obs_shape, self.env.n_agents, self.env.n_actions,
            hidden=config.hidden,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = RODEEpisodeReplay(config.replay_capacity, self.device)
        self.wandb = WandbLogger("mate-i2c", "i2c", dict(config.__dict__), config.wandb_api)

    # ------------------------------------------------------------ collection

    def collect_episode(self, episode_index: int):
        obs, state = self.env.reset(seed=self.config.seed + episode_index)
        builder = RODEEpisodeBuilder(self.env.n_agents, self.env.obs_shape, self.env.n_actions)
        builder.start(obs, state, self.env.get_avail_actions())
        total_reward = 0.0
        n_steps = 0
        for _ in range(self.config.episode_limit):
            n_steps += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.actor(obs_t)          # (1, N, A)
                actions = Categorical(logits=logits).sample()[0]   # (N,)
            if self.config.epsilon > 0 and np.random.random() < self.config.epsilon:
                actions = torch.randint(
                    0, self.env.n_actions, (self.env.n_agents,), device=self.device
                )
            action_array = actions.cpu().numpy()
            reward, done, _ = self.env.step(action_array)
            next_obs = self.env.get_obs()
            next_state = self.env.get_state()
            builder.add(action_array, reward, done, next_obs, next_state,
                        self.env.get_avail_actions())
            obs, state = next_obs, next_state
            total_reward += reward
            if done:
                break
        return builder.finish(), total_reward, total_reward / max(1, n_steps)

    # --------------------------------------------------------------- helpers

    def _blz_distribution(self, obs: torch.Tensor) -> torch.Tensor:
        """Boltzmann-of-Q target over each agent's actions (I2C 'blz').

        Computed from the *current* critic by sweeping each agent's action
        while holding the others' sampled actions fixed (as in the paper's
        ``blz_distribution``).  ``obs`` is ``(B, N, O)``.
        """
        B, N, _ = obs.shape
        A = self.env.n_actions
        with torch.no_grad():
            # sample the others' actions once, then sweep self over all A
            logits, _ = self.actor(obs)
            other_actions = Categorical(logits=logits).sample()   # (B, N)
            q_all = torch.zeros(B, N, A, device=self.device)
            for a in range(A):
                sweep = other_actions.clone()
                for idx in range(N):
                    sweep[:, idx] = torch.full_like(sweep[:, idx], a)
                    q_all[:, idx, a] = self.critic(obs, sweep)[:, idx]
                    sweep[:, idx] = other_actions[:, idx]   # restore others' actions
            # normalize via softmax with beta temperature
            q_all = q_all - q_all.mean(-1, keepdim=True)
            q_all = q_all / q_all.abs().max(-1, keepdim=True)[0].clamp_min(1e-6)
            blz = torch.exp(self.config.blz_beta * q_all)
            blz = blz / blz.sum(-1, keepdim=True)
        return blz

    # ----------------------------------------------------------------- update

    def update(self, batch: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        """One MADDPG update: critic TD + actor PG (+ blz KL), all in TensorFlow-free Torch."""
        cfg = self.config
        obs = batch["obs"][:, :-1]          # (B, T, N, O)
        actions = batch["actions"]          # (B, T, N)
        rewards = batch["rewards"].squeeze(-1)              # (B, T)
        terminated = batch["terminated"].squeeze(-1).float()  # (B, T)
        mask = batch["filled"].squeeze(-1).float()           # (B, T)
        obs_next = batch["obs"][:, 1:]      # (B, T, N, O)

        B, T, N, _ = obs.shape
        obs_flat = obs.reshape(B * T, N, -1)
        obs_next_flat = obs_next.reshape(B * T, N, -1)
        actions_flat = actions.reshape(B * T, N)
        rewards_flat = rewards.reshape(B * T)
        terminated_flat = terminated.reshape(B * T)
        mask_flat = mask.reshape(B * T)

        # ---- critic: TD target with target-actor next actions ----
        with torch.no_grad():
            next_logits, _ = self.target_actor(obs_next_flat)
            next_actions = Categorical(logits=next_logits).sample()
            next_q = self.target_critic(obs_next_flat, next_actions)     # (BT, N)
            next_q = next_q.mean(-1)                                     # shared reward
            targets = rewards_flat + cfg.gamma * (1 - terminated_flat) * next_q
        q_all = self.critic(obs_flat, actions_flat)                      # (BT, N)
        q_shared = q_all.mean(-1)
        critic_loss = (((q_shared - targets.detach()) ** 2) * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)

        # ---- actor: REINFORCE policy gradient + I2C blz KL regularizer ----
        # Categorical policy: the PG must go through log_prob of the sampled
        # action (sampling itself is non-differentiable).  The critic value is
        # detached so the actor graph never depends on the online critic -- the
        # critic optimiser steps inplace between the two backward passes and
        # would otherwise corrupt the actor's autograd graph.
        logits, _ = self.actor(obs_flat)
        dist = Categorical(logits=logits)
        sampled = dist.sample()                                          # (BT, N)
        log_p_sampled = dist.log_prob(sampled)                           # (BT, N)
        with torch.no_grad():
            q_sampled = self.critic(obs_flat, sampled)                   # (BT, N)
        q_shared = q_sampled.mean(-1, keepdim=True)                      # (BT, 1)
        actor_pg = -(log_p_sampled * q_shared * mask_flat.unsqueeze(-1)).sum() / mask_flat.sum().clamp_min(1.0)
        entropy = dist.entropy()
        entropy_loss = -cfg.entropy_coef * (entropy * mask_flat.unsqueeze(-1)).sum() / mask_flat.sum().clamp_min(1.0)
        # blz KL regularizer (observation-only policy vs boltzmann-of-Q target)
        blz_targets = self._blz_distribution(obs_flat)                   # (BT, N, A)
        log_p = dist.logits - torch.logsumexp(dist.logits, dim=-1, keepdim=True)
        blz_log = torch.log(blz_targets.clamp_min(1e-8))
        blz_kl = (blz_targets * (blz_log - log_p)).sum(-1)               # (BT, N)
        blz_loss = cfg.blz_coef * (blz_kl * mask_flat.unsqueeze(-1)).sum() / mask_flat.sum().clamp_min(1.0)

        actor_loss = actor_pg + entropy_loss + blz_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_norm_clip)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_norm_clip)
        self.actor_optimizer.step()

        # polyak target updates
        self._polyak(self.target_actor, self.actor, cfg.polyak)
        self._polyak(self.target_critic, self.critic, cfg.polyak)

        return (float(actor_loss.detach().cpu()),
                float(critic_loss.detach().cpu()),
                float(blz_loss.detach().cpu()))

    @staticmethod
    def _polyak(target: nn.Module, online: nn.Module, tau: float) -> None:
        for t, o in zip(target.parameters(), online.parameters()):
            t.data.mul_(tau).add_(o.data, alpha=1.0 - tau)

    # ----------------------------------------------------------------- train

    def train(self) -> None:
        cfg = self.config
        last = (None, None, None)
        for episode_index in range(cfg.episodes):
            episode, reward, coverage_rate = self.collect_episode(episode_index)
            self.replay.add(episode)
            if (episode_index + 1) % cfg.update_every == 0 and len(self.replay) >= cfg.batch_size:
                last = self.update(self.replay.sample(cfg.batch_size))
            if (episode_index + 1) % 10 == 0:
                msg = f"episode {episode_index + 1}/{cfg.episodes} | reward {reward:.3f}"
                if last[0] is not None:
                    msg += (f" | actor {last[0]:.4f} | critic {last[1]:.4f} | blz {last[2]:.4f}")
                print(msg, flush=True)
                self.wandb.log({"reward": reward,
                                "coverage_rate": coverage_rate,
                                "actor_loss": last[0] if last[0] is not None else float("nan"),
                                "critic_loss": last[1] if last[1] is not None else float("nan"),
                                "blz": last[2] if last[2] is not None else float("nan")},
                               step=episode_index + 1)
        outdir = Path(cfg.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "config": cfg.__dict__}, outdir / "latest.pt")
        self.wandb.finish()
        self.env.close()


def run_i2c(args: Any) -> None:
    """Entry point wired into ``train.py``'s algorithm dispatcher."""
    I2CTrainer(I2CConfig(
        config=args.i2c_config,
        levels=args.i2c_levels,
        episode_limit=args.i2c_episode_limit,
        episodes=args.i2c_episodes,
        update_every=args.i2c_update_every,
        batch_size=args.i2c_batch_size,
        hidden=args.i2c_hidden,
        msg_dim=args.i2c_msg_dim,
        tau=args.i2c_tau,
        blz_coef=args.i2c_blz_coef,
        epsilon=args.i2c_epsilon,
        actor_lr=args.i2c_actor_lr,
        critic_lr=args.i2c_critic_lr,
        device=args.device,
        seed=args.seed,
        outdir=args.i2c_outdir,
        wandb_api=args.wandb_api,
    )).train()
