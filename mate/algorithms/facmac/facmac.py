"""FACMAC — Factored Multi-Agent Centralised Policy Gradients — for MATE.

A compact, modern port of FACMAC (Peng et al., NeurIPS 2021,
arXiv:2003.06709) from its PyMARL implementation.  FACMAC is an off-policy
multi-agent policy-gradient method (MADDPG-style).  The two key pieces:

* a **factorized central critic** — one per-agent Q-value output that
  conditions on the *joint* observation and *joint* action (centralised but
  factored), unlike MADDPG's single global Q;
* a **gumbel-softmax reparameterisation** of the discrete policies, so the
  actor's one-hot actions are differentiable functions of the policy logits
  and the policy gradient flows through the critic::

      actor_loss = -Q( joint_obs(a), a_hat )          (a_hat = gumbel(a))
      critic_loss = MSE( Q(...) - TD(lambda)-target )

MATE is discrete-action and gives a single team reward per step, which is
shared across agents.  Only `numpy` + `torch` are required, and the trainer
runs on the shared camera-team env factory like the other MATE algorithms.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms.rode.replay import RODEEpisodeBuilder, RODEEpisodeReplay
from mate.algorithms._wandb import WandbLogger


# ============================================================================
# Networks
# ============================================================================


class GumbelActor(nn.Module):
    """Shared MLP policy outputting logits; gumbel passes are built in trainer."""

    def __init__(self, obs_shape: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_shape, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., O) -> logits (..., A)."""
        return self.fc(obs)

    def act(self, obs: torch.Tensor, avail: Optional[torch.Tensor] = None,
            greedy: bool = False) -> torch.Tensor:
        """Sample an action (index) per agent from the policy.

        obs: (..., O); avail: (..., A) bool mask, optional. Returns (...,) long.
        """
        logits = self.forward(obs)
        if avail is not None:
            logits = logits + torch.log(avail.float().clamp_min(1e-6))
        if greedy:
            return logits.argmax(-1)
        return Categorical(logits=logits).sample()

    def gumbel_onehot(self, obs: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Differentiable soft one-hot actions via gumbel-softmax.

        obs: (..., O) -> soft actions (..., A). Used to push actor gradients
        through the critic.
        """
        return F.gumbel_softmax(self.forward(obs), tau=tau, hard=False, dim=-1)


class FACMACDiscreteCritic(nn.Module):
    """Centralised-but-factored critic.

    Takes the *joint* observation (all agents' obs concatenated) and the
    *joint* one-hot action as input and outputs one Q-value per agent.
    """

    def __init__(self, n_agents: int, obs_shape: int, n_actions: int,
                 hidden: int = 64) -> None:
        super().__init__()
        self.n_agents = n_agents
        joint_in = n_agents * obs_shape + n_agents * n_actions
        self.fc = nn.Sequential(
            nn.Linear(joint_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents),
        )

    def forward(self, joint_obs: torch.Tensor, joint_act: torch.Tensor) -> torch.Tensor:
        """joint_obs: (B,T,N,O), joint_act: (B,T,N,A) -> Q (B,T,N)."""
        B, T = joint_obs.shape[:2]
        jo = joint_obs.reshape(B, T, -1)
        ja = joint_act.reshape(B, T, -1) if joint_act is not None else \
            torch.zeros(B, T, self.n_agents, device=joint_obs.device)
        x = torch.cat([jo, ja], dim=-1)
        return self.fc(x)


# ============================================================================
# Config + Trainer
# ============================================================================


@dataclass
class FACMACConfig:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 2000
    total_env_steps: Optional[int] = None
    seed: int = 0
    device: str = "cpu"
    hidden: int = 64

    # RL
    gamma: float = 0.99
    td_lambda: float = 0.8
    lr: float = 2.5e-3          # actor lr
    critic_lr: float = 5e-4
    grad_norm_clip: float = 10.0
    gumbel_tau: float = 1.0
    temperature: float = 1.0

    # exploration
    epsilon_start: float = 0.5
    epsilon_finish: float = 0.05
    epsilon_anneal_steps: int = 50000

    # replay / scheduling
    batch_size: int = 32
    buffer_size: int = 5000
    update_every: int = 1        # learn every N env steps
    target_update_interval: int = 200   # hard target updates (critic steps)
    target_update_tau: float = 1.0      # 1.0 = hard, <1 = soft (polyak)

    # io
    save_every: int = 1000
    outdir: str = "ckpts/facmac"
    wandb_api: Optional[str] = None
    num_envs: int = 1


class FACMACTrainer:
    """Off-policy FACMAC trainer on the MATE camera team."""

    def __init__(self, config: FACMACConfig) -> None:
        self.config = config
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.env = make_mate_rode_env(
            config=config.config, levels=config.levels,
            episode_limit=config.episode_limit, render_mode="rgb_array")
        n, obs, na = self.env.n_agents, self.env.obs_shape, self.env.n_actions
        self.n_agents, self.obs_shape, self.n_actions = n, obs, na
        self.rollout = None
        if config.num_envs > 1:
            from mate.algorithms.batched_rollout import BatchedRollout
            self.rollout = BatchedRollout(lambda: _make_facmac_vec_env(config), config.num_envs)

        self.actor = GumbelActor(obs, na, config.hidden).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = FACMACDiscreteCritic(n, obs, na, config.hidden).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = RODEEpisodeReplay(config.buffer_size, self.device)

        self.epsilon = config.epsilon_start
        self._env_steps = 0
        self._critic_steps = 0
        self.wandb = WandbLogger("mate-facmac", "facmac", dict(config.__dict__), config.wandb_api)

    # ------------------------------------------------------- exploration
    def _schedule_epsilon(self) -> None:
        cfg = self.config
        frac = min(1.0, self._env_steps / max(cfg.epsilon_anneal_steps, 1))
        self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_finish - cfg.epsilon_start)

    # ----------------------------------------------------------- collect
    def collect_episode(self, episode_index: int) -> float:
        obs, state = self.env.reset(seed=self.config.seed + episode_index)
        builder = RODEEpisodeBuilder(
            self.env.n_agents, self.env.obs_shape, self.env.n_actions)
        builder.start(obs, state, self.env.get_avail_actions())
        total = 0.0
        n_steps = 0
        for _ in range(self.config.episode_limit):
            n_steps += 1
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                act = self.actor.act(o, greedy=False)
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                act = torch.randint(self.env.n_actions, (self.n_agents,),
                                    device=self.device)
            action_array = act.cpu().numpy()
            reward, done, _ = self.env.step(action_array)
            next_obs, next_state = self.env.get_obs(), self.env.get_state()
            builder.add(action_array, reward, done, next_obs, next_state,
                        self.env.get_avail_actions())
            obs, state = next_obs, next_state
            self._env_steps += 1
            total += reward
            if done:
                break
        self.replay.add(builder.finish())
        return total, total / max(1, n_steps)

    def batched_act(self, payloads):
        """Compute actions for all parallel slots (parent process). Mirrors
        ``collect_episode``'s GumbelActor + epsilon logic, looped per env."""
        actions = []
        for payload in payloads:
            o = torch.as_tensor(payload["obs"], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                act = self.actor.act(o, greedy=False)
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                act = torch.randint(self.n_actions, (self.n_agents,), device=self.device)
            actions.append(act.cpu().numpy())
        return np.stack(actions)

    def collect_batch(self, first_ep):
        """Concurrently collect ``num_envs`` episodes; add each to replay."""
        num = self.rollout.num_envs
        payloads = self.rollout.reset(
            [self.config.seed + first_ep + i for i in range(num)])
        builders = [RODEEpisodeBuilder(self.n_agents, self.obs_shape, self.n_actions)
                    for _ in range(num)]
        for b, p in zip(builders, payloads):
            b.start(p["obs"], p["state"], p["avail_actions"])
        total = np.zeros(num, dtype=np.float64)
        n_steps = np.zeros(num, dtype=np.int64)
        for _ in range(self.config.episode_limit):
            actions = self.batched_act(payloads)
            step = self.rollout.step(actions)
            for i in range(num):
                n_steps[i] += 1
                self._env_steps += 1
                terminal = step.infos[i].get("final_observation")
                fin = terminal if step.done[i] else step.obs[i]
                builders[i].add(actions[i], float(step.rewards[i]), bool(step.done[i]),
                                fin["obs"], fin["state"], fin["avail_actions"])
                total[i] += float(step.rewards[i])
                if step.done[i]:
                    self.replay.add(builders[i].finish())
                    # Auto-reset continuation: re-seed the builder with the
                    # post-reset obs for the fresh episode's first transition.
                    builders[i].start(step.obs[i]["obs"], step.obs[i]["state"],
                                      step.obs[i]["avail_actions"])
            payloads = step.obs
            if all(step.done):
                break
        mean = float(total.mean())
        avg_steps = float(np.mean(np.maximum(n_steps, 1)))
        return mean, mean / max(avg_steps, 1e-8)

    # ----------------------------------------------------------- helpers
    def _onehot(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (..., N) long -> one-hot (..., N, A)."""
        return F.one_hot(actions.long(), self.n_actions).float()

    def _td_lambda_targets(self, rewards, terminated, mask, target_qs):
        """all (B, T, N) -> lambda-return (B, T, N).

        Port of ``utils.rl_utils.build_td_lambda_targets``, aligned so that
        every term is over the reward length T (target_qs already shifted to
        the next timestep by the caller).
        """
        B, T, N = target_qs.shape
        ret = target_qs.new_zeros(B, T, N)
        ret[:, -1] = rewards[:, -1] + self.config.gamma * (1 - terminated[:, -1]) * target_qs[:, -1]
        for t in range(T - 2, -1, -1):
            ret[:, t] = (self.config.td_lambda * self.config.gamma * ret[:, t + 1]
                         + mask[:, t] * (rewards[:, t] + (1 - self.config.td_lambda)
                                         * self.config.gamma * target_qs[:, t + 1]
                                         * (1 - terminated[:, t])))
        return ret

    # ------------------------------------------------------------ update
    def update(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        """One FACMAC actor+critic update over a sampled batch.

        Batch dims: obs (B,T+1,N,O), actions (B,T,N), rewards/terminated/filled
        (B,T,1).  We align over T (the reward length) throughout.
        """
        cfg = self.config
        obs = batch["obs"][:, :-1]                       # (B,T,N,O)
        obs_next = batch["obs"][:, 1:]                   # (B,T,N,O) shifted
        act_onehot = self._onehot(batch["actions"])      # (B,T,N,A)
        rewards = batch["rewards"].expand(-1, -1, self.n_agents)   # (B,T,N)
        terminated = batch["terminated"].float().expand(-1, -1, self.n_agents)
        filled = batch["filled"].float().expand(-1, -1, self.n_agents)
        mask = filled.clone()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # target one-hots on the next observations (greedy target actor)
        with torch.no_grad():
            tgt_act_idx = self.target_actor.act(obs_next, greedy=True)  # (B,T,N)
            tgt_onehot = self._onehot(tgt_act_idx)                       # (B,T,N,A)
            target_vals = self.target_critic(obs_next, tgt_onehot)       # (B,T,N)

        targets = self._td_lambda_targets(rewards, terminated, filled, target_vals)  # (B,T,N)
        q_taken = self.critic(obs, act_onehot)                             # (B,T,N)
        td_error = (q_taken - targets.detach())
        masked_td = td_error * mask
        critic_loss = (masked_td ** 2).sum() / mask.sum().clamp_min(1.0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_norm_clip)
        self.critic_optimizer.step()
        self._critic_steps += 1

        # actor: gumbel-softmax one-hots (differentiable) through critic
        a_hat = self.actor.gumbel_onehot(obs, tau=cfg.gumbel_tau)         # (B,T,N,A)
        chosen_q = self.critic(obs, a_hat)                                 # (B,T,N)
        pg_loss = -((chosen_q * mask).sum()) / mask.sum().clamp_min(1.0)

        self.actor_optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_norm_clip)
        self.actor_optimizer.step()

        self._update_targets()
        return float(critic_loss.detach().cpu()), float(pg_loss.detach().cpu())

    def _update_targets(self) -> None:
        cfg = self.config
        if cfg.target_update_tau >= 1.0:
            if self._critic_steps % cfg.target_update_interval == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_critic.load_state_dict(self.critic.state_dict())
        else:  # soft (polyak)
            tau = cfg.target_update_tau
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # ------------------------------------------------------------- train
    def train(self) -> None:
        cfg = self.config
        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        env_steps_at_last_learn = 0
        ep = 0
        while ep < cfg.episodes:
            if (cfg.total_env_steps is not None
                    and self._env_steps >= cfg.total_env_steps):
                break
            if self.rollout is not None:
                total, coverage_rate = self.collect_batch(ep)
                ep += self.rollout.num_envs
            else:
                total, coverage_rate = self.collect_episode(ep)
                ep += 1
            self._schedule_epsilon()
            if (self._env_steps - env_steps_at_last_learn) >= cfg.update_every \
                    and len(self.replay) >= cfg.batch_size:
                batch = self.replay.sample(cfg.batch_size)
                cl, pl = self.update(batch)
                env_steps_at_last_learn = self._env_steps
                if ep % 20 == 0:
                    print(f"[facmac] ep={ep} rew={total:.2f} "
                          f"critic_loss={cl:.3f} pg_loss={pl:.4f} "
                          f"eps={self.epsilon:.2f}")
                    self.wandb.log({"reward": total, "coverage_rate": coverage_rate,
                                    "critic_loss": cl,
                                    "pg_loss": pl, "epsilon": self.epsilon}, step=ep)
            if ep % cfg.save_every == 0:
                self.save()

    def save(self) -> None:
        out = Path(self.config.outdir)
        torch.save(self.actor.state_dict(), out / "actor.th")
        torch.save(self.critic.state_dict(), out / "critic.th")

    def load(self) -> None:
        out = Path(self.config.outdir)
        self.actor.load_state_dict(torch.load(out / "actor.th", weights_only=True))
        self.critic.load_state_dict(torch.load(out / "critic.th", weights_only=True))

    def close(self) -> None:
        self.wandb.finish()
        self.env.close()
        if self.rollout is not None:
            self.rollout.close()

    def evaluate(self, n_episodes: int = 10) -> float:
        """Mean undiscounted return under the greedy policy (no exploration)."""
        self.actor.eval()
        total = 0.0
        for ep in range(n_episodes):
            obs, state = self.env.reset(seed=1000 + ep)
            ep_rew = 0.0
            for _ in range(self.config.episode_limit):
                o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    act = self.actor.act(o, greedy=True)
                reward, done, _ = self.env.step(act.cpu().numpy())
                obs = self.env.get_obs()
                ep_rew += float(reward)
                if done:
                    break
            total += ep_rew
        self.actor.train()
        return total / max(n_episodes, 1)


def _make_facmac_vec_env(config):
    """Top-level, cloudpickle-safe env factory for the batched rollout worker."""
    from mate.algorithms.vec_adapter import MateRodeVecEnv
    return MateRodeVecEnv(make_mate_rode_env(
        config=config.config, levels=config.levels,
        episode_limit=config.episode_limit))


def run_facmac(args: Any) -> None:
    """Entry point. `args` may be a FACMACConfig or namespace with matching attrs."""
    config = args if isinstance(args, FACMACConfig) else _coerce(args)
    trainer = FACMACTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


def _coerce(args: Any) -> FACMACConfig:
    names = {f for f in FACMACConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in vars(args).items() if k in names}
    return FACMACConfig(**kwargs)
