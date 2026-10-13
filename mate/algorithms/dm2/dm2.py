"""DM² — Distributed Multi-Agent RL for Distribution Matching — for MATE.

A compact, modern port of DM² (Wang et al., arXiv:2206.00233) from the
original PyMARL/MAPPO implementation.  The headline idea: on-policy MAPPO
training where a per-agent GAIL discriminator reshapes the reward so the
learned policy's state/(state, action) distribution matches expert
demonstrations::

    r_hat = -log(1 - D(s,a))
    r     = env_r                     (rew_type="env")
          = r_hat                     (rew_type="gail")
          = r_hat * coef + env_r      (rew_type="mixed")

The discriminator is trained with binary cross-entropy plus a WGAN gradient
penalty; `rew_type="mixed"` is the paper's headline variant.

This port keeps only numpy + torch (the original's `stable_baselines3`
running-mean-stat and SC2/SMAC scaffolding are replaced/inlined) and runs on
the shared MATE camera-team env factory, matching the other MATE algorithms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms._wandb import WandbLogger


# ============================================================================
# Small helpers (inlined from MAPPO's mappo_utils / sb3)
# ============================================================================


def _check(x: Any) -> torch.Tensor:
    """Coerce numpy/tensor input to a torch tensor (MAPPO ``check``)."""
    return torch.as_tensor(x) if not torch.is_tensor(x) else x


def _huber_loss(e: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(e, torch.zeros_like(e), delta=delta)


def _linear_anneal(value: float, step: int, total: int) -> float:
    frac = min(1.0, step / max(total, 1))
    return value * (1.0 - frac)


class RunningMeanStd:
    """Running mean/variance (replaces `stable_baselines3` dependency)."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        batch = np.asarray(x).ravel()
        if batch.size == 0:
            return
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = batch.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = m2 / (total + self.epsilon)
        self.mean, self.var, self.count = new_mean, new_var, total


# ============================================================================
# Networks — shared (homogeneous) actor & critic over per-agent observations
# ============================================================================


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class DM2Actor(nn.Module):
    """Categorical policy over per-agent obs (MLP base)."""

    def __init__(self, obs_shape: int, hidden: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = _mlp(obs_shape, hidden, n_actions)

    def logp_and_entropy(
        self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.trunk(obs)
        logits = logits + torch.log(mask.clamp_min(1e-6))
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return Categorical(logits=self.trunk(obs)).sample()


class DM2Critic(nn.Module):
    """Value head over per-agent obs (local-obs IPPO critic)."""

    def __init__(self, obs_shape: int, hidden: int) -> None:
        super().__init__()
        self.trunk = _mlp(obs_shape, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)


# ============================================================================
# GAIL discriminator — the distribution-matching core
# ============================================================================


class DM2Discriminator(nn.Module):
    """Per-agent discriminator: BC + WGAN gradient penalty, plus pseudo-reward.

    ``gail_state_discrim`` decides whether the input is the observation alone
    (state-matching) or the observation concatenated with the raw action index
    (state/action distribution matching).
    """

    def __init__(
        self,
        obs_shape: int,
        hidden: int,
        n_actions: int,
        state_discrim: bool,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        input_dim = obs_shape if state_discrim else obs_shape + 1
        self.trunk = _mlp(input_dim, hidden, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.trunk.parameters())
        self.ret_rms = RunningMeanStd()
        self.returns: Optional[torch.Tensor] = None
        self.n_actions = n_actions

    # -- input builders ------------------------------------------------
    def _inputs(self, obs: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        x = _check(obs).float().to(self.device)
        if actions is not None:
            a = _check(actions).float().to(self.device)
            if a.dim() < x.dim():
                a = a.unsqueeze(-1)  # action index vector -> column
            x = torch.cat([x, a], dim=-1)
        return x.reshape(-1, x.shape[-1])

    def _grad_pen(
        self, expert_in: torch.Tensor, policy_in: torch.Tensor, lam: float = 10.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.rand(expert_in.size(0), 1, device=self.device)
        mixup = (alpha * expert_in + (1 - alpha) * policy_in).requires_grad_(True)
        d = self.trunk(mixup)
        grad = torch.autograd.grad(
            outputs=d, inputs=mixup, grad_outputs=torch.ones_like(d),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        grad_norm = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return lam * grad_norm, grad_norm

    # -- training ------------------------------------------------------
    def update(
        self, policy_in: torch.Tensor, expert_in: torch.Tensor
    ) -> dict[str, float]:
        self.train()
        pd, ed = self.trunk(policy_in), self.trunk(expert_in)
        loss = F.binary_cross_entropy_with_logits(
            ed, torch.ones_like(ed)) + F.binary_cross_entropy_with_logits(
            pd, torch.zeros_like(pd))
        grad_pen, grad_norm = self._grad_pen(expert_in, policy_in)
        self.optimizer.zero_grad()
        (loss + grad_pen).backward()
        self.optimizer.step()
        with torch.no_grad():
            return {
                "gail_loss": (loss + grad_pen).item(),
                "gail_grad_pen": grad_pen.item(),
                "policy_discr": torch.sigmoid(pd).mean().item(),
                "expert_discr": torch.sigmoid(ed).mean().item(),
            }

    # -- reward --------------------------------------------------------
    def predict_reward(self, obs: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        self.eval()
        x = self._inputs(obs, actions)
        with torch.no_grad():
            s = torch.sigmoid(self.trunk(x))
            r = (s.log() - (1.0 - s).log()).reshape(*obs.shape[:-1], 1)
        return r

    def save(self, path: Path, idx: int) -> None:
        path = Path(path)
        torch.save(self.trunk.state_dict(), path / f"discriminator_{idx}.th")

    def load(self, path: Path, idx: int) -> None:
        path = Path(path)
        self.trunk.load_state_dict(
            torch.load(path / f"discriminator_{idx}.th",
                       map_location=self.device, weights_only=True))


# ============================================================================
# Expert / agent data buffers
# ============================================================================


class DM2Storage:
    """Holds per-agent (obs, action-index) samples, with optional expert file load."""

    def __init__(self, buffer_size: Optional[int], epath: Optional[str] = None,
                 agent_idx: int = 0, exp_eps: int = 1000, same_idxs: bool = True) -> None:
        self.batches: deque[dict[str, np.ndarray]] = deque(maxlen=buffer_size or 0)
        self.exp_eps = exp_eps
        if epath is not None:
            try:
                data = np.load(epath + ".npz", allow_pickle=True, mmap_mode="r")["arr_0"]
            except FileNotFoundError:
                data = np.load(epath + ".npy", allow_pickle=True, mmap_mode="r")
            data = list(data[:exp_eps]) if same_idxs else list(
                data[exp_eps * agent_idx: exp_eps * (agent_idx + 1)])
            self.batches = deque(data, maxlen=exp_eps)

    def store_ep(self, obs: np.ndarray, actions: np.ndarray) -> None:
        self.batches.append({"obs": obs, "actions": actions})

    def sample(self, batch_size: int) -> list[dict[str, np.ndarray]]:
        idx = np.random.choice(len(self.batches), batch_size, replace=False)
        return [self.batches[i] for i in idx]

    def __len__(self) -> int:
        return len(self.batches)


class SILStorage:
    """Self-imitation buffer kept by discounted episode return (min-heap)."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.batches: list[tuple[float, dict[str, np.ndarray]]] = []

    def store_ep(self, obs: np.ndarray, actions: np.ndarray, ret: float) -> None:
        self.batches.append((ret, {"obs": obs, "actions": actions}))
        self.batches.sort(key=lambda x: x[0], reverse=True)
        del self.batches[self.buffer_size:]
        return None

    def sample(self, batch_size: int) -> list[dict[str, np.ndarray]]:
        idx = np.random.choice(len(self.batches), batch_size, replace=False)
        return [self.batches[i][1] for i in idx]

    def __len__(self) -> int:
        return len(self.batches)


# ============================================================================
# Config + Trainer
# ============================================================================


@dataclass
class DM2Config:
    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    seed: int = 0
    device: str = "cpu"
    hidden: int = 64

    # GAIL / reward shaping
    rew_type: str = "mixed"            # "env" | "gail" | "mixed"
    update_gail: bool = True
    gail_rew_coef: float = 1.0
    gail_epoch: int = 3
    gail_batch_size: int = 8
    gail_state_discrim: bool = True
    gail_sil: bool = False
    gail_buffer_size: int = 100        # eps held in agent storage
    gail_data_paths: list[str] = None  # expert demo dirs (per agent or single)
    gail_exp_eps: int = 1000
    gail_exp_use_same_data_idxs: bool = True

    # IPPO
    lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_param: float = 0.2
    ppo_epoch: int = 15
    num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    use_huber_loss: bool = True
    huber_delta: float = 10.0

    # scheduling / io
    use_linear_lr_decay: bool = True
    train_after: int = 16              # episodes collected before first update
    save_every: int = 100
    outdir: str = "ckpts/dm2"
    wandb_api: Optional[str] = None


class DM2Trainer:
    """On-policy (MAPPO-style) DM² trainer on the MATE camera team."""

    def __init__(self, config: DM2Config) -> None:
        self.config = config
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.env = make_mate_rode_env(
            config=config.config, levels=config.levels,
            episode_limit=config.episode_limit, render_mode="rgb_array")
        n, obs, na = self.env.n_agents, self.env.obs_shape, self.env.n_actions
        self.n_agents, self.obs_shape, self.n_actions = n, obs, na

        self.actor = DM2Actor(obs, config.hidden, na).to(self.device)
        self.critic = DM2Critic(obs, config.hidden).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        ds = config.gail_state_discrim
        self.discriminators = [
            DM2Discriminator(obs, config.hidden, na, ds, self.device)
            for _ in range(n)]
        paths = config.gail_data_paths or [None] * n
        if config.gail_exp_use_same_data_idxs and len(paths) > 1:
            paths = [paths[0]] * n
        self.agent_storage = [
            DM2Storage(config.gail_buffer_size if config.update_gail else 0)
            for _ in range(n)]
        self.expert_storage = None
        if config.rew_type in ("gail", "mixed") and not config.gail_sil:
            self.expert_storage = [
                DM2Storage(config.gail_exp_eps, p, i, config.gail_exp_eps,
                           config.gail_exp_use_same_data_idxs)
                for i, p in enumerate(paths)]
        self.sil = [SILStorage(config.gail_buffer_size) for _ in range(n)] \
            if config.gail_sil else None

        # on-policy buffer: accumulates episodes until a PPO update
        self._episodes: list[dict[str, np.ndarray]] = []
        self._env_steps = 0
        self.wandb = WandbLogger("mate-dm2", "dm2", dict(config.__dict__), config.wandb_api)

    # ------------------------------------------------------------ data
    def collect_episode(self, seed: int) -> float:
        obs, _ = self.env.reset(seed=seed)
        obs_l, act_l, rew_l, done_l = [], [], [], []
        total = 0.0
        for _ in range(self.env.episode_limit):
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_l.append(obs)
            act = self.actor.act(o).cpu().numpy()
            act_l.append(act)
            reward, done, _ = self.env.step(act)
            rew_l.append(float(reward))
            total += float(reward)
            obs = self.env.get_obs()
            done_l.append(done)
            self._env_steps += 1
            if done:
                break
        self._episodes.append({
            "obs": np.asarray(obs_l, dtype=np.float32),         # (T, N, O)
            "actions": np.asarray(act_l, dtype=np.int64),       # (T, N)
            "rewards": np.asarray(rew_l, dtype=np.float32)[:, None],  # (T, 1)
            "dones": np.asarray(done_l, dtype=np.float32)[:, None],   # (T, 1)
        })
        return total

    def _buffer_tensors(self) -> dict[str, torch.Tensor]:
        """Concatenate buffered episodes along the time axis."""
        obs = torch.cat([torch.as_tensor(e["obs"], device=self.device)
                         for e in self._episodes], 0)          # (sumT, N, O)
        return {
            "obs": obs,
            "actions": torch.cat([torch.as_tensor(e["actions"], device=self.device)
                                  for e in self._episodes], 0),  # (sumT, N)
            "rewards": torch.cat([torch.as_tensor(e["rewards"], device=self.device)
                                  for e in self._episodes], 0),  # (sumT, 1)
            "dones": torch.cat([torch.as_tensor(e["dones"], device=self.device)
                                for e in self._episodes], 0),    # (sumT, 1)
        }

    def _ppo_update(self, idx: int) -> dict[str, float]:
        cfg = self.config
        buf = self._buffer_tensors()
        obs, acts, env_rew, done = buf["obs"], buf["actions"], buf["rewards"], buf["dones"]
        B = obs.shape[0]
        o = obs[:, idx]                                   # (B, O)
        a = acts[:, idx]                                  # (B,)
        mask = torch.ones((B, self.n_actions), device=self.device)

        # discriminator update
        info: dict[str, float] = {}
        if cfg.update_gail:
            es = self.sil[idx] if cfg.gail_sil else (self.expert_storage or [None] * self.n_agents)[idx]
            if es is not None and len(es) >= cfg.gail_batch_size:
                eb = es.sample(cfg.gail_batch_size)
                pb = self.agent_storage[idx].sample(cfg.gail_batch_size)
                ei = torch.cat([torch.as_tensor(b["obs"], device=self.device) for b in eb])
                pi = torch.cat([torch.as_tensor(b["obs"], device=self.device) for b in pb])
                if not cfg.gail_state_discrim:
                    ei_a = torch.cat([torch.as_tensor(b["actions"], device=self.device)
                                      for b in eb]).float().unsqueeze(-1)  # (sumT,1)
                    pi_a = torch.cat([torch.as_tensor(b["actions"], device=self.device)
                                      for b in pb]).float().unsqueeze(-1)
                    ei = torch.cat([ei, ei_a], -1)
                    pi = torch.cat([pi, pi_a], -1)
                info.update(self.discriminators[idx].update(pi, ei))

        # reward shaping + GAE advantage estimates
        rew = self._shaped_rewards(idx, obs, acts, env_rew)   # (B,1)
        values = self.critic(o)                                # (B,1)
        adv = self._gae(values, rew, done, obs)                # (B,1)
        returns = adv + values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        with torch.no_grad():
            old_logp, _ = self.actor.logp_and_entropy(o, a, mask)
        for _ in range(cfg.ppo_epoch):
            logp, ent = self.actor.logp_and_entropy(o, a, mask)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            self.actor_opt.zero_grad()
            (policy_loss - cfg.entropy_coef * ent.mean()).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()

            v = self.critic(o)
            err = returns - v
            value_loss = (_huber_loss(err, cfg.huber_delta).mean()
                          if cfg.use_huber_loss else err.pow(2).mean())
            self.critic_opt.zero_grad()
            (cfg.value_loss_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_opt.step()
        info["value_loss"] = value_loss.item()
        info["policy_loss"] = policy_loss.item()
        return info

    def _gae(self, values: torch.Tensor, rewards: torch.Tensor,
             dones: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """GAE over the concatenated buffer; returns advantages (B,1).

        The value of the final timestep is bootstrapped from the critic on the
        last observation, masked to 0 if that timestep was terminal.
        """
        cfg = self.config
        B = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        with torch.no_grad():
            bootstrap = 0.0 if (B and dones[-1].item()) else self.critic(
                obs[-1:].detach()).item()
        gae = 0.0
        for t in reversed(range(B)):
            next_v = values[t + 1].item() if t + 1 < B else bootstrap
            mask_t = (1 - dones[t].item())
            delta = rewards[t].item() + cfg.gamma * next_v * mask_t - values[t].item()
            gae = delta + cfg.gamma * cfg.gae_lambda * mask_t * gae
            adv[t] = gae
        return adv

    # --------------------------------------------------------- training

    def _shaped_rewards(self, idx: int, obs: torch.Tensor,
                        acts: torch.Tensor, env_rew: torch.Tensor) -> torch.Tensor:
        rt = self.config.rew_type
        if rt not in ("gail", "mixed"):
            return env_rew
        o = obs[:, idx]                                    # (B, O)
        acts_in = None if self.config.gail_state_discrim else acts[:, idx]
        r_hat = self.discriminators[idx].predict_reward(o, acts_in)  # (B,1)
        return r_hat if rt == "gail" else r_hat * self.config.gail_rew_coef + env_rew

    def step(self, info: dict[str, float]) -> None:
        # feed each collected episode to the per-agent discriminator buffers
        for ep in self._episodes:
            obs_np, act_np = ep["obs"], ep["actions"]      # (T,N,*)
            for i in range(self.n_agents):
                if self.agent_storage[i] is not None:
                    self.agent_storage[i].store_ep(obs_np[:, i], act_np[:, i])
                if self.sil:
                    ret_i = float(np.sum(
                        ep["rewards"][:, 0] * self.config.gamma ** np.arange(obs_np.shape[0])))
                    self.sil[i].store_ep(obs_np[:, i], act_np[:, i], ret_i)

    def train(self) -> None:
        cfg = self.config
        Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
        for ep in range(cfg.episodes):
            rt = self.collect_episode(cfg.seed + ep)
            if len(self._episodes) < cfg.train_after:
                continue
            if cfg.use_linear_lr_decay:
                lr = max(_linear_anneal(cfg.lr, ep, cfg.episodes), 1e-6)
                self.actor_opt.param_groups[0]["lr"] = lr
                self.critic_opt.param_groups[0]["lr"] = lr
            self.step({})
            info = self._ppo_update(0)
            for i in range(1, self.n_agents):
                self._ppo_update(i)
            self._episodes.clear()
            print(f"[dm2] ep={ep} rew={rt:.2f}",
                  {k: round(v, 4) for k, v in info.items()})
            self.wandb.log({"reward": rt, **info}, step=ep)
            if ep % cfg.save_every == 0:
                self.save()

    def save(self) -> None:
        out = Path(self.config.outdir)
        torch.save(self.actor.state_dict(), out / "actor.th")
        torch.save(self.critic.state_dict(), out / "critic.th")
        for i, d in enumerate(self.discriminators):
            d.save(out, i)

    def load(self) -> None:
        out = Path(self.config.outdir)
        self.actor.load_state_dict(torch.load(out / "actor.th", weights_only=True))
        self.critic.load_state_dict(torch.load(out / "critic.th", weights_only=True))
        for i, d in enumerate(self.discriminators):
            d.load(out, i)

    def close(self) -> None:
        self.wandb.finish()
        self.env.close()


def run_dm2(args: Any) -> None:
    """Entry point. `args` may be a DM2Config or a namespace w/ matching attrs."""
    config = args if isinstance(args, DM2Config) else _coerce(args)
    trainer = DM2Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


def _coerce(args: Any) -> DM2Config:
    names = {f for f in DM2Config.__dataclass_fields__}
    kwargs = {k: v for k, v in vars(args).items() if k in names}
    return DM2Config(**kwargs)
