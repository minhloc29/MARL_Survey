"""A simple uniform-sampling replay buffer for the flow-matching trainer.

Stores raw numpy transitions and returns batched torch tensors on ``sample``.
(No prioritization in the skeleton; see the plan for later deviations.)
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions.

    Each transition stores:
        obs:       joint camera observation (num_cameras, obs_dim)
        action:    normalized joint action (num_cameras, act_dim) in [-1, 1]
        reward:    scalar
        next_obs:  joint camera observation (num_cameras, obs_dim)
        done:      bool
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None
        self._pos = 0
        self._size = 0

    def __len__(self):
        return self._size

    def _alloc(self, num_cameras: int, obs_dim: int, act_dim: int):
        if self.obs is not None:
            return
        self.obs = np.zeros((self.capacity, num_cameras, obs_dim), dtype=np.float64)
        self.action = np.zeros((self.capacity, num_cameras, act_dim), dtype=np.float64)
        self.next_obs = np.zeros((self.capacity, num_cameras, obs_dim), dtype=np.float64)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=bool)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        next_obs = np.asarray(next_obs, dtype=np.float64)

        self._alloc(obs.shape[0], obs.shape[1], action.shape[1])

        self.obs[self._pos] = obs
        self.action[self._pos] = action
        self.reward[self._pos] = float(reward)
        self.next_obs[self._pos] = next_obs
        self.done[self._pos] = bool(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device="cpu") -> dict[str, torch.Tensor]:
        """Sample a batch of transitions as stacked torch tensors (float32).

        Returns:
            dict with keys obs, next_obs, action (all (B, C, ·)), reward (B,),
            done (B,).
        """
        if self._size < batch_size:
            raise ValueError(
                f"Not enough transitions ({self._size}) to sample {batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "action": torch.as_tensor(self.action[idx], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(
                self.next_obs[idx], dtype=torch.float32, device=device
            ),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        }
