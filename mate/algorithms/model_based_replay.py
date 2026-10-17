"""Shared bounded joint replay for discrete and continuous model-based agents."""

import numpy as np
import torch


class TransitionReplay:
    keys = ("obs", "actions", "rewards", "next_obs", "terminated")

    def __init__(self, capacity, seed=0, continuous_actions=False):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.continuous_actions = continuous_actions
        self.data = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()
        self.position = 0

    def add(self, obs, actions, rewards, next_obs, terminated):
        raw_actions = np.asarray(actions)
        if not self.continuous_actions and not np.issubdtype(raw_actions.dtype, np.integer):
            raise ValueError("Replay requires integer action indices")
        row = (np.asarray(obs, dtype=np.float32), raw_actions.astype(np.float32 if self.continuous_actions else np.int64),
               np.asarray(rewards, dtype=np.float32), np.asarray(next_obs, dtype=np.float32),
               np.asarray(terminated, dtype=np.float32))
        action_shape = (len(row[0]), raw_actions.shape[-1]) if self.continuous_actions and raw_actions.ndim == 2 else (len(row[0]),)
        if self.continuous_actions and raw_actions.ndim != 2:
            raise ValueError("Continuous replay requires (agents, action_dim) actions")
        if (row[0].ndim != 2 or row[3].shape != row[0].shape
                or row[1].shape != action_shape or row[2].shape != (len(row[0]),)
                or row[4].shape != () or not all(np.isfinite(x).all() for x in row)):
            raise ValueError("Invalid joint transition shapes or non-finite values")
        if row[4] not in (0, 1) or (not self.continuous_actions and np.any(row[1] < 0)):
            raise ValueError("Invalid terminal flag or action indices")
        row = tuple(x.copy() for x in row)
        if len(self.data) < self.capacity:
            self.data.append(row)
        else:
            self.data[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def extend(self, batch):
        arrays = [batch[key].detach().cpu().numpy() for key in self.keys]
        for row in zip(*arrays, strict=True):
            self.add(*row)

    def _batch(self, indices, device):
        return {key: torch.as_tensor(np.stack([self.data[i][k] for i in indices]), device=device)
                for k, key in enumerate(self.keys)}

    def sample(self, count, device="cpu"):
        if not self.data or count < 1:
            raise ValueError("Sampling requires nonempty replay and a positive count")
        return self._batch(self.rng.integers(len(self.data), size=count), device)

    def latest(self, count, device="cpu"):
        if not self.data or count < 1:
            raise ValueError("Reading requires nonempty replay and a positive count")
        count = min(count, len(self.data))
        return self._batch([(self.position - count + i) % len(self.data) for i in range(count)], device)


def mixed_batch(real, model, count, real_ratio, device="cpu"):
    if not 0 <= real_ratio <= 1 or count < 1:
        raise ValueError("Invalid real/model mixture")
    real_count = min(count, max(0, int(count * real_ratio))) if len(model) else count
    parts = []
    if real_count:
        parts.append(real.sample(real_count, device))
    if count > real_count:
        parts.append(model.sample(count - real_count, device))
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in TransitionReplay.keys}
