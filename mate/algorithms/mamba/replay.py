"""Bounded episode replay with explicit observation/action time alignment."""

from collections import deque

import numpy as np
import torch


class SequenceReplay:
    def __init__(self, capacity, sequence_length, n_actions, seed=0):
        if sequence_length < 2 or capacity < sequence_length:
            raise ValueError("Require capacity >= sequence_length >= 2")
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.n_actions = n_actions
        self.episodes = deque()
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, observations, actions, rewards, terminated):
        obs = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=np.float32)
        length = len(actions)
        if (obs.ndim != 3 or actions.shape != (length, obs.shape[1])
                or len(obs) != length + 1 or rewards.shape != (length,)
                or terminated.shape != (length,) or length < 1):
            raise ValueError("Expected T+1 joint observations and T transitions")
        if length > self.capacity:
            raise ValueError("An episode exceeds replay capacity")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError("Replay contains an invalid action")
        if np.any(terminated[:-1]):
            raise ValueError("Replay episodes must not cross terminal boundaries")
        while self.size + length > self.capacity:
            self.size -= len(self.episodes.popleft()[1])
        self.episodes.append(tuple(x.copy() for x in (obs, actions, rewards, terminated)))
        self.size += length

    @property
    def ready(self):
        return any(len(ep[1]) >= self.sequence_length for ep in self.episodes)

    def sample(self, batch_size, device="cpu"):
        eligible = [ep for ep in self.episodes if len(ep[1]) >= self.sequence_length]
        if not eligible or batch_size < 1:
            raise ValueError("Replay needs a complete sequence and a positive batch size")
        # Weight by window count so each valid sequence has equal probability.
        windows = np.asarray([len(ep[1]) - self.sequence_length + 1 for ep in eligible])
        indices = self.rng.choice(len(eligible), batch_size, p=windows / windows.sum())
        samples = []
        for index in indices:
            obs, actions, rewards, terminated = eligible[index]
            start = int(self.rng.integers(windows[index]))
            stop = start + self.sequence_length
            n_agents = obs.shape[1]
            # At index t, obs[t] pairs with PREVIOUS action/reward/done.
            # The upstream objective predicts reward[t+1] from state[t] and
            # next deterministic state[t+1], and reconstructs obs[:-1].
            previous_actions = np.zeros((self.sequence_length + 1, n_agents, self.n_actions), np.float32)
            previous_actions[1:] = np.eye(self.n_actions, dtype=np.float32)[actions[start:stop]]
            if start:
                previous_actions[0] = np.eye(self.n_actions, dtype=np.float32)[actions[start - 1]]
            reward = np.zeros((self.sequence_length + 1, n_agents, 1), np.float32)
            done = np.zeros_like(reward)
            reward[1:] = rewards[start:stop, None, None]
            done[1:] = terminated[start:stop, None, None]
            last = np.zeros_like(reward)
            if stop == len(actions):
                last[-1] = 1.0  # truncations reset recurrence but keep continuation=1
            samples.append(dict(observation=obs[start:stop + 1], action=previous_actions,
                                reward=reward, done=done, fake=np.zeros_like(reward), last=last))
        result = {key: torch.as_tensor(np.stack([s[key] for s in samples], axis=1), device=device)
                  for key in samples[0]}
        result["av_action"] = None  # All MATE grid actions are available.
        return result
