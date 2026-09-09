"""Episode replay for the MATE RODE integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RODEEpisode:
    """One complete multi-agent episode in RODE learner format."""

    obs: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    avail_actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    roles: np.ndarray

    @property
    def length(self) -> int:
        return int(self.rewards.shape[0])


class RODEEpisodeBuilder:
    """Collect transitions and finalize a padded-ready episode."""

    def __init__(self, n_agents: int, obs_shape: int, n_actions: int) -> None:
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.reset()

    def reset(self) -> None:
        self._obs: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._avail_actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[bool] = []
        self._roles: list[np.ndarray] = []

    def start(self, obs: np.ndarray, state: np.ndarray, avail_actions: np.ndarray) -> None:
        self.reset()
        self._obs.append(self._validate_obs(obs))
        self._states.append(self._validate_state(state))
        self._avail_actions.append(self._validate_avail(avail_actions))

    def add(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        next_avail_actions: np.ndarray,
        roles: Optional[np.ndarray] = None,
    ) -> None:
        if not self._obs:
            raise RuntimeError("Call start() before adding transitions.")
        self._actions.append(np.asarray(
            action, dtype=np.int64).reshape(self.n_agents))
        self._rewards.append(float(reward))
        self._terminated.append(bool(terminated))
        self._roles.append(
            np.zeros(self.n_agents, dtype=np.int64)
            if roles is None
            else np.asarray(roles, dtype=np.int64).reshape(self.n_agents)
        )
        self._obs.append(self._validate_obs(next_obs))
        self._states.append(self._validate_state(next_state))
        self._avail_actions.append(self._validate_avail(next_avail_actions))

    def finish(self) -> RODEEpisode:
        if not self._actions:
            raise RuntimeError("Cannot finish an empty episode.")
        episode = RODEEpisode(
            obs=np.asarray(self._obs, dtype=np.float32),
            state=np.asarray(self._states, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.int64),
            avail_actions=np.asarray(self._avail_actions, dtype=np.int32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            terminated=np.asarray(self._terminated, dtype=np.bool_),
            roles=np.asarray(self._roles, dtype=np.int64),
        )
        self.reset()
        return episode

    def _validate_obs(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        expected = (self.n_agents, self.obs_shape)
        if array.shape != expected:
            raise ValueError(
                f"Expected observation shape {expected}, got {array.shape}.")
        return array

    def _validate_state(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != self.n_agents * self.obs_shape:
            raise ValueError(
                "State size must equal the flattened joint observation size.")
        return array

    def _validate_avail(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.int32)
        expected = (self.n_agents, self.n_actions)
        if array.shape != expected:
            raise ValueError(
                f"Expected action-mask shape {expected}, got {array.shape}.")
        return array


class RODEEpisodeReplay:
    """Uniform replay over complete episodes with sequence padding."""

    def __init__(self, capacity: int, device: str | torch.device = "cpu") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.episodes: deque[RODEEpisode] = deque(maxlen=capacity)
        self.device = torch.device(device)

    def add(self, episode: RODEEpisode) -> None:
        self.episodes.append(episode)

    def __len__(self) -> int:
        return len(self.episodes)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not 0 < batch_size <= len(self.episodes):
            raise ValueError(
                "batch_size must not exceed the number of stored episodes")
        indices = np.random.choice(
            len(self.episodes), batch_size, replace=False)
        episodes = [self.episodes[int(index)] for index in indices]
        max_length = max(episode.length for episode in episodes)
        n_agents = episodes[0].actions.shape[1]
        n_actions = episodes[0].avail_actions.shape[2]
        obs_shape = episodes[0].obs.shape[2]
        state_shape = episodes[0].state.shape[1]

        result = {
            "obs": torch.zeros((batch_size, max_length + 1, n_agents, obs_shape), device=self.device),
            "state": torch.zeros((batch_size, max_length + 1, state_shape), device=self.device),
            "actions": torch.zeros((batch_size, max_length, n_agents), dtype=torch.long, device=self.device),
            "avail_actions": torch.zeros((batch_size, max_length + 1, n_agents, n_actions), dtype=torch.bool, device=self.device),
            "rewards": torch.zeros((batch_size, max_length, 1), device=self.device),
            "terminated": torch.ones((batch_size, max_length, 1), dtype=torch.bool, device=self.device),
            "roles": torch.zeros((batch_size, max_length, n_agents), dtype=torch.long, device=self.device),
            "filled": torch.zeros((batch_size, max_length, 1), dtype=torch.bool, device=self.device),
        }
        for batch_index, episode in enumerate(episodes):
            length = episode.length
            result["obs"][batch_index, : length +
                          1] = torch.as_tensor(episode.obs, device=self.device)
            result["state"][batch_index, : length +
                            1] = torch.as_tensor(episode.state, device=self.device)
            result["actions"][batch_index, :length] = torch.as_tensor(
                episode.actions, device=self.device)
            result["avail_actions"][batch_index, : length + 1] = torch.as_tensor(
                episode.avail_actions, dtype=torch.bool, device=self.device)
            result["rewards"][batch_index, :length, 0] = torch.as_tensor(
                episode.rewards, device=self.device)
            result["terminated"][batch_index, :length, 0] = torch.as_tensor(
                episode.terminated, device=self.device)
            result["roles"][batch_index, :length] = torch.as_tensor(
                episode.roles, device=self.device)
            result["filled"][batch_index, :length, 0] = True
        return result
