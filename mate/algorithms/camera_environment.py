"""Shared Gymnasium camera-team boundaries for model-based algorithms."""

import gymnasium as gym
import numpy as np

import mate
from mate.agents import GreedyTargetAgent
from mate.agents.utils import rescale_observation
from mate.utils import Team


class DiscreteCameraEnvironment:
    def __init__(self, config="MATE-4v8-9.yaml", levels=3, episode_limit=64):
        if levels < 2 or episode_limit < 1:
            raise ValueError("levels must be >= 2 and episode_limit must be positive")
        base = gym.make("MultiAgentTracking-v0", config=config)
        discrete = mate.wrappers.DiscreteCamera(base, levels=levels)
        self.action_grid = discrete.normalized_action_grid.copy()
        self.action_high = discrete.action_high.copy()
        self.env = mate.MultiCamera(base, target_agent=GreedyTargetAgent())
        self.n_agents = base.unwrapped.num_cameras
        self.n_actions = len(self.action_grid)
        self.obs_dim = self.env.observation_space.spaces[0].shape[0]
        self.episode_limit = episode_limit
        self.steps = 0

    def _observation(self, obs):
        base = self.env.unwrapped
        return rescale_observation(
            np.asarray(obs), team=Team.CAMERA, num_cameras=base.num_cameras,
            num_targets=base.num_targets, num_obstacles=base.num_obstacles,
        ).astype(np.float32)

    def reset(self, seed=None):
        self.steps = 0
        obs, info = self.env.reset(seed=seed)
        return self._observation(obs), info

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Expected one integer action index per camera")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError("Action index is outside the discrete camera grid")
        obs, reward, terminated, truncated, info = self.env.step(
            self.action_grid[actions] * self.action_high
        )
        self.steps += 1
        return (self._observation(obs), float(reward), bool(terminated),
                bool(truncated or self.steps >= self.episode_limit), info)

    def close(self):
        self.env.close()


class ContinuousCameraEnvironment(DiscreteCameraEnvironment):
    """Normalized continuous camera actions mapped to actual per-camera bounds."""

    def __init__(self, config="MATE-4v8-9.yaml", episode_limit=64):
        if episode_limit < 1:
            raise ValueError("episode_limit must be positive")
        base = gym.make("MultiAgentTracking-v0", config=config)
        self.env = mate.MultiCamera(base, target_agent=GreedyTargetAgent())
        self.n_agents = base.unwrapped.num_cameras
        self.obs_dim = self.env.observation_space.spaces[0].shape[0]
        spaces = self.env.action_space.spaces
        self.action_dim = spaces[0].shape[0]
        self.action_low = np.stack([space.low for space in spaces])
        self.action_high = np.stack([space.high for space in spaces])
        self.episode_limit, self.steps = episode_limit, 0

    def to_environment_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_agents, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError("Expected finite (cameras, action_dim) normalized actions")
        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError("Normalized actions must lie in [-1, 1]")
        return self.action_low + 0.5 * (np.clip(actions, -1., 1.) + 1.) * (self.action_high - self.action_low)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(self.to_environment_actions(actions))
        self.steps += 1
        return (self._observation(obs), float(reward), bool(terminated),
                bool(truncated or self.steps >= self.episode_limit), info)
