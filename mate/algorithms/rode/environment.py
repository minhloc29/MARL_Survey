"""RODE-compatible environment adapter for MATE's camera team.

The original RODE runner uses a small PyMARL environment protocol rather than
the Gymnasium API. This adapter keeps that protocol at the boundary while
leaving MATE's environment and wrappers unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

import mate
from mate.agents import GreedyTargetAgent


class MateRodeEnv:
    """Expose a discrete MATE camera task through RODE's environment API."""

    def __init__(
        self,
        env: gym.Env,
        episode_limit: Optional[int] = None,
        action_grid: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ) -> None:
        self.env = env
        self._episode_limit_override = episode_limit
        self._obs: Optional[np.ndarray] = None
        self._state: Optional[np.ndarray] = None
        self._step_count = 0

        if not isinstance(env.action_space, gym.spaces.Tuple):
            raise TypeError("RODE requires a joint tuple action space.")
        self.n_agents = len(env.action_space.spaces)
        self._action_grid = None if action_grid is None else np.asarray(
            action_grid, dtype=np.float64)
        self._action_high = None if action_high is None else np.asarray(
            action_high, dtype=np.float64)
        if self._action_grid is None:
            if not all(isinstance(space, gym.spaces.Discrete) for space in env.action_space.spaces):
                raise TypeError(
                    "RODE requires discrete actions or an action grid.")
            self.n_actions = int(env.action_space.spaces[0].n)
            if any(int(space.n) != self.n_actions for space in env.action_space.spaces):
                raise ValueError(
                    "RODE requires homogeneous per-agent action spaces.")
        else:
            self.n_actions = len(self._action_grid)

        if not isinstance(env.observation_space, gym.spaces.Tuple):
            raise TypeError("RODE requires one observation space per agent.")
        observation_shapes = tuple(
            tuple(int(size) for size in space.shape)
            for space in env.observation_space.spaces
        )
        if len(observation_shapes) != self.n_agents:
            raise ValueError(
                "RODE requires one fixed-size observation per agent.")
        if not observation_shapes or any(shape != observation_shapes[0] for shape in observation_shapes):
            raise ValueError(
                "RODE requires homogeneous per-agent observations.")
        self.obs_shape = int(np.prod(observation_shapes[0]))
        self.state_shape = self.n_agents * self.obs_shape
        self.episode_limit = int(
            episode_limit
            if episode_limit is not None
            else getattr(env.unwrapped, "max_episode_steps", 10000)
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset the Gymnasium environment and return observations and state."""
        self._step_count = 0
        self._obs, _ = self.env.reset(seed=seed)
        self._obs = self._validate_observation(self._obs)
        self._state = self._make_state(self._obs)
        return self._obs.copy(), self._state.copy()

    def step(self, actions: Any) -> tuple[float, bool, dict[str, Any]]:
        """Step using integer actions and return RODE's three-part result."""
        action_array = np.asarray(
            actions, dtype=np.int64).reshape(self.n_agents)
        if np.any(action_array < 0) or np.any(action_array >= self.n_actions):
            raise ValueError(f"Actions must be in [0, {self.n_actions}).")

        env_actions = action_array
        if self._action_grid is not None:
            env_actions = self._action_high * self._action_grid[action_array]
        obs, reward, terminated, truncated, info = self.env.step(env_actions)
        self._step_count += 1
        self._obs = self._validate_observation(obs)
        self._state = self._make_state(self._obs)

        time_limit = self._step_count >= self.episode_limit
        done = bool(terminated or truncated or time_limit)
        result_info = dict(info) if isinstance(
            info, dict) else {"raw_info": info}
        result_info["episode_limit"] = bool(truncated or time_limit)
        return float(reward), done, result_info

    def get_obs(self) -> np.ndarray:
        return self._require_reset(self._obs).copy()

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self.get_obs()[agent_id]

    def get_obs_size(self) -> int:
        return self.obs_shape

    def get_state(self) -> np.ndarray:
        return self._require_reset(self._state).copy()

    def get_state_size(self) -> int:
        return self.state_shape

    def get_avail_actions(self) -> np.ndarray:
        return np.ones((self.n_agents, self.n_actions), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self) -> int:
        return self.n_actions

    def get_env_info(self) -> dict[str, int]:
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def close(self) -> None:
        self.env.close()

    def render(self) -> Any:
        return self.env.render()

    def _make_state(self, obs: np.ndarray) -> np.ndarray:
        return obs.reshape(-1).astype(np.float32, copy=False)

    def _validate_observation(self, obs: Any) -> np.ndarray:
        array = np.asarray(obs, dtype=np.float32)
        expected_shape = (self.n_agents, self.obs_shape)
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, got {array.shape}."
            )
        return array

    @staticmethod
    def _require_reset(value: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            raise RuntimeError(
                "Call reset() before reading environment state.")
        return value


def make_mate_rode_env(
    config: str = "MATE-4v8-9.yaml",
    levels: int = 5,
    episode_limit: Optional[int] = None,
    **kwargs: Any,
) -> MateRodeEnv:
    """Create a discrete camera-team MATE environment for RODE."""
    base_env = gym.make("MultiAgentTracking-v0", config=config, **kwargs)
    discretizer = mate.wrappers.DiscreteCamera(base_env, levels=levels)
    camera_env = mate.wrappers.CoverageReward(
        mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())
    )
    return MateRodeEnv(
        camera_env,
        episode_limit=episode_limit,
        action_grid=discretizer.normalized_action_grid,
        action_high=discretizer.action_high,
    )
