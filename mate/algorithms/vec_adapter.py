"""Thin Gymnasium adapters making the custom MATE envs VecEnv-compatible.

The MARL trainers in ``mate/algorithms/`` build their own non-Gym env classes:

* ``DiscreteCameraEnvironment`` / ``ContinuousCameraEnvironment``
  (``camera_environment.py``): Gym-style 5-tuple ``step``, joint obs
  ``(n_agents, obs_dim)``.
* ``MateRodeEnv`` (``rode/environment.py``): PyMARL-style 3-tuple ``step``
  ``(reward, done, info)`` with ``obs``/``state``/``avail_actions`` refetched
  via ``get_obs()``/``get_state()``/``get_avail_actions()``.

Neither exposes ``observation_space``/``action_space``/``unwrapped``, which
``gym_agent.core.vec_env`` requires. These adapters add that surface so the
environments can be run in parallel via ``SubprocVecEnv`` (and in-process via
``DummyVecEnv`` for testing), while keeping every underlying transition
contract untouched.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MateCameraVecEnv:
    """Gym adapter for ``DiscreteCameraEnvironment`` / ``ContinuousCameraEnvironment``.

    ``obs`` stays a plain float32 ``(n_agents, obs_dim)`` array; VecEnv batches it
    into ``(num_envs, n_agents, obs_dim)`` and ``unbatch`` slices it back.
    """

    def __init__(self, env, continuous: bool) -> None:
        self.env = env
        self.unwrapped = self
        self.spec = None
        self.metadata = {}
        self.render_mode = None

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (env.n_agents, env.obs_dim), dtype=np.float32
        )
        if continuous:
            self.action_space = spaces.Box(
                -1.0, 1.0, (env.n_agents, env.action_dim), dtype=np.float32
            )
        else:
            self.action_space = spaces.MultiDiscrete([env.n_actions] * env.n_agents)

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, actions):
        # env.step already returns a gym 5-tuple (obs, reward, terminated, truncated, info).
        return self.env.step(actions)

    def close(self) -> None:
        self.env.close()


class MateRodeVecEnv:
    """Gym adapter for ``MateRodeEnv`` (the RODE/PyMARL family).

    Converts the PyMARL 3-tuple ``step`` into a gym 5-tuple and returns a
    ``Dict`` observation payload ``{"obs", "state", "avail_actions"}`` so each
    parallel slot atomically carries everything ``RODEEpisodeBuilder.add`` needs.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.unwrapped = self
        self.spec = None
        self.metadata = {}
        self.render_mode = None

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    -np.inf, np.inf, (env.n_agents, env.obs_shape), dtype=np.float32
                ),
                "state": spaces.Box(
                    -np.inf, np.inf, (env.state_shape,), dtype=np.float32
                ),
                "avail_actions": spaces.MultiBinary((env.n_agents, env.n_actions)),
            }
        )
        self.action_space = spaces.MultiDiscrete([env.n_actions] * env.n_agents)

    def _payload(self) -> dict[str, np.ndarray]:
        return {
            "obs": self.env.get_obs(),
            "state": self.env.get_state(),
            "avail_actions": self.env.get_avail_actions(),
        }

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, state = self.env.reset(seed=seed)
        return self._payload(), {}

    def step(self, actions):
        reward, done, info = self.env.step(actions)
        # Maps the PyMARL 3-tuple onto the gym 5-tuple: terminated = done (MATE's
        # done folds in truncated/time-limit), truncated = False.
        return (self._payload(), float(reward), bool(done), False, info)

    def close(self) -> None:
        self.env.close()
