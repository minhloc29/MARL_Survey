"""Reward wrapper exposing MATE's coverage rate as a [0, 1] team reward.

MATE's raw camera-team reward is scaled by the freight/bounty constants (order
of hundreds) and differs across the algorithms in this survey. To make every
runner learn on the same quantity as PPO/MAPPO (which use ``NetWrapper`` with
``reward_type="coverage_rate"``), this wrapper replaces the step reward with
the shared ``coverage_rate`` read from ``info``, which is naturally bounded in
``[0, 1]`` (fraction of targets currently tracked).
"""

from __future__ import annotations

import gymnasium as gym


class CoverageReward(gym.Wrapper):
    """Replace the raw freight/bounty reward with the shared coverage rate.

    Every camera's entry in ``info['infos']`` carries the same
    ``coverage_rate`` (the team-wide tracked fraction), so this is a single
    shared scalar in ``[0, 1]`` returned as the joint observation.
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        infos = info.get("infos") if isinstance(info, dict) else None
        coverage = float(infos[0]["coverage_rate"]) if infos else 0.0
        return obs, coverage, terminated, truncated, info
