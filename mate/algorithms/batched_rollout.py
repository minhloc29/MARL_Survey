"""True-parallel batched environment rollouts via ``SubprocVecEnv``.

Each of N environments runs in its own OS process and is stepped concurrently
(a real multi-core rollout speedup), while action selection stays in the parent
process where the policy networks live.

The adapter contract: ``env_fn`` returns a ``VecEnv``-compatible env (see
``vec_adapter.py``) whose ``step`` returns a gym 5-tuple ``(obs, reward,
terminated, truncated, info)``. ``obs`` is a plain array (family A) or a Dict
``{"obs", "state", "avail_actions"}`` (family B).

Done handling mirrors what a single-env rollout already does: when a slot hits
``done``, the worker auto-resets it, and the *terminal* observation is made
available in ``infos[i]["final_observation"]`` so the caller records the final
transition before continuing the fresh episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from gym_agent.core.vec_env.subproc_vec_env import SubprocVecEnv


@dataclass
class RolloutStep:
    """Result of one concurrent step across all environment slots."""

    obs: list                      # per-env obs payload; post-reset for finished slots
    rewards: np.ndarray            # (num_envs,)
    done: np.ndarray               # (num_envs,) bool
    infos: list                    # on done, infos[i]["final_observation"] = terminal payload


def _unbatch_payload(batched: Any, n: int) -> list:
    """Split a VecEnv-batched observation (array or Dict) into per-env items."""
    if isinstance(batched, dict):
        return [{k: batched[k][i] for k in batched} for i in range(n)]
    return [batched[i] for i in range(n)]


class BatchedRollout:
    """Thin wrapper over ``SubprocVecEnv`` exposing a per-step primitive."""

    def __init__(
        self,
        env_fn: Callable[[], Any],
        num_envs: int,
        start_method: Optional[str] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        # ``env_fn`` must be a top-level callable: SubprocVecEnv pickles it
        # (cloudpickle) to ship into each worker process.
        self.venv = SubprocVecEnv([env_fn] * self.num_envs, start_method)

    def reset(self, seeds: Optional[list] = None) -> list:
        """Reset all slots. ``seeds`` optionally maps to ``seed + i`` per env."""
        if seeds is not None and len(seeds) == self.num_envs:
            self.venv.seed(seeds[0])
        else:
            self.venv.seed()
        # ``_reset`` returns per-env observations directly (zip of worker results),
        # avoiding VecEnv.reset's dict-stacking so per-env payloads stay distinct.
        obs = self.venv._reset()
        return list(obs)

    def step(self, actions: np.ndarray) -> RolloutStep:
        """Step all slots concurrently. ``actions`` is stacked ``(num_envs, ...)``.

        Finished slots are auto-reset by the worker; their terminal obs is in
        ``infos[i]["final_observation"]``. Uses ``step_async/step_wait`` directly
        so the per-slot ``infos`` dicts (with ``final_observation``) are preserved
        rather than dict-stacked into a single object.
        """
        self.venv.step_async(np.asarray(actions))
        obs, rews, term, trunc, infos = self.venv.step_wait()
        return RolloutStep(
            obs=list(obs),
            rewards=np.asarray(rews),
            done=(np.asarray(term).astype(bool) | np.asarray(trunc).astype(bool)),
            infos=list(infos),
        )

    def close(self) -> None:
        self.venv.close()
