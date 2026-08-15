from typing import TYPE_CHECKING

import numpy as np

from mate.agents.base import CameraAgentBase

if TYPE_CHECKING:
    from mate.agents.utils import TargetStatePublic


class HeuristicCameraAgentBase(CameraAgentBase):
    ALGORITHM: str
    def __init__(self, need_history: bool = False, seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """
        super().__init__(seed=seed)
        self.need_history = need_history


class GoalsBaseAgent(HeuristicCameraAgentBase):
    """A camera agent that executes a single high-level goal target.

    Wraps a goal-driven tracking policy by consuming binary goals from the caller
    (via :meth:`receive_goals`) and steering its camera toward the selected target.
    This bridges the high-level goal generator (which decides *which* target each
    camera should cover) and the low-level executor (the wrapped heuristic policy
    that decides *how* to track that target).
    """

    DEFAULT_TARGET_INDEX = 0  # fallback target when the goal bit is not set

    def __init__(self, executor: HeuristicCameraAgentBase | None = None, seed=None):
        """Initialize the agent.

        This function will be called only once on initialization.

        Args:
            executor: A heuristic policy that produces the underlying camera action.
                If omitted, a :class:`GreedyCameraAgent` is used.
        """
        if executor is None:
            # Deferred import to avoid a circular import at module load time.
            from agents.greedy import GreedyCameraAgent

            executor = GreedyCameraAgent(seed=seed)
        super().__init__(need_history=executor.need_history, seed=seed)

        self.executor = executor
        self.target_index: int = None

    @property
    def _target_states(self) -> list["TargetStatePublic"]:
        return list(self.executor.memory)

    def reset(self, observation: np.ndarray) -> None:
        self.executor.reset(observation)
        # Propagate per-episode attributes that only the executor's CameraAgentBase.reset() sets.
        for attr in ("num_cameras", "num_targets", "num_obstacles", "index", "state"):
            if hasattr(self.executor, attr):
                setattr(self, attr, getattr(self.executor, attr))
        self.target_index = None

    def observe(self, observation: np.ndarray, info: dict | None = None) -> None:
        self.executor.observe(observation, info)

    def receive_goals(self, goal: list[bool] | np.ndarray) -> None:
        """Select which target to track from a binary goal vector.

        This function is called before :meth:`act`. ``goal`` has one entry per
        target; ``goal[i]`` is True when camera ``i`` (this agent) should follow
        target ``i``. The last set target is chosen, falling back to **no target**
        (free scanning) when no goal bit is set.
        """
        if len(goal) != self.num_targets:
            raise ValueError(
                f"Goal length {len(goal)} does not match the number of targets {self.num_targets}."
            )

        selected = None
        for i, bit in enumerate(goal):
            if bit:
                selected = i
        self.target_index = selected

    def act(
        self,
        observation: np.ndarray,
        info: dict | None = None,
        deterministic: bool | None = None,
    ) -> np.ndarray:
        """Track the target selected by :meth:`receive_goals` (track nothing if none)."""
        if self.target_index is not None:
            idea = self._track_goal(self.target_index)
            if idea is not None:
                return idea
        return self.executor.act(observation, info, deterministic)

    def send_responses(self):
        return self.executor.send_responses()

    def receive_responses(self, messages: tuple):
        return self.executor.receive_responses(messages)

    def send_requests(self):
        return self.executor.send_requests()

    def receive_requests(self, messages: tuple):
        return self.executor.receive_requests(messages)

    def _track_goal(self, target_index: int) -> np.ndarray | None:
        """Point the camera at the goal target using the wrapped policy.

        Returns the action, or None when the goal target is not tracked, in which
        case the executor falls back to its own behavior.
        """
        for ts in self._target_states:
            if ts.index == target_index:
                return self.executor.act_from_target_states([ts])
        return None


class HeuristicsCameraAgent(CameraAgentBase):
    """Heuristic Camera Agent Base

    This class is a base class for heuristic camera agents.
    It provides a common interface for heuristic camera agents to implement.
    """

    def __init__(self, heuristics: list[HeuristicCameraAgentBase], seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """
        super().__init__(seed=seed)

        self.heuristics = heuristics

        self.used_heuristic_idx: int = None

    def receive_heuristic_assignments(self, heuristic_assignment: int):
        """Receive the heuristic assignments for each camera.
        This function will be called before act().
        """
        if heuristic_assignment < 0 or heuristic_assignment >= len(self.heuristics):
            raise ValueError(
                f"Heuristic assignment {heuristic_assignment} is out of range for {len(self.heuristics)} heuristics."
            )
        self.used_heuristic_idx = heuristic_assignment

    def reset(self, observation: np.ndarray):
        for heuristic in self.heuristics:
            heuristic.reset(observation)

    def observe(self, observation: np.ndarray, info: dict | None = None) -> None:
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].observe(observation, info)

    def act(
        self,
        observation: np.ndarray,
        info: dict | None = None,
        deterministic: bool | None = None,
    ) -> int | np.ndarray:
        """Act based on the observation and info.
        This function will be called after observe().
        """
        if self.used_heuristic_idx is None:
            raise ValueError("Heuristic assignment not received before act()")

        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].act(observation, info, deterministic)
            elif self.heuristics[i].need_history:
                self.heuristics[i].act(observation, info, deterministic)

        return res

    def send_requests(self):
        """Send requests to other agents.
        This function will be called after act().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].send_requests()
            elif self.heuristics[i].need_history:
                self.heuristics[i].send_requests()

        return res

    def receive_requests(self, messages: tuple):
        """Receive requests from other agents.
        This function will be called after send_requests().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].receive_requests(messages)

    def send_responses(self):
        """Send responses to other agents.
        This function will be called after receive_requests().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].send_responses()
            elif self.heuristics[i].need_history:
                self.heuristics[i].send_responses()

        return res

    def receive_responses(self, messages: tuple):
        """Receive responses from other agents.
        This function will be called after send_responses().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].receive_responses(messages)
