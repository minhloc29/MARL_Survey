from .base import HeuristicCameraAgentBase, HeuristicsCameraAgent
from .built_in_heuristic import CoordinatedCoverageHeuristic
from .greedy import GreedyCameraAgent
from .heuristics import (
    HighValueCoverageHeuristic,
    LoadedInterceptHeuristic,
    PatrolSweepHeuristic,
    ReconTrackingHeuristic,
)

HEURISTIC: list[type[HeuristicCameraAgentBase]] = [
    GreedyCameraAgent,
    CoordinatedCoverageHeuristic,
    HighValueCoverageHeuristic,
    LoadedInterceptHeuristic,
    PatrolSweepHeuristic,
    ReconTrackingHeuristic,
]

_SCORE_BASE_STR = '''from __future__ import annotations

from abc import abstractmethod

import numpy as np

from mate.agents.utils import TargetStatePublic
from mate.constants import MAX_CAMERA_VIEWING_ANGLE
from mate.utils import normalize_angle, sin_deg

from .base import HeuristicCameraAgentBase

class _ScoredTargetHeuristic(HeuristicCameraAgentBase):
    """Base class for heuristics that rank remembered targets with a utility score."""

    def __init__(
        self, memory_period: int = 25, filterout_beyond_range: bool = True, seed=None
    ):
        super().__init__(need_history=False, seed=seed)
        self.memory_period = memory_period
        self.filterout_beyond_range = filterout_beyond_range
        self.range_factor = 1.1
        self.memory = None
        self.time2forget = None
        self.never_loaded = None
        self.prev_action = self.DEFAULT_ACTION
        self.sweep_direction = 1.0

    def reset(self, observation: np.ndarray) -> None:
        super().reset(observation)
        target_states, tracked_bits = self.get_all_opponent_states(observation)
        self.memory = list(target_states)
        self.time2forget = self.memory_period * np.asarray(tracked_bits, dtype=np.int64)
        self.never_loaded = np.ones(self.num_targets, dtype=np.bool_)
        self.prev_action = self.DEFAULT_ACTION
        self.sweep_direction = 1.0

    def observe(self, observation: np.ndarray, info: dict | None = None) -> None:
        self.state, observation, info, _ = self.check_inputs(observation, info)
        self.time2forget = np.maximum(self.time2forget - 1, 0, dtype=np.int64)

        target_states, tracked_bits = self.get_all_opponent_states(observation)
        for target_index in np.flatnonzero(tracked_bits):
            target_state = target_states[target_index]
            self.memory[target_index] = target_state
            self.time2forget[target_index] = self.memory_period
            if target_state.is_loaded:
                self.never_loaded[target_index] = False

    def _action_from_target(
        self, target_state: TargetStatePublic | None
    ) -> np.ndarray | None:
        """Turn a target choice into a bounded camera control action."""

        if target_state is None:
            return None

        distance = (target_state - self.state).norm
        if (
            distance * (1.0 + sin_deg(self.state.min_viewing_angle / 2.0))
            >= self.state.max_sight_range
        ):
            best_viewing_angle = self.state.min_viewing_angle
        else:
            area_product = self.state.viewing_angle * np.square(self.state.sight_range)
            if distance <= np.sqrt(area_product / 180.0) / 2.0:
                best_viewing_angle = min(180.0, MAX_CAMERA_VIEWING_ANGLE)
            else:
                best_viewing_angle = min(180.0, MAX_CAMERA_VIEWING_ANGLE)
                for _ in range(20):
                    sight_range = distance * (
                        1.0 + sin_deg(min(best_viewing_angle / 2.0, 90.0))
                    )
                    best_viewing_angle = area_product / np.square(sight_range)
                best_viewing_angle = np.clip(
                    best_viewing_angle,
                    a_min=self.state.min_viewing_angle,
                    a_max=MAX_CAMERA_VIEWING_ANGLE,
                )

        return np.asarray(
            [
                normalize_angle(
                    (target_state - self.state).angle - self.state.orientation
                ),
                best_viewing_angle - self.state.viewing_angle,
            ]
        ).clip(min=self.action_space.low, max=self.action_space.high)

    def _sweep_action(self) -> np.ndarray:
        """Fallback search motion when no remembered target looks useful."""

        action = np.asarray(
            [
                self.sweep_direction * self.action_space.high[0],
                MAX_CAMERA_VIEWING_ANGLE - self.state.viewing_angle,
            ]
        ).clip(min=self.action_space.low, max=self.action_space.high)
        self.sweep_direction *= -1.0
        return action

    def act(
        self, observation: np.ndarray, info: dict | None = None, deterministic=None
    ):
        self.state, observation, info, _ = self.check_inputs(observation, info)
        tracked_targets = [self.memory[t] for t in np.flatnonzero(self.time2forget)]
        if self.filterout_beyond_range:
            threshold = self.range_factor * self.state.max_sight_range
            tracked_targets = [
                ts for ts in tracked_targets if (ts - self.state).norm < threshold
            ]
        target_state = self.select_target(tracked_targets)
        action = self._action_from_target(target_state)
        if action is None:
            action = np.asarray(
                [
                    self.sweep_direction * self.action_space.high[0],
                    MAX_CAMERA_VIEWING_ANGLE - self.state.viewing_angle,
                ]
            ).clip(min=self.action_space.low, max=self.action_space.high)
            self.sweep_direction *= -1.0
        self.prev_action = action
        return action

    def select_target(
        self, tracked_targets: list[TargetStatePublic]
    ) -> TargetStatePublic | None:
        """Return the highest-scoring target, or none if nothing is worth tracking."""

        if not tracked_targets:
            return None

        best_target = None
        best_score = -np.inf
        for target_state in tracked_targets:
            score = self.score_target(target_state)
            if score > best_score:
                best_score = score
                best_target = target_state
        return best_target

    def target_features(self, target_state: TargetStatePublic) -> dict[str, float]:
        """Compute normalized features shared by the scoring heuristics."""

        distance = float((target_state - self.state).norm)
        angle_error = abs(
            float(
                normalize_angle(
                    (target_state - self.state).angle - self.state.orientation
                )
            )
        )
        freshness = float(self.time2forget[target_state.index]) / max(
            float(self.memory_period), 1.0
        )
        range_ratio = distance / max(float(self.state.max_sight_range), 1.0)
        center_alignment = 1.0 - angle_error / 180.0
        near_ratio = 1.0 - min(range_ratio, 1.0)
        loaded = 1.0 if target_state.is_loaded else 0.0
        unseen = 1.0 if self.never_loaded[target_state.index] else 0.0
        zoom_fit = 1.0 - abs(range_ratio - 0.6)
        return {
            "distance": distance,
            "angle_error": angle_error,
            "freshness": freshness,
            "range_ratio": range_ratio,
            "center_alignment": center_alignment,
            "near_ratio": near_ratio,
            "loaded": loaded,
            "unseen": unseen,
            "zoom_fit": zoom_fit,
        }

    @abstractmethod
    def score_target(self, target_state: TargetStatePublic) -> float:
        """Return the utility score for a remembered target."""
'''

HEURISTIC_STR: dict[type, str] = {
    GreedyCameraAgent: '''"""Built-in greedy rule-based agents."""

import numpy as np

from mate.agents.utils import TargetStatePublic
from mate.constants import MAX_CAMERA_VIEWING_ANGLE
from mate.utils import normalize_angle, sin_deg

from .base import HeuristicCameraAgentBase


class GreedyCameraAgent(HeuristicCameraAgentBase):  # pylint: disable=too-many-instance-attributes
    ALGORITHM = """Track the single most immediately actionable target using local memory.

    The policy keeps a short-lived cache of recently seen targets and always picks the
    nearest candidate that still looks relevant. Once a target is selected, the camera
    rotates to center it and adjusts the field of view to keep that target visible while
    preserving as much sensing area as possible.

    This is a purely myopic strategy: it does not coordinate global coverage or predict
    where targets will move next. If nothing is currently worth tracking, the agent
    falls back to repeating its previous command most of the time, with occasional
    random exploration so the camera does not get stuck forever.
    """

    def __init__(
        self,
        memory_period=25,
        filterout_unloaded=False,
        filterout_beyond_range=True,
        seed=None,
    ):
        """Initialize the agent.
        This function will be called only once on initialization.
        """
        super().__init__(need_history=False, seed=seed)

        self.filterout_unloaded = filterout_unloaded
        self.filterout_beyond_range = filterout_beyond_range
        self.range_factor = 1.1  # 110%

        self.memory = None
        self.time2forget = None
        self.never_loaded = None
        self.memory_period = memory_period
        self.prev_action = self.DEFAULT_ACTION

        self.neighboring_teammate_states = {}
        self.message2send = {}
        self.communication_delay = None

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        target_states, tracked_bits = self.get_all_opponent_states(observation)
        self.memory = list(target_states)
        self.time2forget = self.memory_period * np.asarray(tracked_bits, dtype=np.int64)
        self.never_loaded = np.ones(self.num_targets, dtype=np.bool_)

        self.prev_action = self.DEFAULT_ACTION

        self.neighboring_teammate_states.clear()
        self.message2send.clear()
        self.communication_delay = np.zeros(self.num_teammates, dtype=np.int64)
        self.message2send["state"] = self.state.copy()

    def observe(self, observation, info=None):
        """The agent observe the environment before sending messages.
        This function will be called before send_responses().
        """

        self.state, observation, info, messages = self.check_inputs(observation, info)

        self.process_messages(observation, messages)

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Arbitrarily track the nearest target.
        If no target found, use previous action or generate a new random action.
        """

        self.state, observation, info, _ = self.check_inputs(observation, info)

        tracked_targets = [self.memory[t] for t in np.flatnonzero(self.time2forget)]
        if self.filterout_beyond_range:
            threshold = self.range_factor * self.state.max_sight_range
            tracked_targets = [
                ts for ts in tracked_targets if (ts - self.state).norm < threshold
            ]
        if self.filterout_unloaded:
            tracked_targets = [
                ts
                for ts in tracked_targets
                if ts.is_loaded or self.never_loaded[ts.index]
            ]

        action = None
        if len(tracked_targets) > 0:
            action = self.act_from_target_states(tracked_targets)

        if action is None:
            if self.np_random.binomial(1, 0.1) != 0:
                action = self.action_space.sample()
            else:
                action = self.prev_action

        self.prev_action = action
        return action

    def process_messages(self, observation, messages):  # pylint: disable=unused-argument
        """Process observation and prepare messages to teammates."""

        self.time2forget = np.maximum(self.time2forget - 1, 0, dtype=np.int64)

        target_states, tracked_bits = self.get_all_opponent_states(observation)
        for t in np.flatnonzero(tracked_bits):
            self.time2forget[t] = self.memory_period
            self.memory[t] = target_states[t]
            if target_states[t].is_loaded:
                self.never_loaded[t] = False
            self.message2send.setdefault("target_states", [])
            self.message2send["target_states"].append(target_states[t])

    def act_from_target_states(self, target_states: list[TargetStatePublic]):
        """Place the selected target at the center of the field of view."""

        assert len(target_states) > 0, (
            "You should provide at least one target to compute the action."
        )

        def select_target():
            """Select the nearest target."""

            result = None
            min_result = np.inf

            for target in target_states:
                distance = (target - self.state).norm

                if distance < min_result:
                    min_result = distance
                    result = target

            return result

        def best_orientation(target_state):
            return (target_state - self.state).angle

        def best_viewing_angle(target_state):
            distance = (target_state - self.state).norm

            if (
                distance * (1.0 + sin_deg(self.state.min_viewing_angle / 2.0))
                >= self.state.max_sight_range
            ):
                return self.state.min_viewing_angle

            area_product = self.state.viewing_angle * np.square(self.state.sight_range)
            if distance <= np.sqrt(area_product / 180.0) / 2.0:
                return min(180.0, MAX_CAMERA_VIEWING_ANGLE)

            best = min(180.0, MAX_CAMERA_VIEWING_ANGLE)
            for _ in range(20):
                sight_range = distance * (1.0 + sin_deg(min(best / 2.0, 90.0)))
                best = area_product / np.square(sight_range)
            return np.clip(
                best, a_min=self.state.min_viewing_angle, a_max=MAX_CAMERA_VIEWING_ANGLE
            )

        target_state = select_target()

        if target_state is None:
            return None

        return np.asarray(
            [
                normalize_angle(
                    best_orientation(target_state) - self.state.orientation
                ),
                best_viewing_angle(target_state) - self.state.viewing_angle,
            ]
        ).clip(min=self.action_space.low, max=self.action_space.high)

    def send_responses(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called before receive_responses().

        Send the newest target states to teammates if necessary.
        """

        messages = []

        self.communication_delay = np.maximum(
            self.communication_delay - 1, 0, dtype=np.int64
        )

        if len(self.message2send) > 0:
            for c in range(self.num_cameras):
                if c == self.index or self.communication_delay[c] > 0:
                    continue
                content = self.message2send.copy()
                if "target_states" in content:
                    if (
                        c in self.neighboring_teammate_states
                        and self.filterout_beyond_range
                    ):
                        teammate_state = self.neighboring_teammate_states[c]
                        threshold = self.range_factor * teammate_state.max_sight_range
                        content["target_states"] = [
                            ts
                            for ts in content["target_states"]
                            if (ts - teammate_state).norm < threshold
                        ]
                        if len(content["target_states"]) == 0:
                            del content["target_states"]
                    else:
                        del content["target_states"]
                if len(content) > 0:
                    messages.append(self.pack_message(recipient=c, content=content))
                    delay = self.np_random.integers(
                        self.memory_period // 4, 2 * self.memory_period
                    )
                    self.communication_delay[c] = delay

            self.message2send.clear()

        return messages

    def receive_responses(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().

        Receive and process messages from teammates.
        """

        self.last_responses = tuple(messages)

        for message in self.last_responses:
            if "state" in message.content:
                teammate_state = message.content["state"]
                is_neighboring = True
                if self.filterout_beyond_range:
                    distance = (teammate_state - self.state).norm
                    threshold = (
                        self.state.max_sight_range
                        + self.range_factor * teammate_state.max_sight_range
                    )
                    is_neighboring = distance < threshold
                if is_neighboring:
                    self.neighboring_teammate_states[message.sender] = teammate_state
                elif message.sender in self.neighboring_teammate_states:
                    del self.neighboring_teammate_states[message.sender]
                self.neighboring_teammate_states[message.sender] = teammate_state

            for target_state in message.content.get("target_states", []):
                self.memory[target_state.index] = target_state
                self.time2forget[target_state.index] = self.memory_period
                if target_state.is_loaded:
                    self.never_loaded[target_state.index] = False
''',
    CoordinatedCoverageHeuristic: '''from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np

from mate.constants import MAX_CAMERA_VIEWING_ANGLE
from mate.utils import normalize_angle, polar2cartesian, sin_deg

from .base import HeuristicCameraAgentBase


class CoordinatedCoverageHeuristic(HeuristicCameraAgentBase):
    ALGORITHM = """Coordinated coverage heuristic.

    This is the only multi-camera policy in this module. Every non-controller camera
    sends its latest observation to camera 0, and camera 0 reconstructs a joint view
    of all currently sensed targets and teammate states before choosing new camera poses
    for the full team.

    The controller evaluates a discretized mesh of candidate `(orientation,
    viewing_angle)` states for each camera, estimates how well each state covers each
    target, and then greedily searches camera-order permutations to maximize joint
    target coverage while keeping motion cost low. The selected goal pose for each
    camera is then returned through the response channel, and each camera moves toward
    its assigned pose instead of making an isolated local decision.
    """

    def __init__(self, seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """

        super().__init__(need_history=True, seed=seed)

        self.controller_index = 0
        self.scores = None
        self.state_mesh = None
        self.coord_grid = None
        self.camera_states = None
        self.joint_observation = None
        self.joint_goal_state = None
        self.prev_action = self.DEFAULT_ACTION

    def reset(self, observation):
        """Reset the agent.
        This function will be called immediately after env.reset().
        """

        super().reset(observation)

        results = self.calculate_scores(
            round(float(self.state.max_sight_range), 8),
            round(float(self.state.min_viewing_angle), 8),
        )
        self.state_mesh, self.coord_grid, self.scores = results

        self.camera_states = None
        self.joint_observation = None
        self.joint_goal_state = None
        self.prev_action = self.DEFAULT_ACTION

    def act(self, observation, info=None, deterministic=None):
        """Get the agent action by the observation.
        This function will be called before every env.step().

        Arbitrarily track the nearest target.
        If no target found, use previous action or generate a new random action.
        """

        if self.index == self.controller_index:
            goal_state = self.joint_goal_state[self.index]
        else:
            try:
                goal_state = (
                    self.last_responses[-1].content
                    if self.last_responses
                    else (None, None)
                )
            except IndexError:
                target_states, tracked_bits = self.get_all_opponent_states(
                    self.last_observation
                )
                target_states = [target_states[t] for t in np.flatnonzero(tracked_bits)]
                if len(target_states) > 0:
                    goal_state = self.get_joint_goal_state([self.state], target_states)[
                        self.index
                    ]
                else:
                    goal_state = (None, None)

        if None not in goal_state:
            goal_orientation, goal_viewing_angle = goal_state
            action = np.asarray(
                [
                    normalize_angle(goal_orientation - self.state.orientation),
                    goal_viewing_angle - self.state.viewing_angle,
                ]
            ).clip(min=self.action_space.low, max=self.action_space.high)
        else:
            if self.np_random.binomial(1, 0.1) != 0:
                action = self.action_space.sample()
            else:
                action = self.prev_action

        self.prev_action = action
        return action

    def send_requests(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called after observe() but before receive_requests().
        """

        if self.index == self.controller_index:
            return []

        request = self.pack_message(
            content=self.last_observation, recipient=self.controller_index
        )

        return [request]

    def receive_requests(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_requests().
        """

        self.last_requests = tuple(messages)

        if self.index != self.controller_index:
            return

        self.joint_observation = {self.controller_index: self.last_observation}
        for message in self.last_requests:
            self.joint_observation[message.sender] = message.content

        self.camera_states = {}
        target_states = {}
        unsensed_targets = set(range(self.num_targets))
        for c, observation in self.joint_observation.items():
            camera_state = self.STATE_CLASS(
                observation[self.observation_slices["self_state"]], index=c
            )
            self.camera_states[c] = camera_state

            for t in tuple(unsensed_targets):
                target_state, sensed = self.get_opponent_state(observation, index=t)
                if sensed:
                    target_states[t] = target_state
                    unsensed_targets.remove(t)

        target_states = list(target_states.values())

        self.joint_goal_state = self.get_joint_goal_state(
            list(self.camera_states.values()), target_states
        )

    def send_responses(self):
        """Prepare messages to communicate with other agents in the same team.
        This function will be called after receive_requests().
        """

        if self.index != self.controller_index:
            return []

        responses = []
        for c, goal_state in self.joint_goal_state.items():
            if c == self.index:
                continue
            message = self.pack_message(content=goal_state, recipient=c)
            responses.append(message)

        return responses

    def receive_responses(self, messages):
        """Receive messages from other agents in the same team.
        This function will be called after send_responses() but before act().
        """

        self.last_responses = tuple(messages)

    def get_joint_goal_state(self, camera_states, target_states):  # pylint: disable=too-many-locals
        """Greedily track the targets as many as possible."""

        joint_scores = []
        joint_tracked_bits = []
        num_within_range_targets = []
        for camera_state in camera_states:
            within_range_targets = [
                ts
                for ts in target_states
                if (ts - camera_state).norm <= camera_state.max_sight_range
            ]
            num_within_range_targets.append(len(within_range_targets))

            scores = np.zeros(self.scores.shape[-1], dtype=np.float64)
            tracked_bits = np.zeros(
                (self.scores.shape[-1], self.num_targets), dtype=np.bool_
            )
            for target_state in within_range_targets:
                direction = target_state.location - camera_state.location
                index = np.argmin(
                    np.linalg.norm(direction - self.coord_grid, axis=-1), axis=-1
                )
                tracked_bits[self.scores[index, :] > 0, target_state.index] = True
                scores += self.scores[index, :]

            joint_scores.append(scores)
            joint_tracked_bits.append(tracked_bits)

        permutations = []
        for _ in range(32):
            permutation = self.np_random.permutation(range(len(camera_states)))
            indices = []
            current_tracked_bits = np.zeros((self.num_targets,), dtype=np.bool_)
            total_scores = 0
            total_cost = 0
            for c in permutation:
                camera_state, scores, tracked_bits = (
                    camera_states[c],
                    joint_scores[c],
                    joint_tracked_bits[c],
                )
                untracked_bits = np.logical_and(
                    tracked_bits, np.logical_not(current_tracked_bits)
                )
                index = np.argmax(scores + untracked_bits.sum(axis=-1))

                state_diff = np.abs(
                    self.state_mesh[index, :2]
                    - np.array([camera_state.orientation, camera_state.viewing_angle])
                )
                cost = (state_diff / self.action_space.high).max()

                current_tracked_bits = np.logical_or(
                    current_tracked_bits, tracked_bits[index]
                )
                total_scores = total_scores + scores[index]
                total_cost += cost

                indices.append(index)

            total_scores += current_tracked_bits.sum()
            permutations.append(
                (total_scores, -total_cost, tuple(permutation), tuple(indices))
            )

        _, _, best_permutation, best_indices = max(permutations)
        joint_goal_state = defaultdict(lambda: (None, None))
        for c, index in zip(best_permutation, best_indices):
            if num_within_range_targets[c] > 0:
                goal_orientation, goal_viewing_angle, _ = self.state_mesh[index]
                joint_goal_state[camera_states[c].index] = (
                    goal_orientation,
                    goal_viewing_angle,
                )

        return joint_goal_state

    @staticmethod
    @lru_cache(maxsize=None)
    def calculate_scores(max_sight_range, min_viewing_angle):  # pylint: disable=too-many-locals
        """Calculates the coordinate grid and the weighted scores."""

        state_mesh = np.stack(
            np.meshgrid(
                np.linspace(start=-180.0, stop=+180.0, num=36, endpoint=False),
                np.linspace(
                    start=min_viewing_angle,
                    stop=MAX_CAMERA_VIEWING_ANGLE,
                    num=21,
                    endpoint=True,
                ),
            ),
            axis=-1,
        ).reshape(-1, 2)
        sight_ranges = max_sight_range * np.sqrt(min_viewing_angle / state_mesh[..., 1])
        state_mesh = np.hstack([state_mesh, sight_ranges[:, np.newaxis]])
        rho, phi = (
            np.stack(
                np.meshgrid(
                    np.linspace(start=0.0, stop=max_sight_range, num=41, endpoint=True),
                    np.linspace(start=-180.0, stop=+180.0, num=72, endpoint=False),
                ),
                axis=-1,
            )
            .reshape(-1, 2)
            .transpose()
        )
        coord_grid = polar2cartesian(rho, phi).transpose()

        scores = np.zeros((len(coord_grid), len(state_mesh)), dtype=np.float64)
        # pylint: disable-next=invalid-name
        for s, (orientation, viewing_angle, sight_range) in enumerate(state_mesh):
            half_viewing_angle = viewing_angle / 2.0
            if viewing_angle < 180.0:
                dist_max = sight_range / (1.0 + 1.0 / sin_deg(half_viewing_angle))
            else:
                dist_max = sight_range / 2.0

            delta_angle = np.abs(normalize_angle(phi - orientation))
            within_range = np.logical_and(
                rho <= sight_range, delta_angle <= half_viewing_angle
            )

            dist2boundary1 = np.minimum(rho, sight_range - rho)
            dist2boundary2 = rho * sin_deg(
                np.minimum(half_viewing_angle - delta_angle, 90.0)
            )
            dist2boundary = np.maximum(np.minimum(dist2boundary1, dist2boundary2), 0.0)

            # Target at the incenter:     score1 -> 1.0
            # Target at the boundary:     score1 -> 0.0
            # Target in the sector:       score1 -> 0.0 ~ 1.0
            # Target out of the boundary: score1 -> 0.0
            scores1 = dist2boundary[within_range] / dist_max
            scores2 = 1.0 - rho[within_range] / sight_range

            scores[within_range, s] = scores1 * scores2

        return state_mesh, coord_grid, scores
''',
    HighValueCoverageHeuristic: _SCORE_BASE_STR
    + '''
class HighValueCoverageHeuristic(_ScoredTargetHeuristic):
    ALGORITHM = """Balance target value against field-of-view efficiency instead of just proximity.

    This heuristic also starts from a remembered-target score, but it is less purely
    intercept-oriented than `LoadedInterceptHeuristic`. Loaded targets still receive a
    strong priority bonus, yet the score additionally rewards targets whose distance is
    close to a useful cone-coverage regime, so the selected pose is more likely to keep
    a broad and efficient field of view instead of collapsing into narrow tunnel vision.

    In effect, the policy trades some urgency for better geometric coverage. It tends
    to keep loaded targets under watch when possible, but among several viable options
    it prefers the one whose range, freshness, and alignment make the resulting camera
    pose useful for continued surveillance rather than just immediate interception.
    """

    def score_target(self, target_state: TargetStatePublic) -> float:
        features = self.target_features(target_state)
        return (
            3.2 * features["loaded"]
            + 1.6 * features["zoom_fit"]
            + 1.3 * features["freshness"]
            + 1.0 * features["center_alignment"]
            + 0.9 * (1.0 - abs(features["range_ratio"] - 0.75))
        )''',
    LoadedInterceptHeuristic: _SCORE_BASE_STR
    + '''
class LoadedInterceptHeuristic(_ScoredTargetHeuristic):
    ALGORITHM = """Lock onto loaded targets first, then break ties by reacquisition cost.

    The heuristic maintains a short memory of recently seen targets and computes a score
    for each remembered target at every step. The score is dominated by the `loaded`
    flag, then refined using memory freshness, angular alignment with the current camera
    heading, and distance to the target. That weighting makes currently loaded targets
    the default priority, but still prefers the loaded target that can be reacquired
    with less steering and less delay.

    After selecting the best target, the camera rotates to center it and adapts the
    viewing angle to match the target distance, using a tighter cone for distant targets
    and a wider cone for nearby ones. If no remembered target remains plausible, the
    heuristic falls back to a search sweep instead of repeating stale actions forever.
    """

    def score_target(self, target_state: TargetStatePublic) -> float:
        features = self.target_features(target_state)
        return (
            4.0 * features["loaded"]
            + 1.8 * features["freshness"]
            + 1.5 * features["center_alignment"]
            + 1.2 * features["near_ratio"]
            + 0.4 * features["zoom_fit"]
        )''',
    PatrolSweepHeuristic: _SCORE_BASE_STR
    + '''
class PatrolSweepHeuristic(_ScoredTargetHeuristic):
    ALGORITHM = """Run a patrol-style policy that only interrupts its sweep for high-payoff tracks.

    This heuristic behaves like a guard camera on patrol. It scores remembered targets
    with a loaded-target bias, but also requires enough freshness, alignment, and range
    quality for a target to be worth breaking the ongoing scan pattern. Unloaded targets
    are penalized if they would pull the camera toward awkward or low-value geometry.

    If the best available target does not exceed a minimum score threshold, the policy
    refuses to commit and instead performs a deliberate wide-angle left-right sweep.
    That makes its fallback behavior qualitatively different from the other local
    heuristics: absence of a strong target signal leads to active search, not passive
    persistence.
    """

    def score_target(self, target_state: TargetStatePublic) -> float:
        features = self.target_features(target_state)
        score = (
            3.5 * features["loaded"]
            + 1.7 * features["freshness"]
            + 1.6 * features["center_alignment"]
            + 0.8 * features["zoom_fit"]
            + 0.5 * features["near_ratio"]
        )
        if not target_state.is_loaded:
            score -= 0.8 * abs(features["range_ratio"] - 0.55)
        return score

    def select_target(
        self, tracked_targets: list[TargetStatePublic]
    ) -> TargetStatePublic | None:
        best_target = super().select_target(tracked_targets)
        if best_target is None:
            return None
        if self.score_target(best_target) < 3.5:
            return None
        return best_target
    ''',
    ReconTrackingHeuristic: _SCORE_BASE_STR
    + '''
class ReconTrackingHeuristic(_ScoredTargetHeuristic):
    ALGORITHM = """Prefer targets with weak prior confirmation while still reacting to loaded ones.

    This is the most exploratory scored policy. It uses the same remembered-target
    buffer as the other local heuristics, but gives explicit weight to targets that have
    never previously been observed in the loaded state. That causes the camera to spend
    more time probing uncertain or weakly characterized targets, which can expose new
    movement routes or behavior changes earlier.

    The exploration bias is not unconditional. A currently loaded target still adds
    value to the score, and freshness, steering effort, and distance continue to matter.
    The result is a policy that leans toward information gain, but does not completely
    ignore high-value targets when the opponent is clearly making progress.
    """

    def score_target(self, target_state: TargetStatePublic) -> float:
        features = self.target_features(target_state)
        return (
            2.4 * features["unseen"]
            + 2.0 * (1.0 - features["loaded"])
            + 1.4 * features["freshness"]
            + 1.0 * features["center_alignment"]
            + 0.8 * features["loaded"]
            + 0.6 * features["near_ratio"]
        )
    ''',
}


__all__ = [
    "GreedyCameraAgent",
    "HeuristicsCameraAgent",
    "HeuristicCameraAgentBase",
    "CoordinatedCoverageHeuristic",
    "HighValueCoverageHeuristic",
    "LoadedInterceptHeuristic",
    "PatrolSweepHeuristic",
    "ReconTrackingHeuristic",
    "HEURISTIC",
    "HEURISTIC_STR",
]
