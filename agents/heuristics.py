from __future__ import annotations

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
        )


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
        )


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
