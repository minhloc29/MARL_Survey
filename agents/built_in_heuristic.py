from __future__ import annotations

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
