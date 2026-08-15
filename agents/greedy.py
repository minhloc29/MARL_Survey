"""Built-in greedy rule-based agents."""

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
