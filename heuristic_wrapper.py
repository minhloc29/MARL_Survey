from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mate
from agents import HEURISTIC, HeuristicCameraAgentBase, HeuristicsCameraAgent
from mate.agents import GreedyTargetAgent
from mate.wrappers import MultiCamera


def combine_targets(targets_list: np.ndarray):

    targets = targets_list[0]  # T x target_dim

    n_targets = targets.shape[0]

    for camera_targets in targets_list[1:]:
        for i in range(n_targets):
            target_state = camera_targets[i]
            if target_state[-1] == 1:  # if seen_flag is 1, update the target state
                targets[i] = target_state

    return targets


class HeuristicAsActionWrapper(gym.Wrapper):
    env: MultiCamera

    def __init__(
        self,
        env: MultiCamera,
        heuristics: list[HeuristicCameraAgentBase],
    ):
        super().__init__(env)
        self.n_heuristics = len(heuristics)

        self.agents = [
            HeuristicsCameraAgent(heuristics)
            for _ in range(self.env.unwrapped.num_cameras)
        ]

        self.obs = None
        self.refined_obs = None
        self.info = None

        self.action_space = spaces.MultiDiscrete(
            [self.n_heuristics] * self.env.unwrapped.num_cameras
        )  # action space is the index of the heuristic to use for each camera

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mate.group_reset(self.agents, obs)
        self.info = info

        for i in range(len(self.agents)):
            self.agents[i].reset(obs[i])

        self.obs = obs

        return self.obs, info

    def step(self, cameras_heuristic_assignments: list[int]):
        # actions = np.ndarray([self.unwrapped.num_cameras, 2])

        # # group observe
        for i in range(len(self.agents)):
            self.agents[i].receive_heuristic_assignments(
                cameras_heuristic_assignments[i]
            )
        #     print(self.obs[i])
        #     self.agents[i].observe(self.obs[i], self.info[i] if self.info is not None else None)

        camera_joint_action = mate.group_step(
            self.env.unwrapped, self.agents, self.obs, self.info["infos"]
        )

        # group communicate
        # for i in range(len(self.agents)):
        #     self.env.send_messages(self.agents[i].send_requests())
        # for i in range(len(self.agents)):
        #     self.agents[i].receive_requests(self.env.receive_messages(agent=self.agents[i]))
        # for i in range(len(self.agents)):
        #     self.env.send_messages(self.agents[i].send_responses())
        # for i in range(len(self.agents)):
        #     self.agents[i].receive_responses(self.env.receive_messages(agent=self.agents[i]))

        # # group act
        # for i in range(len(self.agents)):
        #     actions[i] = self.agents[i].act(self.obs[i], self.info[i] if self.info is not None else None)

        obs, reward, done, truncated, info = self.env.step(camera_joint_action)
        self.obs = obs
        self.info = info
        return obs, reward, done, truncated, info


class MATCHAWrapper(gym.Wrapper):
    env: HeuristicAsActionWrapper

    def __init__(
        self,
        env: HeuristicAsActionWrapper,
        skip=4,
        reward_type: Literal["default", "coverage_rate"] = "coverage_rate",
        with_render: bool = False,
    ):
        super().__init__(env)
        self.history_size = self.skip = skip
        self.reward_type = reward_type
        self.with_render = with_render

        self.num_cameras: int = self.env.unwrapped.num_cameras
        self.num_targets: int = self.env.unwrapped.num_targets

        observation_space = self.env.observation_space

        self.observation_space = spaces.Box(
            low=np.repeat(np.repeat(
                observation_space[0].low[np.newaxis, ...], self.skip, axis=0
            )[np.newaxis, ...], self.num_cameras, axis=0),
            high=np.repeat(np.repeat(
                observation_space[0].high[np.newaxis, ...], self.skip, axis=0
            )[np.newaxis, ...], self.num_cameras, axis=0),
            shape=[self.num_cameras, self.skip, *observation_space[0].shape],
        )

        self.cameras_history_space = (
            self.skip,
            self.num_cameras,
            *observation_space[0].shape,
        )

        self.cameras_history = np.ndarray(self.cameras_history_space, dtype=np.float64)

    @classmethod
    def preprocess_history(
        self,
        cameras_history: np.ndarray,
    ) -> np.ndarray:
        # cameras_history: (history_len, num_cameras, camera_state_dim)
        # targets_history: (history_len, num_targets, target_state_dim)

        # Permute to (num_cameras, history_len, camera_state_dim)
        cameras_history = cameras_history.transpose(1, 0, 2)

        return cameras_history

    def run_skip(self, actions):
        total_reward = 0

        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(actions)
            if self.reward_type == "coverage_rate":
                reward = info["infos"][0]["coverage_rate"]

            self.cameras_history = np.roll(self.cameras_history, shift=-1, axis=0)
            self.cameras_history[-1] = obs
            total_reward += reward

            if self.with_render:
                run = self.env.render()
                if not run:
                    break

            if done or truncated:
                break

        final_reward = total_reward
        final_reward = total_reward / self.skip  # mean or total?

        return final_reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.cameras_history = np.ndarray(self.cameras_history_space, dtype=np.float64)
        self.obstacles = None
        self.warehouses = None

        self.cameras_history[-1] = obs

        # run skip first to get history
        _, _, _, info = self.run_skip(actions=self.action_space.sample())

        flatten_cameras_history = self.preprocess_history(self.cameras_history)

        return flatten_cameras_history, info

    def step(self, actions):
        final_reward, done, truncated, info = self.run_skip(actions)

        flatten_cameras_history = self.preprocess_history(self.cameras_history)

        return flatten_cameras_history, final_reward, done, truncated, info


def main():
    base_env = gym.make("MultiAgentTracking-v0", config="MATE-4v8-9.yaml")
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    env = MATCHAWrapper(
        HeuristicAsActionWrapper(env, [heuristic_cls() for heuristic_cls in HEURISTIC]),
        skip=16,
    )

    action_space = env.action_space

    camera_joint_observation, _ = env.reset()


    reward = 0
    for i in range(40000):
        camera_joint_observation, target_team_reward, done, truncated, info = env.step(
            action_space.sample()
        )

        print(camera_joint_observation.shape)

        reward += target_team_reward
        env.render()
        if done:
            env.close()
            break


if __name__ == "__main__":
    main()
