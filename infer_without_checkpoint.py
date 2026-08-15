import gymnasium as gym

import gym_agent as ga
import mate
from agents import HEURISTIC, HEURISTIC_STR
from heuristic_wrapper import HeuristicAsActionWrapper, MATCHAWrapper
from mate.agents import GreedyTargetAgent
from models.main import MATCHA, MATCHAActor, MATCHACritic

# MATE-4v8-9.yaml
# MATE-16v32-16.yaml
env_config = "MATE-4v8-9.yaml"

def env_func(*args, **kwargs):
    base_env = gym.make("MultiAgentTracking-v0", config=env_config, **kwargs)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    env: MATCHAWrapper = MATCHAWrapper(
        HeuristicAsActionWrapper(env, [heuristic_cls() for heuristic_cls in HEURISTIC]),
        skip=16,
    )

    return env


def main():
    temp_env = env_func()

    n_observation = temp_env.observation_space.shape[-1]


    def feature_extractor_factory():
        model = MATCHA(
            observation_dim=n_observation,
        )
        model.register_heuristics([HEURISTIC_STR[_t] for _t in HEURISTIC])
        return model


    def actor_factory():
        return MATCHAActor()


    def critic_factory():
        return MATCHACritic()


    ppo_policy = ga.core.ActorCriticPolicy(
        actor_factory_func=actor_factory,
        critic_factory_func=critic_factory,
        feature_extractor_func=feature_extractor_factory,
        actor_lr=3e-4,
    )


    agent = ga.PPO(
        env=env_func,
        policy=ppo_policy,
    )

    agent.play()

if __name__ == "__main__":
    main()
