import gymnasium as gym
import gym_agent as ga
import mate
from agents import GreedyCameraAgent
from mate.agents import GreedyTargetAgent
from wrapper import NetWrapper, AgentAsActionWrapper
from agents.base import GoalsBaseAgent
from gym_agent.core.polices import ActorCriticPolicy
from net import Net, FeatureExtractor
from common_net.attentions import MHAConfig
import torch.nn as nn
import torch

class Critic(nn.Module):
    def __init__(
        self,
        history_len: int = 8,
        camera_dim: int = 8,
        target_dim: int = 3,
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(num_heads=1),
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            history_len=history_len,
            camera_dim=camera_dim,
            target_dim=target_dim,
            n_linear_attn_blocks=n_linear_attn_blocks,
            n_attn_blocks=n_attn_blocks,
            embed_dim=embed_dim,
            mha_config=mha_config,
        )   # B x C x D

        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        features: torch.Tensor = self.feature_extractor(x)
        avg_features = features.mean(dim=-2)  # B x D
        value = self.linear(avg_features)
        return value

def make_actor_net():
    return Net()


def make_critic_net():
    return Critic()


def make_env(**kwargs):
    base_env = gym.make("MultiAgentTracking-v0", config="MATE-4v8-9.yaml", **kwargs)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    return NetWrapper(
        AgentAsActionWrapper(
            env,
            [GoalsBaseAgent(executor=GreedyCameraAgent())
             for _ in range(env.unwrapped.num_cameras)],
        ),
        reward_type="coverage_rate",
    )

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--actor-lr",
        type=float,
        default=5e-5,
        help="Learning rate for the agent's optimizer."
    )

    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the agent's optimizer."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run training on: 'auto', 'cuda', 'mps' (Apple Silicon), or 'cpu'."
    )


    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments to use for training."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run in each environment per iteration."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training the agent."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total number of timesteps to train the agent."
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10_000,
        help="Save the agent every N timesteps."
    )


    args = parser.parse_args()

    agent = ga.PPO(
        env = make_env,
        policy=ActorCriticPolicy(
            actor_factory_func=make_actor_net,
            critic_factory_func=make_critic_net,
            actor_lr = args.actor_lr,
            critic_lr = args.critic_lr
        ),
        config=ga.PPOConfig(
            num_envs= args.num_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            log_score_type='mean'
        )
    )

    agent.fit(
        total_timesteps=args.total_timesteps,
        save_every=args.save_every,
        wandb_api="wandb_v1_UDhL4VvySJ8joPhdZKFgmNsCohZ"
        # wandb_api="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"
    )

if __name__ == "__main__":
    main()

