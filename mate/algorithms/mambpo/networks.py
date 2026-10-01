"""MAMBPO local Gaussian policies and shared residual world-model ensemble.

Reference: danielwillemsen/MAMBPO, 61593227dfb8ee6cfc21c0e5bb81098f417a6a5a.
Source mapping and adaptations are documented in README.md.
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, hidden, layers=2, activation=nn.ReLU):
    modules = []
    for _ in range(layers):
        modules.extend([nn.Linear(inputs, hidden), activation()])
        inputs = hidden
    return nn.Sequential(*modules, nn.Linear(hidden, outputs))


class GaussianActor(nn.Module):
    """The actor consumes only this camera's own observation."""

    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = mlp(obs_dim, action_dim * 2, hidden)

    def forward(self, obs, greedy=False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-20., 2.).exp())
        raw = mean if greedy else distribution.rsample()
        # Numerically stable tanh Jacobian, including nearly saturated actions.
        correction = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_probability = (distribution.log_prob(raw) - correction).sum(-1)
        return raw.tanh(), log_probability


class MAMBPOAgent(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim, actor_hidden=128, critic_hidden=256, alpha=0.1):
        super().__init__()
        self.actor = GaussianActor(obs_dim, action_dim, actor_hidden)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.critics = nn.ModuleList([mlp(n_agents * (obs_dim + action_dim), 1, critic_hidden) for _ in range(2)])
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))

    def q_values(self, observations, actions, target=False):
        value_input = torch.cat([observations.flatten(1), actions.flatten(1)], dim=-1)
        critics = self.target_critics if target else self.critics
        return torch.stack([critic(value_input).squeeze(-1) for critic in critics], dim=0)


class ResidualModel(nn.Module):
    def __init__(self, obs_size, action_size, hidden):
        super().__init__()
        self.output_dim = obs_size + 1  # joint observation delta + shared reward
        self.net = mlp(obs_size + action_size, 2 * self.output_dim + 1,
                       hidden, layers=4, activation=nn.LeakyReLU)
        self.register_buffer("max_logvar", torch.tensor(-2.))
        self.register_buffer("min_logvar", torch.tensor(-5.))

    def forward(self, inputs):
        result = self.net(inputs)
        mean = result[..., :self.output_dim]
        raw_logvar = result[..., self.output_dim:2 * self.output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar, result[..., -1]


class WorldModel(nn.Module):
    """One ensemble shared by all agents, predicting observation increments."""

    def __init__(self, n_agents, obs_dim, action_dim, hidden=200, members=10):
        super().__init__()
        self.n_agents, self.obs_dim = n_agents, obs_dim
        self.models = nn.ModuleList([ResidualModel(n_agents * obs_dim, n_agents * action_dim, hidden)
                                     for _ in range(members)])

    def forward(self, obs, actions):
        inputs = torch.cat([obs.flatten(1), actions.flatten(1)], dim=-1)
        result = [model(inputs) for model in self.models]
        return tuple(torch.stack([row[k] for row in result], dim=0) for k in range(3))

    def objective(self, batch):
        mean, logvar, done_logits = self(batch["obs"], batch["actions"])
        delta = (batch["next_obs"] - batch["obs"]).flatten(1)
        target = torch.cat([delta, batch["rewards"].mean(-1, keepdim=True)], dim=-1)
        # Match upstream: each independently initialized member sees the batch.
        nll = ((mean - target).square() * (-logvar).exp() + logvar).mean((1, 2)).sum()
        done_target = batch["terminated"].expand_as(done_logits)
        return nll + F.binary_cross_entropy_with_logits(done_logits, done_target)

    @torch.no_grad()
    def validation_error(self, batch):
        mean, _, _ = self(batch["obs"], batch["actions"])
        target = torch.cat([(batch["next_obs"] - batch["obs"]).flatten(1),
                            batch["rewards"].mean(-1, keepdim=True)], dim=-1)
        return (mean - target).square().mean((1, 2))

    @torch.no_grad()
    def predict(self, obs, actions):
        mean, logvar, done_logits = self(obs, actions)
        member = torch.randint(len(self.models), (len(obs),), device=obs.device)
        rows = torch.arange(len(obs), device=obs.device)
        selected = mean[member, rows]
        prediction = selected + torch.randn_like(selected) * (0.5 * logvar[member, rows]).exp()
        next_obs = obs + prediction[:, :-1].reshape_as(obs)
        reward = prediction[:, -1:].expand(-1, self.n_agents)
        terminated = torch.bernoulli(done_logits[member, rows].sigmoid())
        return next_obs, reward, terminated
