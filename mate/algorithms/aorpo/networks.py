"""Modern PyTorch implementation of AORPO's policy and probabilistic ensemble.

Algorithm reference: apexrl/AORPO, commit 6310b678a076cf04385b98474bc710a95a2cbefb.
See README.md for the source mapping and deliberate MATE adaptations.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, hidden):
    # MATE observations are already rescaled. LayerNorm also works for one
    # rollout state; unlike BatchNorm it needs no separate running statistics.
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def discrete_action(logits, stochastic=True):
    action = (F.gumbel_softmax(logits, tau=1.0, hard=True) if stochastic else
              F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype))
    log_prob = (action * F.log_softmax(logits, dim=-1)).sum(-1)
    return action, log_prob


def adaptive_lengths(match_rates, horizon):
    """Upstream's match-rate approximation, with explicit zero-error handling.

    floor(K * accuracy_j / max_accuracy) modeled steps, then query policy j.
    A completely unmatched set gets zero modeled steps. Empty sets support
    single-camera scenarios without a reduction over an empty tensor.
    """
    if horizon < 1 or not torch.isfinite(match_rates).all() or torch.any((match_rates < 0) | (match_rates > 1)):
        raise ValueError("Require positive horizon and match rates in [0, 1]")
    if match_rates.numel() == 0 or match_rates.max() == 0:
        return torch.zeros_like(match_rates, dtype=torch.long)
    return torch.floor(horizon * match_rates / match_rates.max()).long().clamp(0, horizon)


class EnsembleLinear(nn.Module):
    def __init__(self, members, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(members, 1, outputs))
        std = 0.5 / math.sqrt(inputs)
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x):
        return x @ self.weight + self.bias


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_agents, obs_dim, n_actions, hidden=128, members=5):
        super().__init__()
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.members = members
        input_dim = n_agents * (obs_dim + n_actions)
        self.output_dim = n_agents * (obs_dim + 1)
        self.layers = nn.ModuleList([EnsembleLinear(members, input_dim, hidden),
                                     EnsembleLinear(members, hidden, hidden),
                                     EnsembleLinear(members, hidden, hidden),
                                     EnsembleLinear(members, hidden, 2 * self.output_dim + 1)])
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.max_logvar = nn.Parameter(torch.full((self.output_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((self.output_dim,), -10.0))

    def inputs(self, obs, actions):
        onehot = F.one_hot(actions, self.n_actions).to(obs.dtype)
        return torch.cat([obs.flatten(-2), onehot.flatten(-2)], dim=-1)

    @torch.no_grad()
    def fit_stats(self, inputs):
        self.input_mean.copy_(inputs.mean(0))
        std = inputs.std(0, unbiased=False)
        self.input_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))

    def forward(self, inputs):
        x = (inputs - self.input_mean) / self.input_std
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        output = self.layers[-1](x)
        mean = output[..., :self.output_dim]
        raw_logvar = output[..., self.output_dim:2 * self.output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar, output[..., -1]

    def objective(self, batch):
        inputs = self.inputs(batch["obs"], batch["actions"])
        target = torch.cat([batch["next_obs"].flatten(1), batch["rewards"]], -1)
        indices = torch.randint(len(inputs), (self.members, len(inputs)), device=inputs.device)
        mean, logvar, done_logits = self(inputs[indices])
        nll = ((mean - target[indices]).square() * (-logvar).exp() + logvar).mean((1, 2)).sum()
        terminal_loss = F.binary_cross_entropy_with_logits(done_logits, batch["terminated"][indices])
        regularizer = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        for layer, decay in zip(self.layers, (1e-4, 2.5e-4, 2.5e-4, 5e-4)):
            regularizer = regularizer + 0.5 * decay * layer.weight.square().sum()
        return nll + terminal_loss + regularizer

    @torch.no_grad()
    def predict(self, obs, actions):
        mean, logvar, done_logits = self(self.inputs(obs, actions))
        # Sample a member independently for each trajectory, and use standard
        # deviation exp(0.5*logvar), not variance as Normal's scale argument.
        member = torch.randint(self.members, (len(obs),), device=obs.device)
        row = torch.arange(len(obs), device=obs.device)
        prediction = mean[member, row] + torch.randn_like(mean[member, row]) * (0.5 * logvar[member, row]).exp()
        split = self.n_agents * self.obs_dim
        next_obs = prediction[:, :split].reshape(-1, self.n_agents, self.obs_dim)
        reward = prediction[:, split:]
        terminated = torch.bernoulli(done_logits[member, row].sigmoid())
        return next_obs, reward, terminated


class AORPOAgent(nn.Module):
    def __init__(self, index, n_agents, obs_dim, n_actions, hidden, model_hidden, ensemble_size):
        super().__init__()
        self.index, self.n_agents, self.n_actions = index, n_agents, n_actions
        self.other_indices = [j for j in range(n_agents) if j != index]
        self.opponents = nn.ModuleDict({str(j): mlp(obs_dim, n_actions, hidden) for j in self.other_indices})
        self.actor = mlp(obs_dim + (n_agents - 1) * n_actions, n_actions, hidden)
        self.critics = nn.ModuleList([mlp(n_agents * (obs_dim + n_actions), 1, hidden) for _ in range(2)])
        from copy import deepcopy
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.dynamics = DynamicsEnsemble(n_agents, obs_dim, n_actions, model_hidden, ensemble_size)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer("match_rates", torch.zeros(n_agents - 1))
        self.register_buffer("query_counts", torch.zeros(n_agents - 1, dtype=torch.long))

    @torch.no_grad()
    def opponent_actions(self, obs):
        return {j: discrete_action(self.opponents[str(j)](obs[:, j]), stochastic=False)[0]
                for j in self.other_indices}

    def policy(self, obs, stochastic=True):
        others = self.opponent_actions(obs)
        inputs = torch.cat([obs[:, self.index], *[others[j] for j in self.other_indices]], dim=-1)
        logits = self.actor(inputs)
        own, log_prob = discrete_action(logits, stochastic)
        joint = torch.stack([own if j == self.index else others[j] for j in range(self.n_agents)], dim=1)
        return own, log_prob, joint, logits

    def q_values(self, obs, joint_actions, target=False):
        inputs = torch.cat([obs.flatten(1), joint_actions.flatten(1)], dim=-1)
        critics = self.target_critics if target else self.critics
        return torch.stack([critic(inputs).squeeze(-1) for critic in critics], dim=0)
