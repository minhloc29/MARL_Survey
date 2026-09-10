"""MATE-native Deep Coordination Graphs (DCG) trainer.

A compact, modern port of ``Deep Coordination Graphs`` (Boehmer et al., ICML
2020) from the original PyMARL-based implementation.  The joint Q-value is
factored as a sum of per-agent utilities ``f_i`` and per-edge payoff matrices
``f_ij`` over a coordination graph::

    Q(a) = mean_i f_i(a_i) + mean_{(i,j) in E} f_ij(a_i, a_j)

Greedy action selection is done with max-plus message passing (Kok & Vlassis,
2006) with the "anytime" extension.  Learning is off-policy, one-step
double-Q learning (the same as the paper's ``dcg_learner``).

The ``torch_scatter`` dependency of the original is replaced by dense indexed
sums so this module depends only on numpy + torch, matching the rest of MATE.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from random import randrange
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from mate.algorithms.rode.environment import make_mate_rode_env
from mate.algorithms.rode.replay import RODEEpisodeBuilder, RODEEpisodeReplay


# ============================================================================
# Utility helpers
# ============================================================================


def build_edge_list(edges_arg: Any, n_agents: int) -> list[tuple[int, int]]:
    """Build an edge list for a given topology (port of ``_edge_list``).

    ``edges_arg`` is one of:

    * ``"vdn"``    -> no edges (independent Q-learning)
    * ``"line"``, ``"cycle"``, ``"star"``, ``"full"`` -> named topologies
    * an ``int``   -> that many random edges (``<= (n_agents-1)!``)
    * a ``list``   -> of ``[i, j]`` / ``(i, j)`` integer pairs

    Returns a list of ``(int, int)`` edges with ``i != j`` and no duplicates.
    """
    edges: list[tuple[int, int]] = []
    if isinstance(edges_arg, str):
        if edges_arg == "vdn":
            pass
        elif edges_arg == "line":
            edges = [(i, i + 1) for i in range(n_agents - 1)]
        elif edges_arg == "cycle":
            edges = [(i, i + 1) for i in range(n_agents - 1)] + [(n_agents - 1, 0)]
        elif edges_arg == "star":
            edges = [(0, i + 1) for i in range(n_agents - 1)]
        elif edges_arg == "full":
            edges = [
                (j, i + j + 1) for j in range(n_agents - 1) for i in range(n_agents - j - 1)
            ]
        else:
            raise ValueError(
                "cg_edges must be one of {'vdn','line','cycle','star','full'}, "
                f"an int, or a list of edge pairs; got {edges_arg!r}."
            )
    elif isinstance(edges_arg, int):
        if not 0 <= edges_arg <= factorial(n_agents - 1):
            raise ValueError(
                f"Random-edge count {edges_arg} must be in [0, {factorial(n_agents - 1)}]."
            )
        present: set[tuple[int, int]] = set()
        while len(edges) < edges_arg:
            e = (randrange(n_agents), randrange(n_agents))
            if e[0] != e[1] and e not in present and (e[1], e[0]) not in present:
                present.add(e)
                edges.append(e)
    elif isinstance(edges_arg, list):
        for e in edges_arg:
            if (
                not isinstance(e, (list, tuple))
                or len(e) != 2
                or not all(isinstance(i, int) for i in e)
            ):
                raise ValueError(f"Edge {e!r} is not an [i, j] integer pair.")
            edges.append((int(e[0]), int(e[1])))
    else:
        raise ValueError(f"Unsupported cg_edges value: {edges_arg!r}.")
    return edges


def _mlp(input_dim: int, hidden: Optional[list[int]], output: int) -> nn.Sequential:
    """Fully-connected network with ReLU hidden layers, no output activation."""
    hidden = [] if hidden is None else ([hidden] if isinstance(hidden, int) else list(hidden))
    layers: list[nn.Module] = []
    dim = input_dim
    for d in hidden:
        layers.append(nn.Linear(dim, d))
        layers.append(nn.ReLU())
        dim = d
    layers.append(nn.Linear(dim, output))
    return nn.Sequential(*layers)


class DCGRNNFeatureAgent(nn.Module):
    """A GRU that returns its hidden state only (no per-action output).

    Equivalent to PyMARL's ``rnn_feature_agent``.  Returns ``(None, h)`` to keep
    the call convention of a recurrent head.
    """

    def __init__(self, input_shape: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden)
        self.rnn = nn.GRUCell(hidden, hidden)

    def init_hidden(self) -> torch.Tensor:
        return self.fc1.weight.new_zeros(1, self.rnn.hidden_size)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> tuple[None, torch.Tensor]:
        x = nn.functional.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.rnn.hidden_size))
        return None, h


# ============================================================================
# DCG network
# ============================================================================


class DCGNetwork(nn.Module):
    """Utilities, payoffs and (optionally) a duelling state-value head.

    Port of ``DeepCoordinationGraphMAC``.  All operations are dense, so no
    ``torch_scatter`` dependency is required.
    """

    def __init__(
        self,
        obs_shape: int,
        state_shape: int,
        n_agents: int,
        n_actions: int,
        cg_edges: Any = "full",
        cg_utilities_hidden_dim: Optional[int] = None,
        cg_payoffs_hidden_dim: Optional[int] = None,
        cg_payoff_rank: int = 0,
        duelling: bool = False,
        mixing_embed_dim: int = 32,
        rnn_hidden_dim: int = 64,
        obs_last_action: bool = False,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim
        self.obs_last_action = obs_last_action
        self.payoff_rank = cg_payoff_rank
        self.payoff_decomposition = isinstance(cg_payoff_rank, int) and cg_payoff_rank > 0
        self.duelling = duelling

        # Per-agent observation input: obs (+ one-hot last action if requested).
        agent_input_dim = obs_shape
        if obs_last_action:
            agent_input_dim += n_actions

        # Recurrent agent producing hidden states only.
        self.agent = DCGRNNFeatureAgent(agent_input_dim, rnn_hidden_dim)

        # Utility f_i and payoff f_ij heads.
        self.utility_fun = _mlp(rnn_hidden_dim, cg_utilities_hidden_dim, n_actions)
        payoff_out = 2 * self.payoff_rank * n_actions if self.payoff_decomposition else n_actions ** 2
        self.payoff_fun = _mlp(2 * rnn_hidden_dim, cg_payoffs_hidden_dim, payoff_out)

        # Duelling (DCG-V): a state-dependent bias on top of the graph Q.
        if duelling:
            self.state_value = _mlp(int(state_shape), [mixing_embed_dim], 1)

        # Build the coordination graph edges.
        self._set_edges(build_edge_list(cg_edges, n_agents))

    # ------------------------------------------------------------- edges

    def _set_edges(self, edge_list: list[tuple[int, int]]) -> None:
        """Construct the internal edge/node index buffers."""
        n = len(edge_list)
        self.edges_from = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
        self.edges_to = torch.tensor([e[1] for e in edge_list], dtype=torch.long)
        self.n_edges = n
        # Per-node degree used to keep the message normalisation consistent.
        counts = torch.zeros(self.n_agents, dtype=torch.float)
        for e in edge_list:
            counts[e[0]] += 1
            counts[e[1]] += 1
        self.edges_n_in = counts

    # ------------------------------------------------------------- heads

    def utilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Utilities ``f_i``: ``(B, N, H)`` -> ``(B, N, A)``."""
        return self.utility_fun(hidden_states)

    def payoffs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Symmetric per-edge payoff matrices ``f_ij``.

        Args:
            hidden_states: ``(B, N, H)``.

        Returns:
            Payoffs ``(B, E, A, A)``, symmetrised over edge direction.
        """
        n_edges = len(self.edges_from)
        n = self.n_actions
        batch = hidden_states.shape[0]
        # No edges (VDN topology): return an empty payoff tensor.
        if n_edges == 0:
            return hidden_states.new_zeros(batch, 0, n, n)
        # Forward and flipped edge inputs.
        fwd = torch.cat([hidden_states[:, self.edges_from], hidden_states[:, self.edges_to]], dim=-1)
        rev = torch.cat([hidden_states[:, self.edges_to], hidden_states[:, self.edges_from]], dim=-1)
        inputs = torch.stack([fwd, rev], dim=0)                # (2, B, E, 2H)
        inputs = inputs.reshape(2, batch, n_edges, -1)          # (2, B, E, 2H)
        output = self.payoff_fun(inputs)                        # (2, B, E, out)

        if self.payoff_decomposition:
            # De-decompose the rank-K bases into dense payoff matrices.  The
            # head outputs (left, right) base vectors for each of K ranks.
            out = output.view(2, batch, n_edges, self.payoff_rank, 2, n)
            left, right = out[..., 0, :], out[..., 1, :]   # (2, B, E, K, n)
            matrices = torch.einsum("...ki,...kj->...kij", left, right).sum(dim=-3)
        else:
            matrices = output.view(2, batch, n_edges, n, n)
        # transpose the backward-message output and average forward/reverse
        matrices[1] = matrices[1].transpose(-2, -1).clone()
        return matrices.mean(dim=0)

    # ------------------------------------------------------------- values

    def q_values(self, f_i: torch.Tensor, f_ij: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q-value of the given joint action.

        Args:
            f_i: utilities ``(B, N, A)``.
            f_ij: payoffs ``(B, E, A, A)``.
            actions: chosen actions ``(B, N)``.

        Returns:
            Q-values ``(B,)`` -- one scalar per joint action, as in DCG.
        """
        values = f_i.gather(-1, actions.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        if self.n_edges > 0:
            f_flat = f_ij.reshape(f_ij.shape[0], -1, self.n_actions * self.n_actions)
            dev = actions.device
            from_idx = self.edges_from.to(dev).view(1, -1).expand(actions.shape[0], -1)
            to_idx = self.edges_to.to(dev).view(1, -1).expand(actions.shape[0], -1)
            edge_actions = (
                actions.gather(1, from_idx) * self.n_actions
                + actions.gather(1, to_idx)
            )  # (B, E)
            values = values + f_flat.gather(-1, edge_actions.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        return values

    # ----------------------------------------------------- greedy selection

    def greedy(
        self,
        f_i: torch.Tensor,
        f_ij: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        iterations: int = 8,
        normalized: bool = True,
        anytime: bool = True,
    ) -> torch.Tensor:
        """Max-plus message passing to find greedy actions.

        Args:
            f_i: utilities ``(B, N, A)``.
            f_ij: payoffs ``(B, E, A, A)``.
            available_actions: ``(B, N, A)`` 0/1 mask or None (all available).
            iterations: number of message-passing rounds.
            normalized: re-normalise messages each round (Kok & Vlassis, 2006).
            anytime: keep the best solution seen so far.

        Returns:
            Greedy actions ``(B, N, 1)``.
        """
        # Use doubles internally to reduce accumulating precision loss.
        in_f_i, f_i = f_i, f_i.double() / self.n_agents
        in_f_ij, f_ij = f_ij, f_ij.double() / max(self.n_edges, 1)
        if available_actions is not None:
            f_i = f_i.masked_fill(available_actions == 0, -float("inf"))

        best_value = in_f_i.new_empty(in_f_i.shape[0]).fill_(-float("inf"))
        best_actions = in_f_i.new_empty(in_f_i.shape[0], self.n_agents, 1, dtype=torch.long)
        utils = f_i

        if self.n_edges > 0 and iterations > 0:
            dev = f_i.device
            messages = f_i.new_zeros(2, f_i.shape[0], self.n_edges, self.n_actions)
            counts = self.edges_n_in.to(dev).view(1, self.n_agents, 1)
            for _ in range(iterations):
                joint0 = (utils[:, self.edges_to] - messages[1]).unsqueeze(-1) + f_ij
                joint1 = (utils[:, self.edges_from] - messages[0]).unsqueeze(-1) + f_ij.transpose(-2, -1)
                messages[0] = joint0.max(-2)[0]
                messages[1] = joint1.max(-2)[0]
                if normalized:
                    messages -= messages.mean(-1, keepdim=True)
                utils = f_i + self._aggregate_messages(messages, counts)
                if anytime:
                    actions = utils.max(-1, keepdim=True)[1]
                    value = self.q_values(in_f_i, in_f_ij, actions.squeeze(-1))
                    change = value > best_value
                    best_value[change] = value[change]
                    best_actions[change] = actions[change]

        if not anytime or self.n_edges == 0 or iterations <= 0:
            _, best_actions = utils.max(-1, keepdim=True)
        return best_actions

    def _aggregate_messages(self, messages: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Sum incoming messages to each node: ``(2, B, E, A)`` -> ``(B, N, A)``.

        Replaces ``scatter_add`` from ``torch_scatter`` with dense ``index_add_``.
        ``messages[0]`` travel along edges_from -> edges_to and ``messages[1]``
        the reverse.  The result is normalised by node degree.
        """
        batch = messages.shape[1]
        n = self.n_agents
        dev = messages.device
        dtype = messages.dtype
        acc = torch.zeros(batch, n, self.n_actions, device=dev, dtype=dtype)
        acc.index_add_(1, self.edges_to.to(dev), messages[0])
        acc.index_add_(1, self.edges_from.to(dev), messages[1])
        return acc / counts.to(device=dev, dtype=dtype)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        hidden: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        compute_grads: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Step the GRU one time step and return greedy actions or Q-values.

        This is the MATE-native analogue of ``DeepCoordinationGraphMAC.forward``.

        Args:
            obs: observations ``(B, N, O)``.
            state: global state ``(B, S)`` (used only for the duelling head).
            hidden: current GRU hidden states ``(B, N, H)``.
            actions: if given with ``compute_grads=True``, return Q-values for
                these actions (``(B,)``); otherwise return greedy actions
                ``(B, N, 1)``.
            compute_grads: keep gradients on the returned Q-values.

        Returns:
            ``(output, hidden, (f_i, f_ij))``.
        """
        batch = obs.shape[0]
        ctx = torch.enable_grad() if compute_grads else torch.no_grad()
        with ctx:
            inputs = obs
            _, h = self.agent(inputs.reshape(-1, inputs.shape[-1]),
                              hidden.reshape(-1, self.rnn_hidden_dim))
            hidden = h.view(batch, self.n_agents, self.rnn_hidden_dim)
            f_i = self.utilities(hidden)
            f_ij = self.payoffs(hidden)

            # If concrete actions are given, return their Q-values (gradient
            # tracking controlled by `compute_grads`) regardless of grad flag.
            if actions is not None:
                values = self.q_values(f_i, f_ij, actions)
                if self.duelling:
                    values = values + self.state_value(state).squeeze(-1)
                return values, hidden, (f_i, f_ij)

            # Otherwise return greedy actions from max-plus message passing.
            greedy = self.greedy(f_i, f_ij)
            return greedy, hidden, (f_i, f_ij)


# ============================================================================
# Trainer
# ============================================================================


@dataclass
class DCGConfig:
    """Configuration for the MATE-native DCG trainer."""

    config: str = "MATE-4v8-9.yaml"
    levels: int = 3
    episode_limit: int = 64
    episodes: int = 1000
    update_every: int = 8  # train on the buffer after every N collected episodes
    batch_size: int = 8
    replay_capacity: int = 500
    hidden: int = 64              # GRU hidden dim
    cg_edges: Any = "full"        # topology (see build_edge_list)
    cg_utilities_hidden_dim: Optional[int] = None
    cg_payoffs_hidden_dim: Optional[int] = None
    cg_payoff_rank: int = 0
    duelling: bool = False
    mixing_embed_dim: int = 32
    msg_iterations: int = 8
    msg_normalized: bool = True
    msg_anytime: bool = True
    obs_last_action: bool = False
    lr: float = 5e-4
    gamma: float = 0.99
    target_update_every: int = 200   # episodes between target-network updates
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_steps: int = 50000
    grad_norm_clip: float = 10.0
    device: str = "cpu"
    seed: int = 0
    outdir: str = "ckpts/dcg"


class DCGTrainer:
    """Off-policy, one-step double-Q DCG trainer on the MATE camera team."""

    def __init__(self, config: DCGConfig) -> None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        self.env = make_mate_rode_env(
            config=config.config,
            levels=config.levels,
            episode_limit=config.episode_limit,
            render_mode="rgb_array",
        )
        self.network = DCGNetwork(
            self.env.obs_shape,
            self.env.state_shape,
            self.env.n_agents,
            self.env.n_actions,
            cg_edges=config.cg_edges,
            cg_utilities_hidden_dim=config.cg_utilities_hidden_dim,
            cg_payoffs_hidden_dim=config.cg_payoffs_hidden_dim,
            cg_payoff_rank=config.cg_payoff_rank,
            duelling=config.duelling,
            mixing_embed_dim=config.mixing_embed_dim,
            rnn_hidden_dim=config.hidden,
            obs_last_action=config.obs_last_action,
        ).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.replay = RODEEpisodeReplay(config.replay_capacity, self.device)
        self.epsilon = config.epsilon_start
        self._env_steps = 0

    # ------------------------------------------------------ exploration

    def _schedule_epsilon(self) -> None:
        cfg = self.config
        frac = min(1.0, self._env_steps / max(cfg.epsilon_anneal_steps, 1))
        self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_finish - cfg.epsilon_start)

    # ------------------------------------------------------- collection

    def collect_episode(self, episode_index: int):
        obs, state = self.env.reset(seed=self.config.seed + episode_index)
        builder = RODEEpisodeBuilder(self.env.n_agents, self.env.obs_shape, self.env.n_actions)
        builder.start(obs, state, self.env.get_avail_actions())
        hidden = torch.zeros(self.env.n_agents, self.config.hidden, device=self.device)
        total_reward = 0.0
        for _ in range(self.config.episode_limit):
            self._schedule_epsilon()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                greedy, hidden, _ = self.network.forward(obs_t, state_t, hidden.unsqueeze(0))
            greedy = greedy.squeeze(0).squeeze(-1)  # (N,)
            if np.random.random() < self.epsilon:
                actions = torch.randint(
                    0, self.env.n_actions, (self.env.n_agents,), device=self.device
                )
            else:
                actions = greedy
            action_array = actions.cpu().numpy()
            reward, done, _ = self.env.step(action_array)
            next_obs = self.env.get_obs()
            next_state = self.env.get_state()
            builder.add(action_array, reward, done, next_obs, next_state,
                        self.env.get_avail_actions())
            obs, state = next_obs, next_state
            self._env_steps += 1
            total_reward += reward
            if done:
                break
        return builder.finish(), total_reward

    # ------------------------------------------------------------ update

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        """One-step double-Q DCG update over a sampled batch."""
        cfg = self.config
        obs = batch["obs"]         # (B, T+1, N, O)
        state = batch["state"]     # (B, T+1, S)
        actions = batch["actions"]  # (B, T, N)
        rewards = batch["rewards"].squeeze(-1)       # (B, T)
        terminated = batch["terminated"].squeeze(-1).float()  # (B, T)
        mask = batch["filled"].squeeze(-1).float()            # (B, T)

        B, T = actions.shape[0], actions.shape[1]
        hidden_dim = self.config.hidden

        # ---- target: value of greedy actions at t+1 (double-Q) -------------
        # Pass 1: collect greedy actions {t+1} by stepping the target GRU
        # through the sequence once.
        greedy_actions = [None] * (T + 1)
        hidden_t = torch.zeros(B, self.env.n_agents, hidden_dim, device=self.device)
        for t in range(T + 1):
            g, hidden_t, _ = self.target_network.forward(
                obs[:, t], state[:, t], hidden_t, actions=None
            )
            greedy_actions[t] = g.squeeze(-1)  # (B, N)
        # Pass 2: value those greedy actions with the same target net.
        hidden_v = torch.zeros(B, self.env.n_agents, hidden_dim, device=self.device)
        targets_list = []
        for t in range(T):
            val, hidden_v, _ = self.target_network.forward(
                obs[:, t + 1], state[:, t + 1], hidden_v,
                actions=greedy_actions[t + 1], compute_grads=False,
            )
            targets_list.append(val)
        targets = torch.stack(targets_list, dim=1)  # (B, T)
        targets = rewards + cfg.gamma * (1 - terminated) * targets

        # ---- current: Q of the taken actions with gradient ------------------
        hidden_c = torch.zeros(B, self.env.n_agents, hidden_dim, device=self.device)
        mac_out = []
        for t in range(T):
            val, hidden_c, _ = self.network.forward(
                obs[:, t], state[:, t], hidden_c,
                actions=actions[:, t], compute_grads=True,
            )
            mac_out.append(val)
        mac_out = torch.stack(mac_out, dim=1)  # (B, T)

        td_error = mac_out - targets.detach()
        masked = td_error * mask
        loss = (masked ** 2).sum() / mask.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.network.parameters() if p.requires_grad], cfg.grad_norm_clip
        )
        self.optimizer.step()
        return float(loss.detach().cpu())

    # ------------------------------------------------------------ train

    def train(self) -> None:
        cfg = self.config
        last_loss: Optional[float] = None
        for episode_index in range(cfg.episodes):
            episode, reward = self.collect_episode(episode_index)
            self.replay.add(episode)
            if (episode_index + 1) % cfg.update_every == 0 and len(self.replay) >= cfg.batch_size:
                last_loss = self.update(self.replay.sample(cfg.batch_size))
            if (episode_index + 1) % cfg.target_update_every == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            if (episode_index + 1) % 10 == 0:
                msg = f"episode {episode_index + 1}/{cfg.episodes} | reward {reward:.3f}"
                if last_loss is not None:
                    msg += f" | loss {last_loss:.4f}"
                print(msg, flush=True)
        outdir = Path(cfg.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"network": self.network.state_dict(), "config": cfg.__dict__},
                   outdir / "latest.pt")
        self.env.close()


def run_dcg(args: Any) -> None:
    """Entry point wired into ``train.py``'s algorithm dispatcher."""
    DCGTrainer(DCGConfig(
        config=args.dcg_config,
        levels=args.dcg_levels,
        episode_limit=args.dcg_episode_limit,
        episodes=args.dcg_episodes,
        update_every=args.dcg_update_every,
        batch_size=args.dcg_batch_size,
        hidden=args.dcg_hidden,
        cg_edges=args.dcg_cg_edges,
        cg_payoff_rank=args.dcg_payoff_rank,
        duelling=args.dcg_duelling,
        msg_iterations=args.dcg_msg_iterations,
        lr=args.dcg_lr,
        device=args.device,
        seed=args.seed,
        outdir=args.dcg_outdir,
    )).train()
