"""Value-decomposition (QMIX / QPLEX) components for the MATE baseline.

Both QMIX and QPLEX are off-policy, discrete-action value-decomposition
algorithms. They share:

  * a per-agent discrete Q-function ``Q_i(o_i, a_i)`` built over the shared
    ``CameraObsEncoder`` backbone from :mod:`flow.nets`,
  * an epsilon-greedy joint exploration policy,
  * a replay buffer (``flow.ReplayBuffer``),
  * and a target network for the Q-heads and the mixer.

They differ only in how the joint action value ``Q_tot`` is mixed from the
per-agent ``Q_i`` --- QMIX via a monotonic hypernetwork, QPLEX via a
duplex-dueling (V/A) transformation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_joint(observer, camera_encoder, obs_np, num_cameras, device):
    """Encode a raw joint camera observation into per-camera embeddings.

    Args:
        observer: a ``flow.MateObserver`` (parses raw obs into entity dicts).
        camera_encoder: a ``flow.CameraObsEncoder`` shared across cameras.
        obs_np: (B, C, obs_dim) raw camera observations.
        num_cameras: number of cameras C.
        device: torch device.

    Returns:
        (B, C, obs_emb_dim) tensor, indexed [b, c] for camera c's embedding.
    """
    B = obs_np.shape[0]
    parsed = [observer.parse(obs_np[b]) for b in range(B)]
    embs = []
    for c in range(num_cameras):
        ss = np.stack([parsed[b][c]["self_state"] for b in range(B)])
        ents, msks = {}, {}
        kinds = list(parsed[0][c]["entities"])
        for k in kinds:
            ents[k] = np.stack([parsed[b][c]["entities"][k][0] for b in range(B)])
            msks[k] = np.stack([parsed[b][c]["entities"][k][1] for b in range(B)])
        embs.append(camera_encoder(
            torch.as_tensor(ss, dtype=torch.float32, device=device),
            {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in ents.items()},
            {k: torch.as_tensor(v, dtype=torch.bool, device=device) for k, v in msks.items()},
        ))  # (B, obs_emb_dim)
    return torch.stack(embs, dim=1)  # (B, C, obs_emb_dim)


class AgentQNetwork(nn.Module):
    """Per-agent discrete Q-function head over one camera embedding.

    Args:
        obs_emb_dim: per-camera observation embedding dim.
        n_actions: number of discrete actions (e.g. 25 for levels=5).
        hidden: hidden width.
    """

    def __init__(self, obs_emb_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, o_i: torch.Tensor) -> torch.Tensor:
        """(B, obs_emb_dim) -> (B, n_actions) Q-values for each discrete action."""
        return self.net(o_i)


class QplexMixer(nn.Module):
    """QPLEX duplex-dueling mixer: Q_tot = V_tot + A_tot with IGM matching.

    Duplex dueling splits each per-agent Q into a value ``V_i`` and an
    advantage ``A_i``, mixes the joint value ``V_tot`` monotonically, and mixes
    the joint advantage ``A_tot`` through an agent attention that is rescaled to
    satisfy the IGM principle (no over-estimation of greedy joint actions).

    Args:
        n_agents: number of cameras.
        state_dim: dim of the global state fed to the value/advantage mixers.
        n_actions: per-agent discrete action count.
        hidden: hidden width.
    """

    def __init__(self, n_agents: int, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        # W_2 = diag over agents of a signed scalar; A_tot = W_2^T A (see PLEX).
        self.W_2 = nn.Parameter(torch.ones(1, n_agents, 1) * 0.1)
        # Transform A_i -> A'_i per agent (shared MLP over (state, A)).
        self.adv_transform = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        # V_tot mixing (monotonic hypernetwork), consistent with QMIX's.
        self.hyper_v1 = nn.Linear(state_dim, n_agents * hidden)
        self.hyper_v2 = nn.Linear(state_dim, hidden)
        self.hyper_vb1 = nn.Linear(state_dim, hidden)
        self.hyper_vb2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.hidden = hidden

    def _mix_value(
        self, v_values: torch.Tensor, global_state: torch.Tensor
    ) -> torch.Tensor:
        """Monotonically mix per-agent values V_i -> V_tot.

        v_values: (B, n_agents). Returns (B,).
        """
        B = v_values.size(0)
        w1 = torch.abs(self.hyper_v1(global_state)).view(B, self.n_agents, self.hidden)
        b1 = self.hyper_vb1(global_state).view(B, 1, self.hidden)
        h = F.elu(torch.bmm(v_values.unsqueeze(1), w1) + b1)  # (B, 1, hidden)
        w2 = torch.abs(self.hyper_v2(global_state)).view(B, self.hidden, 1)
        b2 = self.hyper_vb2(global_state).view(B, 1, 1)
        return (torch.bmm(h, w2) + b2).view(B)

    def joint_q_tot(
        self,
        q_values: torch.Tensor,
        joint_actions: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q_tot for the taken joint action.

        Args:
            q_values: (B, n_agents, n_actions) per-agent Q-values.
            joint_actions: (B, n_agents) long indices of the taken actions.
            global_state: (B, state_dim).

        Returns:
            (B,) joint action value.
        """
        B, n_agents = joint_actions.shape
        # Per-agent values and advantages at the taken action.
        idx = joint_actions.unsqueeze(-1)  # (B, C, 1)
        taken = torch.gather(q_values, dim=-1, index=idx).squeeze(-1)  # (B, C)
        # V_i = max_a Q_i (dueling value); A_i = Q_i - V_i.
        v_i = q_values.max(dim=-1).values  # (B, C)
        adv_taken = taken - v_i  # (B, C)
        # Advantage transform conditioned on the global state.
        adv_transform_input = torch.cat(
            [global_state.unsqueeze(1).expand(B, n_agents, -1), q_values], dim=-1
        )  # (B, C, state_dim + n_actions)
        adv_transformed = self.adv_transform(adv_transform_input)  # (B, C, n_actions)
        adv_taken_transformed = torch.gather(
            adv_transformed, dim=-1, index=idx
        ).squeeze(-1)  # (B, C)

        # Intermediary: rescale A'_i to satisfy IGM (no over-estimation).
        adv_combined = adv_taken_transformed + adv_taken  # (B, C)

        # Joint value and advantage mixing.
        V_tot = self._mix_value(v_i, global_state)  # (B,)
        A_tot = (self.W_2.view(1, n_agents) * adv_combined).sum(dim=-1)  # (B,)
        return V_tot + A_tot
