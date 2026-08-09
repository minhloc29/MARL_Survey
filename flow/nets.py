"""Network modules and losses for the MAC-Flow (flow-matching CTDE) algorithm.

This mirrors the offline MAC-Flow recipe (Lee, Lee & Zhang, 2026) adapted to
online CTDE training on MATE's cooperative camera team:

  Stage 1  joint flow policy       v_phi(t, o_joint, x_t)  -- flow-matching BC
  Stage 2  per-agent IGM critics   Q_theta_i(o_i, a_i)     + a mixer -> Q_tot
  Stage 3  one-step policies       mu_w_i(o_i, z_i)        distilled from the flow

Working-skeleton scope: three stage-1 encoders over the three MATE entity sets
(targets / obstacles / teammates), uniform-sampling BC, and the IGM avg mixer.
See the repo plan for the deviations (advantage-weighted BC, monotonic mixer,
role conditioning) that are deliberately left for later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Entity / set encoder (permutation-invariant, no slot attention needed:
#    MATE gives already-parsed entity vectors; we only need invariance over
#    a variable-length visible-entity list, not object discovery).
# --------------------------------------------------------------------------
class EntitySetEncoder(nn.Module):
    """Permutation-invariant encoder for a variable-length list of entities.

    Args:
        entity_dim: feature dim of a single entity.
        embed_dim: output embedding dim.
        n_heads: attention heads for the self-attention and pooling layers.
    """

    def __init__(self, entity_dim: int, embed_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(entity_dim, embed_dim)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, entities: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of entity sets.

        Args:
            entities: (B, N_max, entity_dim), zero-padded.
            valid_mask: (B, N_max) bool, True where a real entity is present.

        Returns:
            (B, embed_dim) pooled set embedding.
        """
        x = self.embed(entities)
        # TransformerEncoderLayer expects key_padding_mask=True at PAD positions.
        x = self.self_attn(x, src_key_padding_mask=~valid_mask)
        q = self.pool_query.expand(x.size(0), -1, -1)  # (B, 1, E)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~valid_mask)
        return pooled.squeeze(1)


# --------------------------------------------------------------------------
# MATE -> per-agent obs encoder. A camera observes three entity sets in its
# raw fixed-slot observation (targets, obstacles, teammates), each with its
# own feature dim, plus its self state. We pool each set separately with its
# own EntitySetEncoder and concatenate the results with the self embedding.
# --------------------------------------------------------------------------
class CameraObsEncoder(nn.Module):
    """Encode one camera's raw observation (self-state + 3 entity sets).

    Args:
        self_state_dim: dim of the camera's own state (9 for MATE).
        entity_dims: dict-like of {kind: entity_dim} for the three entity
            sets, e.g. {"targets": 5, "obstacles": 4, "teammates": 7}.
        embed_dim: embedding dim per entity set and for the self projection.
    """

    def __init__(
        self, self_state_dim: int, entity_dims: dict[str, int], embed_dim: int = 128
    ):
        super().__init__()
        # Fixed ordering of entity sets; callers must pass entities/masks in
        # this same order.
        self.kinds = list(entity_dims)
        self.encoders = nn.ModuleDict(
            {k: EntitySetEncoder(d, embed_dim) for k, d in entity_dims.items()}
        )
        self.self_proj = nn.Linear(self_state_dim, embed_dim)
        # self embedding + one pooled embedding per entity set.
        self.out_dim = embed_dim * (len(self.kinds) + 1)

    def forward(
        self,
        self_state: torch.Tensor,
        entities: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode a batched camera observation.

        Args:
            self_state: (B, self_state_dim).
            entities: {kind: (B, N_max_kind, entity_dim_kind)}.
            masks: {kind: (B, N_max_kind) bool}.

        Returns:
            (B, out_dim) flattened observation embedding.
        """
        parts = [self.self_proj(self_state)]
        for k in self.kinds:
            parts.append(self.encoders[k](entities[k], masks[k]))
        return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------
# 2. Stage-1: joint flow policy (velocity field over the JOINT action).
#    The agent tuple is ordered [camera_0, ..., camera_{C-1}], so the joint
#    action is the concatenation of per-camera low-level actions.
# --------------------------------------------------------------------------
class VelocityField(nn.Module):
    def __init__(self, joint_obs_dim: int, joint_act_dim: int, hidden: int = 512):
        super().__init__()
        self.joint_act_dim = joint_act_dim
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + joint_act_dim + 1, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, joint_act_dim),
        )

    def forward(self, t: torch.Tensor, o_joint: torch.Tensor, x_t: torch.Tensor):
        """Predict velocity at time t for state x_t.

        Args:
            t: (B, 1) time in [0, 1].
            o_joint: (B, joint_obs_dim) concatenated joint observation.
            x_t: (B, joint_act_dim) current joint flow state.
        """
        return self.net(torch.cat([o_joint, x_t, t], dim=-1))


@torch.no_grad()
def sample_joint_flow(
    velocity_field: VelocityField,
    o_joint: torch.Tensor,
    act_dim: int,
    steps: int = 10,
) -> torch.Tensor:
    """Euler-sample the joint flow policy (matches MAC-Flow's 10-step default)."""
    B = o_joint.size(0)
    x = torch.randn(B, act_dim, device=o_joint.device)
    dt = 1.0 / steps
    t = torch.zeros(B, 1, device=o_joint.device)
    for _ in range(steps):
        x = x + velocity_field(t, o_joint, x) * dt
        t = t + dt
    return x


def flow_bc_loss(
    velocity_field: VelocityField, o_joint: torch.Tensor, a_joint: torch.Tensor
) -> torch.Tensor:
    """Linear-path flow matching over the joint observation/action (Eq. 6)."""
    x0 = torch.randn_like(a_joint)
    t = torch.rand(a_joint.size(0), 1, device=a_joint.device)
    x_t = (1 - t) * x0 + t * a_joint
    target_v = a_joint - x0
    pred_v = velocity_field(t, o_joint, x_t)
    return F.mse_loss(pred_v, target_v)


def advantage_weighted_flow_bc_loss(
    velocity_field: VelocityField,
    o_joint: torch.Tensor,
    a_joint: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Advantage-weighted flow matching (AWR/IQL-style implicit improvement).

    Instead of fitting the flow uniformly over the buffer (pure imitation),
    weight each transition by ``exp(A(o,a) / beta)`` so the joint flow tracks
    good joint actions in the buffer more than average ones. ``A`` is an
    advantage estimate from the IGM critics (see train_flow.py).

    Args:
        velocity_field: the joint flow VelocityField.
        o_joint: (B, joint_obs_dim) joint observation.
        a_joint: (B, joint_act_dim) target joint action.
        advantages: (B,) per-transition advantage estimates (not normalized
            is fine; weights are softmax-stabilized below).
        beta: advantage-temperature. Smaller beta upweights good transitions
            more aggressively.

    Returns:
        Weighted flow-matching MSE (mean over batch).
    """
    x0 = torch.randn_like(a_joint)
    t = torch.rand(a_joint.size(0), 1, device=a_joint.device)
    x_t = (1 - t) * x0 + t * a_joint
    target_v = a_joint - x0
    pred_v = velocity_field(t, o_joint, x_t)

    per_sample = (pred_v - target_v).pow(2).mean(dim=-1)  # (B,)
    # Numerically stable softmax weights over the batch.
    logits = advantages.detach() / beta
    weights = torch.softmax(logits - logits.max(), dim=0)  # (B,)
    return (weights * per_sample).sum() / weights.sum()


# --------------------------------------------------------------------------
# 3. Stage-2: per-agent IGM critics + mixer.
# --------------------------------------------------------------------------
class IndividualCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, o_i: torch.Tensor, a_i: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim), (B, act_dim) -> (B,)"""
        return self.net(torch.cat([o_i, a_i], dim=-1)).squeeze(-1)


def avg_mixer(q_values: torch.Tensor) -> torch.Tensor:
    """VDN-style average mixer: (B, n_agents) -> (B)."""
    return q_values.mean(dim=-1)


# Deviation option (kept, default off): a QMIX-style monotonic hypernetwork
# mixer as a drop-in replacement for avg_mixer under the IGM principle.
class MonotonicMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden)
        self.hyper_w2 = nn.Linear(state_dim, hidden)
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.hidden = hidden

    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        B = q_values.size(0)
        w1 = torch.abs(self.hyper_w1(global_state)).view(B, self.n_agents, self.hidden)
        b1 = self.hyper_b1(global_state).view(B, 1, self.hidden)
        h = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # (B,1,hidden)
        w2 = torch.abs(self.hyper_w2(global_state)).view(B, self.hidden, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        q_tot = torch.bmm(h, w2) + b2
        return q_tot.view(B)


# --------------------------------------------------------------------------
# 4. Stage-3: decentralized one-step policy (executes at deployment time).
# --------------------------------------------------------------------------
class OneStepPolicy(nn.Module):
    def __init__(self, obs_dim: int, noise_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.noise_dim = noise_dim
        self.act_dim = act_dim
        # Tanh output -> actions in normalized [-1, 1] range.
        self.net = nn.Sequential(
            nn.Linear(obs_dim + noise_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, o_i: torch.Tensor, z_i: torch.Tensor | None = None) -> torch.Tensor:
        """(B, obs_dim) -> (B, act_dim), normalized to [-1, 1]."""
        if z_i is None:
            z_i = torch.randn(o_i.size(0), self.noise_dim, device=o_i.device)
        return self.net(torch.cat([o_i, z_i], dim=-1))


def split_per_agent(joint_action: torch.Tensor, n_agents: int, act_dim: int) -> list[torch.Tensor]:
    """Split a (B, n_agents*act_dim) joint action into per-agent (B, act_dim) slices."""
    return list(joint_action.view(joint_action.size(0), n_agents, act_dim).unbind(dim=1))


def distillation_and_qmax_loss(
    one_step_policies,
    critics,
    mixer,
    obs_list,
    joint_flow_target_slices,
    alpha: float = 3.0,
    global_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq. 9: -Q_tot(one-step actions) + alpha * BC-distill to the joint flow.

    Args:
        one_step_policies: iterable of OneStepPolicy (one per agent).
        critics: iterable of IndividualCritic (one per agent).
        mixer: avg_mixer or a MonotonicMixer instance.
        obs_list: list of per-agent (B, obs_dim) observations.
        joint_flow_target_slices: list of per-agent (B, act_dim) flow targets.
        global_state: (B, state_dim) for a mixer that needs it (MonotonicMixer),
            else None (avg_mixer).
    """
    a_pred = [p(o) for p, o in zip(one_step_policies, obs_list)]
    q_list = torch.stack(
        [c(o, a) for c, o, a in zip(critics, obs_list, a_pred)], dim=-1
    )  # (B, n_agents)
    q_tot = mixer(q_list, global_state) if global_state is not None else mixer(q_list)
    distill = sum(
        F.mse_loss(a_pred[i], joint_flow_target_slices[i])
        for i in range(len(a_pred))
    )
    return -q_tot.mean() + alpha * distill
