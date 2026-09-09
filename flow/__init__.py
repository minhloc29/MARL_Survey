"""Flow-Matching CTDE algorithm package for MATE's cooperative camera team."""

from .nets import (
    CameraObsEncoder,
    DeterministicActor,
    EntitySetEncoder,
    IndividualCritic,
    MonotonicMixer,
    OneStepPolicy,
    VelocityField,
    advantage_weighted_flow_bc_loss,
    avg_mixer,
    distillation_and_qmax_loss,
    flow_bc_loss,
    sample_joint_flow,
    split_per_agent,
)
from .observers import MateObserver
from .replay import ReplayBuffer
from .vd import AgentQNetwork, QplexMixer, encode_joint

__all__ = [
    "CameraObsEncoder",
    "DeterministicActor",
    "EntitySetEncoder",
    "IndividualCritic",
    "MonotonicMixer",
    "OneStepPolicy",
    "VelocityField",
    "advantage_weighted_flow_bc_loss",
    "avg_mixer",
    "distillation_and_qmax_loss",
    "flow_bc_loss",
    "sample_joint_flow",
    "split_per_agent",
    "MateObserver",
    "ReplayBuffer",
    "AgentQNetwork",
    "QplexMixer",
    "encode_joint",
]
