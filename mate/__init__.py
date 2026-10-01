"""MATE: The Multi-Agent Tracking Environment."""


from mate import agents, constants, environment, utils, wrappers
from mate.algorithms.coma import COMAConfig, COMATrainer, run_coma
from mate.algorithms.dcg import DCGConfig, DCGNetwork, DCGTrainer, run_dcg
from mate.algorithms.i2c import I2CConfig, I2CCritic, I2CPolicy, I2CTrainer, run_i2c
from mate.algorithms.mat import MATConfig, MATPolicy, MATTrainer, run_mat
from mate.algorithms.rode import MateRodeEnv, make_mate_rode_env
from mate.algorithms.mamba import MAMBAConfig, MAMBATrainer, run_mamba
from mate.algorithms.aorpo import AORPOConfig, AORPOTrainer, run_aorpo
from mate.algorithms.mambpo import MAMBPOConfig, MAMBPOTrainer, run_mambpo
from mate.agents import (
    GreedyCameraAgent,
    GreedyTargetAgent,
    HeuristicCameraAgent,
    HeuristicTargetAgent,
    RandomCameraAgent,
    RandomTargetAgent,
)
from mate.environment import ASSETS_DIR, MultiAgentTracking
from mate.wrappers import (
    MultiCamera,
    MultiTarget,
    SingleCamera,
    SingleTarget,
    WrapperSpec,
    group_reset,
    group_step,
    group_act,
    group_communicate,
    group_observe
)

__all__ = [
    "make_environment",
    "MultiCamera",
    "MultiTarget",
    "SingleCamera",
    "SingleTarget",
    "GreedyCameraAgent",
    "GreedyTargetAgent",
    "RandomCameraAgent",
    "RandomTargetAgent",
    "HeuristicCameraAgent",
    "HeuristicTargetAgent",
    "COMAConfig",
    "COMATrainer",
    "run_coma",
    "DCGConfig",
    "DCGNetwork",
    "DCGTrainer",
    "run_dcg",
    "I2CConfig",
    "I2CCritic",
    "I2CPolicy",
    "I2CTrainer",
    "run_i2c",
    "MATConfig",
    "MATPolicy",
    "MATTrainer",
    "run_mat",
    "MateRodeEnv",
    "make_mate_rode_env",
    "MAMBAConfig",
    "MAMBATrainer",
    "run_mamba",
    "AORPOConfig",
    "AORPOTrainer",
    "run_aorpo",
    "MAMBPOConfig",
    "MAMBPOTrainer",
    "run_mambpo",

    "group_reset",
    "group_step",
    "group_act",
    "group_communicate",
    "group_observe",
]
__all__.extend(constants.__all__)
__all__.extend(environment.__all__)
__all__.extend(wrappers.__all__)
__all__.extend(agents.__all__)
__all__.extend(utils.__all__)


def make_environment(config=None, wrappers=(), **kwargs):  # pylint: disable=redefined-outer-name
    """Helper function for creating a wrapped environment."""
    import gymnasium as gym  # Ensure gym is defined in this scope

    env = MultiAgentTracking(config, **kwargs)

    for wrapper in wrappers:
        assert (
            isinstance(wrapper, WrapperSpec)
            or callable(wrapper)
            or issubclass(wrapper, gym.Wrapper)
        ), (
            f"You should provide a wrapper class or an instance of `mate.WrapperSpec`. "
            f"Got wrapper = {wrapper!r}."
        )
        env = wrapper(env)

    return env


def auto_register(id, **kwargs):
    import gymnasium as gym  # Ensure gym is defined in this scope
    gym.register(
        id=id,
        entry_point=make_environment,
        disable_env_checker=True,
        order_enforce=False,
        **kwargs
    )


auto_register(
    id="MultiAgentTracking-v0",
)
auto_register(
    id="MATE-v0"
)

auto_register(
    id="MATE-4v2-9-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v2-9.yaml")},
)

auto_register(
    id="MATE-4v2-0-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v2-0.yaml")},
)

auto_register(
    id="MATE-4v4-9-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v4-9.yaml")},
)

auto_register(
    id="MATE-4v4-0-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v4-0.yaml")},
)

auto_register(
    id="MATE-4v8-9-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v8-9.yaml")},
)

auto_register(
    id="MATE-4v8-0-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-4v8-0.yaml")},
)

auto_register(
    id="MATE-8v8-9-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-8v8-9.yaml")},
)

auto_register(
    id="MATE-8v8-0-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-8v8-0.yaml")},
)

auto_register(
    id="MATE-Navigation-v0",
    kwargs={"config": (ASSETS_DIR / "MATE-Navigation.yaml")},
)
