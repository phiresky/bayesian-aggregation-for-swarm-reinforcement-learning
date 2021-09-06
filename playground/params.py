from typing import *

from .KilobotsEnvWrapper import KilobotsEnvParams, KilobotsEnvParamsLegacy
from .ma_envs.point_envs.pursuit import PursuitEvasionMultiEnvParams
from .ma_envs.point_envs.rendezvous import RendezvousEnvParams
from .PettingZooEnvWrapper import PettingZooEnvWrapper
from .policy.embed import EmbeddingParams
from .policy.MlpAggregatingPolicy import PolicyParams
from .sumtype import SumType


def get_pettingzoo_thong(mod, name, args, kwargs):
    import importlib

    module = importlib.import_module(f"pettingzoo.{mod}.{name}")
    return module.parallel_env(*args, **kwargs)


class PettingZooEnvParams(NamedTuple):
    env_name: Tuple[str, str]
    nr_agents: int
    env_kwargs: Dict[str, Union[str, int, float, bool, None]]
    disable_agent_collisions: bool

    type: Literal["PettingZoo"] = "PettingZoo"

    def create_env(self):
        mod, name = self.env_name
        env = PettingZooEnvWrapper(get_pettingzoo_thong(mod, name, [], self.env_kwargs))
        if self.disable_agent_collisions:
            for agent in env.env.aec_env.env.env.world.agents:
                agent.collide = False
        return env


class EnvParams(SumType):
    Rendezvous = RendezvousEnvParams

    PursuitEvasion = PursuitEvasionMultiEnvParams

    PettingZoo = PettingZooEnvParams

    Kilobots = KilobotsEnvParamsLegacy

    KilobotsNew = KilobotsEnvParams

    Type = Union[Rendezvous, PursuitEvasion, PettingZooEnvParams, Kilobots, KilobotsNew]


class FullParams(NamedTuple):
    emb_params: EmbeddingParams
    env_params: EnvParams.Type
    policy_params: PolicyParams
    runname: str
    nr_parallel_envs: int
    env_steps_per_train_step: int
    train_steps: int
    batch_size: int = -1
    training_algorithm: Literal["ppo", "trl", "sac"] = "ppo"
    # only relevant for TRL
    training_algorithm_params: dict[str, Any] = {}
