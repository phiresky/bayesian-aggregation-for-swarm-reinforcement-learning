from typing import Any, List, Literal, NamedTuple, Optional, Type

import gym
import numpy as np
from gym import spaces
from gym_kilobots.envs.yaml_kilobots_env import EnvConfiguration
from kb_learning.envs import NormalizeActionWrapper
from kb_learning.envs._multi_object_env import MODCEConfig
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
)

from playground.FlattenVecWrapper import FlattenVecWrapper
from playground.ma_envs.point_envs.wonkyvec import WonkyVecEnv

from .util import DictSpace, TupleDictSpace


class KilobotsEnvParams(NamedTuple):
    env: EnvConfiguration
    reward_function: str
    swarm_reward: bool
    agent_reward: bool
    agent_type: str
    done_after_steps: int
    m_config: MODCEConfig = MODCEConfig()
    aggregate_clusters_separately: bool = False
    type: Literal["KilobotsNew"] = "KilobotsNew"

    @property
    def nr_agents(self):
        return self.env.kilobots.num

    def create_env(self):
        from kb_learning.envs import MultiObjectDirectControlEnv

        m_config = self.m_config
        if self.aggregate_clusters_separately:
            m_config = self.m_config._replace(disable_observe_object_type=True)

        env = MultiObjectDirectControlEnv(
            configuration=self.env,
            additional_config=m_config,
            reward_function=self.reward_function,
            swarm_reward=self.swarm_reward,
            agent_reward=self.agent_reward,
            agent_type=self.agent_type,
            done_after_steps=self.done_after_steps,
        )

        wrapped_env = FlattenVecWrapper(
            KilobotsEnvWrapper(
                NormalizeActionWrapper(env),
                aggregate_clusters_separately=self.aggregate_clusters_separately,
            )
        )
        return wrapped_env


class KilobotsEnvParamsLegacy(NamedTuple):
    yaml_config: str
    nr_agents: int
    type: Literal["Kilobots"] = "Kilobots"

    def create_env(self):
        import yaml
        from kb_learning.envs import MultiObjectDirectControlEnv, NormalizeActionWrapper

        config = yaml.load(self.yaml_config, Loader=yaml.FullLoader)
        default_params = {
            "sampling": {
                "timesteps_per_batch": 2048,
                "done_after_steps": 1024,
                "num_objects": None,
                "reward_function": None,
                "agent_type": "SimpleAccelerationControlKilobot",
                "swarm_reward": True,
                "agent_reward": True,
                "schedule": "linear",
            }
        }
        config["env_config"].kilobots.num = self.nr_agents
        params = {
            **config["params"],
            "sampling": {**default_params["sampling"], **config["params"]["sampling"]},
        }
        if params["sampling"]["num_objects"]:
            if isinstance(params["sampling"]["num_objects"], int):
                from itertools import cycle, islice

                config["env_config"].objects = list(
                    islice(
                        cycle(config["env_config"].objects),
                        params["sampling"]["num_objects"],
                    )
                )

        # create env
        env = MultiObjectDirectControlEnv(
            configuration=config["env_config"],
            reward_function=params["sampling"]["reward_function"],
            swarm_reward=params["sampling"]["swarm_reward"],
            agent_reward=params["sampling"]["agent_reward"],
            agent_type=params["sampling"]["agent_type"],
            done_after_steps=params["sampling"]["done_after_steps"],
        )

        assert (
            len(env.kilobots) == self.nr_agents
        ), f"{self.nr_agents=} is not {len(env.kilobots)=}"
        wrapped_env = KilobotsEnvWrapper(
            NormalizeActionWrapper(env),
            aggregate_clusters_separately=False,
        )
        return wrapped_env


class KilobotsEnvWrapper(WonkyVecEnv):
    def __init__(
        self, env: NormalizeActionWrapper, aggregate_clusters_separately: bool
    ):
        super().__init__(nr_agents=env.num_kilobots)
        self.env = env

        two_floats = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        single_float = spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        box_sp = {
            "distance": single_float,
            "angle": two_floats,
            "orientation": two_floats,
            "abs_pos": two_floats,
            "type": spaces.Box(  # one hot of cluster
                low=0, high=1, shape=(self.env.env._num_cluster,), dtype=np.float32
            ),
            "valid": single_float,
        }
        if not env.do_observe_object_type:
            del box_sp["type"]
        if not env.do_observe_abs_box_pos:
            del box_sp["abs_pos"]
        aggregatables = {
            "neighbors": TupleDictSpace(
                DictSpace(
                    {
                        "distance": single_float,
                        "angle": two_floats,
                        "orientation": two_floats,
                    }
                ),
                env.num_kilobots - 1,
            ),
        }
        if aggregate_clusters_separately:
            for c in range(env.env._num_cluster):
                aggregatables[f"boxes_cluster{c}"] = TupleDictSpace(
                    DictSpace(box_sp),
                    env.obj_per_cluster,
                )
        else:
            aggregatables[f"boxes"] = TupleDictSpace(
                DictSpace(box_sp),
                len(env.conf.objects),
            )
        one_space = DictSpace(
            {
                "aggregatables": DictSpace(aggregatables),
                "local": DictSpace(
                    {
                        "pos": two_floats,
                        "angle": two_floats,
                        "total_object_count": single_float,
                    }
                ),
            }
        )
        self.unflattened_observation_space = one_space

        total_obs_scalars = spaces.flatdim(one_space)
        assert (
            len(self.env.kilobots),
            total_obs_scalars,
        ) == self.env.observation_space.shape, (
            len(self.env.kilobots),
            total_obs_scalars,
            self.env.observation_space.shape,
        )
        self.observation_space = spaces.Box(
            self.env.observation_space.low.min(),
            self.env.observation_space.high.max(),
            shape=self.env.observation_space.shape[-1:],
        )
        self.action_space = spaces.Box(
            self.env.action_space.low.min(),
            self.env.action_space.high.max(),
            shape=self.env.action_space.shape[-1:],
        )

    def reset(self):
        self.is_done = False
        return self.env.reset()

    def step(self, actions) -> VecEnvStepReturn:
        assert not self.is_done, "tried to step done env"
        return self.env.step(actions)

    def close(self):
        self.env.close()

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:

        raise NotImplementedError()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:

        raise NotImplementedError()

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError()

    def seed(self, seed: Optional[int] = None):
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in indices]

    # taken from stable baselnies dummy_vec_env
