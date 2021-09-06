from typing import Any, List, Optional

import numpy as np
from gym import spaces
from pettingzoo.utils.env import ParallelEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
)

from .EvalCallback import doneify
from .util import DictSpace, TupleDictSpace


class PettingZooEnvWrapper(VecEnv):
    def __init__(self, env: ParallelEnv):
        self.is_actually_a_single_env = True
        self.env = env
        self.innerest_env = env.aec_env.env.env
        fa = self.env.possible_agents[0]
        one_space = self.env.observation_spaces[fa]

        single_float = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        env_kind = self.innerest_env.metadata["name"]
        two_floats = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        if env_kind == "simple_spread_v2":
            agents_landmarks_count = self.innerest_env.num_agents
            neighbors_count = agents_landmarks_count - 1
            self.unflattened_observation_space = DictSpace(
                {
                    "local": DictSpace({"vel": two_floats, "pos": two_floats}),
                    "aggregatables": DictSpace(
                        {
                            "landmarks": TupleDictSpace(
                                DictSpace({"rel_pos": two_floats}),
                                agents_landmarks_count,
                            ),
                            "neighbors": TupleDictSpace(
                                DictSpace(
                                    {
                                        "rel_pos": two_floats,
                                    }
                                ),
                                neighbors_count,
                            ),
                        }
                    ),
                    "unused_comm": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(neighbors_count * self.innerest_env.world.dim_c,),
                    ),
                }
            )
        elif env_kind == "simple_v2":
            self.unflattened_observation_space = DictSpace(
                {
                    "local": DictSpace({"vel": two_floats, "pos": two_floats}),
                }
            )
        else:
            raise Exception(f"unknown env {env_kind}")
        super().__init__(
            num_envs=self.env.max_num_agents,
            observation_space=one_space,
            action_space=self.env.action_spaces[fa],
        )

    def reset(self):
        obs = self.env.reset()
        self.is_done = False

        return self.transform_observations(obs)

    def step_async(self, actions: np.ndarray) -> None:
        assert self.env.agents == self.env.possible_agents, "agents cannot change"
        assert not self.is_done, "tried to step done env"
        self.todo_actions = {
            agent: action for agent, action in zip(self.env.agents, actions)
        }

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.env.step(self.todo_actions)
        assert list(obs.keys()) == self.env.possible_agents
        assert list(reward.keys()) == self.env.possible_agents
        assert list(done.keys()) == self.env.possible_agents

        out_obs = self.transform_observations(obs)
        out_rews = np.array(
            list(reward.values()), dtype="float32"
        )  # must be a np array for EvalCallback to be able to do 0 + rewards
        out_dones = list(done.values())
        out_info = {
            "agent_infos": list(info.values())
        }  # must be a dict because `info["terminal_observation"] = observation` in subprocvecenv

        out_done = doneify(out_dones)
        if out_done:
            self.is_done = True

        return (out_obs, out_rews, out_done, out_info)

    def transform_observations(self, observations):
        """ "aggregatables": {
            "neighbors": [{"data": o2, "valid": 1} for o2 in observations]
        }"""

        return np.asarray(
            list(observations.values())
        )  # [{"local": o} for o in observations]

    def close(self):
        self.env.close()

    def render(self, *a, **kw):
        return self.env.render(*a, **kw)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError()

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

    def seed(self, seed: Optional[int] = None):

        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = self._get_indices(indices)
        return [getattr(self, attr_name) for _ in indices]

    # taken from stable baselnies dummy_vec_env
