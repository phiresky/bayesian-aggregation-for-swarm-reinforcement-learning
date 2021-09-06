from typing import List, Type

import gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices


class WonkyVecEnv(VecEnv):
    """
    a stable baselines VecEnv but for a multi-agent environment
    (from stable-baselines3 perspective each agent is in their own environment)
    """

    def __init__(self, nr_agents: int):
        self.is_actually_a_single_env = True
        self.num_envs = nr_agents

    def close(self):
        pass

    def env_method(self):
        raise NotImplementedError()

    def get_attr(self, arg, _idx):
        return getattr(self, arg)

    def set_attr(self):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray):
        self._todo_actions = actions

    def step_wait(self):
        actions, self._todo_actions = self._todo_actions, None
        return self.step(actions)

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        raise NotImplementedError()
