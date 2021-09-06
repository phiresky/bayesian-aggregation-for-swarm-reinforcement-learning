import gym.spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from .util import structured_to_unstructured


class FlattenVecWrapper(VecEnvWrapper):
    """flattens observations since stable-baselines3 can only handle box obs"""

    def __init__(self, env: VecEnv):
        super().__init__(env)
        if getattr(env, "is_actually_a_single_env", False):
            self.is_actually_a_single_env = True
        self.observation_space = self.venv.observation_space
        # self.observation_space = gym.spaces.flatten_space(self.venv.observation_space)
        self.unflattened_observation_space = getattr(
            self.venv, "unflattened_observation_space", self.venv.observation_space
        )

    def reset(self, **kwargs):
        observation = self.venv.reset(**kwargs)
        return self.observation(observation)

    def step_wait(self):
        observation, reward, done, info = self.venv.step_wait()
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        if observation.dtype.names is None:
            return observation
        else:
            return structured_to_unstructured(observation)
        return [
            gym.spaces.flatten(self.unflattened_observation_space, o)
            for o in observation
        ]
