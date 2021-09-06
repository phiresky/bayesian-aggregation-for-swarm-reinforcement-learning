from gym import ActionWrapper
from gym.spaces import Box

import numpy as np


class NormalizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.action_space, Box)
        # assert len(self.action_space.shape) == 1

        self.mean = (self.action_space.low + self.action_space.high) / 2.
        self.scale = self.action_space.high - self.mean

        # self.mean = np.zeros(self.action_space.shape)
        # scale = np.zeros(self.action_space.shape)
        #
        # for i in range(self.action_space.shape[0]):
        #     low_i = self.action_space.low[0]
        #     high_i = self.action_space.high[0]
        #
        #     self.mean[i] = (low_i + high_i) / 2.
        #     scale[i] = high_i - self.mean[i]

        self.action_space = Box(low=(self.action_space.low - self.mean) / self.scale,
                                high=(self.action_space.high - self.mean) / self.scale,
                                dtype=self.action_space.dtype)
        # self.scale_diag = np.diag(scale)
        # self.inv_scale_diag = np.diag(1/scale)

    def action(self, action):
        if action is None:
            return action
        return action * self.scale + self.mean

    def reverse_action(self, action):
        return (action - self.mean) / self.scale

    def render(self, mode=None):
        return self.env.render(mode)
