import numpy as np

from ._object_absolute_env import ObjectAbsoluteEnv


class PoseControlEnv(ObjectAbsoluteEnv):
    world_size = world_width, world_height = 1.6, 1.2
    screen_size = screen_width, screen_height = 900, 600

    def get_observation(self):
        if self._light_type == 'circular':
            _light_position = (self._light.get_state(),)
        else:
            _light_position = tuple()

        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + _light_position
                              + (self._objects[0].get_pose(),))

    def _sample_init_pos(self):
        _init_position = np.random.rand(2) * self.world_size * [.5, 1.] + [self.world_x_range[0], self.world_y_range[0]]
        _init_position = np.maximum(_init_position, self.world_bounds[0] + self._light_radius)
        _init_position = np.minimum(_init_position, self.world_bounds[1] - self._light_radius)

        return _init_position

    def _get_init_object_pose(self):
        _object_init_orientation = np.random.rand() * np.pi / 2 - np.pi / 4
        self._object_init = np.array([.0, .0, _object_init_orientation])

        return self._object_init

    def _get_desired_object_pose(self):
        # sample the desired position uniformly between [-w/2+ow, w/2-ow] and [-h/2+oh, h/2-oh] (w = width, h = height)
        _object_desired_position = np.random.rand(2) * self.world_size * [.3, 1.] + [.2 * self.world_size[0],
                                                                                     self.world_y_range[0]]
        _object_size = np.array([self._object_width, self._object_height])
        _object_desired_position = np.maximum(_object_desired_position, self.world_bounds[0] + _object_size)
        _object_desired_position = np.minimum(_object_desired_position, self.world_bounds[1] - _object_size)
        # sample the desired orientation uniformly from [-π, +π]
        _object_desired_orientation = np.random.rand() * np.pi / 2 - np.pi / 4

        return np.concatenate((_object_desired_position, [_object_desired_orientation]))
