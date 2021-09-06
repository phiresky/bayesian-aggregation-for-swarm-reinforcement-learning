import numpy as np

from kb_learning.envs import ObjectEnv
from kb_learning.tools import rot_matrix, compute_robust_mean_swarm_position

from gym import spaces


class ObjectAbsoluteEnv(ObjectEnv):
    _observe_objects = True

    def __init__(self,
                 num_kilobots=None,
                 object_shape='quad',
                 object_width=.15,
                 object_height=.15,
                 object_init=None,
                 light_type='circular',
                 light_radius=.2,
                 done_after_steps=350):

        super(ObjectAbsoluteEnv, self).__init__(num_kilobots=num_kilobots,
                                                object_shape=object_shape,
                                                object_width=object_width,
                                                object_height=object_height,
                                                object_init=object_init,
                                                light_type=light_type,
                                                light_radius=light_radius)

        self._desired_pose = None

        self._done_after_steps = done_after_steps

    @property
    def state_space(self):
        _state_space_low = self.kilobots_space.low
        _state_space_high = self.kilobots_space.high
        if self.light_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.light_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.light_state_space.high))
        if self.object_state_space:
            _state_space_low = np.concatenate((_state_space_low, self.object_state_space.low))
            _state_space_high = np.concatenate((_state_space_high, self.object_state_space.high))

        return spaces.Box(low=_state_space_low, high=_state_space_high, dtype=np.float32)

    @property
    def observation_space(self):
        _observation_spaces_low = self.kilobots_space.low
        _observation_spaces_high = self.kilobots_space.high
        if self.light_observation_space:
            _observation_spaces_low = np.concatenate((_observation_spaces_low, self.light_observation_space.low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, self.light_observation_space.high))
        if self.object_observation_space:
            # the objects are observed as x, y, sin(theta), cos(theta)
            objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -1., -1.] * len(self._objects))
            objects_high = np.array([self.world_x_range[1], self.world_y_range[1], 1., 1.] * len(self._objects))
            _observation_spaces_low = np.concatenate((_observation_spaces_low, objects_low))
            _observation_spaces_high = np.concatenate((_observation_spaces_high, objects_high))
            # # for the desired pose
            # _observation_spaces_low = np.concatenate((_observation_spaces_low, self._object_observation_space.low))
            # _observation_spaces_high = np.concatenate((_observation_spaces_high, self._object_observation_space.high))

        return spaces.Box(low=_observation_spaces_low, high=_observation_spaces_high,
                                            dtype=np.float32)

    def get_desired_pose(self):
        return self._desired_pose

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + tuple(o.get_pose() for o in self._objects)
                              + (self._light.get_state(),))

    def get_info(self, state, action):
        return {'desired_pose': self._desired_pose}

    def get_observation(self):
        if self._light_type in ['circular', 'dual']:
            _light_position = (self._light.get_state(),)
        else:
            _light_position = tuple()

        _object_orientation = self._objects[0].get_orientation()
        _object_sin_cos = ((np.sin(_object_orientation), np.cos(_object_orientation)),)

        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              # + (self._objects[0].get_pose(),)
                              + (self._objects[0].get_position(),)
                              + _object_sin_cos
                              + _light_position
                              # + (self._object_desired,)
                              )

    def get_reward(self, old_state, action, new_state):
        # obj pose in frame of desired pose
        old_obj_pose = old_state[-5:-2] - self._desired_pose
        new_obj_pose = new_state[-5:-2] - self._desired_pose
        # swarm_pos = old_state[:-5].reshape(-1, 2)
        #
        # reward_swarm = -np.sum(np.linalg.norm(swarm_pos - old_obj_pose[:2], axis=1)) / swarm_pos.shape[0]
        #
        # reward_obj = -np.linalg.norm(old_obj_pose[:2]) / 2 - np.abs(np.sin(old_obj_pose[2] / 2)) / 2
        #
        # reward = reward_swarm + np.exp(reward_swarm) * reward_obj

        # THIS WAS WORKING
        reward = .0
        # compute polar coordinates of object positions
        r_old = np.linalg.norm(old_obj_pose[:2])
        r_new = np.linalg.norm(new_obj_pose[:2])
        reward += 10 * np.exp(-(new_obj_pose[:2] ** 2).sum() / 2) * (r_old - r_new)

        # compute differences between absolute orientations
        reward += 1 * np.exp(-(new_obj_pose[:2] ** 2).sum() / .05) * (np.abs(old_obj_pose[2]) - np.abs(new_obj_pose[2]))

        return reward

    def has_finished(self, state, action):
        # has finished if object reached goal pose with certain ε
        obj_pose = state[-5:-2]
        dist_obj_pose = self._desired_pose - obj_pose
        dist_obj_pose[2] = np.abs(np.sin(dist_obj_pose[2] / 2))

        l2_norm = dist_obj_pose.dot(dist_obj_pose)
        # print('sq_error_norm: {}'.format(sq_error_norm))

        if l2_norm < .005:
            return True

        if self._sim_steps >= self._done_after_steps * self._steps_per_action:
            # print('maximum number of sim steps.')
            return True

        return False

    def _get_init_object_pose(self):
        # sample initial position as polar coordinates
        # get the min of width, height
        min_extend = max(self.world_size)
        # sample the radius between [min_ext/6, 2min_ext/6]
        radius = np.random.rand() * min_extend / 6 + min_extend / 6
        # sample the angle uniformly from [-π, +π]
        angle = np.random.rand() * np.pi * 2 - np.pi
        _object_init_position = np.array([np.cos(angle), np.sin(angle)]) * radius

        # sample the initial orientation uniformly from [-π, +π]
        _object_init_orientation = np.random.rand() * np.pi * 2 - np.pi
        self._object_init = np.concatenate((_object_init_position, [_object_init_orientation]))
        
        return self._object_init

    def _get_desired_object_pose(self):
        # # sample the desired position uniformly between [-w/2+ow, w/2-ow] and [-h/2+oh, h/2-oh] (w = width, h = height)
        # _object_desired_position = np.random.rand(2) * self.world_size + np.array(self.world_bounds[0])
        # _object_size = np.array([self._object_width, self._object_height])
        # _object_desired_position = np.maximum(_object_desired_position, self.world_bounds[0] + _object_size)
        # _object_desired_position = np.minimum(_object_desired_position, self.world_bounds[1] - _object_size)
        # # sample the desired orientation uniformly from [-π, +π]
        # _object_desired_orientation = np.random.rand() * 2 * np.pi - np.pi
        # self._object_desired = np.concatenate((_object_desired_position, [_object_desired_orientation]))

        return np.zeros(3)

    def _configure_environment(self):
        self._desired_pose = self._get_desired_object_pose()

        super(ObjectAbsoluteEnv, self)._configure_environment()

    def _draw_on_table(self, screen):
        # draw the desired pose as grey square
        vertices = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
        vertices *= np.array([[self._object_width, self._object_height]]) / 2.

        # rotate vertices
        vertices = rot_matrix(self._desired_pose[2]).dot(vertices.T).T

        # translate vertices
        vertices += self._desired_pose[None, :2]

        screen.draw_polygon(vertices=vertices, color=(200, 200, 200), filled=True, width=.005)
        screen.draw_polygon(vertices=vertices[0:3], color=(220, 200, 200), width=.005)
