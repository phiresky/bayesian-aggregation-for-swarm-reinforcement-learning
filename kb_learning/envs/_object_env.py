import numpy as np

from gym_kilobots.envs import KilobotsEnv
from gym_kilobots.lib import SimplePhototaxisKilobot, CircularGradientLight, GradientLight, CompositeLight
from gym import spaces

from gym_kilobots.lib import Quad, CornerQuad, Circle, Triangle, LForm, TForm, CForm

from kb_learning.kernel import angle_from_swarm_mean


class ObjectEnv(KilobotsEnv):
    world_size = world_width, world_height = .8, .8
    screen_size = screen_width, screen_height = 500, 500

    _observe_objects = False
    _observe_light = True

    __steps_per_action = 20

    # _light_max_dist = .4
    # _spawn_angle_mean = np.pi
    # _spawn_angle_variance = .5 * np.pi

    def __init__(self,
                 num_kilobots=None,
                 object_shape='quad',
                 object_width=.15,
                 object_height=.15,
                 object_init=None,
                 # observe_object=False,
                 light_type='circular',
                 light_radius=.2):
        self._num_kilobots = num_kilobots

        if object_init:
            self._object_init = np.array(object_init)
        else:
            self._object_init = np.array([.0, .0, .0])
        self._object_width, self._object_height = object_width, object_height
        self._object_shape = object_shape
        # self._observe_object = observe_object

        self._light_type = light_type
        self._light_radius = light_radius

        self._spawn_type_ratio = 1.
        self._max_spawn_std = .3 * self._light_radius

        self.kilobots_space: spaces.Box = None

        self.light_state_space: spaces.Box = None
        self.light_observation_space: spaces.Box = None

        self.weight_state_space: spaces.Box = None
        self.weight_observation_space: spaces.Box = None

        self.object_state_space: spaces.Box = None
        self.object_observation_space: spaces.Box = None

        super().__init__()

    def _transform_position(self, position):
        return self._objects[0].get_local_point(position)

    def _translate_position(self, position):
        return np.array(position) - self._objects[0].get_position()

    def get_state(self):
        return np.concatenate(tuple(k.get_position() for k in self._kilobots)
                              + (self._light.get_state(),)
                              + tuple(o.get_pose() for o in self._objects))

    def get_info(self, state, action):
        return {'state': self.get_state()}

    def _configure_environment(self):
        self._init_object()

        self._init_light()

        self._init_kilobots()

        # step world once to resolve collisions
        self._step_world()

    def _get_init_object_pose(self):
        return self._object_init

    def _init_object(self):
        object_shape = self._object_shape.lower()

        # TODO limit object initial position to world bounds reflecting object size
        object_init_pose = self._get_init_object_pose()

        if object_shape in ['quad', 'rect']:
            obj = Quad(width=self._object_width, height=self._object_height,
                       position=object_init_pose[:2], orientation=object_init_pose[2],
                       world=self.world)
        elif object_shape in ['corner_quad', 'corner-quad']:
            obj = CornerQuad(width=self._object_width, height=self._object_height,
                       position=object_init_pose[:2], orientation=object_init_pose[2],
                       world=self.world)
        elif object_shape == 'triangle':
            obj = Triangle(width=self._object_width, height=self._object_height,
                           position=object_init_pose[:2], orientation=object_init_pose[2],
                           world=self.world)
        elif object_shape == 'circle':
            obj = Circle(radius=self._object_width, position=object_init_pose[:2],
                         orientation=object_init_pose[2], world=self.world)
        elif object_shape == 'l_shape':
            obj = LForm(width=self._object_width, height=self._object_height,
                        position=object_init_pose[:2], orientation=object_init_pose[2],
                        world=self.world)
        elif object_shape == 't_shape':
            obj = TForm(width=self._object_width, height=self._object_height,
                        position=object_init_pose[:2], orientation=object_init_pose[2],
                        world=self.world)
        elif object_shape == 'c_shape':
            obj = CForm(width=self._object_width, height=self._object_height,
                        position=object_init_pose[:2], orientation=object_init_pose[2],
                        world=self.world)
        else:
            raise UnknownObjectException('Shape of form {} not known.'.format(self._object_shape))

        self._objects.append(obj)

        objects_low = np.array([self.world_x_range[0], self.world_y_range[0], -np.inf] * len(self._objects))
        objects_high = np.array([self.world_x_range[1], self.world_y_range[1], np.inf] * len(self._objects))
        self.object_state_space = spaces.Box(low=objects_low, high=objects_high, dtype=np.float64)
        if self._observe_objects:
            self.object_observation_space = self.object_state_space

    def _init_light(self):
        # determine sampling mode for this episode
        self.__spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        if self._light_type == 'circular':
            self._init_circular_light()
        elif self._light_type == 'linear':
            self._init_linear_light()
            self._observe_light = False
        elif self._light_type == 'dual':
            self._init_dual_light()
        else:
            raise UnknownLightTypeException()

    def _sample_init_pos(self):
        spawn_randomly = np.random.rand() < np.abs(self._spawn_type_ratio)

        if spawn_randomly:
            # select light init position randomly as polar coordinates between .25π and 1.75π
            # light_init_direction = np.random.rand() * np.pi * 1.5 + np.pi * .25
            light_init_direction = np.random.rand() * 2 * np.pi - np.pi
            light_init_radius = np.abs(np.random.normal() * max(self.world_size) / 3)

            light_init = light_init_radius * np.array([np.cos(light_init_direction), np.sin(light_init_direction)])
        else:
            # otherwise, the light starts above the object
            light_init = self._object_init[:2]

        return light_init

    def _init_circular_light(self):
        light_init = self._sample_init_pos()

        light_init = np.maximum(light_init, self.world_bounds[0] * .95)
        light_init = np.minimum(light_init, self.world_bounds[1] * .95)

        light_bounds = np.array(self.world_bounds) * 1.2
        action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01

        self._light = CircularGradientLight(position=light_init, radius=self._light_radius,
                                            bounds=light_bounds, action_bounds=action_bounds)

        self.light_state_space = self._light.observation_space
        if self._observe_light:
            self.light_observation_space = self._light.observation_space

    def _init_linear_light(self):
        # sample initial angle from a uniform between -pi and pi
        init_angle = np.random.rand() * 2 * np.pi - np.pi
        # GradientLight.relative_actions = True
        self._light = GradientLight(angle=init_angle)

        self.light_state_space = self._light.observation_space

    def _init_dual_light(self):
        light1_init = self._sample_init_pos()
        light1_init = np.maximum(light1_init, self.world_bounds[0] * .95)
        light1_init = np.minimum(light1_init, self.world_bounds[1] * .95)

        light2_init = self._sample_init_pos()
        light2_init = np.maximum(light2_init, self.world_bounds[0] * .95)
        light2_init = np.minimum(light2_init, self.world_bounds[1] * .95)

        light_bounds = np.array(self.world_bounds) * 1.2
        action_bounds = np.array([-1, -1]) * .01, np.array([1, 1]) * .01
        # action_bounds = light_bounds

        light1 = CircularGradientLight(position=light1_init, radius=self._light_radius,
                                      bounds=light_bounds, action_bounds=action_bounds, relative_actions=True)
        light2 = CircularGradientLight(position=light2_init, radius=self._light_radius,
                                      bounds=light_bounds, action_bounds=action_bounds, relative_actions=True)

        self._light = CompositeLight(lights=(light1, light2))

        self.light_state_space = self._light.observation_space
        if self._observe_light:
            self.light_observation_space = self._light.observation_space

    def _init_kilobots(self):
        if self._light_type == 'circular':
            # select the mean position of the swarm to be the position of the light
            spawn_mean = self._light.get_state()
            spawn_std = np.random.rand() * self._max_spawn_std
            kilobot_positions = np.random.normal(loc=spawn_mean, scale=spawn_std, size=(self._num_kilobots, 2))

            # spawn_mean1 = self._light.get_state()
            # spawn_mean2 = self._sample_init_pos()
            # spawn_std = np.random.rand() * self._max_spawn_std
            #
            # kilobot_positions = np.random.normal(scale=spawn_std, size=(self._num_kilobots, 2))
            # idx = np.random.permutation(self._num_kilobots)
            # kilobot_positions[idx[:self._num_kilobots // 2]] += spawn_mean1
            # kilobot_positions[idx[self._num_kilobots // 2:]] += spawn_mean2

            # the orientation of the kilobots is chosen randomly
            orientations = 2 * np.pi * np.random.rand(self._num_kilobots)

        elif self._light_type == 'linear':
            spawn_mean = self._sample_init_pos()
            self._light.set_angle(np.arctan2(spawn_mean[1], spawn_mean[0]))
            spawn_mean = np.maximum(spawn_mean, self.world_bounds[0])
            spawn_mean = np.minimum(spawn_mean, self.world_bounds[1])

            # select a spawn std uniformly between zero and half the world width
            spawn_std = np.random.rand() * min(self.world_size) / 5

            # draw the kilobots positions from a normal with mean and variance selected above
            kilobot_positions = np.random.normal(scale=spawn_std, size=(self._num_kilobots, 2))
            kilobot_positions += spawn_mean

            # the orientation of the kilobots is chosen randomly
            orientations = 2 * np.pi * np.random.rand(self._num_kilobots)
        elif self._light_type == 'dual':
            spawn_mean1 = self._light.get_state()[:2]
            spawn_mean2 = self._light.get_state()[2:]
            spawn_std = np.random.rand() * self._max_spawn_std

            kilobot_positions = np.random.normal(scale=spawn_std, size=(self._num_kilobots, 2))
            idx = np.random.permutation(self._num_kilobots)
            kilobot_positions[idx[:self._num_kilobots // 2]] += spawn_mean1
            kilobot_positions[idx[self._num_kilobots // 2:]] += spawn_mean2

            orientations = 2 * np.pi * np.random.rand(self._num_kilobots)
        else:
            raise UnknownLightTypeException()

        # assert for each kilobot that it is within the world bounds and add kilobot to the world
        for position, orientation in zip(kilobot_positions, orientations):
            position = np.maximum(position, self.world_bounds[0] * .95)
            position = np.minimum(position, self.world_bounds[1] * .95)
            self._add_kilobot(SimplePhototaxisKilobot(self.world, position=position, orientation=orientation,
                                                      light=self._light))

        # construct state space, observation space and action space
        kb_low = np.array([self.world_x_range[0], self.world_y_range[0]] * len(self._kilobots))
        kb_high = np.array([self.world_x_range[1], self.world_y_range[1]] * len(self._kilobots))
        self.kilobots_space = spaces.Box(low=kb_low, high=kb_high, dtype=np.float64)


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass
