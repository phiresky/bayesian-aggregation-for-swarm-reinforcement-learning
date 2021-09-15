from ._object_env import ObjectEnv
from ._object_relative_env import ObjectRelativeEnv
from ._object_absolute_env import ObjectAbsoluteEnv
from ._multi_object_env import MultiObjectDirectControlEnv
from ._pose_control_env import PoseControlEnv
from ._eval_env import EvalEnv
from ._env_wrapper import NormalizeActionWrapper

from .sampler import SARSSampler, ParallelSARSSampler

from typing import Union


def _check_parameters(*, num_kilobots, object_width, object_height, light_type, **kwargs):
    if 'weight' in kwargs:
        assert type(kwargs['weight']) is float, "`weight` has to be of type float"
        assert .0 <= kwargs['weight'] <= 1., "`weight` has to be in the interval [0.0, 1.0]"

    assert type(num_kilobots) is int, "`num_kilobots` has to be of type int"
    assert 0 < num_kilobots, "`num_kilobots` has to be a positive integer."

    assert .0 < object_width, "`object_width` has to be a positive float"

    assert .0 < object_height, "`object_height` has to be a positive float"

    assert light_type in ['circular', 'linear', 'dual'], "`light_type` has to be in ['circular', 'linear', 'dual']"
    if light_type == 'circular':
        assert 'light_radius' in kwargs, '`light_radius` is required parameter if `light_type` is `circular`'
        assert kwargs['light_radius'] > .0, "`light_radius` has to be positive."


def register_object_relative_env(weight: float, num_kilobots: int, object_shape: str, object_width: float,
                                 object_height: float, light_type: str, light_radius:
                                                                                                      float = \
    None):
    from gym.envs.registration import register, registry

    _check_parameters(weight=weight, num_kilobots=num_kilobots, object_shape=object_shape, object_width=object_width,
                      object_height=object_height, light_type=light_type, light_radius=light_radius)

    weight_str = '{:03}'.format(int(weight * 100)) if weight is not None else 'RND'
    light_radius_str = '{:03}'.format(int(light_radius * 100)) if light_radius else ''
    _id = 'ObjectRelativeEnv_w{}_kb{}_{}_{:03}x{:03}_{}{}-v0'.format(weight_str, num_kilobots, object_shape,
                                                                     int(object_width * 100), int(object_height * 100),
                                                                     light_type, light_radius_str)

    if _id in registry.env_specs:
        return _id

    register(id=_id, entry_point='kb_learning.envs:ObjectRelativeEnv',
             kwargs=dict(object_shape=object_shape, object_width=object_width, object_height=object_height,
                         num_kilobots=num_kilobots, weight=weight, light_type=light_type, light_radius=light_radius))

    return _id


def register_object_absolute_env(num_kilobots: int, object_shape: str, object_width: float,
                                 object_height: float, light_type: str, light_radius: float = None):
    from gym.envs.registration import register, registry

    _check_parameters(num_kilobots=num_kilobots, object_shape=object_shape, object_width=object_width,
                      object_height=object_height, light_type=light_type, light_radius=light_radius)

    light_radius_str = '{:03}'.format(int(light_radius * 100)) if light_radius else ''
    _id = 'ObjectAbsoluteEnv_kb{}_{}_{:03}x{:03}_{}{}-v0'.format(num_kilobots, object_shape,
                                                                 int(object_width * 100), int(object_height * 100),
                                                                 light_type, light_radius_str)

    if _id in registry.env_specs:
        return _id

    register(id=_id, entry_point='kb_learning.envs:ObjectAbsoluteEnv',
             kwargs=dict(object_shape=object_shape, object_width=object_width, object_height=object_height,
                         num_kilobots=num_kilobots, light_type=light_type, light_radius=light_radius))

    return _id


def register_object_env(*, entry_point: str, num_kilobots, object_shape, object_width, object_height, light_type,
                        **kwargs):
    from gym.envs.registration import register, registry

    _check_parameters(num_kilobots=num_kilobots, object_width=object_width, object_height=object_height,
                      light_type=light_type, **kwargs)

    class_name = entry_point.split(':')[-1]
    _id = class_name

    if class_name == 'ObjectRelativeEnv':
        assert 'weight' in kwargs, "'weight' is a required parameter for 'ObjectRelativeEnv'"
        weight_str = '{:03}'.format(int(kwargs['weight'] * 100)) if kwargs['weight'] is not None else 'RND'
        _id += '_w{}'.format(weight_str)

    _id += '_kb{}_{}_{:03}x{:03}_{}'.format(num_kilobots, object_shape, int(object_width * 100),
                                           int(object_height * 100), light_type)

    if light_type == 'circular':
        light_radius_str = '{:03}'.format(int(kwargs['light_radius'] * 100)) if 'light_radius' in kwargs else ''
        _id += light_radius_str

    _id += '-v0'

    if _id in registry.env_specs:
        return _id

    kwargs.update(dict(num_kilobots=num_kilobots, object_shape=object_shape, object_width=object_width,
                       object_height=object_height, light_type=light_type))

    register(id=_id, entry_point=entry_point, kwargs=kwargs)

    return _id
