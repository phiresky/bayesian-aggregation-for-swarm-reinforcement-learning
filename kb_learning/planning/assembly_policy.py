import numpy as np
import yaml
import math

from typing import List, Union, Dict
from kb_learning.envs import EnvConfiguration


class PolicyContainer:
    def __init__(self):
        self._policies: Dict[Union[str, None], callable] = dict()
        self._policies[None] = {0.0: None}

    def add_policy(self, policy, w_factor, object_type=None):
        if object_type not in self._policies:
            self._policies[object_type] = dict()
        self._policies[object_type][w_factor] = policy

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._policies.get(item, self._policies[None])
        if isinstance(item, tuple):
            object_type, w_factor = item
            obj_policy = self._policies.get(object_type, self._policies[None])
        else:
            w_factor = item
            obj_policy = self._policies[None]

        w_smaller = .0
        w_bigger = 1.
        for k in sorted(obj_policy.keys()):
            if w_factor == k:
                return obj_policy[k]
            if k > w_factor:
                w_bigger = k
                break
            w_smaller = k

        if (w_factor - w_smaller) < (w_bigger - w_factor):
            # print('selecting w = {}'.format(w_smaller))
            return obj_policy[w_smaller]
        else:
            # print('selecting w = {}'.format(w_bigger))
            return obj_policy[w_bigger]


class WayPointConf(yaml.YAMLObject):
    yaml_tag = '!AssemblyWayPoint'

    def __init__(self, obj_conf: EnvConfiguration.ObjectConfiguration, x, y, theta, position_accuracy=0.05,
                 orientation_accuracy=0.05):
        self.obj_conf = obj_conf
        self.x = x
        self.y = y
        self.theta = theta
        self.position_accuracy = position_accuracy
        self.orientation_accuracy = orientation_accuracy

    @property
    def object_type(self):
        _type = self.obj_conf.shape
        if _type in ['corner_quad', 'corner-quad', 'quad']:
            _type = 'square'
        return _type

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def pose(self):
        return np.array([self.x, self.y, self.theta])

    @property
    def accuracy(self):
        return np.r_[self.position_accuracy, self.orientation_accuracy]

    @classmethod
    def to_yaml(cls, dumper: yaml.Dumper, data):
        data = dict(obj_conf=data.obj_conf,  # should emit an anchor if eval_env configuration has been dumped before
                    x=float(data.x), y=float(data.y), theta=float(data.theta),
                    position_accuracy=float(data.position_accuracy),
                    orientation_accuracy=float(data.orientation_accuracy))
        return dumper.represent_mapping(cls.yaml_tag, data, flow_style=cls.yaml_flow_style)


class AssemblyPolicyConf(yaml.YAMLObject):
    yaml_tag = '!AssemblyPolicy'

    def __init__(self, way_points):
        if len(way_points) > 0 and type(way_points[0]) == WayPointConf:
            self.way_points = way_points
        else:
            self.way_points = [WayPointConf(**wp) for wp in way_points]


class WayPoint:
    def __init__(self, pose, tolerances):
        self._pose = np.asarray(pose)
        self._tolerances = np.asarray(tolerances)

    @property
    def pose(self):
        return self._pose

    @property
    def position(self):
        return self._pose[:2]

    @property
    def orientation(self):
        return self._pose[2]

    @property
    def tolerances(self):
        return self._tolerances

    @property
    def position_tolerance(self):
        return self._tolerances[:-1]

    @property
    def orientation_tolerance(self):
        return self._tolerances[-1]

    def __getitem__(self, item):
        return self._pose[item]

    def __sub__(self, other: Union[np.ndarray, 'WayPoint']):
        if isinstance(other, WayPoint):
            return self._pose - other.pose
        return self._pose[:len(other)].__sub__(other)

    def __rsub__(self, other: Union[np.ndarray, 'WayPoint']):
        if isinstance(other, WayPoint):
            return other._pose - self._pose
        return self._pose[:len(other)].__rsub__(other)

    def __add__(self, other: Union[np.ndarray, 'WayPoint']):
        if isinstance(other, WayPoint):
            return self._pose + other.pose
        return self._pose[:len(other)].__add__(other)

    def __radd__(self, other: Union[np.ndarray, 'WayPoint']):
        if isinstance(other, WayPoint):
            return self._pose + other.pose
        return self._pose[:len(other)].__add__(other)

    def __array__(self):
        return self._pose

    def __contains__(self, other: np.ndarray):
        assert other.ndim == 1
        assert other.size in [2, 3]

        if other.size == 3:
            error = self.pose - other
            error[2] = abs((error[2] + np.pi) % (2 * np.pi) - np.pi)
            if self.position_tolerance.size == 1:
                error = np.array([np.linalg.norm(error[:2]), error[2]])
            return np.all(error < self._tolerances)
        else:
            error = self.position - other
            if self.position_tolerance.size == 1:
                error = np.linalg.norm(error)
            return np.all(error < self.position_tolerance)

    def plot(self, axes, color='blue'):
        from matplotlib import patches

        if self.position_tolerance.size == 2:
            p = patches.Rectangle(self.position - self.position_tolerance,
                                  2 * self.position_tolerance[0], 2 * self.position_tolerance[1],
                                  ec=color, fill=False)
        else:
            p = patches.Circle(self.position, self.position_tolerance, ec=color, fill=False)
        axes.add_patch(p)
        if self.orientation_tolerance:
            w = patches.Wedge(self.position, np.min(self.position_tolerance),
                              math.degrees(self.orientation - self.orientation_tolerance),
                              math.degrees(self.orientation + self.orientation_tolerance), self.position_tolerance / 2,
                              color=color)
            axes.add_patch(w)
            return p, w
        return p,

    def update_plot(self, artist):
        if self.orientation_tolerance:
            p, w = artist
            w.center = self.position
            w.set_radius(np.min(self.position_tolerance))
            w.set_theta1(math.degrees(self.orientation - self.orientation_tolerance))
            w.set_theta2(math.degrees(self.orientation + self.orientation_tolerance))
            w.set_width(self.position_tolerance / 2)
        else:
            p = artist

        if self.position_tolerance.size == 2:
            p.set_xy(self.position - self.position_tolerance)
            p.set_width(2 * self.position_tolerance[0])
            p.set_height(2 * self.position_tolerance[1])
        else:
            p.center = self.position
            p.radius = self.position_tolerance


class Path:
    def __init__(self, path_entries: List[WayPoint] = None):
        if path_entries is None:
            self._way_points = []
        else:
            self._way_points = path_entries
        self._idx = 0

    def __iter__(self):
        return self._way_points.__iter__()

    def __len__(self):
        return self._way_points.__len__()

    def clear(self):
        self._way_points = []
        self._idx = 0

    @property
    def remaining(self):
        return self._way_points[self._idx:]

    @property
    def active_way_point(self) -> WayPoint:
        return self._way_points[self._idx]

    @property
    def upcoming_way_point(self) -> WayPoint:
        if self._idx + 1 < len(self._way_points):
            return self._way_points[self._idx + 1]

    # def update_target_position(self, pos, trans_threshold=.025):
    #     """update current trajectory point based on distance"""
    #     # if we are past the last point, clear trajectory and return True
    #     if self._idx == len(self._way_points):
    #         self._way_points = []
    #         self._idx = 0
    #         return True
    #
    #     # compute distance from position to current trajectory point
    #     trans_err = np.linalg.norm(self._way_points[self._idx][0:2] - pos)
    #     update = False
    #     # if we are close enough to the current trajectory point update to next trajectory point
    #     if trans_err < trans_threshold:
    #         self._idx += 1
    #         # check recursively
    #         return self.update_target_position(pos)
    #
    #     # check if object is closer to target than current position and update idx
    #     dist_obj_last = np.linalg.norm(self._way_points[-1][0:2] - pos)
    #     dist_current_last = np.linalg.norm(self._way_points[-1][0:2] - self._way_points[self._idx][0:2])
    #     if dist_obj_last < dist_current_last:
    #         self._idx += 1
    #         # check recursively
    #         return self.update_target_position(pos)
    #
    #     return False
    #
    # def update_target_position_with_orientation(self, pos, orientation, target_position_with_tolerances):
    #     """update current trajectory point based on translational and rotational distance"""
    #     # if we are past the last point, clear trajectory and return True
    #     if self._idx == len(self._way_points):
    #         self._way_points = []
    #         self._idx = 0
    #         return True
    #
    #     trans_err = np.linalg.norm(self._way_points[self._idx] - pos)
    #     rot_err = abs(orientation - target_position_with_tolerances[2] + np.pi) % (2 * np.pi) - np.pi
    #     update = False
    #
    #     if (trans_err < target_position_with_tolerances[3] / 2) & (
    #             abs(rot_err) < target_position_with_tolerances[4] / 2):
    #         self._idx += 1
    #         # check recursively
    #         return self.update_target_position_with_orientation(pos, orientation, target_position_with_tolerances)
    #
    #     # check if object is closer to target than current position and update idx
    #     # dist_obj_last = np.linalg.norm(self.trajectory_points[-1][0:2] - pos)
    #     # dist_current_last = np.linalg.norm(self.trajectory_points[-1][0:2] - self.trajectory_points[self.idx][0:2])
    #     # if dist_obj_last < dist_current_last:
    #     #     self.idx += 1
    #     #     # check recursively
    #     #     return self.update_target_position_with_orientation(pos, orientation, target_position_with_tolerances)
    #
    #     return False

    def update_target(self, pos):
        """Update active entry of the trajectory based on the passed pose and tolerances.

        :type pos: np.ndarray
        :param pos: 2- or 3-dimensional array with position or pose based on which the active entry of the path
        should be updated.

        :returns True if the path has been completed.
        """
        if self._idx == len(self._way_points):
            # self._way_points = []
            # self._idx = 0
            return True

        # if we are close enough to the current path entry update to next path entry
        if pos in self.active_way_point:
            self._idx += 1
            # check recursively
            return self.update_target(pos)

        # # check if object is closer to target than current position and update idx
        # dist_obj_last = np.linalg.norm(self._path_entries[-1][0:2] - pos)
        # dist_current_last = np.linalg.norm(self._path_entries[-1][0:2] - self._path_entries[self._idx][0:2])
        # if dist_obj_last < dist_current_last:
        #     self._idx += 1
        #     # check recursively
        #     return self.update_target_position(pos)

        return False

    def plot(self, axes):
        if self.finished():
            return
        active_wp_plot = self.active_way_point.plot(axes)
        return active_wp_plot,

    def update_plot(self, artists):
        if self.finished() and artists is not None:
            active_wp_plot = artists[0]
            active_wp_plot.remove()
        active_wp_plot = artists[0]
        self.active_way_point.update_plot(active_wp_plot)

    def num_way_points(self) -> int:
        return len(self._way_points)

    def is_empty(self) -> bool:
        return len(self._way_points) == 0

    def finished(self) -> bool:
        return self._idx == len(self._way_points)


class AssemblyWayPoint(WayPoint):
    def __init__(self, obj_idx, obj_shape, obj_width, obj_height, pose: np.ndarray, tolerances):
        self.idx = obj_idx
        self.shape = obj_shape
        self.width = obj_width
        self.height = obj_height

        super(AssemblyWayPoint, self).__init__(pose, tolerances)


class AssemblyPolicy(Path):
    def __init__(self, configuration: AssemblyPolicyConf, obj_conf=None):
        if obj_conf:
            way_points = [AssemblyWayPoint(obj_conf.idx, obj_conf.object_type, obj_conf.width, obj_conf.height,
                                           wp.pose, wp.accuracy) for wp in configuration.way_points]
        else:
            way_points = [AssemblyWayPoint(wp.obj_conf.idx, wp.object_type, wp.obj_conf.width, wp.obj_conf.height,
                                           wp.pose, wp.accuracy) for wp in configuration.way_points]
        super(AssemblyPolicy, self).__init__(way_points)

    def clear(self):
        import warnings
        warnings.warn('AssemblyPolicy cannot be cleared.')

    def reset(self):
        self._idx = 0

    def get_object_idx(self):
        if self._idx == len(self._way_points):
            return -1
        return self._way_points[self._idx].idx

    def get_object_type(self):
        if self._idx == len(self._way_points):
            return None
        return self._way_points[self._idx].shape

    def get_object_width(self):
        if self._idx == len(self._way_points):
            return None
        return self._way_points[self._idx].width

    def get_object_height(self):
        if self._idx == len(self._way_points):
            return None
        return self._way_points[self._idx].height

    def plot_with_object_information(self, axes, objects):
        from matplotlib import patches

        super_artists = self.plot(axes)
        object_pose = objects[self.get_object_idx()].get_pose()
        # line_positions = np.c_[object_pose[:2], self.active_way_point.position]
        # line_object_target, = axes.plot(line_positions[0, :], line_positions[1, :], ls=':', c=(.9, .7, .9))

        if np.abs(self.active_way_point.orientation - object_pose[2]) <= self.active_way_point.orientation_tolerance:
            color = 'green'
        else:
            color = 'red'
        wedge_object_orientation = patches.Wedge(self.active_way_point.position,
                                                 np.min(self.active_way_point.position_tolerance),
                                                 math.degrees(object_pose[2]), math.degrees(object_pose[2]),
                                                 color=color)
        axes.add_patch(wedge_object_orientation)

        return super_artists, wedge_object_orientation

    def update_plot_with_object_information(self, artists, objects):
        super_artists, wedge_object_orientation = artists
        self.update_plot(super_artists)

        object_pose = objects[self.get_object_idx()].get_pose()
        # line_positions = np.c_[object_pose[:2], self.active_way_point.position]
        # line_object_target.set_xdata(line_positions[0, :])
        # line_object_target.set_ydata(line_positions[1, :])

        if np.abs(self.active_way_point.orientation - object_pose[2]) <= self.active_way_point.orientation_tolerance:
            color = 'green'
        else:
            color = 'red'

        wedge_object_orientation.center = self.active_way_point.position
        wedge_object_orientation.set_radius(np.min(self.active_way_point.position_tolerance))
        wedge_object_orientation.set_theta1(math.degrees(object_pose[2]))
        wedge_object_orientation.set_theta2(math.degrees(object_pose[2]))
        wedge_object_orientation.set_color(color)
