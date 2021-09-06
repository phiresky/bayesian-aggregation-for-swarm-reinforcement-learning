from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from kb_learning.planning.assembly_policy import AssemblyPolicy, Path, WayPoint, PolicyContainer
from kb_learning.envs import EvalEnv


Line = namedtuple('Line', ['a', 'b', 'c'])


def _line_through_pose(pose):
    a = -np.sin(pose[2])
    b = np.cos(pose[2])
    c = -a * pose[0] - b * pose[1]
    return Line(a, b, c)


def _closest_point_on_line(point, line: Line):
    ab = np.array([line.a, line.b])
    ba = np.flipud(ab)
    abab = ab.dot(ab)
    p_x = (line.b * (point * [1, -1]).dot(ba) - line.a * line.c) / abab
    p_y = (line.a * (point * [-1, 1]).dot(ba) - line.b * line.c) / abab

    return np.array([p_x, p_y])


def _which_side_of_line(point, line: Line):
    return np.sign(line.a * point[0] + line.b * point[1] + line.c)


class PoseController:
    def __init__(self, assembly_policy: AssemblyPolicy, env: EvalEnv, k_p, k_t, C):
        self.assembly_policy: AssemblyPolicy = assembly_policy
        self.env = env

        self.k_p = k_p
        self.k_t = k_t
        self.C = C

        # compute line through active waypoint
        self._line_through_target = _line_through_pose(self.assembly_policy.active_way_point)

        self._ax = None
        self._x = np.linspace(-.8, .8, 100)

        self._target_patch = None
        self._intermediate_patch = None
        self._o_p_line_plot = None
        self._target_line_plot = None
        self._target_swarm_trace = None

    def reset(self):
        self.assembly_policy.reset()
        self._line_through_target = _line_through_pose(self.assembly_policy.active_way_point)

        self._ax = None

        self._target_patch = None
        self._intermediate_patch = None
        self._o_p_line_plot = None
        self._target_line_plot = None
        self._target_swarm_trace = None

    def set_axes(self, axes):
        self._ax = axes

        self._target_patch = self._plot_patch(self.assembly_policy.active_way_point.pose,
                                              fill=False, clip_on=False, ls='--', ec='grey')
        self._intermediate_patch = self._plot_patch(self.assembly_policy.active_way_point.pose,
                                                    fill=False, clip_on=False, ls=':', ec='grey')
        self._target_line_plot, = self._plot_line(self._line_through_target)

    def _plot_line(self, line: Line, ls=':'):
        return plt.plot(self._x, -line.a / line.b * self._x - line.c / line.b, ls)

    def _update_line(self, line_plot: plt.Line2D, line: Line):
        if not line_plot:
            return
        line_plot.set_ydata(-line.a / line.b * self._x - line.c / line.b)

    def _plot_patch(self, pose, **kwargs):
        _a = -np.sin(pose[2])
        _b = np.cos(pose[2])

        _object_width = self.assembly_policy.get_object_width()
        _object_height = self.assembly_policy.get_object_height()

        left = -_b *  _object_width/ 2 - _a * _object_height / 2 + pose[0]
        bottom = _a * _object_width / 2 - _b * _object_height / 2 + pose[1]

        patch = patches.Rectangle((left, bottom), _object_width, _object_height, np.rad2deg(pose[2]), **kwargs)
        self._ax.add_patch(patch)
        # self._ax.plot(*pose[:2], 'xr')

        return patch

    def _update_patch(self, patch: patches.Rectangle, pose):
        if not patch:
            return
        _a = -np.sin(pose[2])
        _b = np.cos(pose[2])

        _object_width = self.assembly_policy.get_object_width()
        _object_height = self.assembly_policy.get_object_height()

        left = -_b * _object_width / 2 - _a * _object_height / 2 + pose[0]
        bottom = _a * _object_width / 2 - _b * _object_height / 2 + pose[1]

        patch.set_xy((left, bottom))
        patch.angle = np.rad2deg(pose[2])

    def _pose_control(self, object_pose, goal_pose):
        # compute closest point of object on line (a, b, c)
        p = _closest_point_on_line(object_pose[:2], self._line_through_target)
        # d_theta = goal_pose[2] - object_pose[2]

        # distance to line
        d_o_p = object_pose[:2] - p
        d = np.linalg.norm(d_o_p)

        if d < self.C:
            # if distance below threshold, we do line following along (a, b, c)
            p_s = _closest_point_on_line(object_pose[:2], self._line_through_target)
            theta = goal_pose[2]

            self._update_patch(self._intermediate_patch, goal_pose)
            if self._o_p_line_plot:
                # TODO specify line outside of plot
                self._update_line(self._o_p_line_plot, Line(.0, .1, .1))
        else:
            # if distance above threshold, compute line perpendicular to (a,b,c) through p (thus through object pos)
            theta = goal_pose[2] - np.pi / 2 * _which_side_of_line(object_pose[:2], self._line_through_target)
            a_t = -np.sin(theta)
            b_t = np.cos(theta)
            c_t = -a_t * p[0] - b_t * p[1]
            line_through_p = Line(a_t, b_t, c_t)

            self._update_patch(self._intermediate_patch, (*p, theta))

            if self._o_p_line_plot:
                self._update_line(self._o_p_line_plot, line_through_p)
            else:
                self._o_p_line_plot, = self._plot_line(line_through_p)

            # compute closest point of swarm on line (a_t, b_t, c_t)
            p_s = _closest_point_on_line(object_pose[:2], line_through_p)

        sc_theta = np.array([-np.sin(object_pose[2]), np.cos(object_pose[2])])

        # if k_p * d_o_p[0] + k_t * np.abs(d_theta) > .1:

        d_o_p_s = object_pose[:2] - p_s
        d_theta = theta - object_pose[2]

        return object_pose[:2] - self.k_p * d_o_p_s - max(min(self.k_t * d_theta, .15), -.15) * sc_theta

    def compute_action(self, kilobots, objects, light):
        # update assembly policy
        if self.assembly_policy.update_target(objects[self.assembly_policy.get_object_idx()]):
            return None

        # update line through current target
        self._line_through_target = _line_through_pose(self.assembly_policy.active_way_point)
        self._update_line(self._target_line_plot, self._line_through_target)
        # update target patch
        self._update_patch(self._target_patch, self.assembly_policy.active_way_point)

        object_pose = objects[self.assembly_policy.get_object_idx()]
        goal_pose = self.assembly_policy.active_way_point.pose

        target_swarm = self._pose_control(object_pose, goal_pose)

        if self._target_swarm_trace:
            self._target_swarm_trace.set_xdata(np.append(self._target_swarm_trace.get_xdata(), target_swarm[0]))
            self._target_swarm_trace.set_ydata(np.append(self._target_swarm_trace.get_ydata(), target_swarm[1]))
        else:
            self._target_swarm_trace, = self._ax.plot(target_swarm[0], target_swarm[1], '.k', markersize=.5)

        action = target_swarm - light
        action = np.minimum(action, np.array([.005, .005]))
        action = np.maximum(action, -np.array([.005, .005]))

        return action
