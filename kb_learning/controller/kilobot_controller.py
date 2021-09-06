import gym_kilobots
import numpy as np
import enum

from matplotlib.axes import Axes

from kb_learning.planning.a_star import AStar, GridWithObstacles
from kb_learning.planning.assembly_policy import AssemblyPolicy, Path, WayPoint, PolicyContainer
from kb_learning.envs import EvalEnv
from kb_learning.tools import compute_swarm_mean_in_light


class KilobotController:
    class PolicyState(enum.Enum):
        MOVE_SWARM_TO_OBJECT = 0
        MOVE_OBJECT_ALONG_TRAJECTORY = 1

    def __init__(self, pushing_policies: PolicyContainer, assembly_policy: AssemblyPolicy, env: EvalEnv):
        # object pushing policies
        self.pushing_policies: PolicyContainer = pushing_policies
        # object assembly policy
        self.assembly_policy: AssemblyPolicy = assembly_policy

        self.policy_state = self.PolicyState.MOVE_SWARM_TO_OBJECT

        # auxiliary path for swarm
        self.use_a_star_for_swarm = True
        self.swarm_path = Path()
        # auxiliary path for object
        self.use_a_star_for_object = False
        self.object_path = Path()

        self.a_star = AStar(GridWithObstacles(env.world_width - .1, env.world_height - .1, 50,
                                              offset=-env.world_bounds[0]))
        self.env = env

        self._ax = None
        self._target_patch = None
        self._target_patch_body_idx = None

    def reset(self):
        self.assembly_policy.reset()
        self.object_path.clear()
        self.swarm_path.clear()

        self.a_star.grid.clear_obstacles()

        self._ax = None
        self._target_patch = None
        self._target_patch_body_idx = None

    def set_axes(self, axes):
        self._ax = axes

        self._target_patch_body_idx = self.assembly_policy.get_object_idx()
        body_shape = self.assembly_policy.get_object_type()
        body_width = self.assembly_policy.get_object_width()
        body_height = self.assembly_policy.get_object_height()
        body_pose = self.assembly_policy.active_way_point.pose
        self._target_patch, _ = gym_kilobots.kb_plotting.plot_body_from_shape(self._ax, body_shape, body_width,
                                                                              body_height, body_pose,
                                                                              color=(.6, .6, .6, .3))

    def update_plot(self):
        if self._target_patch is None:
            return
        if self.assembly_policy.finished():
            self._target_patch.remove()
            self._target_patch = None
            return
        if self._target_patch_body_idx == self.assembly_policy.get_object_idx():
            body_shape = self.assembly_policy.get_object_type()
            body_width = self.assembly_policy.get_object_width()
            body_height = self.assembly_policy.get_object_height()
            body_pose = self.assembly_policy.active_way_point.pose
            body = gym_kilobots.kb_plotting.get_body_from_shape(body_shape, body_width, body_height, body_pose)
            gym_kilobots.kb_plotting.update_body(body, self._target_patch)
        else:
            self._target_patch.remove()
            self._target_patch_body_idx = self.assembly_policy.get_object_idx()
            body_shape = self.assembly_policy.get_object_type()
            body_width = self.assembly_policy.get_object_width()
            body_height = self.assembly_policy.get_object_height()
            body_pose = self.assembly_policy.active_way_point.pose
            self._target_patch, _ = gym_kilobots.kb_plotting.plot_body_from_shape(self._ax, body_shape, body_width,
                                                                                  body_height, body_pose, zorder=-1)

    def _kilobots_aux_target(self, objects):
        obj_pos = objects[self.assembly_policy.get_object_idx()]
        # compute aux target for kilobots 15cm behind object wrt next target
        vec_wp_to_obj = obj_pos[:2] - self.assembly_policy.active_way_point[:2]
        if np.linalg.norm(vec_wp_to_obj) < .01 and self.assembly_policy.upcoming_way_point:
            vec_wp_to_obj = obj_pos[:2] - self.assembly_policy.upcoming_way_point[:2]
        target_pos_offset = 0.15 * vec_wp_to_obj / (np.linalg.norm(vec_wp_to_obj) + 1e-6)
        return obj_pos[0:2] + target_pos_offset

    def _update_a_star_with_objects(self, exclude_object_idx=None):
        obstacle_points = []
        for i, o in enumerate(self.env.get_objects()):
            if i == exclude_object_idx:
                continue
            for part in o.vertices:
                for v1, v2 in zip(part, np.roll(part, 1, axis=0)):
                    _v = v1 - v2
                    _d = np.linalg.norm(_v)
                    steps = np.arange(.0, _d, .07) / _d

                    boundary_points = v2 + np.tile(_v, (steps.size, 1)) * steps[:, None]
                    obstacle_points.append(boundary_points)
        obstacle_points = np.vstack(obstacle_points)
        self.a_star.set_obstacles(obstacle_points)

    def _move_swarm_to_object(self, kilobots, objects, light):
        # compute position of swarm (mean of all agents)
        # swarm_pos = kilobots.mean(axis=0)
        swarm_pos = compute_swarm_mean_in_light(kilobots[:, :2], light, .2)

        # is auxiliary policy is empty, compute new path with A*
        if self.swarm_path.is_empty():
            self._update_a_star_with_objects()
            aux_target_pos = self._kilobots_aux_target(objects)

            print('Started A* search')
            aux_way_points = self.a_star.get_path(swarm_pos, aux_target_pos)
            self.swarm_path._way_points = [WayPoint(wp, (.025, .5)) for wp in aux_way_points]
            print('Finished A* search')

        target_pos = self.swarm_path.active_way_point
        action = target_pos - light[0:2]
        return action

    def _move_object_along_trajectory(self, kilobots, objects, light):
        _target_object_idx = self.assembly_policy.get_object_idx()

        # get object position and orientation
        obj_pos = objects[_target_object_idx, :2]
        obj_orientation = objects[_target_object_idx, 2]

        # translate light and kilobots into object frame
        translated_light = light - objects[_target_object_idx, :2]
        translated_kilobots = np.array([kb[:2] - objects[_target_object_idx, :2] for kb in kilobots])
        translated_state = np.concatenate([translated_kilobots, np.array([translated_light])], axis=0)

        # compute direction into which to push and rotate state accordingly (we want to push into x-direction)
        push_direction = self.object_path.active_way_point - obj_pos
        direction_angle = -np.arctan2(push_direction[1], push_direction[0])

        rotation_matrix = np.array([[np.cos(direction_angle), -np.sin(direction_angle)],
                                    [np.sin(direction_angle), np.cos(direction_angle)]])

        rotated_state = translated_state.dot(rotation_matrix.T)

        # compute rotation error of object to check if we want to rotate cw or ccw
        rotation_error = obj_orientation - self.assembly_policy.active_way_point.orientation
        rotation_direction = -np.sign(rotation_error)
        if rotation_direction == 0:
            rotation_direction = 1

        # mirror the state on the x-axis if we want to rotate cw
        transformed_state = rotated_state * np.array([[1., rotation_direction]])

        # compute translation and rotation error to estimate closest w-factor
        # translation_err = np.maximum(np.linalg.norm(push_direction) - self.assembly_policy.next_way_point.position_tolerance, 0)
        # rotation_error = np.sign(rotation_error) * np.maximum(np.abs(rotation_error) -
        #                                                       self.assembly_policy.next_way_point.orientation_tolerance, 0)
        translation_error = np.linalg.norm(push_direction)

        # controller
        error_ratio = np.abs(rotation_error * .05) / (np.abs(translation_error) + np.abs(rotation_error * .05) +
                                                       1.e-8)
        # policy_idx = int((len(self.pushing_policies) - 1) * error_ratio + 0.5)

        object_type = self.assembly_policy.get_object_type()
        action = self.pushing_policies[object_type, error_ratio].get_mean(np.c_[transformed_state.reshape(1, -1),
                                                                                obj_orientation - direction_angle])

        # rotate action
        rotation_matrix = np.array([[np.cos(-direction_angle), -np.sin(-direction_angle) * rotation_direction],
                                    [np.sin(-direction_angle), np.cos(-direction_angle) * rotation_direction]])
        transformed_action = action.dot(rotation_matrix.T).flatten()
        # action = self.env.transform_object_to_world_point(action.flatten(), _target_object_idx)

        # print('action: {} transformed_action: {}'.format(action, transformed_action))

        return np.array(transformed_action)

    def _update_policy_state(self, kilobots: np.ndarray, objects: np.ndarray, light):
        # swarm_pos = kilobots[:, :2].mean(axis=0)
        swarm_pos = compute_swarm_mean_in_light(kilobots[:, :2], light, .2)

        object_pose = objects[self.assembly_policy.get_object_idx()]
        # obj_position = objects[_target_object_idx, :2]
        # obj_orientation = objects[_target_object_idx, 2]

        if self.assembly_policy.finished():
            return True

        if self.policy_state == self.PolicyState.MOVE_SWARM_TO_OBJECT:
            self.swarm_path.update_target(swarm_pos)

            if self.swarm_path.is_empty():
                self.update_plot()
                return False

            # if we reached the end of the swarm path, we can start pushing the object
            if self.swarm_path.finished():
                self.update_plot()
                self.swarm_path.clear()
                self.policy_state = self.PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY
                return self._update_policy_state(kilobots, objects, light)

        elif self.policy_state == self.PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
            # if we are too far away from the object, we want to reposition the swarm
            if np.linalg.norm(swarm_pos - object_pose[:2]) > 0.3:
                # change policy state to reposition the swarm
                self.policy_state = self.PolicyState.MOVE_SWARM_TO_OBJECT
                self.object_path.clear()
                # recursive update
                return self._update_policy_state(kilobots, objects, light)
            else:
                finished = self.assembly_policy.update_target(object_pose)
                # if the assembly policy is updated, we want to reposition the swarm
                if finished:
                    self.update_plot()
                    return True
                else:
                    if self.object_path.is_empty():
                        self.update_plot()
                        if self.use_a_star_for_object:
                            self._update_a_star_with_objects(self.assembly_policy.get_object_idx())

                            # print('Started A* search')
                            aux_way_points = self.a_star.get_path(object_pose[:2],
                                                                  self.assembly_policy.upcoming_way_point)
                            self.object_path.trajectory_points = [WayPoint(wp, (.05, .5)) for wp in aux_way_points]
                            # print('Finished A* search')
                        else:
                            # take next point from assembly policy
                            self.object_path._way_points = [self.assembly_policy.active_way_point]
                        self.object_path.update_target(object_pose)
                        return self._update_policy_state(kilobots, objects, light)
                    else:
                        self.object_path.update_target(object_pose)
                        if self.object_path.finished():
                            self.update_plot()
                            self.object_path.clear()
                            # self.policy_state = self.PolicyState.MOVE_SWARM_TO_OBJECT
                            return self._update_policy_state(kilobots, objects, light)
        return False

    def compute_action(self, kilobots, objects, light):
        if self._update_policy_state(kilobots, objects, light):
            return None

        if self.policy_state == self.PolicyState.MOVE_SWARM_TO_OBJECT:
            self.env.path = self.swarm_path
            return self._move_swarm_to_object(kilobots, objects, light)
        elif self.policy_state == self.PolicyState.MOVE_OBJECT_ALONG_TRAJECTORY:
            self.env.path = self.object_path
            return self._move_object_along_trajectory(kilobots, objects, light)
