from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gym import spaces
from playground.fast_dict_space import FastDictSpace

from ...util import DictSpace, TupleDictSpace

if TYPE_CHECKING:
    from ..point_envs.pursuit import PursuitEvasionMultiEnv

from ..point_envs.base import Agent


class PointAgent(Agent):
    def __init__(self, experiment: PursuitEvasionMultiEnv):
        super(PointAgent, self).__init__()
        self.experiment = experiment
        self.comm_radius = experiment.comm_radius
        self.obs_radius = experiment.obs_radius
        self.obs_is_limited = self.obs_radius <= 100  # todo: separate flag
        self.obs_mode = experiment.agent_params.obs_mode
        if self.obs_mode != "relative":
            raise Exception(f"unknown bos mode {self.obs_mode}")
        self.torus = experiment.torus
        self.n_evaders = experiment.nr_evaders
        self.world_size = experiment.world_size
        self.r_matrix = None
        self.graph_feature = None
        self.see_evader = None
        self.max_lin_velocity = 10  # cm/s
        self.max_ang_velocity = 2 * np.pi
        if self.experiment.agent_params.dynamics == "unicycle_acc":
            raise Exception("acc not supported, vel not added to obs")
        self.observation_space = self._observation_space()

    def _observation_space(self):
        single_float = spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        two_floats = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        evader_space = DictSpace(
            {
                # should have 7 things
                "distance": single_float,
                "bearing": two_floats,
                "valid": single_float,
            }
        )
        neighbor_space = DictSpace(
            {
                # should have 5 things
                "distance": single_float,
                "bearing": two_floats,
                "their_bearing": two_floats,
                "valid": single_float,
            }
        )
        local_space = {"ratio_of_agents_visible": single_float}
        if not self.torus:
            local_space["is_at_wall"] = single_float
        return FastDictSpace(
            DictSpace(
                {
                    "local": DictSpace(local_space),
                    "aggregatables": DictSpace(
                        {
                            "neighbors": TupleDictSpace(
                                neighbor_space, self.experiment.nr_agents - 1
                            ),
                            "evaders": TupleDictSpace(
                                evader_space, self.experiment.nr_evaders
                            ),
                        }
                    ),
                }
            )
        )

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        self.state.p_vel = np.zeros(2)
        self.state.w_vel = np.zeros(2)
        self.graph_feature = np.inf

    def get_observation(self, dm, my_orientation, their_orientation, feat, out_obs):
        evader_dists = dm[-self.n_evaders :]
        evader_bearings = my_orientation[-self.n_evaders :]
        pursuer_dists = dm[: -self.n_evaders]
        pursuer_bearings = my_orientation[: -self.n_evaders]

        in_range = (evader_dists <= self.obs_radius) & (0 <= evader_dists)
        nr_neighboring_evaders = np.sum(in_range)
        dist_to_evader = evader_dists[in_range] / self.obs_radius

        pursuers_in_range = (pursuer_dists <= self.comm_radius) & (0 <= pursuer_dists)
        nr_neighbors = np.sum(pursuers_in_range)
        local_obs = out_obs["local"]

        local_obs["ratio_of_agents_visible"] = nr_neighbors / (
            self.experiment.nr_agents - 1
        )

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                local_obs["is_at_wall"] = 1
            else:
                local_obs["is_at_wall"] = 0
        if self.obs_is_limited:
            shortest_path_to_evader = (
                self.graph_feature / (5 * self.comm_radius)
                if self.graph_feature < (5 * self.comm_radius)
                else 1.0
            )

            local_obs["shortest_path_to_evader"] = shortest_path_to_evader

        num_visible_evaders = len(dist_to_evader)
        e_obs = out_obs["aggregatables"]["evaders"][:num_visible_evaders]
        e_obs[:]["valid"] = 1
        e_obs[:]["distance"] = dist_to_evader
        e_obs[:]["bearing"][:, 0] = np.cos(evader_bearings[in_range])
        e_obs[:]["bearing"][:, 1] = np.sin(evader_bearings[in_range])
        out_obs["aggregatables"]["evaders"][num_visible_evaders:][:] = 0

        if self.obs_is_limited:
            dists_in_range = np.array(feat)[pursuers_in_range]
            dists_in_range_capped = np.where(
                dists_in_range <= 5 * self.comm_radius,
                dists_in_range / (5 * self.comm_radius),
                1.0,
            )
            # todo: add to obs
        n_obs = out_obs["aggregatables"]["neighbors"][:nr_neighbors]

        n_obs[:]["valid"] = 1
        n_obs[:]["distance"] = pursuer_dists[pursuers_in_range] / self.comm_radius
        n_obs[:]["bearing"][:, 0] = np.cos(pursuer_bearings[pursuers_in_range])
        n_obs[:]["bearing"][:, 1] = np.sin(pursuer_bearings[pursuers_in_range])
        n_obs[:]["their_bearing"][:, 0] = np.cos(their_orientation[pursuers_in_range])
        n_obs[:]["their_bearing"][:, 1] = np.sin(their_orientation[pursuers_in_range])

        out_obs["aggregatables"]["neighbors"][nr_neighbors:] = 0

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self.position = x_2

    def set_angle(self, phi):
        assert phi.shape == (1,)
        self.angle = phi
        r_matrix_1 = np.squeeze(
            [
                [np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
                [np.sin(-np.pi / 2), np.cos(-np.pi / 2)],
            ]
        )
        r_matrix_2 = np.squeeze(
            [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
        )

        self.r_matrix = np.dot(r_matrix_1, r_matrix_2)
