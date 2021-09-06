from __future__ import annotations

import math
from typing import *

import numba
import numpy as np
from gym import spaces
from gym.spaces.space import Space

from ...fast_dict_space import FastDictSpace
from ...util import DictSpace, TupleDictSpace

if TYPE_CHECKING:
    from ..point_envs.rendezvous import RendezvousEnv

from ..point_envs.base import Agent

# import fast_histogram as fh
sq2 = math.sqrt(2)


ObsVal = Union["Obs", List["Obs"], List[float], np.ndarray]
Obs = Dict[str, ObsVal]

if TYPE_CHECKING:
    from ma_envs.point_envs.rendezvous import RendezvousEnv


class RelativeObs:
    def __init__(self, agent: PointAgent):
        self.agent = agent
        self.experiment = agent.experiment
        self.do_walls = False
        self.observation_space = self._observation_space()

    def _observation_space(self) -> Space:
        single_float = spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        local_space: Dict[str, Space] = {
            "ratio_of_agents_visible": single_float,
        }
        if self.agent.add_vel_to_obs:
            local_space["linear_velocity"] = single_float
            local_space["angular_velocity"] = single_float

        if self.agent.params.add_walls_to_obs and not self.experiment.torus:
            local_space["nearest_wall"] = DictSpace(
                {
                    "distance": single_float,
                    "angle_cos": single_float,
                    "angle_sin": single_float,
                }
            )
        neighbor_space = DictSpace(
            {
                "distance": single_float,
                "bearing_cos": single_float,
                "bearing_sin": single_float,
                "their_bearing_cos": single_float,
                "their_bearing_sin": single_float,
                # todo: -1 to 1?
                "relative_velocity": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "ratio_of_agents_visible": single_float,
                "valid": single_float,
            }
        )
        return FastDictSpace(
            DictSpace(
                {
                    "local": DictSpace(local_space),
                    "aggregatables": DictSpace(
                        {
                            "neighbors": TupleDictSpace(
                                neighbor_space, self.experiment.nr_agents - 1
                            ),
                        }
                    ),
                }
            )
        )

    def get_local_observation(self, nr_neighbors, obs: np.ndarray) -> None:
        obs["ratio_of_agents_visible"] = nr_neighbors / (self.experiment.nr_agents - 1)

        if self.agent.add_vel_to_obs:
            obs["linear_velocity"] = (
                self.agent.state.p_vel[0] / self.agent.max_lin_velocity
            )

            obs["angular_velocity"] = (
                self.agent.state.p_vel[1] / self.agent.max_ang_velocity
            )

        if self.do_walls and not self.experiment.torus:
            if np.any(self.agent.state.p_pos <= 1) or np.any(
                self.agent.state.p_pos >= self.experiment.world_size - 1
            ):
                wall_dists = np.array(
                    [
                        self.experiment.world_size - self.agent.state.p_pos[0],
                        self.experiment.world_size - self.agent.state.p_pos[1],
                        self.agent.state.p_pos[0],
                        self.agent.state.p_pos[1],
                    ]
                )
                wall_angles = (
                    np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi])
                    - self.agent.state.p_orientation
                )
                closest_wall = np.argmin(wall_dists)
                obs["nearest_wall"]["distance"] = [
                    wall_dists[closest_wall] / self.experiment.world_size
                ]
                obs["nearest_wall"]["angle_cos"] = [np.cos(wall_angles[closest_wall])]
                obs["nearest_wall"]["angle_sin"] = [np.sin(wall_angles[closest_wall])]

            else:
                obs["nearest_wall"]["distance"] = 1
                obs["nearest_wall"]["angle_cos"] = 0
                obs["nearest_wall"]["angle_sin"] = 0

        return obs

    def get_observation(
        self, agent_distances, my_orientation, their_orientation, vels, nh_size, obs
    ) -> None:
        # true for agents within comm radius (except self)
        agent_is_in_range = (agent_distances < self.experiment.comm_radius) & (
            0 < agent_distances
        )
        nr_neighbors = np.sum(agent_is_in_range)

        self.get_local_observation(nr_neighbors, obs["local"])

        neighbors_indices = agent_is_in_range.nonzero()[0]
        rel_vels = self.agent.state.w_vel - vels

        neighbors = obs["aggregatables"]["neighbors"][0 : len(neighbors_indices)]
        invalid_neighbors = obs["aggregatables"]["neighbors"][len(neighbors_indices) :]

        neighbors["distance"] = agent_distances[neighbors_indices] / (
            self.experiment.world_size * sq2
        )

        neighbors["bearing_cos"] = np.cos(my_orientation[neighbors_indices])
        neighbors["bearing_sin"] = np.sin(my_orientation[neighbors_indices])
        neighbors["their_bearing_cos"] = np.cos(their_orientation[neighbors_indices])
        neighbors["their_bearing_sin"] = np.sin(their_orientation[neighbors_indices])
        neighbors["relative_velocity"] = (
            rel_vels[neighbors_indices] / 2 * self.agent.max_lin_velocity
        )
        neighbors["ratio_of_agents_visible"] = (
            # todo: why subtract own number of neighbors?
            (nh_size[neighbors_indices] - nr_neighbors)
            / (self.experiment.nr_agents - 2)
            if self.experiment.nr_agents > 2
            else 0
        )

        neighbors["valid"] = 1
        invalid_neighbors["distance"] = 0
        invalid_neighbors["bearing_cos"] = 0
        invalid_neighbors["bearing_sin"] = 0
        invalid_neighbors["their_bearing_cos"] = 0
        invalid_neighbors["their_bearing_sin"] = 0
        invalid_neighbors["relative_velocity"] = np.zeros(2, dtype=np.float32)
        invalid_neighbors["ratio_of_agents_visible"] = 0
        invalid_neighbors["valid"] = 0


class UniformObs:
    def __init__(self, agent: PointAgent):
        self.agent = agent
        self.experiment = agent.experiment

    def observation_space(self) -> Space:
        single_float = spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)
        local_space_data = {
            "position": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "orientation_cossin": spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            ),
        }
        if self.agent.add_vel_to_obs:
            local_space_data["linear_velocity"] = single_float
            local_space_data["angular_velocity"] = single_float

        local_space = DictSpace(
            {"valid": single_float, "data": DictSpace(local_space_data)}
        )

        return DictSpace(
            {"agents": TupleDictSpace(local_space, self.experiment.nr_agents)}
        )

    def get_observation(self, agent_distances, _c, _a, _b, _d):
        # agent is in range (including self agent)
        agent_is_in_range = agent_distances < self.experiment.comm_radius

        def order(arg):
            (i, agent) = arg
            # self first
            if agent == self.agent:
                return -1
            # then neighbors
            if agent_is_in_range[i]:
                return 0
            # finally others
            return 1

        agents_sorted = sorted(enumerate(self.agent.experiment.agents), key=order)

        agents = [
            {"valid": [1], "data": agent.obsy.get_local_observation()}
            if agent_is_in_range[i]
            else {
                "valid": [0],
                "data": {
                    "position": [0, 0],
                    "orientation_cossin": [0, 0],
                    "linear_velocity": [0],
                    "angular_velocity": [0],
                },
            }
            for i, agent in agents_sorted
        ]
        return {"agents": agents}

    def get_local_observation(self):
        obs = {
            "position": self.agent.state.p_pos / self.experiment.world_size,
            "orientation_cossin": [
                np.cos(self.agent.state.p_orientation),
                np.sin(self.agent.state.p_orientation),
            ],
        }
        if self.agent.add_vel_to_obs:
            obs["linear_velocity"] = [
                self.agent.state.p_vel[0] / self.agent.max_lin_velocity
            ]
            obs["angular_velocity"] = [
                self.agent.state.p_vel[1] / self.agent.max_ang_velocity
            ]
        return obs


class PointAgent(Agent):
    def __init__(self, experiment: RendezvousEnv):
        super().__init__()
        self.experiment = experiment
        self.params = experiment.agent_params
        self.add_vel_to_obs = self.params.dynamics == "unicycle_acc"
        self.r_matrix = None  # needed for direct dynamics
        if self.params.obs_mode == "relative":
            self.obsy = RelativeObs(self)
        if self.params.obs_mode == "uniform":
            self.obsy = UniformObs(self)

    @property
    def observation_space(self) -> Space:
        return self.obsy.observation_space

    @property
    def action_space(self):
        return spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def set_velocity(self, vel):
        self.velocity = vel

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        self.state.p_vel = np.zeros(2)
        self.state.w_vel = np.zeros(2)

    def get_observation(
        self, agent_distances, my_orientation, their_orientation, vels, nh_size, out_obs
    ) -> Obs:
        return self.obsy.get_observation(
            agent_distances, my_orientation, their_orientation, vels, nh_size, out_obs
        )
