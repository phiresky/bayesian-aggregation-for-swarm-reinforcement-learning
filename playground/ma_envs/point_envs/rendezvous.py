from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding
from matplotlib import patches
from playground.fast_dict_space import FastDictSpace

from ...FlattenVecWrapper import FlattenVecWrapper
from ...util import throw_if_not_contains
from ..commons import utils as U

# from ma_envs.envs.environment import MultiAgentEnv
from ..point_agents.rendezvous_agent import PointAgent
from . import base
from .wonkyvec import WonkyVecEnv

ObsMode = Literal["relative", "uniform"]


class RendezvousAgentParams(NamedTuple):
    dynamics: base.Dynamics
    # in relative mode, the neighbors obs contains angles, positions etc relative to the current agent,
    # in uniform mode, neighbors and the self agent have the same observation format (absolute angles / positions, ...)
    obs_mode: ObsMode

    add_walls_to_obs: bool  # only applies if torus = False


class RendezvousEnvParams(NamedTuple):
    agent_params: RendezvousAgentParams
    torus: bool
    nr_agents: int = 5
    comm_radius: float = 40
    world_size: int = 100
    type: Literal["Rendezvous"] = "Rendezvous"

    def create_env(self):
        return FlattenVecWrapper(RendezvousEnv(params=self))


class RendezvousEnv(WonkyVecEnv):
    metadata = {"render.modes": ["human", "animate"]}

    def __init__(self, params: RendezvousEnvParams):
        super().__init__(nr_agents=params.nr_agents)
        self.params = params
        self.nr_agents = params.nr_agents
        self.world_size = params.world_size
        self.world = base.World(
            params.world_size, params.torus, params.agent_params.dynamics
        )
        self.torus = params.torus
        self.comm_radius = params.comm_radius
        self.hist = None
        self.agent_params = params.agent_params
        self.world.agents = [PointAgent(self) for _ in range(self.nr_agents)]
        # self.seed()

        self.vel_hist = []
        self.state_hist = []
        self.timestep = 0
        self.ax = None

    @property
    def observation_space(self) -> FastDictSpace:
        return self.agents[0].observation_space

    @property
    def action_space(self):
        return self.agents[0].action_space

    @property
    def agents(self) -> list[PointAgent]:
        return self.world.policy_agents

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        # if (np.max(U.get_upper_triangle(self.world.distance_matrix,
        #                                 subtract_from_diagonal=-1)) < 1
        #     and np.mean([agent.state.p_vel**2 for agent in self.agents]) < 0.1**2)\
        #         or self.timestep >= self.timestep_limit:
        if self.timestep >= self.timestep_limit:
            # if self.ax:
            #     plt.close()
            return True
        else:
            return False

    def reset(self):
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = np.random.randint(2, 10)
        # self.nr_agents = 10
        agent_states = np.random.default_rng().random((self.nr_agents, 3))
        agent_states[:, 0:2] = self.world_size * (
            (0.95 - 0.05) * agent_states[:, 0:2] + 0.05
        )
        agent_states[:, 2:3] = 2 * np.pi * agent_states[:, 2:3]

        self.world.agent_states = agent_states

        agent_list = [PointAgent(self) for _ in range(self.nr_agents)]

        self.world.agents = agent_list
        self.world.reset()

        obs = self._get_observation()
        throw_if_not_contains(self.observation_space, obs[0])

        return obs

    def _get_observation(self):
        obs = self.observation_space.create_nans(self.nr_agents)
        velocities = np.vstack([agent.state.w_vel for agent in self.agents])
        nr_agents_sensed = np.sum(
            (0 < self.world.distance_matrix)
            & (self.world.distance_matrix < self.comm_radius),
            axis=1,
        )  # / (self.nr_agents - 1)
        for i, bot in enumerate(self.agents):
            bot.get_observation(
                self.world.distance_matrix[i, :],
                self.world.angle_matrix[i, :],
                self.world.angle_matrix[:, i],
                velocities,
                nr_agents_sensed,
                obs[i],
            )
        return obs

    def step(self, actions):
        if self.is_terminal:
            raise Exception("tried to step done env")

        self.timestep += 1

        # assert len(actions) == self.nr_agents
        # print(actions)
        clipped_actions = np.clip(
            actions[0 : self.nr_agents, :],
            self.agents[0].action_space.low,
            self.agents[0].action_space.high,
        )

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action

        self.world.step()

        next_obs = self._get_observation()

        rewards = self.get_reward(actions)

        done = self.is_terminal

        # if rewards[0] > -0.1 / self.comm_radius:  # distance of 0.1 in world coordinates, scaled by the reward scaling factor
        #     done = True

        # if done:
        #     rewards += 100
        # info = dict()
        info = {
            "state": self.world.agent_states,
            "actions": actions,
            "action_penalty": 0.05 * np.mean(actions ** 2),
            "velocities": np.vstack([agent.state.p_vel for agent in self.agents]),
        }

        return next_obs, rewards, done, info

    def get_mean_distances(self):
        all_distances = U.get_upper_triangle(
            self.world.distance_matrix, subtract_from_diagonal=-1
        )
        all_distances_cap = np.where(
            all_distances > self.comm_radius, self.comm_radius, all_distances
        )
        dist_rew = np.mean(all_distances_cap)
        return dist_rew

    def get_reward(self, actions):

        dist_rew = self.get_mean_distances() / self.comm_radius
        action_pen = 0.001 * np.mean(actions ** 2)
        r = -dist_rew - action_pen
        r = np.ones((self.nr_agents,)) * r
        # print(dist_rew, action_pen)

        return r

    def get_plot(self):
        import matplotlib
        import matplotlib.figure

        dpi = 300
        fig = matplotlib.figure.Figure(figsize=(600 / dpi, 600 / dpi), dpi=dpi)
        ax = fig.subplots()
        # else:
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim((0, self.world_size))
        ax.set_ylim((0, self.world_size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        comm_circles = []
        ax.scatter(
            self.world.agent_states[:, 0], self.world.agent_states[:, 1], c="b", s=10
        )
        for i in range(self.nr_agents):
            comm_circles.append(
                plt.Circle(
                    (self.world.agent_states[i, 0], self.world.agent_states[i, 1]),
                    self.comm_radius,
                    color="g" if i != 0 else "b",
                    fill=False,
                )
            )

            ax.add_artist(comm_circles[i])

        ax.text(
            95,
            5,
            f"{self.timestep}/{self.timestep_limit}",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="large",
        )
        fig.tight_layout(pad=0.05)
        return fig

    def render(
        self, mode="human"
    ):  # , close=True):  check if works with older gym version
        if mode == "animate":
            output_dir = Path("/tmp/rz_video/")
            if self.timestep == 0:
                import os
                import shutil

                if output_dir.exists():
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)

        if not self.ax:
            fig, ax = plt.subplots()
            self.ax = ax

        # else:
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim((0, self.world_size))
        self.ax.set_ylim((0, self.world_size))
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # [ax.clear() for ax in self.axes]
        comm_circles = []
        self.ax.scatter(
            self.world.agent_states[:, 0], self.world.agent_states[:, 1], c="b", s=10
        )
        # self.ax.scatter(self.nodes_all[:, 0], self.nodes_all[:, 1], c="k")
        # self.ax.scatter(self.center_of_mass[0], self.center_of_mass[1], c="g")
        # self.ax.scatter(
        #    self.center_of_mass_torus[0], self.center_of_mass_torus[1], c="r"
        # )
        for i in range(self.nr_agents):

            comm_circles.append(
                plt.Circle(
                    (self.world.agent_states[i, 0], self.world.agent_states[i, 1]),
                    self.comm_radius,
                    color="g" if i != 0 else "b",
                    fill=False,
                )
            )

            self.ax.add_artist(comm_circles[i])

            # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
            #              i, ha='center',
            #              va='center', size=25)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])
        # self.axes[0].imshow(self.agents[0].histogram[0, :, :], vmin=0, vmax=10)
        # self.axes[1].imshow(self.agents[0].histogram[1, :, :], vmin=0, vmax=1)

        self.ax.text(
            95,
            5,
            f"{self.timestep}/{self.timestep_limit}",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        if mode == "human":
            plt.pause(0.01)
        elif mode == "animate":
            # if self.timestep % 2 == 0:
            #    plt.savefig(output_dir + format(self.timestep // 2, "04d"))
            plt.savefig(output_dir / f"{self.timestep:04d}")
            if self.is_terminal:
                import os

                os.system(
                    f"ffmpeg -r 24 -i '{output_dir}/%04d.png' -c:v libx264 -pix_fmt yuv420p -y /tmp/rz_out.mp4"
                )


if __name__ == "__main__":
    n_ag = 10
    env = RendezvousEnv(
        nr_agents=n_ag,
        obs_mode="3d_rbf",
        comm_radius=40,
        world_size=100,
        distance_bins=8,
        bearing_bins=8,
        dynamics="unicycle",
        torus=False,
    )
    for e in range(100):
        o = env.reset()
        dd = False
        flip = -1
        for t in range(10):
            a = 2 * np.random.default_rng().random((n_ag, 2)) - 1
            # print(t, flip, env.agents[0].state.p_vel)
            # if t % 50 == 0:
            #     flip = -flip
            # a[:, 0] = 1 * flip
            # a[:, 1] = 0
            # if t >= 150:
            #     a = np.zeros([20, 2])
            #     print(t, flip, env.agents[0].state.p_vel)
            a[:, 0] = 1
            # a[:, 1] = 1
            # if t >= 60:
            #     a = np.zeros([20, 2])
            o, rew, dd, _ = env.step(a)
            # if rew.sum() < 0:
            #     print(rew[0])
            if t % 1 == 0:
                env.render(mode="human")
                # time.sleep(0.5)
            if dd:

                break
        print(np.mean(env.agents[0].neighborhood_size_hist))
