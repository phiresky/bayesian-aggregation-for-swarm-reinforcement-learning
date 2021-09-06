import itertools
from typing import Literal, NamedTuple, Union

import networkx as nwx
import numpy as np
from gym.utils import seeding
from playground.fast_dict_space import FastDictSpace

from ...FlattenVecWrapper import FlattenVecWrapper
from ...util import throw_if_not_contains
from ..commons import utils as U
from ..point_agents.evader_agent import Evader
from ..point_agents.pursuit_agent import PointAgent
from . import base
from .rendezvous import ObsMode
from .wonkyvec import WonkyVecEnv

try:
    import matplotlib
    import matplotlib.animation as mpla
    import matplotlib.patches as patches

    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon, Wedge
except:
    pass


class PursuitEvasionAgentParams(NamedTuple):
    dynamics: base.Dynamics
    obs_mode: ObsMode


histlen = 10


class PursuitEvasionMultiEnvParams(NamedTuple):
    agent_params: PursuitEvasionAgentParams
    torus: bool
    nr_agents: int
    nr_evaders: int
    comm_radius: float
    obs_radius: float
    torus: bool
    world_size: int = 100
    reward_mode: Union[
        Literal["min_distance"], Literal["count_catches"]
    ] = "min_distance"
    type: Literal["PursuitEvasion"] = "PursuitEvasion"

    def create_env(self):
        return FlattenVecWrapper(PursuitEvasionMultiEnv(params=self))


class PursuitEvasionMultiEnv(WonkyVecEnv):
    metadata = {"render.modes": ["human", "animate"]}

    def __init__(self, params: PursuitEvasionMultiEnvParams):
        super().__init__(nr_agents=params.nr_agents)
        self.params = params
        self.nr_agents = params.nr_agents
        self.nr_evaders = params.nr_evaders
        self.comm_radius = params.comm_radius
        self.obs_radius = params.obs_radius
        self.torus = params.torus
        self.world_size = params.world_size
        self.agent_params = params.agent_params
        self.world = base.World(
            params.world_size, params.torus, params.agent_params.dynamics
        )
        self.world.agents = [PointAgent(self) for _ in range(self.nr_agents)]
        [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]
        self.timestep = None
        self.ax = None
        self.obs_comm_matrix = None
        self.catch_locations = []

        # self.seed()

    @property
    def observation_space(self) -> FastDictSpace:
        return self.agents[0].observation_space

    @property
    def action_space(self):
        return self.agents[0].action_space

    @property
    def agents(self):
        return self.world.policy_agents

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        return 1024

    @property
    def is_terminal(self):
        if self.timestep >= self.timestep_limit:
            if self.ax:
                plt.close()
            return True
        return False

    def reset(self):
        self.was_done = False
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = 100  # np.random.randint(2, 10)
        # self.nr_evaders = 10
        # self.world.agents = [
        #     PointAgent(self)
        #     for _ in
        #     range(self.nr_agents)
        # ]
        #
        # [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]

        self.obs_comm_matrix = self.obs_radius * np.ones(
            [self.nr_agents + self.nr_evaders, self.nr_agents + self.nr_evaders]
        )
        self.obs_comm_matrix[
            0 : -self.nr_evaders, 0 : -self.nr_evaders
        ] = self.comm_radius

        pursuers = np.random.rand(self.nr_agents, 3)
        pursuers[:, 0:2] = self.world_size * ((0.95 - 0.05) * pursuers[:, 0:2] + 0.05)
        pursuers[:, 2:3] = 2 * np.pi * pursuers[:, 2:3]

        evader = (0.95 - 0.05) * np.random.rand(self.nr_evaders, 2) + 0.05
        evader = self.world_size * evader

        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()
        self.catch_locations = []
        self.historical_agent_states = [np.copy(self.world.agent_states)]
        self.historical_landmark_states = [np.copy(self.world.landmark_states)]

        if self.obs_radius < self.world_size * np.sqrt(2):
            self.graph_feature()

        obs = self._get_observation()

        throw_if_not_contains(self.observation_space, obs[0])

        return obs

    def _get_observation(self):
        feats = [p.graph_feature for p in self.agents]
        # velocities = np.vstack([agent.state.w_vel for agent in self.agents])

        obs = self.observation_space.create_nans(self.nr_agents)

        for i, bot in enumerate(self.world.policy_agents):
            bot.get_observation(
                self.world.distance_matrix[i, :],
                self.world.angle_matrix[i, :],
                self.world.angle_matrix[:, i],
                feats,
                # velocities
                obs[i],
            )
        return obs

    def step(self, actions):
        if self.was_done:
            raise Exception("tried to step done env")

        self.timestep += 1

        assert len(actions) == self.nr_agents
        # print(actions)
        clipped_actions = np.clip(
            actions, self.agents[0].action_space.low, self.agents[0].action_space.high
        )

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action[0:2]

        if self.params.reward_mode == "count_catches":
            caught = 0
            for i in range(self.nr_evaders):
                if (
                    np.sum(
                        self.world.distance_matrix[
                            -self.nr_evaders + i, : -self.nr_evaders
                        ]
                        < 2
                    )
                    > 0
                ):

                    evader = (0.95 - 0.05) * np.random.rand(2) + 0.05
                    evader = self.world_size * evader

                    self.catch_locations.append(
                        (*self.world.scripted_agents[i].state.p_pos,)
                    )
                    self.world.scripted_agents[i].reset(evader)

                    caught += 1

        self.world.step()

        if self.obs_radius < self.world_size * np.sqrt(2):
            sets = self.graph_feature()

        next_obs = self._get_observation()

        done = self.is_terminal
        if self.params.reward_mode == "count_catches":
            rewards = caught * np.ones((self.nr_agents,))  # / self.nr_evaders
        else:
            rewards = self.get_dense_reward(actions)
            if (
                rewards[0] > -1 / self.obs_radius
            ):  # distance of 1 in world coordinates, scaled by the reward scaling factor
                done = True
        self.was_done = done

        # info = dict()
        info = {
            "pursuer_states": self.world.agent_states,
            "evader_states": self.world.landmark_states,
            "actions": actions,
        }
        # info = {'state': np.concatenate([s_a_next, np.array([[self.evader[0], self.evader[1], 0]])],
        #                                 axis=0)}
        self.historical_agent_states.append(np.copy(self.world.agent_states))
        self.historical_agent_states = self.historical_agent_states[-histlen:]
        self.historical_landmark_states.append(np.copy(self.world.landmark_states))
        self.historical_landmark_states = self.historical_landmark_states[-histlen:]

        return next_obs, rewards, done, info

    def get_dense_reward(self, actions):
        if self.nr_evaders > 1:
            # raise Exception("can only handle one with dense reward (use count_catches)")
            pass
        r = (
            -np.minimum(
                np.min(
                    self.world.distance_matrix[-self.nr_evaders :, : -self.nr_evaders]
                ),
                self.obs_radius,
            )
            / self.obs_radius
        )  # - 0.05 * np.sum(np.mean(actions**2, axis=1))
        # r = -np.minimum(np.partition(self.world.distance_matrix[-1, :-self.nr_evaders], 2)[2], self.obs_radius) / self.world_size
        # r = - 1
        # print(np.min(self.world.distance_matrix[-1, :-self.nr_evaders]))
        r = np.ones((self.nr_agents,)) * r

        return r

    def graph_feature(self):
        adj_matrix = np.array(
            self.world.distance_matrix < self.obs_comm_matrix, dtype=float
        )
        # visibles = np.sum(adj_matrix, axis=0) - 1
        # print("mean neighbors seen: ", np.mean(visibles[:-1]))
        # print("evader seen by: ", visibles[-1])
        sets = U.dfs(adj_matrix, 2)

        g = nwx.Graph()

        for set_ in sets:
            l_ = list(set_)
            if self.nr_agents in set_:
                # points = self.nodes[set_, 0:2]
                # dist_matrix = self.get_euclid_distances(points, matrix=True)

                # determine distance and adjacency matrix of subset
                dist_matrix = np.array(
                    [
                        self.world.distance_matrix[x]
                        for x in list(itertools.product(l_, l_))
                    ]
                ).reshape([len(l_), len(l_)])

                obs_comm_matrix = np.array(
                    [self.obs_comm_matrix[x] for x in list(itertools.product(l_, l_))]
                ).reshape([len(l_), len(l_)])

                adj_matrix_sub = np.array(
                    (0 <= dist_matrix) & (dist_matrix < obs_comm_matrix), dtype=float
                )
                connection = np.where(adj_matrix_sub == 1)
                edges = [
                    [x[0], x[1]]
                    for x in zip(
                        [l_[c] for c in connection[0]], [l_[c] for c in connection[1]]
                    )
                ]

                g.add_nodes_from(l_)
                g.add_edges_from(edges)
                for ind, e in enumerate(edges):
                    g[e[0]][e[1]]["weight"] = dist_matrix[
                        connection[0][ind], connection[1][ind]
                    ]

        for i in range(self.nr_agents):
            try:
                self.agents[i].graph_feature = nwx.shortest_path_length(
                    g, source=i, target=self.nr_agents, weight="weight"
                )
            except:
                self.agents[i].graph_feature = np.inf

        return sets

    def get_plot(self, reuse=False):
        if not reuse:
            self.ax = None
        self.render_inner()
        return self.fig

    def render_historical(self, historical_states, color):
        r, g, b, _ = matplotlib.colors.to_rgba(color)
        for i, states in enumerate(historical_states):
            ratio = (i + 1) / len(historical_states)
            self.ax.scatter(
                states[:, 0],
                states[:, 1],
                color=(
                    r * ratio + 1 * (1 - ratio),
                    g * ratio + 1 * (1 - ratio),
                    b * ratio + 1 * (1 - ratio),
                ),
                s=20,
            )

    def render_inner(self):
        if not self.ax:
            dpi = 300
            fig, ax = plt.subplots(figsize=(600 / dpi, 600 / dpi), dpi=dpi)
            self.fig = fig
            self.ax = ax
            self.fig.tight_layout(pad=0.05)
        else:
            self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim((0, self.world_size))
        self.ax.set_ylim((0, self.world_size))
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.render_historical(self.historical_landmark_states, "r")
        self.render_historical(self.historical_agent_states, "b")
        self.ax.scatter(
            [x for x, y in self.catch_locations],
            [y for x, y in self.catch_locations],
            c="g",
            s=40,
            marker="x",
        )
        """
        for i in range(self.nr_agents):
            self.ax.add_patch(
                plt.Circle(
                    (self.world.agent_states[i, 0], self.world.agent_states[i, 1]),
                    self.comm_radius,
                    color="g",
                    fill=False,
                )
            )
            self.ax.add_patch(
                plt.Circle(
                    (self.world.agent_states[i, 0], self.world.agent_states[i, 1]),
                    self.obs_radius,
                    color="g",
                    fill=False,
                )
            )
        """

        # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
        #              "{}".format(i), ha='center',
        #              va='center', size=20)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])
        import matplotlib.patheffects as patheffects

        tx1 = self.ax.text(
            96,
            2,
            f"{self.timestep}/{self.timestep_limit}",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="large",
        )
        tx2 = self.ax.text(
            96,
            98,
            f"{len(self.catch_locations)} caught",
            horizontalalignment="right",
            verticalalignment="top",
            fontsize="large",
        )
        blackonwhite = [
            patheffects.Stroke(linewidth=2, foreground="white"),
            patheffects.Normal(),
        ]
        tx1.set_path_effects(blackonwhite)
        tx2.set_path_effects(blackonwhite)
        self.fig.tight_layout(pad=0.05)

    def render(self, mode="human"):
        if mode == "animate":
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import os
                import shutil

                try:
                    shutil.rmtree(output_dir)
                except FileNotFoundError:
                    pass
                os.makedirs(output_dir, exist_ok=True)

        self.render_inner()

        if mode == "human":
            plt.pause(0.01)
        elif mode == "animate":
            if self.timestep % 1 == 0:
                plt.savefig(output_dir + format(self.timestep // 1, "04d"))

            if self.is_terminal:
                import os

                os.system(
                    "ffmpeg -r 10 -i "
                    + output_dir
                    + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4"
                )


if __name__ == "__main__":
    nr_pur = 5
    env = PursuitEvasionMultiEnv(
        params=PursuitEvasionMultiEnvParams(
            nr_pursuers=nr_pur,
            nr_evaders=2,
            comm_radius=200 * np.sqrt(2),
            world_size=100,
            torus=True,
            agent_params=PursuitEvasionAgentParams(
                dynamics="unicycle",
                obs_mode="fix",
            ),
        )
    )
    for ep in range(1):
        o = env.reset()
        dd = False
        for t in range(1024):
            a = 1 * np.random.randn(nr_pur, env.world.agents[0].dim_a)
            a[:, 0] = 1
            # a[:, 1] = 0
            o, rew, dd, _ = env.step(a)
            # if rew.sum() < 0:
            #     print(rew[0])
            if t % 1 == 0:
                env.render()

            if dd:
                break
