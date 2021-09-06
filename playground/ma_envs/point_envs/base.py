from typing import Literal

import numba
import numpy as np

from ..commons import utils as U

dynamics = ["point", "unicycle", "box2d", "direct", "unicycle_acc"]


class EntityState(object):
    """physical/external base state of all entities"""

    def __init__(self):
        # physical position
        self.p_pos: np.ndarray = None
        self.p_orientation: np.ndarray = None
        # physical velocity
        self.p_vel: np.ndarray = None
        # velocity in world coordinates
        self.w_vel: np.ndarray = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()


class Action(object):
    """action of the agent"""

    def __init__(self):
        # physical action
        self.u = None


class Entity(object):
    """properties and state of physical world entity"""

    def __init__(self):
        # name
        self.name = ""
        # max speed and accel
        self.max_speed = None
        # state
        self.state = EntityState()


class Landmark(Entity):
    """properties of landmark entities"""

    def __init__(self):
        super(Landmark, self).__init__()


class Agent(Entity):
    """properties of agent entities"""

    def __init__(self):
        super(Agent, self).__init__()
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # physical damping
        self.lin_damping = 0.01  # 0.025  # 0.05
        self.ang_damping = 0.01  # 0.05
        self.max_lin_velocity = 10  # cm/s
        self.max_ang_velocity = np.pi  # 2 * np.pi  # rad/s
        self.max_lin_acceleration = 10  # 25  # 100  # cm/s**2
        self.max_ang_acceleration = np.pi  # 60  # rad/s**2


Dynamics = Literal["direct", "unicycle", "unicycle_acc"]


class World(object):
    def __init__(self, world_size, torus, agent_dynamic: Dynamics):
        self.nr_agents = None
        # world is square
        self.world_size = world_size
        # dynamics of agents
        assert agent_dynamic in dynamics
        self.agent_dynamic: Dynamics = agent_dynamic
        # periodic or closed world
        self.torus = torus
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # matrix containing agent states
        self.agent_states = None
        # matrix containing landmark states
        self.landmark_states = None
        # x,y of everything
        self.nodes = None
        self.distance_matrix = None
        self.angle_matrix = None
        # position dimensionality
        self.dim_p = 2
        # simulation timestep
        self.dt = 0.01
        self.action_repeat = 10
        self.timestep = 0
        # physical damping
        self.damping = 0.25

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def reset(self):
        self.timestep = 0
        self.nr_agents = len(self.policy_agents)
        self.total_nr_agents = len(self.agents)

        for i, agent in enumerate(self.policy_agents):
            agent.reset(self.agent_states[i, :])

        for i, agent in enumerate(self.scripted_agents):
            agent.reset(self.landmark_states[i, :])

        self.nodes = (
            np.vstack([self.agent_states[:, 0:2], self.landmark_states])
            if self.landmark_states is not None
            else self.agent_states[:, 0:2]
        )

        self.distance_matrix = U.get_distance_matrix(
            self.nodes, torus=self.torus, world_size=self.world_size, add_to_diagonal=-1
        )

        self.angle_matrix = World._angles(
            self.nodes,
            self.torus,
            self.world_size,
            self.agent_states,
            self.total_nr_agents,
        )

    def step(self):

        self.timestep += 1

        for i, agent in enumerate(self.scripted_agents):
            action = agent.action_callback(agent, self)
            if agent.dynamics == "direct":
                next_coord = (
                    agent.state.p_pos
                    + action * agent.max_speed * self.dt * self.action_repeat
                )
                if self.torus:
                    next_coord = np.where(
                        next_coord < 0, next_coord + self.world_size, next_coord
                    )
                    next_coord = np.where(
                        next_coord > self.world_size,
                        next_coord - self.world_size,
                        next_coord,
                    )
                else:
                    next_coord = np.where(next_coord < 0, 0, next_coord)
                    next_coord = np.where(
                        next_coord > self.world_size, self.world_size, next_coord
                    )
                agent.state.p_pos = next_coord
            else:
                raise Exception(f"unknown scripted agent dynamic {agent.dynamics}")

            self.landmark_states[i, :] = agent.state.p_pos

        if self.agent_dynamic == "direct":
            agents = self.policy_agents
            actions = np.zeros([self.nr_agents, 2])
            for i, agent in enumerate(agents):
                actions[i, 0] = agent.action.u[0]
                actions[i, 1] = agent.action.u[1]

            # direct state manipulation
            action_norm = np.linalg.norm(actions, axis=1)
            # print(actions)
            # print("")
            scaled_actions = np.empty_like(actions)
            scaled_actions[:, 0] = np.where(
                action_norm <= 1, actions[:, 0], actions[:, 0] / action_norm
            )
            scaled_actions[:, 1] = np.where(
                action_norm <= 1, actions[:, 1], actions[:, 1] / action_norm
            )
            for i, action in enumerate(scaled_actions):
                actions[i, :] = np.dot(agents[i].r_matrix, actions[[i], :].T).T
            next_coord = self.actors[:, 0:2] + actions * self.dt
            if self.torus:
                next_coord = np.where(
                    next_coord < 0, next_coord + self.world_size, next_coord
                )
                next_coord = np.where(
                    next_coord > self.world_size,
                    next_coord - self.world_size,
                    next_coord,
                )
            else:
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(
                    next_coord > self.world_size, self.world_size, next_coord
                )
            agent_states_next = next_coord  # + np.ones((self.nr_pursuers, 2)) * np.random.rand(self.nr_pursuers, 2) * 1e-6
            self.actors[:, 0:2] = agent_states_next
            for i, agent in enumerate(agents):
                agent.set_position(agent_states_next[i, :])

            self.nodes = self.actors[:, 0:2]

        elif self.agent_dynamic == "unicycle":
            self._unicycle_step()
        elif self.agent_dynamic == "unicycle_acc":
            self.agent_states = self._unicycle_acc_step()

        elif self.agent_dynamic == "box2d":
            for i, bot in enumerate(self.bots):
                bot.set_motor(actions[i, 0], actions[i, 1])

            for j in range(int(self.frame_skip)):
                [bot.set_velocities() for bot in self.bots]
                self.world.Step(self.time_step, 10, 10)

            next_coord = np.array([bot.get_real_position() for bot in self.bots])
            next_angle = np.array([bot.body.angle for bot in self.bots]) % (2 * np.pi)

            agent_states_next = np.concatenate(
                [next_coord, next_angle[:, None]], axis=1
            )
            self.actors = agent_states_next

            self.nodes = np.vstack([agent_states_next[:, 0:2], self.source, self.sink])

        self.nodes = (
            np.vstack([self.agent_states[:, 0:2], self.landmark_states])
            if self.landmark_states is not None
            else self.agent_states[:, 0:2]
        )

        self.distance_matrix = U.get_distance_matrix(
            self.nodes, torus=self.torus, world_size=self.world_size, add_to_diagonal=-1
        )

        self.angle_matrix = World._angles(
            self.nodes,
            self.torus,
            self.world_size,
            self.agent_states,
            self.total_nr_agents,
        )

    @numba.njit
    def _angles(nodes, torus, world_size, agent_states, total_nr_agents):
        angles = np.empty((len(agent_states), total_nr_agents), dtype=np.float32)
        for i, a in enumerate(agent_states):
            angles[i] = (
                U.get_angle(nodes, a[0:2], torus=torus, world_size=world_size) - a[2]
            )

        angles_shift = -angles % (2 * np.pi)
        return np.where(angles_shift > np.pi, angles_shift - 2 * np.pi, angles_shift)

    def _unicycle_step(self):
        # unicycle dynamics

        scaled_actions = np.zeros([self.nr_agents, 2])
        for i, agent in enumerate(self.policy_agents):
            scaled_actions[i, 0] = agent.action.u[0] * agent.max_lin_velocity
            scaled_actions[i, 1] = agent.action.u[1] * agent.max_ang_velocity

        for i in range(self.action_repeat):
            step = np.concatenate(
                [
                    scaled_actions[:, [0]] * np.cos(self.agent_states[:, 2:3]),
                    scaled_actions[:, [0]] * np.sin(self.agent_states[:, 2:3]),
                ],
                axis=1,
            )
            next_coord = self.agent_states[:, 0:2] + step * self.dt
            next_angle = (
                self.agent_states[:, 2:3] + scaled_actions[:, [1]] * self.dt
            ) % (2 * np.pi)

            if self.torus:
                next_coord = np.where(
                    next_coord < 0, next_coord + self.world_size, next_coord
                )
                next_coord = np.where(
                    next_coord > self.world_size,
                    next_coord - self.world_size,
                    next_coord,
                )
            else:
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(
                    next_coord > self.world_size, self.world_size, next_coord
                )

            agent_states_next = np.concatenate([next_coord, next_angle], axis=1)

            self.agent_states = agent_states_next

        for i, agent in enumerate(self.policy_agents):
            agent.state.p_pos = agent_states_next[i, 0:2]
            agent.state.p_orientation = agent_states_next[i, 2:3]
            agent.state.p_vel = step[i, :]

    def _unicycle_acc_step(self):
        agents = self.policy_agents
        actions = np.array([agent.action.u for agent in agents])
        max_lin_vel = np.array([agent.max_lin_velocity for agent in agents])
        max_ang_vel = np.array([agent.max_ang_velocity for agent in agents])
        max_lin_acc = np.array([agent.max_lin_acceleration for agent in agents])
        max_ang_acc = np.array([agent.max_ang_acceleration for agent in agents])

        damping = np.asarray(
            [[agent.lin_damping, agent.ang_damping] for agent in agents]
        )
        velocities = np.asarray([agent.state.p_vel for agent in agents])

        agent_states_next, velocities, step = World._unicycle_acc_step_jit(
            self.action_repeat,
            self.dt,
            self.nr_agents,
            self.torus,
            self.world_size,
            actions,
            max_lin_acc,
            max_ang_acc,
            max_lin_vel,
            max_ang_vel,
            self.agent_states,
            damping,
            velocities,
        )
        for i, agent in enumerate(agents):
            agent.state.p_pos = agent_states_next[i, 0:2]
            agent.state.p_orientation = agent_states_next[i, 2:3]
            agent.state.p_vel = velocities[i, :]
            agent.state.w_vel = step[i, :]
        return agent_states_next

    @numba.njit
    def _unicycle_acc_step_jit(
        action_repeat,
        dt,
        nr_agents,
        torus,
        world_size,
        actions,
        max_lin_acc,
        max_ang_acc,
        max_lin_vel,
        max_ang_vel,
        agent_states,
        damping,
        velocities,
    ):
        # unicycle dynamics with acceleration

        scaled_actions = np.zeros((nr_agents, 2), np.float32)
        for i, action in enumerate(actions):
            scaled_actions[i, 0] = action[0] * max_lin_acc[i]
            scaled_actions[i, 1] = action[1] * max_ang_acc[i]

        agent_states_next = np.copy(agent_states)

        for i in range(action_repeat):
            velocities = velocities * (1 - damping)

            velocities = velocities + scaled_actions * dt

            velocities[:, 0] = np.where(
                np.abs(velocities[:, 0]) > max_lin_vel,
                np.sign(velocities[:, 0]) * max_lin_vel,
                velocities[:, 0],
            )

            velocities[:, 1] = np.where(
                np.abs(velocities[:, 1]) > max_ang_vel,
                np.sign(velocities[:, 1]) * max_ang_vel,
                velocities[:, 1],
            )

            step = np.stack(
                (
                    velocities[:, 0] * np.cos(agent_states_next[:, 2]),
                    velocities[:, 0] * np.sin(agent_states_next[:, 2]),
                ),
                axis=-1,
            )

            turn = velocities[:, 1:2]

            next_coord = agent_states_next[:, 0:2] + step * dt
            next_angle = agent_states_next[:, 2:3] + turn * dt

            if torus:
                next_coord = np.where(
                    next_coord < 0, next_coord + world_size, next_coord
                )
                next_coord = np.where(
                    next_coord > world_size,
                    next_coord - world_size,
                    next_coord,
                )
            else:
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(next_coord > world_size, world_size, next_coord)

            agent_states_next = np.concatenate(
                (next_coord, next_angle, velocities), axis=1
            )

        return agent_states_next, velocities, step
