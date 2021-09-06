import gym_kilobots
import numpy as np
from matplotlib.axes import Axes

import kb_learning


def compute_value_function_grid(state_action_features, policy, theta, num_kilobots, x_range, y_range, resolution=40,
                                extra_dims=None):
    if type(resolution) is not tuple:
        resolution = (resolution, resolution)

    x_space = np.linspace(*x_range, resolution[0])
    y_space = np.linspace(*y_range, resolution[1])
    [X, Y] = np.meshgrid(x_space, y_space)
    X = X.flatten()
    Y = -Y.flatten()

    # kilobots at light position
    states = np.tile(np.c_[X, Y], [1, num_kilobots + 1])
    if extra_dims is not None:
        states = np.c_[states, np.tile(extra_dims, (states.shape[0], 1))]

    # get mean actions
    actions = policy.get_mean(states)

    value_function = state_action_features(states, actions).dot(theta).reshape((resolution[1], resolution[0]))

    return value_function


def compute_policy_quivers(policy, num_kilobots, x_range, y_range, resolution=40, extra_dims=None):
    if type(resolution) is not tuple:
        resolution = (resolution, resolution)

    [X, Y] = np.meshgrid(np.linspace(*x_range, resolution[0]), np.linspace(*y_range, resolution[1]))
    X = X.flatten()
    Y = Y.flatten()

    # kilobots at light position
    states = np.tile(np.c_[X, Y], [1, num_kilobots + 1])
    if extra_dims is not None:
        states = np.c_[states, np.tile(extra_dims, (states.shape[0], 1))]

    # get mean actions
    mean_actions, sigma_actions = policy.get_mean_sigma(states)
    mean_actions = mean_actions.reshape((resolution[1], resolution[0], mean_actions.shape[1]))
    sigma_actions = sigma_actions.reshape((resolution[1], resolution[0]))

    return mean_actions, sigma_actions
