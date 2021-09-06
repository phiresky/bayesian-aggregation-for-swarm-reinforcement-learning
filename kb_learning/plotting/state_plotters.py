from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.patches import Circle

from gym_kilobots.envs import KilobotsEnv

import numpy as np


def plot_objects_from_env(axes: Axes, env: KilobotsEnv, alpha=.7, **kwargs):
    for o in env.get_objects():
        o.plot(axes, alpha=alpha, **kwargs)


def plot_kilobot_kernel_distribution(axes: Axes, kilobots: np.ndarray, kernel, resolution=40):
    X, Y = np.meshgrid(np.linspace(-.4, .4, resolution), np.linspace(-.4, .4, resolution))
    XY = np.c_[X.flat, Y.flat]

    K = kernel(XY, kilobots).sum(axis=1).reshape(resolution, resolution) / kilobots.shape[0]

    axes.contourf(X, Y, K, cmap=cm.BuPu)
    axes.scatter(kilobots[:, 0], kilobots[:, 1], s=10)


def plot_light(axes: Axes, light: np.ndarray, radius: float = .2):
    axes.add_patch(Circle(light, radius=radius, color=(0.4, 0.7, 0.3, 0.3), fill=False))
    axes.plot(light[0], light[1], 'kx', markersize=5)
