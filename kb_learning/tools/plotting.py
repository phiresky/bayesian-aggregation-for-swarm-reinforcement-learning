import os

import matplotlib
import numpy as np
import pandas as pd
from gym_kilobots.envs import KilobotsEnv
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# matplotlib.rc('font', family='Oswald')


class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''

    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                          'Stdout: {1} StdError: {2}. It may help to re-run '
                          'with --verbose-debug.'.format(e, out, err))


def plot_trajectories(axes: Axes, trajectories: pd.DataFrame):
    # light_states.index.shape
    num_episodes, num_steps = trajectories.index.levshape

    line_collections = []

    for _, traj in trajectories.groupby(level=0):
        segments = np.r_['2,3,0', traj[:-1], traj[1:]].swapaxes(1, 2)
        # segments[:, :, 1] *= -1
        lc = LineCollection(segments, cmap='viridis', norm=Normalize(0, num_steps))
        color = np.arange(num_steps)
        lc.set_array(color)
        lc.set_linewidth(1.)
        line_collections.append(lc)

        axes.add_collection(lc)

    return line_collections


def plot_value_function(axes: Axes, V, x_range, y_range, **kwargs):
    im = axes.matshow(V, extent=x_range+y_range, **kwargs, norm=Normalize(-0.4, 0.4))
    return im


def plot_policy(axes: Axes, actions, x_range, y_range, **kwargs):
    if type(actions) is tuple:
        A, sigma = actions
        # color = (sigma - sigma.min()) / (sigma.max() - sigma.min()) / 2 + .5
        color = sigma
    else:
        A = actions
        color = 1.

    [X, Y] = np.meshgrid(np.linspace(*x_range, A.shape[1]), np.linspace(*y_range, A.shape[0]))
    if A.shape[2] == 2:
        return axes.quiver(X, Y, A[..., 0], A[..., 1], color, angles='xy', **kwargs)
    else:
        A_cos = np.cos(A.squeeze())
        A_sin = np.sin(A.squeeze())
        return axes.quiver(X, Y, A_cos, A_sin, color, angles='xy', **kwargs)


def plot_objects(axes: Axes, env: KilobotsEnv, alpha=.7, **kwargs):
    for o in env.get_objects():
        o.plot(axes, alpha=alpha, **kwargs)


def plot_trajectory_reward_distribution(axes: Axes, reward: pd.DataFrame):
    mean_reward = reward.groupby(level=1).mean()
    std_reward = reward.groupby(level=1).std()

    x = np.arange(mean_reward.shape[0])

    axes.plot(x, reward.unstack(level=0), 'k-', alpha=.2)
    axes.fill_between(x, mean_reward - 2 * std_reward, mean_reward + 2 * std_reward, alpha=.5)
    axes.plot(x, mean_reward)


# def save_plot_as_html(figure, filename=None, path=None, overwrite=True):
#     if path is None:
#         import tempfile
#         path = tempfile.gettempdir()
#     if filename is None:
#         filename = 'plot.html'
#
#     html_full_path = os.path.join(path, filename)
#
#     if overwrite and os.path.exists(html_full_path):
#         os.remove(html_full_path)
#     elif os.path.exists(html_full_path):
#         root, ext = os.path.splitext(html_full_path)
#         root_i = root + '_{}'
#         i = 1
#         while os.path.exists(html_full_path):
#             html_full_path = root_i.format(i) + ext
#             i = i + 1
#
#     html_data = mpld3.fig_to_html(figure)
#
#     with open(html_full_path, mode='w') as html_file:
#         # mpld3.save_html(figure, html_file)
#         html_file.write(html_data)
#
#     return html_full_path

browser_controller = None


# def show_plot_in_browser(figure, filename=None, path=None, overwrite=True, browser='google-chrome',
#                          save_only=False):
#     html_full_path = save_plot_as_html(figure, filename, path, overwrite)
#
#     if save_only:
#         return
#
#     global browser_controller
#     if browser_controller is None:
#         import webbrowser
#         browser_controller = webbrowser.get(browser)
#
#     browser_controller.open(html_full_path)


def save_plot_as_pdf(figure, filename=None, path=None, overwrite=True):
    if path is None:
        import tempfile
        path = tempfile.gettempdir()
    if filename is None:
        filename = 'plot.pdf'

    pdf_full_path = os.path.join(path, filename)

    if overwrite and os.path.exists(pdf_full_path):
        os.remove(pdf_full_path)
    elif os.path.exists(pdf_full_path):
        root, ext = os.path.splitext(pdf_full_path)
        root_i = root + '_{}'
        i = 1
        while os.path.exists(pdf_full_path):
            pdf_full_path = root_i.format(i) + ext
            i = i + 1

    figure.savefig(pdf_full_path, bbox_inches='tight')

    return pdf_full_path


def show_plot_as_pdf(figure, filename=None, path=None, overwrite=True, save_only=False, browser='google-chrome'):
    pdf_full_path = save_plot_as_pdf(figure, filename, path, overwrite)

    if save_only:
        return

    global browser_controller
    if browser_controller is None:
        import webbrowser
        browser_controller = webbrowser.get(browser)

    browser_controller.open(pdf_full_path)
