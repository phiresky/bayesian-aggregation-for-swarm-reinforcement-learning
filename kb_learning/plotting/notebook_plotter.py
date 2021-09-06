import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib import cm, animation
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

from itertools import cycle

from gym_kilobots.kb_plotting import get_body_from_shape

import plot_work

from kb_learning.learner import ACRepsLearner

from .iteration_results_plotters import *
from .plotting_helper import *

cmap_plasma = cm.get_cmap('plasma')
cmap_gray = cm.get_cmap('gray')


def reward_figure(R) -> plt.Figure:
    f = plt.figure()
    ax_R = f.add_subplot(111)
    ax_R.set_xlabel('time step')
    ax_R.set_ylabel('reward')
    reward_distribution_plot(R, ax_R)

    return f

    # out = widgets.Output()
    # with out:
    #     clear_output(wait=True)
    #     display(f)
    #
    # plt.close(f)
    #
    # return out


def value_function_figure(V, x_range, y_range, obj=None, S=None, cb_label=None) -> plt.Figure:
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('x')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('y')
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_ylabel('value')
    cax.yaxis.set_label_position('right')

    value_function_plot(V, x_range, y_range, axes=ax, cm_axes=cax, cmap=cmap_plasma, S=S, cb_label=cb_label)
    if obj is not None:
        obj.plot(ax, alpha=.5, fill=True, edgecolor=(0, 0, 0), linewidth=2)

    return f

    # out = widgets.Output()
    # with out:
    #     clear_output(wait=True)
    #     display(f)
    #
    # plt.close(f)
    #
    # return out


def trajectory_figure(T, x_range, y_range, V=None, obj=None, color=None, cb_label=None) -> plt.Figure:
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.xaxis.tick_top()
    ax.set_xlabel('x')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('y')
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if V is not None:
        value_function_plot(V, x_range, y_range, axes=ax, cmap=cmap_gray)
    trajectories_plot(T, x_range, y_range, ax, cb_axes=cax, cb_label=cb_label, color=color)
    if obj is not None:
        obj.plot(ax, alpha=.5, fill=True, edgecolor=(0, 0, 0), linewidth=2, zorder=20)

    return f

    # out = widgets.Output()
    # with out:
    #     clear_output(wait=True)
    #     display(f)
    #
    # plt.close(f)
    #
    # return out


def policy_figure(P, x_range, y_range, V=None, obj=None) -> plt.Figure:
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('x')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.set_ylabel('variance')
    cax.yaxis.set_label_position('right')

    if V is not None:
        value_function_plot(V, x_range, y_range, axes=ax, cmap=cmap_gray)
    policy_plot(P, x_range, y_range, ax, cm_axes=cax, cmap=cmap_plasma)
    if obj is not None:
        obj.plot(ax, alpha=.5, fill=True, edgecolor=(0,0,0), linewidth=2)

    return f

    # out = widgets.Output()
    # with out:
    #     clear_output(wait=True)
    #     display(f)
    #
    # plt.close(f)
    #
    # return out


@plot_work.register_iteration_plot_function('animate_trajectories')
def trajectory_animation_output(learner: ACRepsLearner, args):
    params = learner._params

    if args and 'samples' in args:
        num_episodes = params['sampling']['num_episodes']
        num_steps = params['sampling']['num_steps_per_episode']

        kb_T = learner.it_info[learner.kilobots_columns].values.reshape((num_episodes * num_steps, -1, 2))
        light_T = learner.it_info.S.loc[:, 'light'].values
        reward_T = learner.it_sars.R.values

        object_T = learner.it_info.S.loc[:, 'object'].values
    else:
        num_episodes = params['eval']['num_episodes']
        num_steps = params['eval']['num_steps_per_episode']

        kb_T = learner.eval_info[learner.kilobots_columns].values.reshape((num_episodes * num_steps, -1, 2))
        light_T = learner.eval_info.S.loc[:, 'light'].values
        reward_T = learner.eval_sars.R.values

        object_T = learner.eval_info.S.loc[:, 'object'].values

    from sklearn.gaussian_process.kernels import RBF
    print(learner.policy.kernel.kilobots_dist.bandwidth)
    kernel = learner.policy.kernel.variance[0] * RBF(
        length_scale=np.sqrt(learner.policy.kernel.kilobots_dist.bandwidth))
    kernel_l = learner.policy.kernel.variance[0] * RBF(length_scale=np.sqrt(learner.policy.kernel.light_dist.bandwidth))

    N = 30
    X, Y = np.meshgrid(np.linspace(-.4, .4, N), np.linspace(-.4, .4, N))
    XY = np.c_[X.flat, Y.flat]

    f = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.yaxis.tick_right()

    range_R = reward_T.min(), reward_T.max()

    out = widgets.Output()
    # with out:
    #     display(f)

    if args and 'mp4' in args:
        def update(i=0):
            # draw kilobot density
            K = kernel(XY, kb_T[i]).sum(axis=1).reshape(N, N) * 2 / kb_T.shape[1]
            K_l = kernel_l(XY, light_T[[i]]).reshape(N, N)

            obj = get_body_from_shape(object_shape=params['sampling']['object_shape'],
                                      object_width=params['sampling']['object_width'],
                                      object_height=params['sampling']['object_height'],
                                      object_init=object_T[i])

            ax.clear()
            c1 = ax.contourf(X, Y, K, cmap=cm.BuPu)
            c2 = ax.contour(X, Y, K_l, cmap=cm.YlGn_r, linewidths=.5)
            o = obj.plot(ax, alpha=.5, fill=True)
            s1 = ax.plot(kb_T[i, :, 0], kb_T[i, :, 1], '.', markersize=10)
            p1 = ax.add_patch(Circle(light_T[i], radius=.2, color=(0.4, 0.7, 0.3, 0.3), fill=False))
            m1 = ax.plot(light_T[i, 0], light_T[i, 1], 'kx', markersize=10)
            ax.set_xlim([-.4, .4])
            ax.set_ylim([-.4, .4])

            cax.clear()
            b1 = cax.bar(0, reward_T[i], color='r' if reward_T[i] < 0 else 'g')
            cax.set_xlim([-.5, .5])
            cax.set_ylim([*range_R])
            cax.set_xticks([])

            return tuple(c1.collections) + tuple(c2.collections)

        anim = animation.FuncAnimation(f, update, frames=kb_T.shape[0], interval=1000. / 25, blit=True)
        return anim.to_html5_video()
        # return anim

    else:
        def update(event):
            if isinstance(event, dict):
                i = event['new']
            else:
                i = event
            # draw kilobot density
            K = kernel(XY, kb_T[i]).sum(axis=1).reshape(N, N) * 2 / kb_T.shape[1]
            K_l = kernel_l(XY, light_T[[i]]).reshape(N, N)

            obj = get_body_from_shape(object_shape=params['sampling']['object_shape'],
                                      object_width=params['sampling']['object_width'],
                                      object_height=params['sampling']['object_height'],
                                      object_init=object_T[i])

            with out:
                ax.clear()
                ax.contourf(X, Y, K, cmap=cm.BuPu)
                ax.contour(X, Y, K_l, cmap=cm.YlGn_r, linewidths=.5)
                obj.plot(ax, alpha=.5, fill=True)
                ax.plot(kb_T[i, :, 0], kb_T[i, :, 1], '.', markersize=10)
                ax.add_patch(Circle(light_T[i], radius=.2, color=(0.4, 0.7, 0.3, 0.3), fill=False))
                ax.plot(light_T[i, 0], light_T[i, 1], 'kx', markersize=5)
                ax.set_xlim([-.4, .4])
                ax.set_ylim([-.4, .4])

                cax.clear()
                cax.bar(0, reward_T[i], color='r' if reward_T[i] < 0 else 'g')
                cax.set_xlim([-.5, .5])
                cax.set_ylim([*range_R])
                cax.set_xticks([])
                # f.canvas.draw()
                # f.show()
                # plt.show()

                clear_output(wait=True)
                display(f)

        update(0)

        play = widgets.Play(value=0, min=0, step=1, max=kb_T.shape[0])
        slider = widgets.IntSlider(value=0, min=0, step=1, max=kb_T.shape[0], continuous_update=False, disabled=False)
        widgets.jslink((play, 'value'), (slider, 'value'))
        play.observe(update, names='value')

        display(widgets.VBox(children=[out, widgets.HBox([play, slider])]))
        return


@plot_work.register_iteration_plot_function('fixed_weight')
def plot_fixed_weight_iteration(learner: ACRepsLearner, args=None):
    params = learner._params

    # def state_action_features(state, action):
    #     if state.ndim == 1:
    #         state = state.reshape((1, -1))
    #     if action.ndim == 1:
    #         action = action.reshape((1, -1))
    #     return learner.state_action_kernel(np.c_[state, action], learner.lstd_samples.values)

    x_range = (-.4, .4)
    y_range = (-.4, .4)

    obj = get_body_from_shape(object_shape=params['sampling']['object_shape'],
                              object_width=params['sampling']['object_width'],
                              object_height=params['sampling']['object_height'],
                              object_init=(.0, .0, .0))
    num_kilobots = params['sampling']['num_kilobots']
    light_type = params['sampling']['light_type']

    extra_dims = None
    if 'observe_object' in params['sampling'] and params['sampling']['observe_object']:
        if params['sampling']['observe_object'] == 'orientation':
            extra_dims = (np.sin(np.pi / 4), np.cos(np.pi / 4))
        elif params['sampling']['observe_object'] == 'position':
            extra_dims = (.0, .0)
        elif params['sampling']['observe_object'] == 'pose' or params['sampling']['observe_object'] is True:
            extra_dims = (.0, .0, .0, .1)

    if args and 'samples' in args:
        num_episodes = params['sampling']['num_episodes']
        num_steps = params['sampling']['num_steps_per_episode']

        if light_type == 'circular':
            T = learner.it_sars['S']['light'].values.reshape((num_episodes, num_steps, 2))
            S = learner.lstd_samples.S.light.values
        elif light_type == 'linear':
            T = learner.it_sars[learner.kilobots_columns].values.reshape(
                (num_episodes, num_steps, num_kilobots, 2)).mean(axis=2)
            S = learner.lstd_samples.S.values.reshape((-1, num_kilobots, 2)).mean(axis=1)
        R = learner.it_sars['R'].unstack(level=0).values.T
        O = learner.it_info['S']['object'][['x', 'y']].values.reshape((num_episodes, num_steps, 2))
    else:
        num_episodes = params['eval']['num_episodes']
        num_steps = params['eval']['num_steps_per_episode']

        if light_type == 'circular':
            T = learner.eval_sars['S']['light'].values.reshape((num_episodes, num_steps, 2))
            S = learner.lstd_samples.S.light.values
        elif light_type == 'linear':
            T = learner.eval_sars[learner.kilobots_columns].values.reshape(
                (num_episodes, num_steps, num_kilobots, 2)).mean(axis=2)
            S = learner.lstd_samples.S.values.reshape((-1, num_kilobots, 2)).mean(axis=1)
        R = learner.eval_sars['R'].unstack(level=0).values.T
        O = learner.eval_info['S']['object'][['x', 'y']].values.reshape((num_episodes, num_steps, 2))

    # V = compute_value_function_grid(state_action_features, learner.policy, learner.theta, num_kilobots=num_kilobots,
    #                                 x_range=x_range, y_range=y_range)
    # learner.state_action_kernel.extra_dim_bandwidth *= 1000
    # learner.policy.kernel.extra_dim_bandwidth *= 1000

    V = compute_value_function_grid(lambda s, a: learner.state_action_kernel(np.c_[s, a],
                                                                             learner.lstd_samples.values),
                                    learner.policy, learner.theta, num_kilobots=num_kilobots,
                                    x_range=x_range, y_range=y_range, extra_dims=extra_dims)
    P = compute_policy_quivers(learner.policy, num_kilobots, x_range, y_range, extra_dims=extra_dims,
                               resolution=20)

    # reward plot
    # R_out = reward_figure(R)
    # R_out.axes[0].set_title('Rewards')

    # value function plot
    V_out = value_function_figure(V, x_range, y_range, obj=obj, S=S, cb_label='value')

    # light/swarm trajectories plot
    T_out = trajectory_figure(T, x_range, y_range, V=V, obj=obj, color=R, cb_label='reward')

    # object trajectories plot
    O_out = trajectory_figure(O, x_range, y_range, color=R, cb_label='reward')

    # new policy plot
    P_out = policy_figure(P, x_range, y_range, V=V, obj=obj)

    return V_out, T_out, P_out, O_out

    # box = widgets.VBox(children=[R_out, V_out, T_out, P_out])
    #
    # # save and show plot
    # return box


@plot_work.register_file_provider('ppo_policy_file')
def provide_ppo_policy_file(learner: ACRepsLearner, config: dict, args=None):
    from plot_work import DownloadFile
    prefix = config['name']
    model_path = os.path.join(learner._log_path_rep, 'make_model.pkl')
    model_link_text = 'make_model'
    model_file_name = prefix + '_make_model.pkl'
    parameter_path = os.path.join(learner._log_path_it, 'model_parameters')
    parameter_link_text = 'model_parameters'
    parameter_file_name = prefix + '_model_parameters_rep{:02d}_it{:02d}.pkl'.format(learner._rep, learner._it)
    return [DownloadFile(path=model_path, link_text=model_link_text, file_name=model_file_name),
            DownloadFile(path=parameter_path, link_text=parameter_link_text, file_name=parameter_file_name)]


@plot_work.register_file_provider('trpo_policy_file')
def provide_trpo_policy_file(learner: ACRepsLearner, config: dict, args=None):
    from plot_work import DownloadFile
    prefix = config['name']
    policy_path = os.path.join(learner._log_path_it, 'policy.pkl')
    policy_link_text = 'policy_rep00.pkl'
    policy_file_name = prefix + '_policy.pkl'
    return DownloadFile(path=policy_path, link_text=policy_link_text, file_name=policy_file_name)
