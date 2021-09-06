"""
stable baselines SubprocVecEnv but allows wrapping a multi-agent "vec-env"
"""
import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    if "terminal_observation" in info:
                        print(
                            "warning: auto resetting twice? wrong observation returned"
                        )
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space, env.num_envs))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class FlatteningSubprocVecEnv(VecEnv):
    """
    like stable-baselines3 SubprocVecEnv, but flattens the observations from each inner vecenv
    useful because each multi-agent environment is a vecenv so we need to flatten multiple vecenvs
    """

    def __init__(
        self, env_fns: List[Callable[[], VecEnv]], start_method: Optional[str] = None
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns)
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker,
                args=args,
                daemon=True,
                name=f"VecEnv {i}/{len(env_fns)}",
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space, num_inner_envs_per_env = self.remotes[0].recv()
        self.num_inner_envs_per_env = num_inner_envs_per_env
        VecEnv.__init__(
            self, len(env_fns) * num_inner_envs_per_env, observation_space, action_space
        )

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(
            self.remotes,
            _unflatten_actions(
                actions,
                len(self.remotes),
                self.num_inner_envs_per_env,
            ),
        ):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        remotes, epe = len(self.remotes), self.num_inner_envs_per_env
        return (
            _flatten_obs(obs, self.observation_space),
            _flatten_rews(rews, remotes, epe),
            _flatten_dones(dones, remotes, epe),
            _flatten_infos(infos, remotes, epe),
        )

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(
    obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space
) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.
    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict
        ), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        # concatenate instead of stack to merge the first and second dimension
        return np.concatenate(obs, axis=0)


def _unflatten_actions(actions: np.ndarray, remotes_count: int, envs_per_env: int):
    first_dim, *dims = actions.shape
    assert first_dim == remotes_count * envs_per_env
    return actions.reshape((remotes_count, envs_per_env, *dims))


def _flatten_rews(rews: Tuple[np.ndarray], remotes_count: int, envs_per_env: int):
    assert len(rews) == remotes_count
    out = np.asarray(rews).flatten()
    assert len(out) == remotes_count * envs_per_env
    return out


def _flatten_dones(rews: Tuple[bool], remotes_count: int, envs_per_env: int):
    """returns only one done per env, so duplicate that info"""
    assert len(rews) == remotes_count
    return (
        np.asarray(rews)
        .reshape(remotes_count, 1)
        .repeat(envs_per_env, axis=1)
        .reshape(remotes_count * envs_per_env)
    )


def _flatten_infos(infos: Tuple[Dict[str, Any]], remotes_count: int, envs_per_env: int):
    """returns only one info per env, so duplicate that info"""
    assert len(infos) == remotes_count
    return [info for info in infos for i in range(envs_per_env)]


class FlatteningDummyVecWrapper(VecEnvWrapper):
    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step_wait(self):
        observation, reward, done, info = self.venv.step_wait()
        epe = self.venv.num_envs
        return (
            observation,
            _flatten_rews([reward], 1, epe),
            _flatten_dones([done], 1, epe),
            _flatten_infos([info], 1, epe),
        )


# needed because of https://github.com/DLR-RM/stable-baselines3/issues/426
class NothingWrapper(VecEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.venv, "is_actually_a_single_env", False):
            self.is_actually_a_single_env = True

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step_wait(self):
        return self.venv.step_wait()


class AutoResetWrapper(VecEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.venv, "is_actually_a_single_env", False):
            self.is_actually_a_single_env = True

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        if done:
            if "terminal_observation" in info:
                print("warning: auto resetting twice? wrong observation returned")
            info["terminal_observation"] = obs
            obs = self.reset()
        return obs, rew, done, info
