import datetime
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional, cast

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from tap import Tap

from playground.default_params import Task, get_default_params
from playground.inference import render_ep_random
from playground.trl.notppo import NotPPO

from . import serde
from .EvalCallback import EvalCallback
from .logging_util import setup_logging
from .ma_envs.cool_subproc_vec_env import (
    AutoResetWrapper,
    FlatteningDummyVecWrapper,
    FlatteningSubprocVecEnv,
    NothingWrapper,
)
from .params import FullParams
from .policy.embed import EmbeddingParamsRuntime
from .policy.MlpAggregatingPolicy import MlpAggregatingPolicy, PolicyParamsRuntime


@contextmanager
def timed(msg: str):
    print(msg)
    tstart = time.perf_counter()
    yield
    print("done in %.3f seconds" % (time.perf_counter() - tstart))


class CmdArgs(Tap):
    runname: str
    task: Optional[str] = None
    algo: Literal["ppo", "trl", "trl_kl"] = "ppo"
    agg: str = "bayes"  # default_params agg_*
    show_env: bool = False
    params_from: Optional[str] = None
    weights_from: Optional[str] = None
    single_threaded: bool = False


def runname_with_time(runname: str):
    runtime = datetime.datetime.now().isoformat("_", "seconds").replace(":", ".")
    return f"{runtime}-{runname}"


class LogTimingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_start_time = None
        self.train_start_time = None

    def _on_rollout_start(self) -> None:
        import torch

        print(torch.cuda.memory_summary(abbreviated=True))
        self.rollout_start_time = time.perf_counter()
        if self.train_start_time is not None:
            print(f"Train/rest took {time.perf_counter() - self.train_start_time:.2f}s")

    def _on_rollout_end(self) -> None:
        print(
            f"Collecting rollouts took {time.perf_counter() - self.rollout_start_time:.2f}s"
        )
        self.rollout_start_time = None
        self.train_start_time = time.perf_counter()

    def _on_step(self):
        return True


def train(
    params: FullParams,
    eval_result_callback: Optional[Callable[[int, float], None]] = None,
    weights_from: Optional[str] = None,
    single_threaded=False,
):
    runname = runname_with_time(params.runname)
    run_dir_orig = Path("runs") / runname
    run_dir = run_dir_orig
    _suffix = 1
    while run_dir.exists():
        _suffix += 1
        run_dir = run_dir_orig.parent / (run_dir_orig.name + f"-{_suffix}")
    run_dir.mkdir(parents=True)
    setup_logging(run_dir)

    env_params = params.env_params
    # env = env_params.create_env()

    if single_threaded:
        env = env_params.create_env()
        full_obs_space = env.unflattened_observation_space
        env = FlatteningDummyVecWrapper(AutoResetWrapper(env))
        eval_env = AutoResetWrapper(NothingWrapper(env_params.create_env()))
    else:
        env = FlatteningSubprocVecEnv(
            [lambda: env_params.create_env() for i in range(params.nr_parallel_envs)]
        )
        eval_env = FlatteningSubprocVecEnv(
            [lambda: env_params.create_env() for i in range(params.nr_parallel_envs)]
        )
        full_obs_space = env.get_attr("unflattened_observation_space", indices=0)[0]

    (run_dir / "full_params.json").write_text(serde.serialize(params))

    it_steps = int(
        params.env_steps_per_train_step / env_params.nr_agents / params.nr_parallel_envs
    )
    total_steps = params.env_steps_per_train_step * params.train_steps
    print(
        f"{params.train_steps=} PPO steps, {params.env_steps_per_train_step=} agent steps each ({it_steps=}, {total_steps=})"
    )
    print(f"using params {params}")
    ppr = PolicyParamsRuntime(
        config=params.policy_params,
        emb_params=EmbeddingParamsRuntime(
            config=params.emb_params,
            full_obs_space=full_obs_space,
        ),
    )
    algo = params.training_algorithm
    if algo == "ppo":
        model = PPO(
            env=env,
            policy=MlpAggregatingPolicy,
            policy_kwargs=dict(params=ppr),
            verbose=1,
            device="cuda",
            n_steps=it_steps,
            tensorboard_log=str(run_dir),
            batch_size=params.batch_size,
        )
    elif algo == "a2c":
        model = A2C(
            env=env,
            policy=MlpAggregatingPolicy,
            policy_kwargs=dict(
                params=ppr,
                optimizer_class=RMSpropTFLike,
                optimizer_kwargs=dict(eps=1e-5),
            ),
            verbose=1,
            device="cuda",
            n_steps=it_steps,
            tensorboard_log=str(run_dir),
        )
    elif algo == "trl":
        import torch as th

        from .trl.projections.projection_factory import get_projection_layer

        proj_params = dict(
            proj_type="w2",
            mean_bound=0.03,
            cov_bound=0.001,
            trust_region_coeff=8.0,
            scale_prec=True,
            entropy_schedule=False,
            action_dim=env.action_space.shape[0],
            total_train_steps=-1,
            target_entropy=-1.0,
            temperature=0.5,
            entropy_eq=False,
            entropy_first=False,
            do_regression=False,
            regression_iters=5,
            regression_lr=3e-4,
            optimizer_type_reg="adam",
            cpu=False,
            dtype=th.float32,
        )
        proj_params = {**proj_params, **params.training_algorithm_params}
        projection = get_projection_layer(**proj_params)

        model = NotPPO(
            env=env,
            policy=MlpAggregatingPolicy,
            policy_kwargs=dict(params=ppr),
            verbose=1,
            device="cuda",
            n_steps=it_steps,
            tensorboard_log=str(run_dir),
            batch_size=params.batch_size,
            clip_range=0,
            projection=projection,
        )
    elif algo == "SAC":
        raise Exception(":o")
        model = SAC(
            env=env,
            policy=MlpSacAggregatingPolicy,
            policy_kwargs=dict(params=ppr),
            verbose=1,
            device="cuda",
            train_freq=it_steps,
            tensorboard_log=str(run_dir),
            batch_size=params.batch_size,
        )
    else:
        raise Exception(f"unknown training algorithm {algo}")

    if weights_from:
        import zipfile
        from os.path import relpath

        import torch as th

        print(f"setting parameters from weights file {weights_from}")
        (run_dir / "weights_from").symlink_to(
            relpath(Path(weights_from), run_dir), target_is_directory=True
        )

        with zipfile.ZipFile(weights_from) as archive:
            with archive.open("policy.pth", mode="r") as f:
                policy_params = th.load(f)
            with archive.open("policy.optimizer.pth", mode="r") as f:
                optimizer_params = th.load(f)
        model.set_parameters(
            {
                "policy": policy_params,
                # "policy.optimizer": optimizer_params
            },
            exact_match=False,
        )
    # with allow_empty():
    #    dim = gym.spaces.flatdim(ppr.emb_params.full_obs_space)
    #    print("summary", summary(model.policy, input_size=(dim,)))

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=params.nr_parallel_envs,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=it_steps,
        deterministic=True,
        render=False,
        verbose=True,
        on_eval_result=eval_result_callback,
    )
    log_timing_callback = LogTimingCallback()
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([log_timing_callback, eval_callback]),
    )
    return model


def main():
    args = CmdArgs().parse_args()

    if args.params_from:
        params = cast(
            FullParams,
            serde.deserialize(Path(args.params_from).read_text(), FullParams),
        )
    elif args.task:
        params = get_default_params(args.task, args.agg, args.algo)
    else:
        raise Exception("either task or params_from must be given")

    if args.show_env:
        env = params.env_params.create_env()
        render_ep_random(env)
        return

    params = params._replace(runname=args.runname)

    model = train(
        params, weights_from=args.weights_from, single_threaded=args.single_threaded
    )


if __name__ == "__main__":
    main()
