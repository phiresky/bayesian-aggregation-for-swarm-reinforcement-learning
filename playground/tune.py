"""tune hyperparameters with optuna"""

import math

import numpy as np
import optuna
from tap import Tap

from playground.ma_envs.point_envs.pursuit import PursuitEvasionAgentParams
from playground.ma_envs.point_envs.rendezvous import ObsMode

from . import serde
from .params import EnvParams, FullParams
from .policy.embed import EmbeddingParams
from .policy.MlpAggregatingPolicy import PolicyParams
from .train import train


class TuneArgs(Tap):
    db: str
    task: str
    fixed: str
    study_name: str
    runname_suffix: str
    tune_batchsize: bool = False


fixed = {
    "none": {},
    "mean": {
        "emb_params.agg_mode": "Mean",
        "policy_params.share_vf_pi_params": False,
    },
    "bayes": {
        "emb_params.agg_mode": "Bayesian",
        "emb_params.agg_mode.separate": False,
        "emb_params.agg_mode.learnable_sigma_z_0": "one_per_dim",
        "emb_params.agg_mode.learnable_mu_z_0": "one_per_dim",
        "emb_params.agg_mode.exp_sigma_z_0": False,
        "emb_params.agg_mode.variance_rectifier": "softplus",
        "policy_params.share_vf_pi_params": False,
    },
    "self_attention": {
        "emb_params.agg_mode": "MultiHeadAttention",
        "policy_params.share_vf_pi_params": False,
    },
}


def current_task_params(task: str):
    from .default_params import get_default_params

    params = get_default_params(task, agg="mean", algo="ppo")
    return params


def get_optuna_storage(fname: str):
    import sqlite3

    return optuna.storages.RDBStorage(
        url="sqlite:////",
        engine_kwargs=dict(
            creator=lambda: sqlite3.connect(
                # special locking to make it not depend on working NFS locking
                # f"file:{fname}?vfs=unix-dotfile",
                f"file:{fname}",
                uri=True,
                # wait up to 60 seconds when other process is writing to db before throwing "database is locked"
                timeout=60,
            )
        ),
    )


def main():
    args = TuneArgs().parse_args()

    study = optuna.create_study(
        storage=get_optuna_storage(fname=args.db),
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),  # optuna.pruners.MedianPruner(n_warmup_steps=),
    )
    fixes = fixed[args.fixed]
    study.sampler = optuna.samplers.PartialFixedSampler(fixes, study.sampler)
    if "default_full_params" in study.user_attrs:
        task_params: FullParams = serde.from_dict(
            study.user_attrs["default_full_params"], FullParams
        )
        print(f"used existing env params from study: {task_params}")
    else:
        task_params = current_task_params(args.task)
        study.set_user_attr("default_full_params", serde.to_dict(task_params))

        print(f"saved env params to new study: {task_params}")

    def objective(trial: optuna.Trial):
        steps = []
        results = []

        def get_value():
            # average results over time since we want to optimize how quickly it learns
            return np.mean(results)

        def eval_result(step: int, result: float):
            steps.append(step)
            results.append(result)
            value = np.mean(results)
            print(f"reporting to optuna {step=}, {result=}, {value=}")
            # if len(results) > 10:
            trial.report(step=step, value=value)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if args.tune_batchsize:
            # batchsize = trial.suggest_int("batch_size", 512, 512 * 10 * 2, 512)
            batch_size = int(trial.suggest_loguniform("batch_size", 512, 10240))
        else:
            batch_size = task_params.batch_size
        params = FullParams(
            env_params=task_params.env_params,
            policy_params=PolicyParams.choose_optuna("policy_params", trial),
            emb_params=EmbeddingParams.choose_optuna("emb_params", trial),
            nr_parallel_envs=task_params.nr_parallel_envs,
            runname=f"opt-{args.study_name}-t{trial.number}-r{args.runname_suffix}",
            env_steps_per_train_step=task_params.env_steps_per_train_step,
            train_steps=task_params.train_steps,
            batch_size=batch_size,
        )
        trial.set_user_attr("full_params", serde.to_dict(params))
        train(params, eval_result_callback=eval_result)
        value = get_value()
        print(
            f"training complete. last step={steps[-1]}, last result={results[-1]}, {value=}"
        )
        return value

    study.optimize(objective, n_trials=1)


if __name__ == "__main__":
    main()
