import time
from pathlib import Path
from typing import Optional

from gym_kilobots.lib import body
from stable_baselines3 import PPO
from tap import Tap

from . import serde
from .params import EnvParams, FullParams, PettingZooEnvParams
from .policy.embed import EmbeddingParamsRuntime
from .policy.MlpAggregatingPolicy import MlpAggregatingPolicy, PolicyParamsRuntime

body._world_scale = 25.0


def render_ep_random(env):
    obs = env.reset()
    for i in range(10000):
        action = [env.action_space.sample() for _ in range(env.nr_agents)]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            return


def render_plot(timestep, plt, output_dir=Path("/tmp/foobar")):
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / f"{timestep:04d}")


def render_ep_to_file(
    env, model, max_step=1024, output_dir=Path("/tmp/foobar"), module=10
):
    import multiprocessing
    import os
    import shutil

    from tqdm import tqdm, trange

    for f in output_dir.glob("*"):
        f.unlink()
    obs = env.reset()
    todos = []
    for i in trange(max_step, desc="stepping"):
        if i % module == 0:
            # todos.append((env.timestep, env.get_plot(), output_dir))
            if i == 0:
                env.get_plot(
                    reuse=True
                )  # double render first frame, otherwise pursuit renders weirdly
            render_plot(env.timestep, env.get_plot(reuse=True), output_dir)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            break
    pool = multiprocessing.Pool()

    pool.starmap(render_plot, tqdm(todos, desc="rendering"))

    # os.system(
    #    f"ffmpeg -r 24 -i '{output_dir}/%04d.png' -c:v libx264 -pix_fmt yuv420p -y /tmp/rz_out.mp4"
    # )
    # os.system(
    #    f"ffmpeg -r 24 -i '{output_dir}/%04d.png' -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -y /tmp/rz_out.webm"
    # )


def render_ep_to_file_o(env, model):
    obs = env.reset()
    env.render(mode="animate")
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode="animate")
        if done:
            return


def render_ep(env, model, sleep_s=0, skip_every=1, **kwargs):
    obs = env.reset()
    for i in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print("step", i, "rew", reward[0])
        if i % skip_every == 0:
            env.render(**kwargs)

        time.sleep(sleep_s)
        # print(f"step {i}, done={done}")
        if done:
            return


def legacy_conversions(params):
    import re

    paramsout = {}

    mappings = [
        (r"(.*)\.neighbor_embedding\.0\.(.*)", r"\1.neighbor_embedding.nn.0.\2"),
    ]
    for param in params:
        print("have param", param)
        for regex, rep in mappings:
            param2 = re.sub(regex, rep, param)
            if param2 != param:
                print("migrating", param, "to", param2)
            paramsout[param2] = params[param]

    return paramsout


def main():
    interesting = dict(
        # mean_agg="2020-12-16_12.40.01-mean-agg-longrun",
        # mean_agg="2021-01-21_15.54.49-mean",
        # bayesian="2021-01-21_16.03.26-bayesian",  # "2021-01-21_17.16.39-bayesian-vf_pi_unshared-bayesianshared"
        # bayesian="2020-12-16_22.32.56-bayesian",
        # attention_relative="2021-01-07_12.21.52-attention-2",
        # attention_uniform="2021-01-12_19.29.33-are-64+later-layer",
    )

    # chosen = "2021-01-21_18.02.45-are-uniform-unshared"
    #  chosen = "2021-01-28_20.13.50-debug"

    class Args(Tap):
        run: str
        cpu: bool = False
        render: Optional[str] = None

    args = Args().parse_args()
    rundir = Path(args.run)
    modelfile = rundir / "best_model"

    print("loading model from", modelfile)

    params = serde.deserialize((rundir / "full_params.json").read_text(), FullParams)
    # model = PPO.load(modelfile)

    # obs = env.reset

    env_params = params.env_params

    if isinstance(env_params, EnvParams.KilobotsNew):
        max_res = 1200 / env_params.env.width
        if env_params.env.resolution > max_res:
            env_params.env.resolution = max_res

    # runs = [
    # "runs/2021-02-04_18.06.02-simple-nocoll-3agents",
    # "runs/2021-02-04_19.08.10-simple-nocoll-3agents-sigmaz0",
    # ]
    # env = CoolSubprocVecEnv(
    #     [
    #         serde.deserialize(
    #             (Path(rundir) / "full_params.json").read_text(), FullParams
    #         ).env_params.create_env
    #         for rundir in runs
    #     ]
    # )

    env = env_params.create_env()

    model = PPO(
        env=env,
        policy=MlpAggregatingPolicy,
        policy_kwargs=dict(
            params=PolicyParamsRuntime(
                config=params.policy_params,
                emb_params=EmbeddingParamsRuntime(
                    config=params.emb_params,
                    full_obs_space=env.unflattened_observation_space,
                ),
            )
        ),
        verbose=1,
        device="cuda",
    )

    import zipfile

    import torch as th

    device = dict(map_location=th.device("cpu")) if args.cpu else dict()

    with zipfile.ZipFile(modelfile.with_suffix(".zip")) as archive:
        with archive.open("policy.pth", mode="r") as f:
            policy_params = th.load(f, **device)
        with archive.open("policy.optimizer.pth", mode="r") as f:
            optimizer_params = th.load(f, **device)
    policy_params = legacy_conversions(policy_params)
    model.set_parameters(
        {"policy": policy_params, "policy.optimizer": optimizer_params}
    )
    # model.get_parameters()
    if args.render:
        render_ep_to_file(env, model, output_dir=Path(args.render))
    else:
        for i in range(0, 10):
            print(f"ep {i}")
            if isinstance(env_params, PettingZooEnvParams):
                render_ep(env, model, sleep_s=0.1, skip_every=1)
            else:
                render_ep(env, model)


if __name__ == "__main__":
    main()
"""
import typedload
import typedload.datadumper

dpd = typedload.datadumper.Dumper(hidedefault=False).dump(params)
print(dpd)


print(typedload.load(dpd, FullParams))
# def isspace(t):
#    return isinstance(t, gym.Space)


# dumper = typedload.datadumper.Dumper()
# dumper.handlers.append((isspace, lambda l, v: v.to_jsonable()))

# dumper.dump(params)
"""
