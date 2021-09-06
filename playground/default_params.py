import math
from typing import Literal

from kb_learning.envs._multi_object_env import MODCEConfig

from playground.ma_envs.point_envs.pursuit import PursuitEvasionAgentParams
from playground.policy.MlpAggregatingPolicy import PolicyParams

from .ma_envs.point_envs.rendezvous import RendezvousAgentParams
from .params import EnvParams, FullParams
from .policy.embed import AggMode, EmbeddingParams

"""
sets of default params for the diffferent agg methods and tasks
"""

nr_parallel_envs = 10

Task = Literal["clustering", "assembly", "rendezvous", "pursuit"]

bayesian_config = AggMode.Bayesian(
    separate=False,
    learnable_sigma_z_0="one_per_dim",
    learnable_mu_z_0="one_per_dim",
    exp_sigma_z_0=True,
    variance_rectifier="softplus",
)


old_bayesian_emb = EmbeddingParams(
    # agg_mode=AggMode.Mean(),
    agg_mode=bayesian_config,
    latent_embedding_dim=64,
    latent_dim=64,
    obs_mode="relative",
    activation_fn="ReLU",
)

emb_mean = EmbeddingParams(
    agg_mode=AggMode.Mean(),
    latent_embedding_dim=60,
    latent_embedding_hidden_layer_sizes=[120],  # [50, 100],
    latent_dim=160,
    latent_out_hidden_layer_sizes=[],
    obs_mode="relative",
    activation_fn="LeakyReLU",
)
emb_max = emb_mean._replace(agg_mode=AggMode.Mean(actual_mode="max"))

# optuna-pursuit-bayes-500steps-countcatches trial 6, 17
emb_bayes = EmbeddingParams(
    agg_mode=bayesian_config,
    latent_embedding_dim=60,
    latent_embedding_hidden_layer_sizes=[120],  # [50, 100],
    latent_dim=160,
    latent_out_hidden_layer_sizes=[],
    obs_mode="relative",
    activation_fn="LeakyReLU",
)
emb_bayes_sepvar = emb_bayes._replace(agg_mode=bayesian_config._replace(separate=True))
emb_bayes_samespace = emb_bayes._replace(agg_into_space="same")
emb_bayes_outputvariance = emb_bayes._replace(
    agg_mode=bayesian_config._replace(output_variance=True)
)

emb_bayes_simple = EmbeddingParams(
    agg_mode=bayesian_config,
    latent_embedding_dim=64,
    latent_embedding_hidden_layer_sizes=[],  # [50, 100],
    latent_dim=64,
    latent_out_hidden_layer_sizes=[],
    obs_mode="relative",
    activation_fn="LeakyReLU",
)
emb_bayes_simple_samespace = emb_bayes_simple._replace(agg_into_space="same")
emb_bayes_rendezopt = EmbeddingParams(
    agg_mode=bayesian_config,
    latent_embedding_dim=120,
    latent_embedding_hidden_layer_sizes=[146],
    latent_out_hidden_layer_sizes=[19, 177],
    latent_dim=162,
    obs_mode="relative",
    activation_fn="Tanh",
)
emb_mean_rendezopt = EmbeddingParams(
    agg_mode=AggMode.Mean(),
    latent_embedding_dim=120,
    latent_embedding_hidden_layer_sizes=[146],
    latent_out_hidden_layer_sizes=[19, 177],
    latent_dim=162,
    obs_mode="relative",
    activation_fn="Tanh",
)
emb_mean_pursuitopt = EmbeddingParams(
    agg_mode=AggMode.Mean(),
    latent_embedding_dim=96,
    latent_embedding_hidden_layer_sizes=[174, 226],
    latent_out_hidden_layer_sizes=[],
    latent_dim=97,
    obs_mode="relative",
    activation_fn="LeakyReLU",
)

emb_bayes_randomforclus = EmbeddingParams(
    agg_mode=bayesian_config,
    latent_embedding_dim=120,
    latent_embedding_hidden_layer_sizes=[146],
    latent_out_hidden_layer_sizes=[60, 177],
    latent_dim=162,
    obs_mode="relative",
    activation_fn="LeakyReLU",
)

emb_mean_randomforclus = EmbeddingParams(
    agg_mode=AggMode.Mean(),
    latent_embedding_dim=120,
    latent_embedding_hidden_layer_sizes=[146],
    latent_out_hidden_layer_sizes=[60, 177],
    latent_dim=162,
    obs_mode="relative",
    activation_fn="Tanh",
)

emb_mean_simple = EmbeddingParams(
    agg_mode=AggMode.Mean(),
    latent_embedding_dim=64,
    latent_embedding_hidden_layer_sizes=[],  # [50, 100],
    latent_dim=64,
    latent_out_hidden_layer_sizes=[],
    obs_mode="relative",
    activation_fn="LeakyReLU",
)


# from optuna-pursuit-selfatt trial 21 and 91
emb_selfatt = EmbeddingParams(
    agg_mode=AggMode.MultiHeadAttention(num_heads=2, residual=True),
    latent_embedding_hidden_layer_sizes=[],
    latent_embedding_dim=72,
    latent_dim=200,
    latent_out_hidden_layer_sizes=[132],
    obs_mode="relative",
    activation_fn="LeakyReLU",
    agg_into_space="separate",
)


def task_rendezvous():
    obs_mode = "relative"
    env_params = EnvParams.Rendezvous(
        torus=False,
        nr_agents=20,
        comm_radius=100 * math.sqrt(2),
        agent_params=RendezvousAgentParams(
            dynamics="unicycle_acc",
            obs_mode=obs_mode,
            add_walls_to_obs=False,
        ),
    )

    full_params = FullParams(
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=164000,
        train_steps=160,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=1000,
    )
    return full_params


def task_rendezvous_notorus():
    params = task_rendezvous()
    params = params._replace(env_params=params.env_params._replace(torus=False))
    return params


def task_singlepursuit():
    obs_mode = "relative"
    worldsize = 100
    env_params = EnvParams.PursuitEvasion(
        agent_params=PursuitEvasionAgentParams(dynamics="unicycle", obs_mode=obs_mode),
        reward_mode="min_distance",
        world_size=worldsize,
        comm_radius=worldsize * math.sqrt(2),
        obs_radius=worldsize * math.sqrt(2),
        nr_evaders=1,
        nr_agents=10,
        torus=True,
    )
    full_params = FullParams(
        training_algorithm="ppo",
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=1020 * 100,
        train_steps=500,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=10200,
    )
    return full_params


# multi evader pursuit
def task_pursuit():
    obs_mode = "relative"
    worldsize = 100
    env_params = EnvParams.PursuitEvasion(
        agent_params=PursuitEvasionAgentParams(dynamics="unicycle", obs_mode=obs_mode),
        reward_mode="count_catches",
        world_size=worldsize,
        comm_radius=worldsize * math.sqrt(2),
        obs_radius=worldsize * math.sqrt(2),
        nr_evaders=5,
        nr_agents=50,
        torus=True,
    )
    full_params = FullParams(
        training_algorithm="ppo",
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=1020 * 100,
        train_steps=500,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=10200,
    )
    return full_params


def task_pursuit_smallbatch():
    params = task_pursuit()
    params = params._replace(batch_size=1020)
    return params


def env_clustering2():
    from gym_kilobots.envs.yaml_kilobots_env import EnvConfiguration as Kb

    return EnvParams.KilobotsNew(
        env=Kb(
            width=1.0,
            height=1.0,
            resolution=600,
            objects=[
                dict(
                    idx=0,
                    shape="square",
                    width=0.1,
                    height=0.1,
                    init="random",
                    color=None,
                    symmetry=None,
                )
                for _ in range(4)
            ],
            kilobots=dict(mean="random", std=0.03, num=10),
            light=None,
        ),
        aggregate_clusters_separately=True,
        m_config=MODCEConfig(num_cluster=2, observe_abs_box_pos=True),
        reward_function="object_clustering_amp",
        agent_type="SimpleVelocityControlKilobot",
        swarm_reward=True,
        agent_reward=True,
        done_after_steps=1024,
    )


def env_clustering3(*, boxes=12, bots=20, size=2.0):
    from gym_kilobots.envs.yaml_kilobots_env import EnvConfiguration as Kb

    return EnvParams.KilobotsNew(
        env=Kb(
            width=size,
            height=size,
            resolution=300,
            objects=[
                dict(
                    idx=0,
                    shape="square",
                    width=0.1,
                    height=0.1,
                    init="random",
                    color=None,
                    symmetry=None,
                )
                for _ in range(boxes)
            ],
            kilobots=dict(mean="random", std=0.03, num=bots),
            light=None,
        ),
        aggregate_clusters_separately=True,
        m_config=MODCEConfig(num_cluster=3, observe_abs_box_pos=True),
        reward_function="object_clustering_amp",
        agent_type="SimpleVelocityControlKilobot",
        swarm_reward=True,
        agent_reward=True,
        done_after_steps=1024,
    )


def task_clustering2():
    env_params = env_clustering2()
    params = FullParams(
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=512000,
        train_steps=2000,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=1024 * 10,
    )
    return params


def task_clustering3(**kwargs):
    env_params = env_clustering3(**kwargs)
    params = FullParams(
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=250 * 1024 * 5,
        train_steps=500,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=1024 * 10,
    )
    return params


def task_clustering3_explicit():
    params = task_clustering3()
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit"
        )
    )
    return params


def task_clustering3_simple():
    params = task_clustering3(boxes=3, bots=5)
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit"
        )
    )
    return params


def task_clustering3_simple_oneagg():
    params = task_clustering3(boxes=3, bots=5)
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit",
            aggregate_clusters_separately=False,
        ),
    )
    return params
def task_clustering3_simple_twoagg():
    params = task_clustering3(boxes=6, bots=5, size=1.0)
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit",
            aggregate_clusters_separately=False,
        ),
    )
    return params


def task_clustering3_explicit_indiv():
    params = task_clustering3()
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit_indiv"
        )
    )
    return params


def task_clustering2_explicit():
    params = task_clustering2()
    params = params._replace(
        env_params=params.env_params._replace(
            reward_function="object_clustering_explicit"
        )
    )
    return params


def env_assembly():
    from gym_kilobots.envs.yaml_kilobots_env import EnvConfiguration as Kb

    return EnvParams.KilobotsNew(
        env=Kb(
            width=1.0,
            height=1.0,
            resolution=300,
            objects=[
                dict(
                    idx=0,
                    shape="square",
                    width=0.1,
                    height=0.1,
                    init="random",
                    color=None,
                    symmetry=None,
                )
                for _ in range(4)
            ],
            kilobots=dict(mean="random", std=0.03, num=10),
            light=None,
        ),
        aggregate_clusters_separately=True,
        m_config=MODCEConfig(observe_abs_box_pos=True),
        reward_function="assembly",
        agent_type="SimpleVelocityControlKilobot",
        swarm_reward=True,
        agent_reward=True,
        done_after_steps=512,
    )


def task_assembly():

    env_params = env_assembly()
    params = FullParams(
        env_params=env_params,
        runname="??",
        nr_parallel_envs=nr_parallel_envs,
        env_steps_per_train_step=50 * 1024 * 5,
        train_steps=200,
        emb_params=None,
        policy_params=PolicyParams(
            share_vf_pi_params=False,
        ),
        batch_size=1024 * 5,
    )
    return params


def get_default_params(task: Task, agg: str, algo: str) -> FullParams:
    fn = globals().get(f"task_{task}", None)
    if not fn:
        raise Exception(f"unknown task {task}")
    full_params: FullParams = fn()

    emb = globals().get(f"emb_{agg}", None)
    if not emb:
        raise Exception(f"unknown agg {agg}")

    training_algorithm = algo
    training_algorithm_params = {}
    if algo == "trl_kl":
        training_algorithm = "trl"
        training_algorithm_params = {"proj_type": "kl"}
    full_params = full_params._replace(
        training_algorithm=training_algorithm,
        emb_params=emb,
        training_algorithm_params=training_algorithm_params,
    )
    return full_params
