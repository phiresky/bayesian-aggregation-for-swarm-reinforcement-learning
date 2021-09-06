from typing import TYPE_CHECKING, Literal, NamedTuple, Type, Union

import gym.spaces as spaces
import optuna
import torch as th
import torch.nn
import torch.nn.functional
from torch import nn

from ..ma_envs.point_envs.rendezvous import ObsMode
from ..sumtype import SumType
from ..util import namedshape, stripnames
from .mlp import ActivationFn, get_activation_fn, mlp
from .NeatSpaces import NeatSpaces


def sumtype_dict(st: Type[SumType]):
    classes = st.Type.__args__
    name_map = {c.__name__: c for c in classes}
    return name_map


SigmaZ0Variant = Literal[False, True, "one_per_dim"]
VarianceRectifierVariant = Literal["unset", "exp", "softplus"]


class AggMode(SumType):
    class Mean(NamedTuple):
        """mean aggregation, like max's paper https://www.jmlr.org/papers/volume20/18-476/18-476.pdf"""

        type: Literal["Mean"] = "Mean"
        actual_mode: Literal["mean", "max"] = "mean"

        @staticmethod
        def choose_optuna(name_prefix: str, trial: optuna.Trial):
            return AggMode.Mean()

    class Bayesian(NamedTuple):
        """bayesian aggregation"""

        separate: bool
        learnable_sigma_z_0: SigmaZ0Variant = False
        # legacy, use variance_rectifier
        exp_sigma_z_0: bool = False
        # the function to applied to learned std-dev values
        # if unset, use exp_sigma_z_0 for variable and exp for bayesian layer output
        # exp as done by https://github.com/iclr2021637/iclr2021637/blob/9068072b9d3dd27ac39086a27b5f570bcf6dfe52/bayesian_aggregation/encoder_network.py#L95
        # to make values positive, net learns log std
        # softplus as suggested by prof. neumann

        variance_rectifier: VarianceRectifierVariant = "unset"
        learnable_mu_z_0: SigmaZ0Variant = False
        output_variance: bool = False

        type: Literal["Bayesian"] = "Bayesian"

        @staticmethod
        def choose_optuna(name_prefix: str, trial: optuna.Trial):
            return AggMode.Bayesian(
                separate=trial.suggest_categorical(
                    f"{name_prefix}.separate", [False, True]
                ),
                learnable_sigma_z_0=trial.suggest_categorical(
                    f"{name_prefix}.learnable_sigma_z_0", SigmaZ0Variant.__args__
                ),
                exp_sigma_z_0=False,
                learnable_mu_z_0=trial.suggest_categorical(
                    f"{name_prefix}.learnable_mu_z_0", SigmaZ0Variant.__args__
                ),
                variance_rectifier=trial.suggest_categorical(
                    f"{name_prefix}.variance_rectifier",
                    VarianceRectifierVariant.__args__,
                ),
            )

    class MultiHeadAttention(NamedTuple):
        """multi head attention aggregation, similar to openai emergent behaviour paper"""

        num_heads: int
        residual: bool
        dropout: float = 0.0
        post_att_layers: list[int] = []
        type: Literal["MultiHeadAttention"] = "MultiHeadAttention"

        @staticmethod
        def choose_optuna(name_prefix: str, trial: optuna.Trial):
            return AggMode.MultiHeadAttention(
                num_heads=trial.suggest_int(f"{name_prefix}.num_heads", 1, 4, 1),
                dropout=0.0,
                residual=trial.suggest_categorical(
                    f"{name_prefix}.residual", [False, True]
                ),
            )

    class AttentiveRelational(NamedTuple):
        """
        based on the paper https://ieeexplore.ieee.org/document/9049415
        legacy, unused
        """

        type: Literal["AttentiveRelational"] = "AttentiveRelational"

    Type = Union[Mean, Bayesian, MultiHeadAttention, AttentiveRelational]


def choose_int_list(
    trial: optuna.Trial, prefix: str, min_len: int, max_len: int, min: int, max: int
):
    length = trial.suggest_int(f"{prefix}.len", low=min_len, high=max_len)
    return [
        trial.suggest_int(f"{prefix}[{i}]", low=min, high=max) for i in range(length)
    ]


class EmbeddingParams(NamedTuple):
    latent_dim: int
    agg_mode: AggMode.Type
    obs_mode: ObsMode
    latent_out_hidden_layer_sizes: list[int] = []
    latent_embedding_hidden_layer_sizes: list[int] = []
    latent_embedding_dim: int = 64
    activation_fn: ActivationFn = "Tanh"
    local_obs_aggregation_space: bool = False
    agg_into_space: Literal["separate", "same"] = "separate"

    @staticmethod
    def choose_optuna(name_prefix: str, trial: optuna.Trial):
        agg_modes = sumtype_dict(AggMode)
        del agg_modes["AttentiveRelational"]
        return EmbeddingParams(
            latent_dim=trial.suggest_int(f"{name_prefix}.latent_dim", low=1, high=256),
            agg_mode=agg_modes[
                trial.suggest_categorical(f"{name_prefix}.agg_mode", agg_modes.keys())
            ].choose_optuna(f"{name_prefix}.agg_mode", trial),
            obs_mode="relative",
            latent_out_hidden_layer_sizes=choose_int_list(
                trial, f"{name_prefix}.latent_out_hidden_layer_sizes", 0, 2, 10, 256
            ),
            latent_embedding_hidden_layer_sizes=choose_int_list(
                trial,
                f"{name_prefix}.latent_embedding_hidden_layer_sizes",
                0,
                2,
                20,
                256,
            ),
            latent_embedding_dim=trial.suggest_int(  # divisible by 2, 3, 4 for multi head attention
                f"{name_prefix}.latent_embedding_dim", low=12, high=132, step=12
            ),
            activation_fn=trial.suggest_categorical(
                f"{name_prefix}.activation_fn", ActivationFn.__args__
            ),
            agg_into_space=trial.suggest_categorical(
                f"{name_prefix}.agg_into_space", ["separate", "same"]
            ),
        )


class EmbeddingParamsRuntime(NamedTuple):
    """created from EmbeddingParams with new information gathered at runtime"""

    config: EmbeddingParams
    full_obs_space: spaces.Dict


class SampleOfEmbeddedLatentSpace(NamedTuple):
    embedded: th.Tensor  # dimensions: (batch, neighbor, feature)
    visible_count: th.Tensor


class EmbedVisiblesToLatentSpace(nn.Module):
    """
    Dense NN that embeds observations into a latent space, ignoring those observations that are marked as non-visible
    """

    def __init__(self, nn: nn.Sequential, output_dim: int):
        super().__init__()
        self.nn = nn
        self.output_dim: int = output_dim

    def forward(self, obs: "NeatSpaces.Aggregatable") -> SampleOfEmbeddedLatentSpace:
        partial_visibility = False
        if partial_visibility:
            raise Exception("needs inspection")
            # todo: check why / if this is much slower than without the visibility thing
            # todo: check interaction with bayesian agg
            # todo: alternative implementation - pass through everything but multiply with valid afterwards (masking basically, code is simpler and it might be faster)

            # get data only for those. this merges the neighbor and batch dimensions since number of neighbors can vary between each batch sample
            neighbor_valid_indices = obs.valid > 0
            # th.nonzero(
            #    obs.neighbor_valid.rename(None), as_tuple=True
            # )
            neighbor_valid_data = stripnames(obs.data, "batch", "neighbor", "feature")[
                neighbor_valid_indices
            ]

            neighbor_embedded_flat = self.nn(neighbor_valid_data)

            neighbor_embedded = th.zeros(
                (*obs.data.shape[:-1], self.output_dim),
                device=neighbor_embedded_flat.device,
            )

            neighbor_embedded[neighbor_valid_indices] = neighbor_embedded_flat
            neighbor_embedded = neighbor_embedded.refine_names(
                "batch", "neighbor", "feature"
            )
            visible_neighbors_count = th.count_nonzero(
                stripnames(obs.valid, "batch", "neighbor", "feature"),
                1,  # axis="neighbor"
            ).refine_names("batch")
        else:
            neighbor_embedded = self.nn(
                stripnames(obs.data, "batch", "neighbor", "feature")
            ).refine_names("batch", "neighbor", "feature")
            visible_neighbors_count = torch.as_tensor(
                namedshape(neighbor_embedded)["neighbor"]
            )
        return SampleOfEmbeddedLatentSpace(
            embedded=neighbor_embedded, visible_count=visible_neighbors_count
        )


class Embed(nn.Module):
    def __init__(
        self,
        params: EmbeddingParamsRuntime,
        agg_info: NeatSpaces.AggregatableInfo,
        out_dim: int,
    ):
        super().__init__()
        config = params.config
        hidden_sizes = config.latent_embedding_hidden_layer_sizes
        activation_fn = get_activation_fn(config.activation_fn)
        self.embedding = EmbedVisiblesToLatentSpace(
            mlp([agg_info.obs_dim, *hidden_sizes, out_dim], activation_fn), out_dim
        )
        print(
            f"embedder {agg_info.name}, max {agg_info.max_num} eles: obs_dim={agg_info.obs_dim}, out_dim={out_dim}"
        )

    def forward(self, obs: NeatSpaces.Aggregatable) -> SampleOfEmbeddedLatentSpace:
        self._forward_pass_debug_info = {}
        return self.embedding(obs)


if __name__ == "__main__":
    import optuna

    def objective(trial):
        params = EmbeddingParams.choose_optuna("", trial)
        print(params)
        return 1

    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
