import math
from typing import *

import torch as th
import torch.nn
import torch.nn.functional
from torch import nn

from ..util import ns, stripnames
from .embed import (
    AggMode,
    Embed,
    EmbeddingParams,
    EmbeddingParamsRuntime,
    SampleOfEmbeddedLatentSpace,
    SigmaZ0Variant,
)
from .mlp import get_activation_fn, mlp
from .NeatSpaces import NeatSpaces


class MultiHeadSelfAttention(nn.Module):
    """
    self-attention module.
    input: (batch, neighbor, feature)
    output: (batch, neighbor, feature) (the same shape, but feature dimension are the "attended" features)
    """

    def __init__(self, embed_dim: int, config: EmbeddingParams):
        super().__init__()
        agg_mode: AggMode.MultiHeadAttention = config.agg_mode
        # S = L = sequence length
        # input: (S, N, E) where S is sequence length, N is batch size, E is embedding dimension
        self.mha = nn.MultiheadAttention(
            num_heads=agg_mode.num_heads, dropout=agg_mode.dropout, embed_dim=embed_dim
        )
        if len(agg_mode.post_att_layers) > 0:
            self.post_att_layers = mlp(
                agg_mode.post_att_layers, get_activation_fn(config.activation_fn)
            )
        else:
            self.post_att_layers = None
        self.residual = agg_mode.residual

    def forward(self, orig_data: th.Tensor):
        # reorder to align with (S, N, E) (S = sequence = neighbor, N = batch, E = embedding = feature)
        data = orig_data.align_to("neighbor", "batch", "feature")
        data = stripnames(data, "neighbor", "batch", "feature")
        output, no = self.mha(query=data, key=data, value=data, need_weights=False)
        assert no is None
        output = output.refine_names("neighbor", "batch", "feature")
        output = output.align_to("batch", "neighbor", "feature")
        if self.post_att_layers:
            output = self.post_att_layers(output)
        if self.residual:
            output = orig_data + output
        # todo: residual after first dense layer?

        return output


def get_variance_rectifier(
    rectifier: Literal["none", "exp", "softplus"]
) -> Callable[[th.Tensor], th.Tensor]:
    if rectifier == "none":

        def fn(x):
            return x

    if rectifier == "exp":
        fn = th.exp
    if rectifier == "softplus":
        fn = th.nn.functional.softplus
    if not fn:
        raise Exception(f"unknown variance rectifier {rectifier}")
    return fn


class Aggregate(nn.Module):
    """
    aggregate a SampleOfEmbeddedLatentSpace into a single vector using one of the aggregation methods (mean, bayesian, ...)
    converts a vector with the dimensions (batch, neighbor, feature) into a vector of (batch, feature)
    """

    def __init__(self, params: EmbeddingParamsRuntime):
        super().__init__()
        self._forward_pass_debug_info = None
        config = params.config
        self.agg_mode = config.agg_mode

        if isinstance(self.agg_mode, AggMode.Mean):
            self.latent_embedding_dim = config.latent_embedding_dim
            self.input_dim = self.latent_embedding_dim
        elif isinstance(self.agg_mode, AggMode.Bayesian):

            def param_variant(name: str, d: SigmaZ0Variant, default: float):
                if d:
                    dim = config.latent_embedding_dim if d == "one_per_dim" else 1
                    print(f"adding {name}_z₀ of dim {dim}")
                    return nn.Parameter(th.ones(dim))
                else:
                    return default

            self.sigma_z_0_sq = param_variant("σ", self.agg_mode.learnable_sigma_z_0, math.inf)

            self.mean_z_0 = param_variant("μ", self.agg_mode.learnable_mu_z_0, 0)
            variance_rectifier_r = self.agg_mode.variance_rectifier
            variance_rectifier_z0 = self.agg_mode.variance_rectifier
            if variance_rectifier_r == "unset":
                variance_rectifier_r = "exp"
            if variance_rectifier_z0 == "unset":
                variance_rectifier_z0 = "exp" if self.agg_mode.exp_sigma_z_0 else "none"
            self.variance_rectifier_r = get_variance_rectifier(variance_rectifier_r)
            self.variance_rectifier_z0 = get_variance_rectifier(variance_rectifier_z0)

            self.latent_feature_count = config.latent_embedding_dim
            if self.agg_mode.output_variance:
                self.latent_embedding_dim = 2 * self.latent_feature_count
            else:
                self.latent_embedding_dim = self.latent_feature_count

            if not self.agg_mode.separate:
                # 2 outputs per feature: mean and stddev
                self.input_dim = 2 * self.latent_feature_count
            else:
                raise Exception("not supported right now")
                self.input_dim = self.latent_feature_count
                # todo: add flag to have two separate encoders
                self.neighbor_embedding
                self.neighbor_embedding_variance

        elif isinstance(self.agg_mode, AggMode.MultiHeadAttention):
            self.latent_embedding_dim = config.latent_embedding_dim
            self.input_dim = self.latent_embedding_dim
            self.attention = MultiHeadSelfAttention(
                embed_dim=self.latent_embedding_dim,
                config=config,
            )
        else:
            raise Exception(f"unknown agg mode {self.agg_mode}")

    def forward(self, obs: SampleOfEmbeddedLatentSpace):
        self._forward_pass_debug_info = {}
        neighbor_embedded, visible_neighbors_count = obs

        # assert th.equal(neighbor_embedded_flat[0], neighbor_embedded[0][0])

        if isinstance(self.agg_mode, AggMode.Mean):
            if self.agg_mode.actual_mode == "mean":
                sum_neighbor_embedded = neighbor_embedded.sum(axis="neighbor").refine_names(
                    "batch", "feature"
                )
                aggregated_neighbor_embedded = sum_neighbor_embedded / visible_neighbors_count
            elif self.agg_mode.actual_mode == "max":
                aggregated_neighbor_embedded = neighbor_embedded.max(
                    axis="neighbor"
                ).values.refine_names("batch", "feature")
            else:
                raise Exception(f"unknown agg {self.agg_mode.actual_mode}")

        elif isinstance(self.agg_mode, AggMode.Bayesian):
            if not self.agg_mode.separate:
                # https://en.wikipedia.org/wiki/Inverse-variance_weighting, same as
                # https://github.com/iclr2021637/iclr2021637/blob/9068072b9d3dd27ac39086a27b5f570bcf6dfe52/bayesian_aggregation/aggregator.py#L165

                unf = neighbor_embedded.unflatten(
                    "feature",
                    OrderedDict(feature=self.latent_feature_count, mean_or_std=2),
                )
                r = ns(unf, mean_or_std=0)  # ns[:, :, :, 0]
                # ns(unf, batch=slice(0,1), mean_or_std=0) # ns[0:10, :, :, 0]
                cov_r = ns(unf, mean_or_std=1)
            else:
                neighbor_embedded_cov, _ = self.neighbor_embedding_variance(obs)
                r = neighbor_embedded
                cov_r = neighbor_embedded_cov

            cov_r = self.variance_rectifier_r(
                stripnames(cov_r, "batch", "neighbor", "feature")
            ).refine_names("batch", "neighbor", "feature")
            # cov_r is sigma_r_n² in https://openreview.net/pdf?id=ufZN2-aehFa formula 8
            sigma_z_0_sq = self.variance_rectifier_z0(th.as_tensor(self.sigma_z_0_sq))
            sigma_z_sq = 1 / (1 / sigma_z_0_sq + th.sum(1 / cov_r, dim="neighbor"))
            # self._forward_pass_debug_info["avg_sigma_z_sq"] = (
            #     sigma_z_sq.detach().mean(dim=("batch", "feature")).item()
            # )
            mean_z_0 = self.mean_z_0
            aggregated_neighbor_embedded = mean_z_0 + sigma_z_sq * th.sum(
                (r - mean_z_0) / cov_r, dim="neighbor"
            )
            if self.agg_mode.output_variance:
                aggregated_neighbor_embedded = th.cat(
                    [aggregated_neighbor_embedded, sigma_z_sq], axis="feature"
                )
        elif isinstance(self.agg_mode, AggMode.MultiHeadAttention):
            neighbor_embedded = self.attention(neighbor_embedded)
            # actually a mean embedding, todo: flexible?
            sum_neighbor_embedded = neighbor_embedded.sum(axis="neighbor").refine_names(
                "batch", "feature"
            )
            aggregated_neighbor_embedded = sum_neighbor_embedded / visible_neighbors_count
        else:
            raise Exception(f"agg mode {self.agg_mode} unknown")

        return aggregated_neighbor_embedded

    def get_debug_info_from_last_forward_pass(self):
        return self._forward_pass_debug_info


class FullAggregatingEmbedder(nn.Module):
    """
    1. Unflatten input observations from Box to NeatSpace
    2. Embed() the observations that are marked as aggregatable
    3. Aggregate the observations either into separate latent spaces or the same latent space (mean, bayesian, selfatt, ...)
    4. Concatenate all aggregateds with the self-observations and pass through another dense NN
    """

    def __init__(self, params: EmbeddingParamsRuntime):
        super().__init__()
        self.spaces = NeatSpaces(params)
        local_obs_dim = self.spaces.local_obs_dim
        self.agg_into_space = params.config.agg_into_space

        aggregator = None  # only used when aggregating in same space
        aggregators = None  # only used when aggregating in separate spaces
        if params.config.agg_into_space == "same":
            aggregator = Aggregate(params)
            latent_agg_space_dims = [aggregator.latent_embedding_dim]
            embedder_out_dim = aggregator.input_dim
        elif params.config.agg_into_space == "separate":
            aggregators = torch.nn.ModuleDict(
                {name: Aggregate(params) for name, info in self.spaces.aggregatables_info.items()}
            )
            latent_agg_space_dims = [
                aggregator.latent_embedding_dim for aggregator in aggregators.values()
            ]
            embedder_out_dim = None
            for agg in aggregators.values():
                assert (
                    embedder_out_dim is None or agg.input_dim == embedder_out_dim
                ), "different embedders have different out dims"
                embedder_out_dim = agg.input_dim
        else:
            raise Exception(f"unknown config {params.config.agg_into_space=}")

        self.embedders = cast(
            Dict[str, Callable[[NeatSpaces.Aggregatable], SampleOfEmbeddedLatentSpace]],
            torch.nn.ModuleDict(
                {
                    name: Embed(params, info, out_dim=embedder_out_dim)
                    for name, info in self.spaces.aggregatables_info.items()
                }
            ),
        )
        self.aggregator = aggregator
        self.aggregators = aggregators

        config = params.config
        activation_fn = get_activation_fn(config.activation_fn)
        self.later_layer = mlp(
            [
                local_obs_dim + sum(latent_agg_space_dims),
                *config.latent_out_hidden_layer_sizes,
                config.latent_dim,
            ],
            activation_fn,
        )
        self.output_dim = config.latent_dim

    def forward(self, _obs):
        obs = self.spaces.unflatten(_obs)
        embeddeds = {
            name: embedder(obs.aggregatables[name]) for name, embedder in self.embedders.items()
        }
        if self.agg_into_space == "same":
            # concat the embedded samples along the "neighbor" dimension, then aggregate them into one space
            flonk = SampleOfEmbeddedLatentSpace(
                embedded=th.cat([a.embedded for a in embeddeds.values()], dim="neighbor"),
                visible_count=th.stack([a.visible_count for a in embeddeds.values()], dim=0).sum(
                    axis=0
                ),
            )
            aggregateds = [self.aggregator(flonk)]
        elif self.agg_into_space == "separate":
            # a separate aggregated per list of embeddeds
            aggregateds = [self.aggregators[name](embedded) for name, embedded in embeddeds.items()]
        else:
            raise Exception("unknown agg into mode")
        concatted = th.cat((obs.local_data, *aggregateds), axis="feature")

        output = self.later_layer(stripnames(concatted, "batch", "feature"))
        return output
