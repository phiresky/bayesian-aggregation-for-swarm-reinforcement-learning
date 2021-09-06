import torch as th
from playground.policy.embed import AggMode, EmbeddingParamsRuntime
from playground.policy.mlp import get_activation_fn, mlp
from torch import nn

from .embed import EmbedVisiblesToLatentSpace
from .NeatSpaces import NeatSpaces

"""
Attentive Relatonal Encoder based on https://ieeexplore.ieee.org/document/9049415
Does not seem to work very well
"""


class AttentiveRelationalModuleRelativeObs(nn.Module):
    def __init__(self, params: EmbeddingParamsRuntime):
        super().__init__()
        self.params = params
        config = params.config
        self.spaces = NeatSpaces(params)
        if not isinstance(config.agg_mode, AggMode.AttentiveRelational):
            raise Exception("wrong agg mode")
        self.agg_mode = config.agg_mode
        hidden_layer_sizes_ef = [32, config.latent_dim]
        hidden_layer_sizes_ec = [32, 32]

        activation_fn = get_activation_fn(config.activation_fn)
        # E_c
        self.encoder_communication = EmbedVisiblesToLatentSpace(
            mlp(
                [self.spaces.neighbor_obs_dim, *hidden_layer_sizes_ec],
                activation_fn,
            )
        )
        # E_f
        self.encoder_feature = EmbedVisiblesToLatentSpace(
            mlp(
                [self.spaces.neighbor_obs_dim, *hidden_layer_sizes_ef],
                activation_fn,
            )
        )
        ec_out_size = hidden_layer_sizes_ec[-1]
        ef_out_size = hidden_layer_sizes_ef[-1]
        # is fed ec(self), ec(currentneighbor), mean(ec(otherneighbors)) so 3*ec_out_size
        # outputs one attention value per encoded feature: "we set the dimension of the attention vector a_ij to be the same with e_f_j"
        self.encoder_attention = mlp(
            [2 * ec_out_size, ef_out_size],
            nn.Softmax(dim=-2),  # softmax in the neighbor dimension
        )

        self.latent_dim = config.latent_dim

        self.later_layer = nn.Sequential(
            nn.Linear(ef_out_size + self.spaces.local_obs_dim, config.latent_dim),
            activation_fn,
        )

    def forward(self, _obs):
        obs = self.spaces.unflatten(_obs)
        enc_feature, visible_neighbor_dim = self.encoder_feature(obs)
        # enc_feature_self = self.encoder_feature.nn(obs.local_data)
        # enc_comm_self = self.encoder_comm()
        enc_comm, _ = self.encoder_communication(obs)

        # calc the mean embedding, copy it for each neighbor
        enc_comm_mean = (
            enc_comm.mean(axis="neighbor")
            .rename(None)
            .unsqueeze(1)
            .repeat(1, 19, 1)
            .refine_names("batch", "neighbor", "feature")
        )
        assert enc_comm_mean.shape == enc_comm.shape

        # todo: add encoded self
        enc_att = self.encoder_attention(
            th.cat((enc_comm, enc_comm_mean), axis="feature")
        )

        weighted_neighbor_embedded = (enc_att * enc_feature).refine_names(
            "batch", "neighbor", "feature"
        )
        assert th.allclose(
            enc_att[0, :, 0].sum(), th.as_tensor(1, dtype=th.float32)
        )  # sum of attention weights over each neighbor for the first feature is 1
        aggregated_neighbor_embedded = weighted_neighbor_embedded.sum(axis="neighbor")

        concatted = th.cat(
            (obs.local_data, aggregated_neighbor_embedded), axis="feature"
        )
        output = self.later_layer(concatted)

        return output.rename(None)


class AttentiveRelationalModuleUniformObs(nn.Module):
    def __init__(self, params: EmbeddingParamsRuntime):
        super().__init__()
        self.params = params
        config = params.config
        if not isinstance(config.agg_mode, AggMode.AttentiveRelational):
            raise Exception("wrong agg mode")
        self.agg_mode = config.agg_mode
        hidden_layer_sizes_ef = [config.latent_dim]
        hidden_layer_sizes_ec = [32]

        activation_fn = get_activation_fn(config.activation_fn)

        self.spaces = NeatSpacesUniform(params)
        # E_c
        self.encoder_communication = EmbedVisibleNeighbors(
            mlp(
                [self.spaces.agent_obs_dim, *hidden_layer_sizes_ec],
                activation_fn,
            )
        )
        # E_f
        self.encoder_feature = EmbedVisibleNeighbors(
            mlp(
                [self.spaces.agent_obs_dim, *hidden_layer_sizes_ef],
                activation_fn,
            )
        )
        ec_out_size = hidden_layer_sizes_ec[-1]
        ef_out_size = hidden_layer_sizes_ef[-1]
        # is fed ec(self), ec(currentneighbor), mean(ec(otherneighbors)) so 3*ec_out_size
        # outputs one attention value per encoded feature: "we set the dimension of the attention vector a_ij to be the same with e_f_j"
        self.encoder_attention = mlp(
            [3 * ec_out_size, ef_out_size],
            nn.Softmax(dim=-2),  # softmax in the neighbor dimension
        )
        self.latent_dim = config.latent_dim
        self.use_later_layer = True
        if self.use_later_layer:
            self.later_layer = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                activation_fn,
            )

    def forward(self, _obs):
        obs = self.spaces.unflatten(_obs)
        assert obs.neighbor_data.shape[1] == self.spaces.num_agents
        assert obs.neighbor_data.shape[2] == self.spaces.agent_obs_dim
        enc_feature, visible_neighbor_dim = self.encoder_feature(obs)
        # enc_feature_self = self.encoder_feature.nn(obs.local_data)
        enc_comm_self = self.encoder_communication.nn(obs.local_data)
        enc_comm, _ = self.encoder_communication(obs)
        agent_count = self.spaces.num_agents

        # copy self embedding for each neighbor
        enc_comm_self_rep = (
            enc_comm_self.rename(None)
            .unsqueeze(1)
            .repeat(1, agent_count, 1)
            .refine_names("batch", "neighbor", "feature")
        )
        # calc the mean embedding, copy it for each neighbor
        enc_comm_mean = (
            enc_comm.mean(axis="neighbor")
            .rename(None)
            .unsqueeze(1)
            .repeat(1, agent_count, 1)
            .refine_names("batch", "neighbor", "feature")
        )
        assert (
            enc_comm_mean.shape == enc_comm.shape
        ), "mean duped to same shape as comm emb"
        assert enc_comm_self_rep.shape == enc_comm.shape

        enc_att = self.encoder_attention(
            th.cat((enc_comm_self_rep, enc_comm, enc_comm_mean), axis="feature")
        )

        weighted_neighbor_embedded = (enc_att * enc_feature).refine_names(
            "batch", "neighbor", "feature"
        )
        assert th.allclose(
            enc_att[0, :, 0].sum(), th.as_tensor(1, dtype=th.float32)
        )  # sum of attention weights over each neighbor for the first feature is 1
        aggregated_neighbor_embedded = weighted_neighbor_embedded.sum(axis="neighbor")

        output = aggregated_neighbor_embedded.refine_names("batch", "feature")

        if self.use_later_layer:
            output = self.later_layer(output)

        return output.rename(None)
