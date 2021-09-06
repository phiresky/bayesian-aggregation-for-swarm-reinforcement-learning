from typing import Callable, NamedTuple

import gym.spaces as spaces
import optuna
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

from .aggregate import FullAggregatingEmbedder
from .AttentiveRelationalEncoder import (
    AttentiveRelationalModuleRelativeObs,
    AttentiveRelationalModuleUniformObs,
)
from .embed import AggMode, EmbeddingParamsRuntime


class DoublyBubbly(nn.Module):
    """
    create separate value and policy networks based on the same nn.Module
    """

    def __init__(
        self,
        module: Callable[[EmbeddingParamsRuntime], nn.Module],
        params: EmbeddingParamsRuntime,
    ):
        super().__init__()
        print("vf and pi nets are separate")
        self.pi_network = module(params)
        self.vf_network = module(params)
        self.latent_dim_pi = self.pi_network.output_dim
        self.latent_dim_vf = self.vf_network.output_dim

    def forward(self, obs):
        return self.pi_network(obs), self.vf_network(obs)


class SingleyPingly(nn.Module):
    """
    create a common value and policy network based on a nn.Module
    """

    def __init__(
        self,
        module: Callable[[EmbeddingParamsRuntime], nn.Module],
        params: EmbeddingParamsRuntime,
    ):
        super().__init__()
        print("vf and pi nets are shared")
        self.shared_network = module(params)
        self.latent_dim_pi = self.shared_network.output_dim
        self.latent_dim_vf = self.shared_network.output_dim

    def forward(self, obs):
        out = self.shared_network(obs)
        return out, out


class PolicyParams(NamedTuple):
    share_vf_pi_params: bool

    @staticmethod
    def choose_optuna(name_prefix: str, trial: optuna.Trial):
        return PolicyParams(
            share_vf_pi_params=trial.suggest_categorical(
                f"{name_prefix}.share_vf_pi_params", [False, True]
            )
        )


class PolicyParamsRuntime(NamedTuple):
    config: PolicyParams
    emb_params: EmbeddingParamsRuntime
    # net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None


class ThrowMod(nn.Module):
    def __init__(self):
        raise Exception("should not be used")


import torch as ch


class ActorCriticGaussianDiagPolicy(ActorCriticPolicy):
    """add a few functions that are needed only for TRL (not PPO) but are not part of stable-baselines"""

    contextual_std = False
    is_diag = True

    def maha(self, mean: ch.Tensor, mean_other: ch.Tensor, std: ch.Tensor):
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def precision(self, std: ch.Tensor):
        return (1 / self.covariance(std).diagonal(dim1=-2, dim2=-1)).diag_embed()

    def covariance(self, std: ch.Tensor):
        return std.pow(2)

    def entropy(self, p):
        # TODO: compare with DiagGaussianPolicy -> distributions.Normal().entropy()
        import numpy as np

        _, std = p
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return 0.5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor):
        """
        Returns the log determinant of a diagonal matrix
        Args:
            std: a diagonal matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)


class MlpAggregatingPolicy(ActorCriticGaussianDiagPolicy):
    def __init__(
        self,
        # stable-baselines passes these as unnamed params :/
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *,
        params: PolicyParamsRuntime,
        **kwargs,
    ):
        self.params = params
        # # TODO: do this?
        self.ortho_init = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=ThrowMod,
            # Pass remaining arguments to base class
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        constructor = (
            SingleyPingly if self.params.config.share_vf_pi_params else DoublyBubbly
        )
        module = (
            (
                AttentiveRelationalModuleUniformObs
                if self.params.emb_params.config.obs_mode == "uniform"
                else AttentiveRelationalModuleRelativeObs
            )
            if isinstance(
                self.params.emb_params.config.agg_mode, AggMode.AttentiveRelational
            )
            else FullAggregatingEmbedder
        )
        self.mlp_extractor = constructor(module, self.params.emb_params)
        print("full policy", self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"full policy has {total_params} parameters, {trainable_params} trainable"
        )

    def collect_debug_info_from_last_forward_pass(self):
        out_dict = {}
        for name, module in self.named_modules():
            if getinfo := getattr(
                module, "get_debug_info_from_last_forward_pass", None
            ):
                for pname, val in getinfo().items():
                    out_dict[f"{name}.{pname}"] = val
        return out_dict
