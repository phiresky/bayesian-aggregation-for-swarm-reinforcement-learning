from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, NamedTuple

import gym.spaces as spaces
import numpy as np
import torch as th
from playground.fast_dict_space import FastDictSpace

from .. import util
from ..util import unflatten_batch

if TYPE_CHECKING:
    from .embed import EmbeddingParamsRuntime

lax_obs_asserts = False


@contextmanager
def allow_empty():
    global lax_obs_asserts
    lax_obs_asserts = True
    yield None
    lax_obs_asserts = False


def map_aggs(name: str, space: spaces.Tuple, getter: Callable[[dict], np.ndarray]):
    assert isinstance(space, spaces.Tuple)
    each_neighbor_space = space.spaces[0]
    assert isinstance(
        each_neighbor_space, spaces.Dict
    ), "aggregatable space must be dict"

    obs_dim = spaces.flatdim(each_neighbor_space)

    has_valid_indicator = "valid" in each_neighbor_space.spaces
    if has_valid_indicator:
        assert (
            list(each_neighbor_space.spaces)[-1] == "valid"
        ), "valid space must be last"
        obs_dim -= 1  # -1 because valid is not passed

    total_neighbor_count = len(space.spaces)
    mapped_space = spaces.Box(
        0,
        1,
        shape=(
            total_neighbor_count,
            spaces.flatdim(each_neighbor_space),
        ),
        dtype="float32",
    )
    info = NeatSpaces.AggregatableInfo(
        name=name,
        obs_dim=obs_dim,
        max_num=total_neighbor_count,
        has_valid_indicator=has_valid_indicator,
        get_from=getter,
    )
    return mapped_space, info


class NeatSpaces:
    """
    stable-baselines3 only supports Box observation spaces. Those are really annoying to work with since you can't name your observations.
    This is a utility that allows using a DictSpace instead which is much easier to understand
    """

    class AggregatableInfo(NamedTuple):
        name: str
        obs_dim: int
        max_num: int
        # whether or not there is a valid: 0/1 observation that indicates whether the aggregatable is valid / visible
        has_valid_indicator: bool
        get_from: Callable[[dict], np.ndarray]

    def __init__(self, params: EmbeddingParamsRuntime):
        self.params = params
        if isinstance(params.full_obs_space, FastDictSpace):
            self.full_obs_space = params.full_obs_space.as_dict_space
        elif isinstance(params.full_obs_space, spaces.Dict):
            self.full_obs_space = params.full_obs_space
        else:
            raise Exception(
                f"unknown full obs space type {type(params.full_obs_space)}"
            )
        self.local_space = self.full_obs_space["local"]

        self.local_obs_dim = spaces.flatdim(self.local_space)

        self.aggregatables_info: Dict[str, NeatSpaces.AggregatableInfo] = {}

        self.neat_space = util.DictSpace(
            {
                name: (
                    self._init_agg_spaces(space)
                    if name == "aggregatables"
                    else spaces.flatten_space(space)
                )
                for name, space in self.full_obs_space.spaces.items()
            }
        )

        if params.config.local_obs_aggregation_space:
            self.aggregatables_info["$local"] = map_aggs(
                "$local", self.full_obs_space.spaces["local"], lambda obs: obs["local"]
            )

    def _init_agg_spaces(self, spaces):

        mapped_aggs = {}
        for name, space in self.full_obs_space.spaces["aggregatables"].spaces.items():
            mapped_aggs[name], self.aggregatables_info[name] = map_aggs(
                name, space, lambda obs, name=name: obs["aggregatables"][name]
            )

        return util.DictSpace(mapped_aggs)

    class Aggregatable(NamedTuple):
        valid: th.Tensor
        data: th.Tensor

    class UnflattenReturn(NamedTuple):
        local_data: th.Tensor
        aggregatables: Dict[str, "NeatSpaces.Aggregatable"]

    def _get_aggregatable_spaces(self, obs):
        return [
            (name, obs["aggregatables"][name])
            for name in self.neat_spaces["aggregatables"].spaces
        ]

    def unflatten(self, _obs: th.Tensor):
        """convert a Box observation into a a dict of aggregatables and the local observations"""

        obs: dict = unflatten_batch(self.neat_space, _obs)
        aggregatables = {}
        if "aggregatables" in self.neat_space.spaces:
            for name, info in self.aggregatables_info.items():

                full = th.as_tensor(info.get_from(obs)).refine_names(
                    "batch", "neighbor", "feature"
                )
                if info.has_valid_indicator:
                    aggregatables[name] = NeatSpaces.Aggregatable(
                        valid=full[:, :, -1],  # last column is valid 1 or 0
                        data=full[:, :, :-1],
                    )
                else:
                    aggregatables[name] = NeatSpaces.Aggregatable(valid=1, data=full)
        return NeatSpaces.UnflattenReturn(
            aggregatables=aggregatables,
            local_data=th.as_tensor(obs["local"]).refine_names("batch", "feature"),
        )


class NeatSpacesUniform:
    """unused"""

    def __init__(self, params: EmbeddingParamsRuntime):
        self.params = params
        self.num_agents = len(self.params.full_obs_space.spaces["agents"].spaces)
        agent_full_dim = spaces.flatdim(
            self.params.full_obs_space.spaces["agents"].spaces[0]
        )
        self.agent_obs_dim = agent_full_dim - 1  # -1 because valid is not passed
        self.neat_space = util.DictSpace(
            {
                "agents": spaces.Box(
                    -1,
                    1,
                    shape=(
                        self.num_agents,
                        agent_full_dim,
                    ),
                    dtype="float32",
                ),
            }
        )

    class UnflattenReturn(NamedTuple):
        neighbor_valid: th.Tensor
        neighbor_data: th.Tensor
        local_data: th.Tensor

        def size(self):
            return [0, 0]

    def unflatten(self, _obs):
        obs = unflatten_batch(self.neat_space, _obs)
        neighbor_full = th.as_tensor(obs["agents"]).refine_names(
            "batch", "neighbor", "feature"
        )
        if not lax_obs_asserts:
            assert (neighbor_full[:, 0, 0] == 1).all(), "self is not visible"
        return NeatSpacesUniform.UnflattenReturn(
            neighbor_valid=neighbor_full[:, :, 0],  # first column is valid 1 or 0
            neighbor_data=neighbor_full[:, :, 1:],
            local_data=neighbor_full[:, 0, 1:],
        )
