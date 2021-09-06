from pprint import pformat
from typing import Any, OrderedDict, Union

import numpy as np
import numpy.lib.recfunctions as rfn
import torch as th
from gym import spaces


# indexing for named dimensions
# https://github.com/pytorch/pytorch/issues/29023
# ns(t, foo=1:5) is what t[foo=1:5] should be
def ns(t: th.Tensor, **idx: Union[slice, int]):
    slices = [
        slice(None) for _ in t.shape
    ]  # by default, select every element in each dimension
    for dimension_name, selector in idx.items():
        if dimension_name not in t.names:
            raise Exception(
                f"dimension {dimension_name} does not exist in tensor {t} of shape {namedshapestr(t)}"
            )
        # for each given dimension, set the wanted slice
        slices[t.names.index(dimension_name)] = selector
    return t[tuple(slices)]


def namedshapestr(t: th.Tensor):
    return ",".join(f"{n}={s}" for n, s in zip(t.names, t.shape))


def namedshape(t: th.Tensor):
    return dict(zip(t.names, t.shape))


def stripnames(tensor: th.Tensor, *names) -> th.Tensor:
    """many th ops sadly don't support named tensors. this remove names safely by checking them first"""
    return tensor.refine_names(*names).rename(None)


def structured_to_unstructured(arr: np.ndarray):
    return rfn.structured_to_unstructured(arr, casting="no")


def unstructured_to_structured(arr: np.ndarray, dtype: np.dtype):
    return rfn.unstructured_to_structured(arr, dtype, casting="no")


def structured_to_debug_dict(arr: np.ndarray):
    import numpy as np

    if np.ndim(arr) == 0:
        if arr.dtype.names == None:
            return arr.item()
        # accessing by int does *not* work when arr is a zero-dimensional array!
        return {k: structured_to_debug_dict(arr[k]) for k in arr.dtype.names}
    return [structured_to_debug_dict(v) for v in arr]


def unflatten_batch(space, x) -> Any:
    """Unflatten a data point from a space.
    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.
    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    import numpy as np
    from gym.spaces import (
        Box,
        Dict,
        Discrete,
        MultiBinary,
        MultiDiscrete,
        Tuple,
        flatdim,
    )

    if isinstance(space, Box):
        return th.as_tensor(x).reshape((-1, *space.shape))
    elif isinstance(space, Discrete):
        batch, indz = np.nonzero(x, as_tuple=True)
        assert np.array_equal(batch, np.arange(0, len(batch)))
        return indz  # assume only one per dim
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = th.split(x, dims, dim=1)
        list_unflattened = [
            unflatten_batch(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = th.split(x, dims, dim=1)
        return {
            key: unflatten_batch(s, flattened)
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        }
    elif isinstance(space, MultiBinary):
        return th.as_tensor(x).reshape((-1, *space.shape))
    elif isinstance(space, MultiDiscrete):
        return th.as_tensor(x).reshape((-1, *space.shape))
    else:
        raise NotImplementedError(f"noti {type(space)}")


def does_not_contain_reason(space: spaces.Space, x, prefix=""):
    """the same as space.contains(x) but better readable output than integrated one"""
    if isinstance(space, spaces.Tuple):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        if not isinstance(x, tuple):
            return f"{prefix} is {type(x)} not tuple"
        if len(x) != len(space.spaces):
            return f"{prefix} has {len(x)=} not {len(space.spaces)}"
        for i, (space, part) in enumerate(zip(space.spaces, x)):
            if reason := does_not_contain_reason(space, part, prefix=f"{prefix}[{i}]"):
                return reason
        return None
    if isinstance(space, spaces.Dict):
        if not isinstance(x, dict):
            return prefix + " not dict"
        if len(x) != len(space.spaces):
            toomuch = set(x) - set(space.spaces)
            missing = set(space.spaces) - set(x)
            return f"{prefix} has {toomuch} too much, {missing} missing"
        for k, space in space.spaces.items():
            if k not in x:
                return f"{prefix}.{k} is missing"
            if reason := does_not_contain_reason(space, x[k], prefix=f"{prefix}.k"):
                return reason
        return None
    return prefix if not space.contains(x) else None


def throw_if_not_contains(space: spaces.Space, x):
    if not space.contains(x):
        reason = does_not_contain_reason(space, x)
        raise Exception(
            f"integrity error: observation space does not contain observation: {reason}"
        )
    return


def doneify(dones):
    import numpy as np

    dones = np.asarray(dones)
    a = dones.any()
    b = dones.all()
    if a != b:
        raise Exception("not all done at same time")
    return a


spaces.Dict.__repr__ = lambda self: pformat(dict(self.spaces))

innerinit = spaces.Dict.__init__


def _initwrap(self, spaces=None, **kwargs):
    if spaces is None or (
        isinstance(spaces, dict) and not isinstance(spaces, OrderedDict)
    ):
        raise Exception(
            f"passing spaces as normal dict reorders them, disallowed (use OrderedDict or list of tuples)"
        )
    innerinit(self, spaces, **kwargs)


spaces.Dict.__init__ = _initwrap  # just for safety, don't allow passing unordered dict


def DictSpace(spaces_faces: dict):
    # different behaviour than spaces.Dict: don't reorder (only makes sense in python 3.7+)
    return spaces.Dict(OrderedDict(spaces_faces))


def TupleDictSpace(space: spaces.Dict, count: int):
    return spaces.Tuple(tuple([space for _ in range(count)]))
