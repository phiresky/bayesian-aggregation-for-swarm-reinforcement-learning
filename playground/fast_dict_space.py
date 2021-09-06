from typing import OrderedDict

import numpy as np
import numpy.lib.recfunctions as rfn
from gym import spaces


def structured_to_unstructured(arr: np.ndarray):
    return rfn.structured_to_unstructured(arr, casting="no")


def walk_dtype(space: spaces.Space) -> tuple[np.dtype, tuple[int]]:
    """convert a gym Space into a numpy struct dtype"""
    if isinstance(space, spaces.Dict):
        return (
            np.dtype([(name, *walk_dtype(s)) for name, s in space.spaces.items()]),
            (),
        )

    if isinstance(space, spaces.Tuple):
        if not all(a == space.spaces[0] for a in space.spaces):
            raise Exception("tuple with different values not supported")
        res, innerc = walk_dtype(space.spaces[0])
        return np.dtype(res), (len(space.spaces), *innerc)
    if isinstance(space, spaces.Box):
        return np.dtype(space.dtype), space.shape
    raise Exception(f"can't convert space to dtype: {space}")


class FastDictSpace(spaces.Box):
    """like a Dict observation space, but *much* faster since it is based on numpy struct arrays"""

    def __init__(self, dict_space: spaces.Dict):
        tmp = spaces.flatten_space(dict_space)
        super().__init__(tmp.low, tmp.high, tmp.shape, dtype=tmp.dtype)
        self.as_dict_space = dict_space
        self._dtype, self._shape = walk_dtype(self.as_dict_space)

    def get_dtype(self):
        if self._shape != ():
            raise Exception("expected empty tuple")
        return self._dtype

    def create_nans(self, shape):
        # create obs that's in this space, fill with NaNs
        dtype = self.get_dtype()
        return np.full(shape, fill_value=np.nan, dtype=dtype)

    def create_instance(self):
        # create obs that's in this space
        dtype = self.get_dtype()
        data = np.full((), np.nan, dtype=dtype)
        return data

    def contains(self, x):
        return super().contains(structured_to_unstructured(x))


def structured_to_dict(arr: np.ndarray):
    import numpy as np

    if np.ndim(arr) == 0:
        if arr.dtype.names == None:
            return arr.item()
        # accessing by int does *not* work when arr is a zero-dimensional array!
        return {k: structured_to_dict(arr[k]) for k in arr.dtype.names}
    return [structured_to_dict(v) for v in arr]


def ds(d):
    return [(k, v) for k, v in d.items()]


if __name__ == "__main__":

    box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    dict_space = spaces.Dict(
        OrderedDict(
            {
                "foo": box,
                "bar": box,
                "neighbors": spaces.Tuple(
                    [spaces.Dict(OrderedDict({"x": box, "y": box})) for _ in range(3)]
                ),
            }
        )
    )
    d = FastDictSpace(dict_space)

    o = d.create_instance()
    o["foo"] = 1
    o["bar"] = 2
    print(o["foo"], o["bar"])
    # set x of every neighbor
    o["neighbors"][:]["x"] = 10
    # set y of one neighbor
    o["neighbors"][1]["y"] = 20

    print("full", structured_to_dict(o))
