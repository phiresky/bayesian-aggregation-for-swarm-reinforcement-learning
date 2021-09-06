from typing import List, Literal

import torch.nn

nn = torch.nn


ActivationFn = Literal["Tanh", "LeakyReLU", "ReLU"]


def get_activation_fn(af_name: ActivationFn):
    return getattr(nn, af_name)()


def mlp(
    sizes: List[int],
    inner_activation: nn.Module,
    last_activation: nn.Module = None,
):
    """
    return a simple dense NN with the given layer sizes
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = (
            inner_activation
            if j < len(sizes) - 2 or not last_activation
            else last_activation
        )
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]

    return nn.Sequential(*layers)
