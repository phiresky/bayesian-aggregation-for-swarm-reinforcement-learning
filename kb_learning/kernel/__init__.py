import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from ._kilobot_distance import EmbeddedSwarmDistance, EmbeddingCovariance
from ._mahalanobis_distance import MahaDist, MeanSwarmDist, MeanCovSwarmDist
from ._distances import PeriodicDist
from ._kernel import KilobotEnvKernel
from ._preprocessors import compute_median_bandwidth, compute_median_bandwidth_kilobots, \
    select_reference_set_randomly, select_reference_set_by_kernel_activation, compute_mean_position, \
    compute_mean_and_cov_position, compute_mean_position_pandas, angle_from_swarm_mean, step_towards_center

__all__ = [
    'EmbeddedSwarmDistance',
    'MahaDist',
    'MeanSwarmDist',
    'MeanCovSwarmDist',
    'PeriodicDist',
    'KilobotEnvKernel',
    'compute_median_bandwidth',
    'compute_median_bandwidth_kilobots',
    'select_reference_set_randomly',
    'select_reference_set_by_kernel_activation',
    'compute_mean_position',
    'compute_mean_and_cov_position',
    'compute_mean_position_pandas',
    'angle_from_swarm_mean',
    'step_towards_center'
]
