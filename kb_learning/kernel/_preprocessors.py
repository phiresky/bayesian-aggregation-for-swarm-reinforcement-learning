import numpy as np
import pandas as pd
import scipy.spatial as spatial


def compute_median_bandwidth(data, quantile=.5, sample_size=1000, preprocessor=None):
    """Computes a bandwidth for the given data set using the median heuristic.
    Other quantiles can be chosen with the quantile keyword argument.

    Arguments:
    data -- a DataFrame with the variables in the columns
    quantile -- scalar or list of scalars from the range (0,1)
    sample_size -- maximum number of sample to compute the point-wise distances

    Returns:
    bandwidths -- an array with the bandwidth for each variable
    """
    num_data_points = data.shape[0]

    if sample_size > num_data_points:
        data_points = data.values
    else:
        data_points = data.sample(sample_size).values

    if preprocessor:
        data_points = preprocessor(data_points)

    if data_points.ndim > 1:
        num_variables = data_points.shape[1]

        bandwidths = np.zeros(num_variables)
        for i in range(num_variables):
            distances = spatial.distance.pdist(data_points[:, i:i + 1])
            if quantile == .5:
                bandwidths[i] = np.median(distances)
            else:
                bandwidths[i] = pd.DataFrame(distances).quantile(quantile)
    else:
        distances = spatial.distance.pdist(data_points[:, None])
        if quantile == .5:
            bandwidths = np.median(distances)
        else:
            bandwidths = pd.DataFrame(distances).quantile(quantile)

    return bandwidths


def compute_median_bandwidth_kilobots(data, quantile=.5, sample_size=1000):
    """Computes a bandwidth for the given data set using the median heuristic.
    Other quantiles can be chosen with the quantile keyword argument.

    Arguments:
    data -- a DataFrame with the variables in the columns
    quantile -- scalar or list of scalars from the range (0,1)
    sample_size -- maximum number of sample to compute the point-wise distances

    Returns:
    bandwidths -- an array with the bandwidth for each variable
    """
    num_data_points = data.shape[0]

    if sample_size > num_data_points:
        data_points = data.values
    else:
        data_points = data.sample(sample_size).values

    data_points = data_points.reshape((sample_size, -1, 2))

    bandwidths = np.zeros(2)
    for i in range(sample_size):
        for j in range(2):
            distances = spatial.distance.pdist(data_points[i, :, [j]].T)
            if quantile == .5:
                bandwidths[j] += np.median(distances)
            else:
                bandwidths[j] += pd.DataFrame(distances).quantile(quantile)

    return bandwidths / sample_size


def select_reference_set_randomly(data, size, consecutive_sets=1, group_by=None):
    """selects a random reference set from the given DataFrame. Consecutive sets are computed from the first random
    reference set, where it is assured that only data points are chosen for the random set that have the required
    number of successive data points. Using the group_by argument allows to ensure that all consecutive samples are
    from the same group.

    :param data: a pandas.DataFrame with the samples to choose from
    :param size: the number of samples in the reference set
    :param consecutive_sets: the number of consecutive sets returned by this function (default: 1)
    :param group_by: a group_by argument to ensure that the consecutive samples are from the same group as the first
    random sample
    :return: a tuple with the reference sets
    """
    weights = np.ones(data.shape[0])

    if group_by is not None:
        gb = data.groupby(level=group_by)
        last_windows_idx = [ix[-i] for _, ix in gb.indices.items() for i in range(1, consecutive_sets)]
        weights[last_windows_idx] = 0
    else:
        last_windows_idx = [data.index[-i] for i in range(1, consecutive_sets + 1)]
        weights[last_windows_idx] = 0

    # select reference set
    if weights.sum() <= size:
        # if there is not enough data, we take all data points
        reference_set1 = data.loc[weights == 1].index.sort_values()
    else:
        # otherwise we chose a random reference set from the data
        reference_set1 = data.sample(n=size, weights=weights).index.sort_values()

    if consecutive_sets > 1:
        reference_set = [reference_set1]
        for i in range(1, consecutive_sets):
            if type(reference_set1) is pd.MultiIndex:
                reference_set_i = pd.MultiIndex.from_tuples([*map(lambda t: (*t[:-1], t[-1] + i),
                                                                  reference_set1.values)])
                reference_set_i.set_names(reference_set1.names, inplace=True)
                reference_set.append(reference_set_i)
            else:
                reference_set_i = pd.Index(data=reference_set1 + i, name=reference_set1.name)
                reference_set.append(reference_set_i)

    else:
        reference_set = reference_set1

    return tuple(reference_set)


def select_reference_set_by_kernel_activation(data: pd.DataFrame, size: int, kernel_function,
                                              batch_size: int = 1, start_from: pd.Index = None) -> pd.Index:
    """
    Iteratively selects a subset from the given data by applying a heuristic that is based on the kernel activations of
    the data with the already selected data points. The returned If the consecutive_sets parameter is greater than 1,
    multiple

    :param data: a pandas.DataFrame with the data from which the subset should be selected
    :param size: the size of the subset (if data has less data points, all data points are selected into the subset.)
    :param kernel_function: the kernel function for computing the kernel activations
    :param batch_size: number of reference samples to choose at once
    :param start_from:
    :return: a tuple of
    """
    num_reference_data_points = data.shape[0]

    # if we have not enough data to select a reference set, we take all data points
    if num_reference_data_points <= size:
        reference_set = data.index.sort_values()
    else:
        if start_from is not None:
            reference_set = start_from.tolist()
        else:
            reference_set = data.sample(batch_size).index.tolist()

        samples_to_add = size - len(reference_set)
        kernel_matrix = np.zeros((size + 1, num_reference_data_points))

        if samples_to_add < 0:
            reference_set = reference_set[:samples_to_add]
        elif samples_to_add > 0:

            kernel_matrix[:len(reference_set), :] = kernel_function(data.loc[reference_set].values,
                                                                    data.values)
            kernel_matrix[-1, reference_set] = 1000

            for i in range(samples_to_add//batch_size):
                max_kernel_activations = kernel_matrix.max(0)
                for j in range(batch_size):
                    next_reference_point = np.argmin(max_kernel_activations)
                    max_kernel_activations[next_reference_point] += 1000

                    reference_set.append(next_reference_point)

                # compute kernel activations for last chosen kernel samples
                kernel_matrix[i:i + batch_size, :] = kernel_function(data.loc[reference_set[-batch_size:]].values,
                                                                     data.values)

            reference_set = pd.Index(data=reference_set, name=data.index.names)

    return reference_set


def compute_mean_position(data):
    # number of samples in data
    q = data.shape[0]
    # number of kilobots in data
    num_kb = data.shape[1] // 2

    data_reshaped = data.reshape(q, num_kb, 2)
    return np.mean(data_reshaped, axis=1)


def compute_mean_position_pandas(data):
    mean_df = pd.DataFrame(index=data.index, columns=['x', 'y'])

    for d in mean_df.columns:
        mean_df[d] = data.loc[:, (slice(None), d)].mean(axis=1)

    return mean_df


def compute_mean_and_cov_position(data):
    # number of samples in data
    q = data.shape[0]
    # number of kilobots in data
    num_kb = data.shape[1] // 2

    data_reshaped = data.reshape(q, num_kb, 2)
    data_mean = np.mean(data_reshaped, axis=1, keepdims=True)
    data_norm = data_reshaped - data_mean
    data_cov = np.einsum('qni,qnk->qik', data_norm, data_norm)
    data_cov = data_cov[:, [0, 0, 1], [0, 1, 1]]
    return np.c_[data_mean.squeeze(axis=1), data_cov]


class angle_from_swarm_mean:
    def __init__(self, swarm_coordinates, output_tiling=1):
        self.swarm_coordinates = swarm_coordinates
        self.output_tiling = output_tiling

    def __call__(self, states):
        mean_data = -compute_mean_position(states[:, self.swarm_coordinates])
        return np.tile(np.arctan2(mean_data[:, 1], mean_data[:, 0]).reshape((-1, 1)), (1, self.output_tiling))


class step_towards_center:
    def __init__(self, coordinates, step_size: float=.01):
        self.coordinates = coordinates
        self.step_size = step_size

    def __call__(self, states):
        light_direction = -states[:, self.coordinates]
        light_norm = np.linalg.norm(light_direction, keepdims=True)
        return np.divide(light_direction, light_norm, where=light_norm != .0) * self.step_size
        # return light_direction / np.linalg.norm(light_direction, axis=1, keepdims=True) * .02

