from typing import Tuple
import numpy as np
from scipy import linalg

from kb_learning.kernel import KilobotEnvKernel

import logging
logger = logging.getLogger('kb_learning.gp')


class SparseWeightedGP:
    def __init__(self, kernel: KilobotEnvKernel, output_dim: int, noise_variance=1e-6,
                 mean_function=None, output_bounds: Tuple[np.ndarray, np.ndarray]=None, eval_mode=False):
        # TODO add documentation
        """

        :param kernel:
        :param output_bounds:
        """
        self.mean_function = mean_function
        self.noise_variance = noise_variance
        self.min_variance = 1e-8
        self.cholesky_regularizer = 1e-9

        self.Q_Km = None
        self.alpha = None
        self.trained = False

        self.sparse_inputs = None
        self.k_cholesky = None
        self.kernel = kernel
        self.output_dim = output_dim
        self.output_bounds = output_bounds

        self.eval_mode = eval_mode

    def to_dict(self):
        input_dict = dict()
        input_dict['class'] = 'SparseWeightedGP'
        input_dict['kernel'] = self.kernel.to_dict()
        input_dict['noise_variance'] = self.noise_variance
        input_dict['mean_function'] = self.mean_function
        input_dict['output_bounds'] = self.output_bounds
        input_dict['output_dim'] = self.output_dim
        input_dict['eval_mode'] = self.eval_mode

        if self.trained:
            gp_dict = dict()
            gp_dict['Q_Km'] = self.Q_Km
            gp_dict['alpha'] = self.alpha
            gp_dict['sparse_inputs'] = self.sparse_inputs
            gp_dict['k_cholesky'] = self.k_cholesky
            input_dict['gp'] = gp_dict

        return input_dict

    @staticmethod
    def from_dict(input_dict: dict):
        policy_class = input_dict.pop('class')
        assert policy_class == 'SparseWeightedGP'
        gp_dict = input_dict.pop('gp', None)
        kernel_dict = input_dict.pop('kernel')
        input_dict['kernel'] = KilobotEnvKernel.from_dict(kernel_dict)
        spwgp = SparseWeightedGP(**input_dict)

        if gp_dict:
            spwgp.Q_Km = gp_dict['Q_Km']
            spwgp.alpha = gp_dict['alpha']
            spwgp.sparse_inputs = gp_dict['sparse_inputs']
            spwgp.k_cholesky = gp_dict['k_cholesky']
            spwgp.trained = True

        return spwgp

    def train(self, inputs, outputs, weights, sparse_inputs, *args, **kwargs):
        """
        :param inputs: N x d_input
        :param outputs: N x d_output
        :param weights: N
        :param sparse_inputs: M x d_output
        :return:
        """
        if outputs.ndim == 1:
            outputs = outputs.reshape((-1, 1))

        self.sparse_inputs = sparse_inputs

        weights /= weights.max()

        # kernel matrix on subset of samples
        K_m = self.kernel(sparse_inputs)
        K_mn = self.kernel(sparse_inputs, inputs)

        # fix cholesky with regularizer
        reg_I = self.cholesky_regularizer * np.eye(K_m.shape[0])
        while True:
            try:
                K_m_c = np.linalg.cholesky(K_m), True
                logger.debug('regularization for cholesky: {}'.format(reg_I[0, 0]))
                break
            except np.linalg.LinAlgError:
                K_m += reg_I
                reg_I *= 2
        else:
            raise Exception("SparseGPPolicy: Cholesky decomposition failed")

        L = self.kernel.diag(inputs) - np.sum(K_mn * linalg.cho_solve(K_m_c, K_mn), axis=0).squeeze() \
            + self.noise_variance * (1 / weights)
        L = 1 / L

        Q = K_m + (K_mn * L).dot(K_mn.T)
        if self.mean_function:
            outputs -= self.mean_function(inputs)

        self.alpha = (np.linalg.solve(Q, K_mn) * L).dot(outputs)

        self.Q_Km = np.linalg.pinv(K_m) - np.linalg.pinv(Q)

        self.trained = True

    def get_mean(self, inputs, return_k=False):
        if self.trained:
            k = self.kernel(inputs, self.sparse_inputs)
            mean = k.dot(self.alpha)
        else:
            k = None
            mean = np.zeros((inputs.shape[0], self.output_dim))

        if self.mean_function:
            mean += self.mean_function(inputs)

        if return_k:
            return mean, k

        return mean

    def get_mean_sigma(self, inputs):
        mean, k = self.get_mean(inputs, True)

        if self.trained:
            variance = self.kernel.diag(inputs) - np.sum(k.T * self.Q_Km.dot(k.T), axis=0) \
                        + self.noise_variance
        else:
            variance = self.kernel.variance + self.noise_variance

        if np.isscalar(variance):  # single number
            variance = np.array([variance])

        variance[variance < self.min_variance] = self.min_variance

        sigma = np.sqrt(variance)

        return mean, sigma

    def sample(self, inputs):
        if inputs.ndim == 1:
            inputs = np.array([inputs])

        gp_mean, gp_sigma = self.get_mean_sigma(inputs)

        if gp_mean.ndim > 1:
            samples = np.random.normal(gp_mean, gp_sigma[:, None])
        else:
            samples = np.random.normal(gp_mean, gp_sigma)

        if self.output_bounds:
            # check samples against bounds from action space
            samples = np.minimum(samples, self.output_bounds[1])
            samples = np.maximum(samples, self.output_bounds[0])

        return np.reshape(samples, (-1, self.output_dim))

    def __call__(self, inputs):
        if self.eval_mode:
            return self.get_mean(inputs)
        return self.sample(inputs)
