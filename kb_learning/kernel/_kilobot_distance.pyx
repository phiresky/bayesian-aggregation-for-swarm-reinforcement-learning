from cython.parallel import parallel, prange
import cython
cimport cython

import numpy as np
cimport numpy as np

from math import exp
from libc.math cimport exp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class EmbeddingCovariance:
    def __init__(self):
        self.bandwidth = np.array([1., 1.])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double[:, :] get_cov_matrix(self, double[:, :, :] a, double[:, :, :] b=None):
        """
        :param a: q x n x d matrix of kilobot positions
        :param b: r x m x d matrix of kilobot positions
        :return:  q x r matrix if a and b are given, q x 1 if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, n, d
        cdef Py_ssize_t r, m
        cdef Py_ssize_t i, j, k, l, s

        q = a.shape[0]
        n = a.shape[1]
        d = a.shape[2]

        cdef double[:, :] sq_dist
        if b is None:
            sq_dist = np.empty((q, 1))
        else:
            assert b.shape[2] == d
            r = b.shape[0]
            m = b.shape[1]
            sq_dist = np.empty((q, r))

        if len(self.bandwidth) > d:
            self.bandwidth = self.bandwidth.reshape((-1, d)).mean(axis=0)

        assert d == len(self.bandwidth)

        cdef double[:] bw = 1 / self.bandwidth

        cdef DTYPE_t kb_sum_1 = .0, kb_sum_2 = .0
        with nogil, parallel():
            if b is None:
                for i in prange(q, schedule='guided'):
                    sq_dist[i, 0] = .0
                    for j in range(n):
                        for k in range(n):
                            kb_sum_1 = .0
                            for s in range(d):
                                kb_sum_1 += (a[i, j, s] - a[i, k, s])**2 * bw[s]
                            sq_dist[i, 0] += exp(-kb_sum_1 / 2)
                    sq_dist[i, 0] /= n**2
            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        sq_dist[i, j] = .0
                        for k in range(n):
                            for l in range(m):
                                kb_sum_2 = .0
                                for s in range(d):
                                    kb_sum_2 += (a[i, k, s] - b[j, l, s])**2 * bw[s]
                                sq_dist[i, j] += exp(-kb_sum_2 / 2)
                        sq_dist[i, j] /= n * m
                        sq_dist[i, j] *= 2

        return sq_dist

    def get_gram_diag(self, np.ndarray data):
        return np.ones(data.shape[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef double[:, :, :] get_cov_matrix_gradient(self, double[:, :, :] a, double[:, :, :] b=None):
        """
        :param a: q x n x d matrix of kilobot positions
        :param b: r x m x d matrix of kilobot positions
        :return:  q x r x 2 matrix if a and b are given, q x 1 x 2 if only a is given
        """
        # assert a.dtype == DTYPE

        cdef Py_ssize_t q, n, d
        cdef Py_ssize_t r, m
        cdef Py_ssize_t i, j, k, l, s

        q = a.shape[0]
        n = a.shape[1]
        d = a.shape[2]

        cdef double[:, :, :] d_sq_dist_d_bw
        if b is None:
            d_sq_dist_d_bw = np.empty((q, 1, d))
        else:
            assert b.shape[2] == d
            r = b.shape[0]
            m = b.shape[1]
            d_sq_dist_d_bw = np.empty((q, r, d))

        if len(self.bandwidth) > d:
            self.bandwidth = self.bandwidth.reshape((-1, d)).mean(axis=0)

        assert d == len(self.bandwidth)

        cdef double[:] bw = 1 / self.bandwidth

        cdef DTYPE_t kb_sum_1 = .0, sq_dist_1 =.0, kb_sum_2 = .0, sq_dist_2 = .0
        with nogil, parallel():
            if b is None:
                for i in prange(q, schedule='guided'):
                    for s in range(d):
                        d_sq_dist_d_bw[i, 0, s] = .0
                    for k in range(n):
                        for l in range(n):
                            kb_sum_1 = .0
                            for s in range(d):
                                kb_sum_1 += (a[i, k, s] - a[i, l, s])**2 * bw[s]
                            kb_sum_1 = exp(-kb_sum_1 / 2) / (2 * n**2)
                            for s in range(d):
                                d_sq_dist_d_bw[i, 0, s] += kb_sum_1 * (a[i, k, s] - a[i, l, s])**2 * bw[s]**2

            else:
                for i in prange(q, schedule='guided'):
                    for j in range(r):
                        for s in range(d):
                            d_sq_dist_d_bw[i, j, s] = .0
                        for k in range(n):
                            for l in range(m):
                                kb_sum_2 = .0
                                for s in range(d):
                                    kb_sum_2 += (a[i, k, s] - b[j, l, s])**2 * bw[s]
                                kb_sum_2 = exp(-kb_sum_2 / 2) / (n * m)
                                for s in range(d):
                                    d_sq_dist_d_bw[i, j, s] += kb_sum_2 * (a[i, k, s] - b[j, l, s])**2 * bw[s]**2

        return d_sq_dist_d_bw


cdef class EmbeddedSwarmDistance:
    def __init__(self):
        self._kernel_func = EmbeddingCovariance()

    cpdef np.ndarray get_distance_matrix(self, np.ndarray k1, np.ndarray k2=None):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
        :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
        :return: q x r matrix with the distances between the configurations in k1 and k2
        """
        cdef int num_kb_1, num_kb_2
        cdef int q, r
        cdef int i, j

        # number of samples in k1
        q = k1.shape[0]

        # number of kilobots in k1
        num_kb_1 = k1.shape[1] // 2

        # reshape matrices
        cdef np.ndarray k1_reshaped, k2_reshaped

        k1_reshaped = k1.reshape(q, num_kb_1, 2)
        if k2 is not None:
            # number of samples in k2
            r = k2.shape[0]
            # number of kilobots in k2
            num_kb_2 = k2.shape[1] // 2

            k2_reshaped = k2.reshape(r, num_kb_2, 2)
        else:
            r = q

        cdef double[:, :] k_n, k_m, k_nm
        k_n = self._kernel_func.get_cov_matrix(k1_reshaped)
        if k2 is not None:
            k_m = self._kernel_func.get_cov_matrix(k2_reshaped)
            k_nm = self._kernel_func.get_cov_matrix(k1_reshaped, k2_reshaped)
        else:
            k_m = k_n
            k_nm = self._kernel_func.get_cov_matrix(k1_reshaped, k1_reshaped)

        for i in range(q):
            for j in range(r):
                k_nm[i, j] = -k_nm[i, j] + k_n[i, 0] + k_m[j, 0]

        return np.asarray(k_nm)

    cpdef np.ndarray get_distance_matrix_gradient(self, np.ndarray k1, np.ndarray k2=None):
        """Computes the kb distance matrix between any configuration in k1 and any configuration in k2.

        :param k1: q x 2*d1 matrix of q configurations with each d1 kilobots
        :param k2: r x 2*d2 matrix of r configurations with each d2 kilobots
        :return: q x r matrix with the distances between the configurations in k1 and k2
        """
        cdef int num_kb_1, num_kb_2
        cdef int q, r
        cdef int i, j

        # number of samples in k1
        q = k1.shape[0]

        # number of kilobots in k1
        num_kb_1 = k1.shape[1] // 2

        # reshape matrices
        cdef np.ndarray k1_reshaped, k2_reshaped

        k1_reshaped = k1.reshape(q, num_kb_1, 2)
        if k2 is not None:
            # number of samples in k2
            r = k2.shape[0]
            # number of kilobots in k2
            num_kb_2 = k2.shape[1] // 2

            k2_reshaped = k2.reshape(r, num_kb_2, 2)
        else:
            r = q

        cdef double[:, :, :] k_n, k_m, k_nm
        k_n = self._kernel_func.get_cov_matrix_gradient(k1_reshaped)
        if k2 is not None:
            k_m = self._kernel_func.get_cov_matrix_gradient(k2_reshaped)
            k_nm = self._kernel_func.get_cov_matrix_gradient(k1_reshaped, k2_reshaped)
        else:
            k_m = k_n
            k_nm = self._kernel_func.get_cov_matrix_gradient(k1_reshaped, k1_reshaped)

        for i in range(q):
            for j in range(r):
                for d in range(2):
                    k_nm[i, j, d] = -k_nm[i, j, d] + k_n[i, 0, d] + k_m[j, 0, d]

        return np.asarray(k_nm)

    def __call__(self, k1, k2=None):
        return self.get_distance_matrix(k1, k2)

    def diag(self, X):
        if len(X.shape) > 1:
            return np.zeros((X.shape[0],))
        return np.zeros(1)

    @property
    def bandwidth(self):
        return self._kernel_func.bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        if np.isscalar(bandwidth):
            self._kernel_func.bandwidth = np.array([bandwidth] * 2)
        else:
            self._kernel_func.bandwidth = bandwidth

