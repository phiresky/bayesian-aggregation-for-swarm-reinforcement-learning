cimport numpy as np


cdef class EmbeddingCovariance:
    cdef public np.ndarray bandwidth
    cdef double[:, :] get_cov_matrix(self, double[:, :, :] a, double[:, :, :] b=?)
    cdef double[:, :, :] get_cov_matrix_gradient(self, double[:, :, :] a, double[:, :, :] b=?)

cdef class EmbeddedSwarmDistance:
    cdef EmbeddingCovariance _kernel_func
    cpdef np.ndarray get_distance_matrix(self, np.ndarray k1, np.ndarray k2=?)
    cpdef np.ndarray get_distance_matrix_gradient(self, np.ndarray k1, np.ndarray k2=?)
