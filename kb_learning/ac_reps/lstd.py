import numpy as np
from ..tools import np_chunks


class LeastSquaresTemporalDifference:
    def __init__(self):
        self.discount_factor = 0.98
        self.lstd_regularization_factor = 1e-8
        self.lstd_projection_regularization_factor = 1e-6

    def learn_q_function(self, phi, phi_next, rewards):
        """compute the parameters theta of an approximation to the Q-function Q(s,a) = phi(s,a) * theta

        :param phi: the state-action features for samples s, a
        :param phi_next: the next state action features for samples s', a'
        :param rewards: the reward for taking action a in state s
        :return: the parameters theta of the Q-function approximation
        """
        _A = phi.T.dot(phi - self.discount_factor * phi_next)
        _b = (phi * rewards).sum(axis=0).T

        _I = np.eye(phi.shape[1])

        _C = np.linalg.solve(phi.T.dot(phi) + self.lstd_regularization_factor * _I, phi.T).T
        _X = _C.dot(_A + self.lstd_regularization_factor * _I)
        _y = _C.dot(_b)

        return np.linalg.solve(_X.T.dot(_X) + self.lstd_projection_regularization_factor * _I, _X.T.dot(_y))
