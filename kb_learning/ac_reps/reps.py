import numpy as np
from numexpr import evaluate as ev
from scipy.optimize import minimize

import logging

logger = logging.getLogger('kb_learning.reps')


class ActorCriticReps:
    def __init__(self):
        # upper bound on the KL between the old and new state-action distribution
        self.epsilon = 0.5

        # regularization of parameters theta in dual function
        self.alpha = 0.0

        self.max_iter_optim = 100
        self.max_iter_reps = 40
        self.tolerance_g = 1e-6
        self.tolerance_f = 1e-6

    def _dual_function(self, Q, phi, phi_hat, theta, eta, num_features):
        epsilon = self.epsilon

        v = phi.dot(theta)
        v_hat = phi_hat.dot(theta)

        advantage = Q - v
        max_advantage = advantage.max()
        q_norm = Q - max_advantage
        log_z = (q_norm - v) / eta

        g = 0
        g_dot = np.zeros(phi.shape[1] + 1)

        if log_z.max() > 500:
            g = 1e30 - eta
            g_dot[-2] = -1
            return g, g_dot

        z = ev('exp(log_z)')
        sum_z = z.sum()

        realmin = np.finfo(np.double).tiny
        if sum_z < realmin:
            sum_z = realmin

        g_log_part = (1.0 / phi.shape[0]) * sum_z

        g += eta * np.log(g_log_part) + v_hat + max_advantage
        g += eta * epsilon + self.alpha * (theta.dot(theta))

        # gradient
        if (eta * sum_z) == 0:
            g_dot_eta = 1e100
        else:
            g_dot_eta = epsilon + np.log(g_log_part) - (z * (q_norm - v)).sum() / (eta * sum_z)
        g_dot[-1] = g_dot_eta

        g_dot_theta = phi_hat - (phi * z[:, None]).sum(0) / sum_z + 2 * self.alpha * theta
        g_dot[0:num_features] = g_dot_theta

        return g, 0.5 * g_dot

    def _dual_eta(self, Q, phi, phi_hat, theta, eta):
        epsilon = self.epsilon
        v = phi.dot(theta)
        v_hat = phi_hat.dot(theta)

        advantage = Q - v
        max_advantage = advantage.max()
        log_z = (advantage - max_advantage) / eta

        z = ev('exp(log_z)')
        sum_z = z.sum(axis=0)
        mean_z = sum_z / phi.shape[0]

        g = eta * epsilon + max_advantage + eta * np.log(mean_z) + self.alpha * (theta.dot(theta)) + v_hat
        g_dot_eta = epsilon + np.log(mean_z) - (z * (advantage - max_advantage)).sum() / (eta * sum_z)

        return g, 0.5 * g_dot_eta

    def _dual_theta(self, Q, phi, phi_hat, theta, eta):
        epsilon = self.epsilon

        v = phi.dot(theta)
        v_hat = phi_hat.dot(theta)

        advantage = Q - v
        max_advantage = advantage.max()
        log_z = (advantage - max_advantage) / eta

        z = ev('exp(log_z)')
        sum_z = z.sum(axis=0)
        mean_z = sum_z / phi.shape[0]

        g = eta * epsilon + max_advantage + eta * np.log(mean_z) + self.alpha * (theta.dot(theta)) + v_hat
        g_dot_theta = phi_hat - (phi * z[:, None]).sum(0) / sum_z + 2 * self.alpha * theta

        return g, 0.5 * g_dot_theta

    def _numerical_dual_gradient(self, Q, phi, phi_hat, theta, eta):
        params = np.r_[theta, eta]
        g_dot_numeric = np.zeros(params.size)

        num_features = phi.shape[1]

        g, g_dot = self._dual_function(Q, phi, phi_hat, theta, eta, num_features)

        step_size = np.maximum(np.minimum(abs(params) * 1e-4, 1e-6), 1e-6)
        for i in range(params.size):
            params_temp = params
            params_temp[i] = params[i] - step_size[i]

            g1, tmp = self._dual_function(Q, phi, phi_hat, params_temp[:-1], params_temp[-1], num_features)

            params_temp = params
            params_temp[i] = params[i] + step_size[i]

            g2, tmp = self._dual_function(Q, phi, phi_hat, params_temp[:-1], params_temp[-1], num_features)
            g_dot_numeric[i] = (g2 - g1) / (step_size[i] * 2)

        return g_dot, g_dot_numeric

    @staticmethod
    def _compute_weights_from_theta_and_eta(Q, phi, theta, eta):
        advantage = Q - phi.dot(theta)
        max_advantage = advantage.max()

        w = ev('exp((advantage - max_advantage) / eta)')
        return w / w.sum()

    @staticmethod
    def _get_KL_divergence(weighting, num_samples):
        p = weighting / weighting.sum()

        return np.nansum(p * np.log(p * num_samples))

    def _optimize_dual_function(self, Q, phi, phi_hat, theta, eta):
        lower_bound = np.r_[-1e5 * np.ones(phi.shape[1]), 1e-10]
        upper_bound = np.r_[+1e5 * np.ones(phi.shape[1]), 1e5]
        bounds = list(map(tuple, np.c_[lower_bound, upper_bound]))

        # start_params = np.r_[theta, eta]
        #
        # num_features = phi.shape[1]

        # test gradient
        # g_dot, g_dot_numeric = self._numerical_dual_gradient(Q=Q, phi=phi, phi_hat=phi_hat, theta=theta, eta=eta)
        # logger.info('Gradient error: {:f}'.format(abs(g_dot - g_dot_numeric).max()))

        # def optim_func(params):
        #     return self._dual_function(Q=Q, phi=phi, phi_hat=phi_hat,
        #                                theta=params[0:num_features], eta=params[-1], num_features=num_features)
        #
        # res = minimize(optim_func, start_params, method='L-BFGS-B',
        #                bounds=bounds, jac=True,
        #                options={'maxiter': self.max_iter_optim,
        #                         'gtol':    self.tolerance_g,
        #                         'ftol':    self.tolerance_f,
        #                         'disp':    False})
        #
        # return res.x[0:phi.shape[1]], res.x[-1]

        def optim_dual_eta(params, theta):
            return self._dual_eta(Q=Q, phi=phi, phi_hat=phi_hat, theta=theta, eta=params)

        def optim_dual_theta(params, eta):
            return self._dual_theta(Q=Q, phi=phi, phi_hat=phi_hat, theta=params, eta=eta)

        LBFGSB_options = {'maxiter': self.max_iter_optim,
                          'gtol':    self.tolerance_g,
                          'ftol':    self.tolerance_f,
                          'disp':    False}

        SLSQP_options = {'maxiter': self.max_iter_optim,
                         'ftol':    self.tolerance_f,
                         'disp':    False}

        for _ in range(3):
            res = minimize(optim_dual_eta, eta, method='SLSQP', jac=True,
                           args=theta, bounds=((1e-8, 1e8),), options=SLSQP_options)

            eta = res.x

            res = minimize(optim_dual_theta, theta, method='L-BFGS-B', jac=True,
                           args=eta, bounds=bounds[:-1], options=LBFGSB_options)

            theta = res.x

        return theta, eta

    def compute_weights(self, Q, phi):
        num_samples, num_features = phi.shape

        # self.Q = Q
        # self.PHI_S = phi
        # self.PHI_HAT = phi.mean(0)
        phi_hat = phi.mean(0)

        # initial params
        theta = np.zeros(num_features)
        eta = max(1.0, Q.std() * 0.1)

        best_feature_error = np.Inf
        last_feature_error = np.Inf
        without_improvement = 0

        return_weights = np.ones(num_samples) / num_samples

        for i in range(self.max_iter_reps):
            theta, eta = self._optimize_dual_function(Q, phi, phi_hat, theta, eta)

            weights = self._compute_weights_from_theta_and_eta(Q, phi, theta, eta)
            kl_divergence = self._get_KL_divergence(weights, num_samples)

            if kl_divergence > 3 or np.isnan(kl_divergence):
                logger.warning('KL_divergence warning')

            state_feature_difference = phi_hat - (phi * weights.reshape((-1, 1))).sum(0)

            feature_error = abs(state_feature_difference).max()
            logger.info('Feature Error: {:f}, KL: {:f}'.format(feature_error, kl_divergence))

            if not np.isinf(best_feature_error) and i >= 10 and feature_error >= best_feature_error:
                without_improvement = without_improvement + 1
            if without_improvement >= 3:
                logger.info('No improvement within the last 3 iterations.')
                break

            if abs(kl_divergence - self.epsilon) < 0.05 \
                    and feature_error < 0.01 \
                    and feature_error < best_feature_error:

                logger.info('Accepted solution.')
                without_improvement = 0
                return_weights = weights

                best_feature_error = feature_error

                if abs(kl_divergence - self.epsilon) < 0.05 \
                        and feature_error < 0.001:
                    logger.info('Found sufficient solutions.')
                    break

            if (abs(state_feature_difference) - last_feature_error).max() > -0.000001:
                logger.info('Solution unchanged or degrading, restart from new point')
                theta = np.random.random(theta.shape) * 2.0 - 1.0
                last_feature_error = np.Inf
            else:
                last_feature_error = feature_error

        return return_weights
