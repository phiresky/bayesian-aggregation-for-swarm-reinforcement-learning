import numpy as np
from GPy import Param
from GPy.kern.src.kern import Kern
from paramz.transformations import Logexp

from . import EmbeddedSwarmDistance, MeanSwarmDist, MeanCovSwarmDist, MahaDist, PeriodicDist


class KilobotEnvKernel(Kern):
    _name = 'kilobot_env'

    def __init__(self, kilobots_dim, light_dim=0, extra_dim=0, action_dim=0, rho=.5, variance=1.,
                 kilobots_bandwidth=None, light_bandwidth=None, extra_dim_bandwidth=None, action_bandwidth=None,
                 kilobots_dist_class=None, light_dist_class=None, extra_dim_dist_class=None, action_dist_class=None,
                 active_dims=None):
        super(KilobotEnvKernel, self).__init__(input_dim=kilobots_dim + light_dim + extra_dim + action_dim,
                                               active_dims=active_dims, name=self._name)
        self.kilobots_dim = kilobots_dim
        self.light_dim = light_dim
        self.extra_dim = extra_dim
        self.action_dim = action_dim

        if isinstance(kilobots_dist_class, str):
            if kilobots_dist_class == 'embedded':
                self.kilobots_dist = EmbeddedSwarmDistance()
            elif kilobots_dist_class == 'mean':
                self.kilobots_dist = MeanSwarmDist()
            elif kilobots_dist_class == 'mean_cov':
                self.kilobots_dist = MeanCovSwarmDist()
            else:
                raise UnknownDistClassException()
        else:
            self.kilobots_dist = kilobots_dist_class() if kilobots_dist_class else EmbeddedSwarmDistance()

        if kilobots_bandwidth is None:
            self.kilobots_bandwidth = Param('kilobots_bandwidth', np.array([1.] * 2), Logexp())
        else:
            self.kilobots_bandwidth = Param('kilobots_bandwidth', kilobots_bandwidth, Logexp())
        self.kilobots_dist.bandwidth = self.kilobots_bandwidth
        self.kilobots_bandwidth.add_observer(self, self.__kilobots_bandwidth_observer)
        self.link_parameter(self.kilobots_bandwidth)

        if light_dim:
            if isinstance(light_dist_class, str):
                if light_dist_class == 'maha':
                    self.light_dist = MahaDist()
                elif light_dist_class == 'periodic':
                    self.light_dist = PeriodicDist()
                else:
                    raise UnknownDistClassException
            else:
                self.light_dist = light_dist_class() if light_dist_class else MahaDist()

            if light_bandwidth is None:
                self.light_bandwidth = Param('light_bandwidth', np.array([1.] * light_dim), Logexp())
            else:
                self.light_bandwidth = Param('light_bandwidth', light_bandwidth, Logexp())
            self.light_dist.bandwidth = self.light_bandwidth
            self.light_bandwidth.add_observer(self, self.__light_bandwidth_observer)
            self.link_parameter(self.light_bandwidth)

        if extra_dim:
            if isinstance(extra_dim_dist_class, str):
                if extra_dim_dist_class == 'maha':
                    self.extra_dim_dist = MahaDist()
                elif extra_dim_dist_class == 'periodic':
                    self.extra_dim_dist = PeriodicDist()
                else:
                    raise UnknownDistClassException
            else:
                self.extra_dim_dist = extra_dim_dist_class() if extra_dim_dist_class else MahaDist()

            if extra_dim_bandwidth is None:
                self.extra_dim_bandwidth = Param('extra_dim_bandwidth', np.array([1.] * extra_dim), Logexp())
            else:
                self.extra_dim_bandwidth = Param('extra_dim_bandwidth', extra_dim_bandwidth, Logexp())
            self.extra_dim_dist.bandwidth = self.extra_dim_bandwidth
            self.extra_dim_bandwidth.add_observer(self, self.__extra_dim_bandwidth_observer)
            self.link_parameter(self.extra_dim_bandwidth)

        if action_dim:
            if isinstance(action_dist_class, str):
                if action_dist_class == 'maha':
                    self.action_dist = MahaDist()
                elif action_dist_class == 'periodic':
                    self.action_dist = PeriodicDist()
                else:
                    raise UnknownDistClassException
            else:
                self.action_dist = action_dist_class() if action_dist_class else MahaDist()

            if action_bandwidth is None:
                self.action_bandwidth = Param('action_bandwidth', np.array([1.] * action_dim), Logexp())
            else:
                self.action_bandwidth = Param('action_bandwidth', action_bandwidth, Logexp())
            self.action_dist.bandwidth = self.action_bandwidth
            self.action_bandwidth.add_observer(self, self.__action_bandwidth_observer)
            self.link_parameter(self.action_bandwidth)

        self.rho = Param('rho', np.array([rho]))
        self.rho.constrain_bounded(.1, .9)
        # self.rho.fix()

        self.variance = Param('variance', np.array([variance]), Logexp())
        # self.variance.fix()

        self.link_parameters(self.rho, self.variance)

    def __kilobots_bandwidth_observer(self, param, which):
        self.kilobots_dist.bandwidth = self.kilobots_bandwidth.values

    def __light_bandwidth_observer(self, param, which):
        self.light_dist.bandwidth = self.light_bandwidth.values

    def __extra_dim_bandwidth_observer(self, param, which):
        self.extra_dim_dist.bandwidth = self.extra_dim_bandwidth.values

    def __action_bandwidth_observer(self, param, which):
        self.action_dist.bandwidth = self.action_bandwidth.values

    def to_dict(self):
        input_dict = dict()
        input_dict['kilobots_dim'] = self.kilobots_dim
        input_dict['light_dim'] = self.light_dim
        input_dict['extra_dim'] = self.extra_dim
        input_dict['action_dim'] = self.action_dim
        input_dict['kilobots_bandwidth'] = self.kilobots_bandwidth.values
        if isinstance(self.kilobots_dist, EmbeddedSwarmDistance):
            input_dict['kilobots_dist_class'] = 'embedded'
        elif isinstance(self.kilobots_dist, MeanSwarmDist):
            input_dict['kilobots_dist_class'] = 'mean'
        else:
            input_dict['kilobots_dist_class'] = 'mean_cov'

        if self.light_dim:
            input_dict['light_bandwidth'] = self.light_bandwidth.values
            input_dict['light_dist_class'] = 'maha' if isinstance(self.light_dist, MahaDist) else 'periodic'
        if self.extra_dim:
            input_dict['extra_dim_bandwidth'] = self.extra_dim_bandwidth.values
            input_dict['extra_dim_dist_class'] = 'maha' if isinstance(self.extra_dim_dist, MahaDist) else 'periodic'
        if self.action_dim:
            input_dict['action_bandwidth'] = self.action_bandwidth.values
            input_dict['action_dist_class'] = 'maha' if isinstance(self.action_dist, MahaDist) else 'periodic'

        input_dict['rho'] = self.rho[0]
        input_dict['variance'] = self.variance[0]

        return input_dict

    @staticmethod
    def from_dict(input_dict):
        import copy
        input_dict = copy.deepcopy(input_dict)
        return KilobotEnvKernel(**input_dict)

    def K(self, X, X2=None, return_components=False):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.extra_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_extra_dim = X_splits[2]
        X_action = X_splits[3]

        if X2 is not None:
            Y_splits = np.split(X2, np.cumsum([self.kilobots_dim, self.light_dim, self.extra_dim]), axis=1)
            Y_kilobots = Y_splits[0]
            Y_light = Y_splits[1]
            Y_extra_dim = Y_splits[2]
            Y_action = Y_splits[3]
        else:
            Y_kilobots = None
            Y_light = None
            Y_extra_dim = None
            Y_action = None

        k_kilobots = self.kilobots_dist(X_kilobots, Y_kilobots)
        if self.light_dim:
            k_light = self.light_dist(X_light, Y_light)
        else:
            k_light = .0
        if self.extra_dim:
            k_extra_dim = self.extra_dim_dist(X_extra_dim, Y_extra_dim)
        else:
            k_extra_dim = .0
        if self.action_dim:
            k_action = self.action_dist(X_action, Y_action)
        else:
            k_action = .0

        if return_components:
            return (self.variance * np.exp((self.rho - 1) * k_kilobots - self.rho * (k_light + k_extra_dim + k_action)),
                    k_kilobots, k_light, k_extra_dim, k_action)
        return self.variance * np.exp((self.rho - 1) * k_kilobots - self.rho * (k_light + k_extra_dim + k_action))

    def Kdiag(self, X):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.extra_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_extra_dim = X_splits[2]
        X_action = X_splits[3]

        k_kilobots = self.kilobots_dist.diag(X_kilobots)
        if self.light_dim:
            k_light = self.light_dist.diag(X_light)
        else:
            k_light = .0
        if self.extra_dim:
            k_extra_dim = self.extra_dim_dist.diag(X_extra_dim)
        else:
            k_extra_dim = .0
        if self.action_dim:
            k_action = self.action_dist.diag(X_action)
        else:
            k_action = .0

        return self.variance * np.exp((self.rho - 1) * k_kilobots - self.rho * (k_light + k_extra_dim + k_action))

    def update_gradients_full(self, dL_dK, X, Y=None):
        X_splits = np.split(X, np.cumsum([self.kilobots_dim, self.light_dim, self.extra_dim]), axis=1)
        X_kilobots = X_splits[0]
        X_light = X_splits[1]
        X_extra_dim = X_splits[2]
        X_action = X_splits[3]

        if Y is not None:
            Y_splits = np.split(Y, np.cumsum([self.kilobots_dim, self.light_dim, self.extra_dim]), axis=1)
            Y_kilobots = Y_splits[0]
            Y_light = Y_splits[1]
            Y_extra_dim = Y_splits[2]
            Y_action = Y_splits[3]
        else:
            Y_kilobots = None
            Y_light = None
            Y_extra_dim = None
            Y_action = None

        k, k_kilobots, k_light, k_extra_dim, k_action = self.K(X, Y, return_components=True)

        # compute gradient w.r.t. kernel bandwidths
        dK_kb_d_bw = self.kilobots_dist.get_distance_matrix_gradient(X_kilobots, Y_kilobots)
        dK_kb_d_bw *= (dL_dK * k)[..., None] * (self.rho - 1)
        self.kilobots_bandwidth.gradient = np.sum(dK_kb_d_bw, axis=(0, 1))
        if self.light_dim:
            dK_l_d_bw = self.light_dist.get_distance_matrix_gradient(X_light, Y_light)
            dK_l_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.light_bandwidth.gradient = np.sum(dK_l_d_bw, axis=(0, 1))
        if self.extra_dim:
            dK_w_d_bw = self.extra_dim_dist.get_distance_matrix_gradient(X_extra_dim, Y_extra_dim)
            dK_w_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.extra_dim_bandwidth.gradient = np.sum(dK_w_d_bw, axis=(0, 1))
        if self.action_dim:
            dK_a_d_bw = self.action_dist.get_distance_matrix_gradient(X_action, Y_action)
            dK_a_d_bw *= (dL_dK * k)[..., None] * (-self.rho)
            self.action_bandwidth.gradient = np.sum(dK_a_d_bw, axis=(0, 1))

        # compute gradient w.r.t. rho
        self.rho.gradient = np.sum(dL_dK * k * (k_kilobots - k_light - k_extra_dim - k_action))

        # compute gradient w.r.t. variance
        self.variance.gradient = np.sum(dL_dK * k) / self.variance

    def update_gradients_diag(self, dL_dKdiag, X):
        # compute gradient w.r.t. kernel bandwidths
        self.kilobots_bandwidth.gradient = np.zeros(2)
        if self.light_dim:
            self.light_bandwidth.gradient = np.zeros(self.light_dim)
        if self.extra_dim:
            self.extra_dim_bandwidth.gradient = np.zeros(self.extra_dim)
        if self.action_dim:
            self.action_bandwidth.gradient = np.zeros(self.action_dim)
        # compute gradient w.r.t. rho
        self.rho.gradient = 0
        # compute gradient w.r.t. variance
        self.variance.gradient = np.sum(dL_dKdiag)

    def gradients_X(self, dL_dK, X, X2):
        # return np.zeros((dL_dK.shape[0], 1))
        pass

    def gradients_X_diag(self, dL_dKdiag, X):
        # return np.zeros((dL_dKdiag.shape[0], 1))
        pass

    def __call__(self, X, Y=None):
        return self.K(X, Y)

    def diag(self, X):
        return self.Kdiag(X)


class UnknownDistClassException(Exception):
    pass