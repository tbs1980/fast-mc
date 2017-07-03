import numpy as np
from math import log


class RiemannianManifoldHamiltonianSampler(object):
    """
    http://dx.doi.org/10.1111/j.1467-9868.2010.00765.x
    """

    def __init__(self, model, num_dims, max_epsilon=1., max_leapfrog_steps=5, num_fixed_point_steps=5):
        self._model = model
        if num_dims < 1:
            raise ValueError("Number of dimensions should be at least one.")
        self._num_dims = num_dims
        if max_epsilon <= 0. or max_epsilon > 2.:
            raise ValueError("Maximum value of epsilon should be 0 < eps <= 2.")
        self._max_epsilon = max_epsilon
        if max_leapfrog_steps == 0:
            raise ValueError("Maximum number of leap-frog steps should be greater than zero.")
        self._max_leapfrog_stps = max_leapfrog_steps
        if num_fixed_point_steps == 0:
            raise ValueError("Maximum number of fixed-point steps should be greater than zero.")
        self._num_fixed_point_steps = num_fixed_point_steps

        self._samples = None
        self._weights = None
        self._acceptance = None

    def run(self, num_samples, q_0):
        num_accepted = 0
        num_rejected = 0
        norm_threshold = 1e-4

        self._samples = np.empty(shape=[num_samples, self._num_dims], dtype=float)
        self._weights = np.empty(shape=num_samples, dtype=float)

        while num_accepted < num_samples:
            metric_tensor = self._model.compute_metric_tensor(q_0)
            p_0 = np.random.multivariate_normal(mean=np.zeros(self._num_dims), cov=metric_tensor)
            step_size = np.random.uniform(low=0., high=self._max_epsilon)
            num_leap_frog_steps = np.random.randint(1, self._max_leapfrog_stps+1)

            det_metric_tensor = np.linalg.det(metric_tensor)
            inv_metric_tensor = np.linalg.inv(metric_tensor)

            h_0 = -self._model.compute_log_posterior(q_0) + log(det_metric_tensor) \
                + 0.5*(np.dot(p_0, np.dot(inv_metric_tensor, p_0)))

            if not np.isfinite(h_0):
                raise ValueError('Initial Hamiltonian is not finite.')

            deriv_metric_tensor = self._model.compute_deriv_metric_tensor(q_0)

            q_1 = np.copy(q_0)
            p_1 = np.copy(p_0)
            for _ in range(num_leap_frog_steps):
                d_g_inv_g = np.dot(deriv_metric_tensor, inv_metric_tensor)
                trace = np.trace(d_g_inv_g)
                grad_inv_metric_tensor = np.dot(inv_metric_tensor, d_g_inv_g)

                p_0 = np.copy(p_1)
                for __ in range(self._num_fixed_point_steps):
                    nu = np.dot(p_1, np.dot(grad_inv_metric_tensor, p_1))
                    p_1 = p_0 - step_size*0.5*(-self._model.compute_grad_log_posterior(q_0) + 0.5*trace - 0.5*nu)

                    if np.linalg.norm(p_0/np.linalg.norm(p_0) - p_1/np.linalg.norm(p_1)) < norm_threshold:
                        break

                q_0 = np.copy(q_1)
                inv_metric_tensor_1 = np.copy(inv_metric_tensor)
                for __ in range(self._num_fixed_point_steps):
                    q_1 = q_0 + step_size*0.5*np.dot(inv_metric_tensor + inv_metric_tensor_1, p_1)
                    inv_metric_tensor_1 = np.linalg.inv(self._model.compute_metric_tensor(q_1))

                    if np.linalg.norm(q_0/np.linalg.norm(q_0) - q_1/np.linalg.norm(q_1)) < norm_threshold:
                        break

                inv_metric_tensor = np.copy(inv_metric_tensor_1)
                deriv_metric_tensor = self._model.compute_deriv_metric_tensor(q_1)
                d_g_inv_g = np.dot(deriv_metric_tensor, inv_metric_tensor)
                trace = np.trace(d_g_inv_g)
                grad_inv_metric_tensor = np.dot(inv_metric_tensor, d_g_inv_g)
                nu = np.dot(p_1, np.dot(grad_inv_metric_tensor, p_1))

                p_1 = p_1 - step_size*0.5*(-self._model.compute_grad_log_posterior(q_1) + 0.5*trace - 0.5*nu)

            log_post_x_1 = self._model.compute_log_posterior(q_1)
            det_metric_tensor = np.linalg.det(metric_tensor)
            h_1 = -log_post_x_1 + log(det_metric_tensor) + 0.5*(np.dot(p_0, np.dot(inv_metric_tensor, p_0)))

            if log(np.random.uniform()) < (h_1 - h_0):
                q_0 = np.copy(q_1)
                self._samples[num_accepted, :] = q_0
                self._weights[num_accepted] = log_post_x_1
                num_accepted += 1
            else:
                num_rejected += 1

        self._acceptance = float(num_accepted)/float(num_accepted + num_rejected)

    @property
    def samples(self):
        return self._samples

    @property
    def weights(self):
        return self._weights





