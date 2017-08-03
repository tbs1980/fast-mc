import numpy as np
from math import log


class HamiltonianMonteCarloSampler(object):
    """
    https://arxiv.org/abs/1206.1901v1
    """

    def __init__(self, model, num_dims, diag_mass_mat, max_epsilon=1., max_leapfrog_steps=10):
        self._model = model
        if num_dims < 1:
            raise ValueError("Number of dimensions should be at least one.")
        self._num_dims = num_dims

        if np.linalg.norm(diag_mass_mat) <= 0:
            raise ValueError("Diagonal elements of the mass matrix should be >0.")
        self._diag_mass_mat = diag_mass_mat

        if max_epsilon <= 0. or max_epsilon > 2.:
            raise ValueError("Maximum value of epsilon should be 0 < eps <= 2.")
        self._max_epsilon = max_epsilon

        if max_leapfrog_steps == 0:
            raise ValueError("Maximum number of leap-frog steps should be greater than zero.")
        self._max_leapfrog_stps = max_leapfrog_steps

        self._samples = None
        self._weights = None
        self._acceptance = None

    def run(self, num_samples, q_0):
        num_accepted = 0
        num_rejected = 0

        self._samples = np.empty(shape=[num_samples, self._num_dims], dtype=float)
        self._weights = np.empty(shape=num_samples, dtype=float)

        log_post_q_0 = self._model.compute_log_posterior(q_0)
        while num_accepted < num_samples:
            p_0 = np.random.normal(loc=np.zeros(self._num_dims), scale=np.ones(self._num_dims))
            epsilon = np.random.uniform(low=0., high=self._max_epsilon)
            num_leap_frog_steps = np.random.randint(1, self._max_leapfrog_stps + 1)
            h_0 = -log_post_q_0 + 0.5 * (np.dot(p_0, self._diag_mass_mat*p_0))
            if not np.isfinite(h_0):
                raise ValueError('Initial Hamiltonian is not finite.')

            q_1 = q_0
            p_1 = p_0
            grad_q_1 = self._model.compute_grad_log_posterior(q_1)
            grad_p_1 = -self._diag_mass_mat*p_1

            p_1 += 0.5*epsilon*grad_q_1
            for _ in range(num_leap_frog_steps):
                grad_p_1 = -self._diag_mass_mat*p_1
                q_1 -= epsilon*grad_p_1
                grad_q_1 = self._model.compute_grad_log_posterior(q_1)
                p_1 += epsilon*grad_q_1
            p_1 -= 0.5*epsilon*grad_q_1

            log_post_q_1 = self._model.compute_log_posterior(q_1)
            h_1 = -log_post_q_1 + 0.5 * (np.dot(p_1, self._diag_mass_mat*p_1))
            delta_h = h_1 - h_0
            if log(np.random.uniform()) < -delta_h:
                q_0 = q_1
                log_post_q_0 = log_post_q_1
                self._samples[num_accepted, :] = q_0
                self._weights[num_accepted] = log_post_q_1
                num_accepted += 1
            else:
                num_rejected += 1

        self._acceptance = float(num_accepted) / float(num_accepted + num_rejected)

    @property
    def samples(self):
        return self._samples

    @property
    def weights(self):
        return self._weights

    @property
    def acceptance(self):
        return self._acceptance
