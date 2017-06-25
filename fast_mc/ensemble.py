import numpy as np
from math import log
from fast_mc.chain import Chain


class EnsembleSampler(object):
    """
    https://arxiv.org/pdf/1202.3665.pdf
    """

    def __init__(self, model, num_walkers, num_dims, a=2.):
        self._model = model
        if num_walkers % 2 != 0:
            raise ValueError("Number of walkers should be an even number")
        self._num_walkers = num_walkers
        self._num_dims = num_dims
        if self._num_walkers < self._num_dims*2:
            raise ValueError("Number of walkers should be greater than twice the number of dimensions.")
        self._a = a
        self._samples = None
        self._weights = None
        self._acceptance = None

    def run(self, num_samples, start_points):
        if start_points.shape[0] != self._num_walkers:
            raise ValueError("Number of rows in 'start_points' should be identical to 'num_walkers'")
        if start_points.shape[1] != self._num_dims:
            raise ValueError("Number of columns in 'start_points' should be identical to 'num_dims'")

        self._samples = np.empty(shape=[num_samples, self._num_walkers, self._num_dims], dtype=float)
        self._weights = np.empty(shape=[num_samples, self._num_walkers], dtype=float)

        log_probs = np.array([self._model.compute_log_posterior(x) for x in start_points])

        num_accepted = 0
        num_rejected = 0
        half_k = int(self._num_walkers/2)
        while num_accepted < num_samples:
            acc = np.zeros(self._num_walkers, dtype=bool)
            for i in range(2):
                for k in range(half_k*i, half_k*i + half_k):
                    j = np.random.randint((1-i)*half_k, (1-i)*half_k+half_k)
                    x_j = start_points[j, :]
                    x_k = start_points[k, :]
                    u = np.random.uniform()
                    z = ((u*(self._a - 1.) + 1.)**2.)/self._a
                    # lp_x_k = log_probs[k]
                    y = x_j + z*(x_k - x_j)
                    lp_y = self._model.compute_log_posterior(y)
                    # log_q = (self._num_dims - 1)*log(z) + lp_y - log_probs[k]
                    # log_r = log(np.random.uniform())
                    if log(np.random.uniform()) <= (self._num_dims - 1)*log(z) + lp_y - log_probs[k]:
                        start_points[k, :] = y
                        log_probs[k] = lp_y
                        acc[k] = True

            if acc.any():
                self._samples[num_accepted, :, :] = start_points
                self._weights[num_accepted, :] = log_probs
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

