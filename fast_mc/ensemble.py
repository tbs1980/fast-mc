import numpy as np


class EnsembleSampler(object):
    """
    https://arxiv.org/pdf/1202.3665.pdf
    """

    def __init__(self, model, num_walkers, num_dims):
        self._model = model
        if num_walkers % 2 != 0:
            raise ValueError("Number of walkers should be an even number")
        self._num_walkers = num_walkers
        self._num_dims = num_dims
        if self._num_walkers < self._num_dims*2:
            raise ValueError("Number of walkers should be greater than twice the number of dimensions.")


    def run(self, num_samples, start_points):
        if start_points.shape[0] != self._num_walkers:
            raise ValueError("Number of rows in 'start_points' should be identical to 'num_walkers'")
        if start_points.shape[1] != self._num_dims:
            raise ValueError("Number of columns in 'start_points' should be identical to 'num_dims'")

        log_probs = np.array([self._model.compute_log_posterior(state) for state in start_points])

        num_accepted = 0
        num_rejected = 0
        max_k = self._num_walkers/2
        while num_accepted < num_samples:
            for i in range(2):
                for k in range(max_k):
                    j = np.random.uniform((1-i)*max_k, (1-i)*max_k+max_k)




