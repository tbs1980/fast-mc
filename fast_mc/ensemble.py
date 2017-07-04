import numpy as np


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

    def run(self, num_samples, start_points):
        if start_points.shape[0] != self._num_walkers:
            raise ValueError("Number of rows in 'start_points' should be identical to 'num_walkers'")
        if start_points.shape[1] != self._num_dims:
            raise ValueError("Number of columns in 'start_points' should be identical to 'num_dims'")

        self._samples = np.empty(shape=[num_samples, self._num_walkers, self._num_dims], dtype=float)
        self._weights = np.empty(shape=[num_samples, self._num_walkers], dtype=float)

        log_probs = np.array([self._model.compute_log_posterior(x) for x in start_points])

        for sample in range(num_samples):
            half_num_walkers = int(self._num_walkers / 2)
            first, second = slice(half_num_walkers), slice(half_num_walkers, self._num_walkers)

            for slice_1, slice_2 in [(first, second), (second, first)]:
                set_1 = np.atleast_2d(start_points[slice_1])
                log_probs_set_1 = log_probs[slice_1]
                num_walkers_1 = len(set_1)
                set_2 = np.atleast_2d(start_points[slice_2])
                num_walkers_2 = len(set_2)

                rand_z_val = ((self._a - 1.) * np.random.rand(num_walkers_1) + 1) ** 2. / self._a
                random_selection_2 = np.random.randint(num_walkers_2, size=(num_walkers_1,))

                proposal = set_2[random_selection_2] - rand_z_val[:, np.newaxis] * (set_2[random_selection_2] - set_1)
                new_log_probs = np.array([self._model.compute_log_posterior(q_i) for q_i in proposal])

                delta_log_probs = (self._num_dims - 1.) * np.log(rand_z_val) + new_log_probs - log_probs_set_1
                accept = (delta_log_probs > np.log(np.random.rand(len(delta_log_probs))))

                if np.any(accept):
                    log_probs[slice_1][accept] = new_log_probs[accept]
                    start_points[slice_1][accept] = proposal[accept]
                    self._samples[sample, :, :] = start_points
                    self._weights[sample, :] = log_probs

    @property
    def samples(self):
        return self._samples

    @property
    def weights(self):
        return self._weights
