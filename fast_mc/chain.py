import numpy as np

class Chain:
    def __init__(self, num_samples, num_dims):
        self._num_samples = num_samples
        self._num_dims = num_dims
        self._coords = np.zeros(shape=(num_samples, num_dims))
        self._weights = np.ones(shape=(num_samples))

    def