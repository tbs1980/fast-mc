import numpy as np
from fast_mc.model import Model
from fast_mc.ensemble import EnsembleSampler
import matplotlib.pyplot as plt


class NormalDist(Model):
    def __init__(self):
        pass

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)

num_dims = 5
num_walkers = 20

nrm_dst = NormalDist()
sampler = EnsembleSampler(nrm_dst, num_walkers=num_walkers, num_dims=num_dims)

start_points = np.array([np.random.rand(num_dims) for _ in range(num_walkers)])
num_samples = 10000
sampler.run(num_samples=num_samples, start_points=start_points)
# samples = sampler.samples.reshape(-1, 5)
#
# plt.hist(samples[:, 0])
# plt.show()