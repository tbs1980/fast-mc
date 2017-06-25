import numpy as np
import emcee
import time
from fast_mc.model import Model
from fast_mc.ensemble import EnsembleSampler
from scipy import stats

class NormalDist(Model):
    def __init__(self):
        pass

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)


def log_prost(x):
    return -0.5*np.sum(x**2)

num_dims = 5
num_walkers = 20
num_samples = 10000

em_sampler = emcee.EnsembleSampler(num_walkers, num_dims, log_prost)
start_points = np.array([np.random.rand(num_dims) for _ in range(num_walkers)])

strt = time.time()
em_sampler.run_mcmc(start_points, num_samples)
print("emcee took {0} seconds".format(time.time() - strt))
samples = em_sampler.chain[:, 2000:, :].reshape((-1, num_dims))
print(stats.kstest(samples[:, 0], 'norm'))


nrm_dst = NormalDist()
start_points = np.array([np.random.rand(num_dims) for _ in range(num_walkers)])
sampler = EnsembleSampler(nrm_dst, num_walkers=num_walkers, num_dims=num_dims)

strt = time.time()
sampler.run(num_samples=num_samples, start_points=start_points)
print("fast-mc took {0} seconds".format(time.time() - strt))
samples = sampler.samples[2000:, :, :].reshape(-1, num_dims)
print(stats.kstest(samples[:, 0], 'norm'))
