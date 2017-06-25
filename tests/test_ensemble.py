import numpy as np
from unittest import TestCase
from fast_mc.ensemble import EnsembleSampler
from fast_mc.model import Model
from scipy import stats


class NormalDist(Model):
    def __init__(self):
        pass

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)


class TestMvp(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ensemble(self):
        num_dims = 5
        num_walkers = 20
        nrm_dst = NormalDist()
        sampler = EnsembleSampler(nrm_dst, num_walkers=num_walkers, num_dims=num_dims)
        start_points = np.array([np.random.rand(num_dims) for _ in range(num_walkers)])
        num_samples = 10000
        sampler.run(num_samples=num_samples, start_points=start_points)

        samples = sampler.samples[5000:,:,:].reshape(-1, 5)
        print(samples[:, 0].shape)
        print(samples[:, 0].mean(), samples[:, 0].var())
        # print(stats.norm.rvs(size=2000).shape)
        print(stats.kstest(samples[:, 0], 'norm'))
        print(stats.kstest(stats.norm.rvs(size=100), 'norm'))

