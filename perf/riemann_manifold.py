import numpy as np
from fast_mc.model import Model
from fast_mc.rmhmc import RiemannianManifoldHamiltonianSampler
import time
from math import sqrt

np.random.seed(31415)


class NormalDist(Model):
    def __init__(self, num_dims):
        if num_dims <= 0:
            raise ValueError("Number of dimensions should be greater than zero.")
        self._num_dims = num_dims
        self._metric_tensor = np.eye(self._num_dims)
        self._deriv_metric_tensor = np.zeros(shape=[self._num_dims, self._num_dims, self._num_dims])

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)

    def compute_grad_log_posterior(self, x, **kwargs):
        return -x

    def compute_metric_tensor(self, x, **kwargs):
        return self._metric_tensor

    def compute_deriv_metric_tensor(self, x, **kwargs):
        return self._deriv_metric_tensor


num_dims = 5

nrm_dst = NormalDist(num_dims=num_dims)
max_epsilon = 1.
max_leapfrog_steps = 5
num_fixed_point_steps = 5

sampler = RiemannianManifoldHamiltonianSampler(model=nrm_dst, num_dims=num_dims, max_epsilon=max_epsilon,
                                               max_leapfrog_steps=max_leapfrog_steps,
                                               num_fixed_point_steps=num_fixed_point_steps)


num_samples = 10000
q_0 = np.random.normal(size=num_dims)
start = time.time()
sampler.run(num_samples=num_samples, q_0=q_0)
print("fast-mc took {0} seconds".format(time.time() - start))
print("acceptance  = ", sampler.acceptance)

samples = sampler.samples

print('mean = ', samples[:, 0].mean())
print('std-dvn = ', sqrt(samples[:, 0].var()))
