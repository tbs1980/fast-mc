import numpy as np
from fast_mc.model import Model
from fast_mc.hmc import HamiltonianMonteCarloSampler
import matplotlib.pyplot as plt

class NormalDist(Model):
    def __init__(self, num_dims):
        if num_dims <= 0:
            raise ValueError("Number of dimensions should be greater than zero.")
        self._num_dims = num_dims

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)

    def compute_grad_log_posterior(self, x, **kwargs):
        return -x

    def compute_metric_tensor(self, x, **kwargs):
        return np.eye(self._num_dims)

    def compute_deriv_metric_tensor(self, x, **kwargs):
        return np.zeros(shape=[self._num_dims, self._num_dims, self._num_dims])


num_dims = 100000

nrm_dst = NormalDist(num_dims=num_dims)
diag_mass_mat = np.ones(num_dims)
max_epsilon = 0.03
max_leapfrog_steps = 10

sampler = HamiltonianMonteCarloSampler(model=nrm_dst, num_dims=num_dims, diag_mass_mat=diag_mass_mat,
                                       max_epsilon=max_epsilon, max_leapfrog_steps=max_leapfrog_steps)

num_samples = 10000
q_0 = np.random.normal(size=num_dims)
sampler.run(num_samples=num_samples, q_0=q_0)
samples = sampler.samples

# plt.hist(samples[:, 0])
# plt.show()
plt.plot(samples[:, 0])
plt.show()

print('acceptance = ', sampler.acceptance)
