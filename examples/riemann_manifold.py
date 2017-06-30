import numpy as np
from fast_mc.model import Model

class NormalDist(Model):
    def __init__(self):
        pass

    def compute_log_posterior(self, x, **kwargs):
        return -0.5*np.sum(x**2)

    def compute_grad_log_posterior(self, x, **kwargs):
        return -x

    def compute_metric_tensor(self, x, **kwargs):

