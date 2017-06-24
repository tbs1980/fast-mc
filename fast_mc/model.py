class Model(object):
    def compute_log_posterior(self, x, **kwargs):
        raise NotImplementedError("The function 'compute_log_posterior' must be implemented by the sub-classes")

