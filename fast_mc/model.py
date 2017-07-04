class Model(object):
    def compute_log_posterior(self, x, **kwargs):
        raise NotImplementedError("The function 'compute_log_posterior' must be implemented by the sub-classes")

    def compute_grad_log_posterior(self, x, **kwargs):
        raise NotImplementedError("The function 'compute_grad_log_posterior' must be implemented by the sub-classes")

    def compute_metric_tensor(self, x, **kwargs):
        raise NotImplementedError("The function 'compute_metric_tensor' must be implemented by the sub-classes")

    def compute_deriv_metric_tensor(self, x, **kwargs):
        raise NotImplementedError("The function 'compute_deriv_metric_tensor' must be implemented by the sub-classes")
