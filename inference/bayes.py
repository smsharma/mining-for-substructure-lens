import numpy as np


class Posterior:
    def __init__(self, llr, param_grid):
        self.llr = llr  # (n_grid, n_events)
        self.thetas = param_grid  # (n_grid, n_parameters_

    def posterior_based_on_expected_llr(self, n, prior_fn):
        self.prior = prior_fn(self.thetas)  # (n_grid,)
        self.prior /= np.sum(self.prior)

        delta_llr = self.llr[np.newaxis, :, :] - self.llr[:, np.newaxis, :]  # (n_grid, n_grid', n_events
        delta_llr = n * np.mean(delta_llr, axis=2)  # (n_grid, n_grid)

        inv = self.prior[np.newaxis, :] / self.prior[:, np.newaxis] * np.exp(delta_llr)  # (n_grid, n_grid')
        inv = np.sum(inv, axis=1)  # (n_grid,)

        posterior = 1. / inv  # (n_grid,)
        return posterior
