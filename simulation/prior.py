import numpy as np
from scipy.stats import uniform, norm


def draw_params_from_prior(n):
    f_sub = uniform(0.001, 0.199).rvs(size=n)
    beta = uniform(-2.5, 1.0).rvs(size=n)
    return f_sub, beta


def get_reference_point():
    return 0.05, -1.9


def get_grid(resolution=25, fine=False):
    if fine:
        f_sub_1d = np.linspace(0.03, 0.07, resolution)
        beta_1d = np.linspace(-2.0, -1.8, resolution)
    else:
        f_sub_1d = np.linspace(0.001, 0.200, resolution)
        beta_1d = np.linspace(-2.5, -1.5, resolution)

    theta0, theta1 = np.meshgrid(f_sub_1d, beta_1d)
    return np.vstack((theta0.flatten(), theta1.flatten())).T


def get_grid_point(i, resolution=25):
    return get_grid(resolution)[i]


def get_grid_midpoint_index(resolution=25):
    return ((resolution - 1) // 2) * resolution + ((resolution - 1) // 2)
