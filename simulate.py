from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

sys.path.append('../')

import logging
import numpy as np
import argparse

from simulation.units import *
from simulation.population_sim import SubhaloSimulator

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)


def simulate_train(n=500, n_prior_samples=500, alpha_mean=200., alpha_std=50., beta_mean=-1.9, beta_std=0.2,
                   mass_base_unit=1.e9 * M_s, m_sub_min=100., resolution=64):
    alpha_train = np.random.normal(loc=alpha_mean, scale=alpha_std, size=n_train)
    beta_train = np.random.normal(loc=beta_mean, scale=beta_std, size=n_train)
    theta_train = np.vstack((alpha_train, beta_train)).T
    assert np.min(alpha_train) > 0.
    assert np.max(beta_train) < -1.

    sim = SubhaloSimulator(
        mass_base_unit=mass_base_unit,
        resolution=resolution,
        m_sub_min=m_sub_min,
    )

    y0 = np.zeros(n // 2)
    x0, t_xz0, log_r_xz0, log_r_xz_uncertainties0, latents0 = sim.rvs_score_ratio_to_evidence(
        alpha_train,
        beta_train,
        alpha_mean,
        alpha_std,
        beta_mean,
        beta_std,
        n // 2,
        n_prior_samples
    )

    y1 = np.ones(n // 2)
    x1, t_xz1, log_r_xz1, log_r_xz_uncertainties1, latents1 = sim.rvs_score_ratio_to_evidence(
        alpha_train,
        beta_train,
        alpha_mean,
        alpha_std,
        beta_mean,
        beta_std,
        n // 2,
        n_prior_samples
    )

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    theta = np.vstack((theta_train, theta_train))
    log_r_xz = np.hstack((log_r_xz0, log_r_xz1))
    t_xz = np.vstack((t_xz0, t_xz1))
    latents = latents0 + latents1
    r_xz = np.exp(log_r_xz, dtype=np.float64)
    n_subs = np.array([v[0] for v in latents])
    avg_m_subs = np.array([np.mean(v[1]) for v in latents])

    return x, y, theta, r_xz, t_xz, n_subs, avg_m_subs


def save_train(data_dir, name, x, y, theta, r_xz, t_xz, n_subs, avg_m_subs):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/samples".format(data_dir)):
        os.mkdir("{}/samples".format(data_dir))

    np.save("{}/samples/x_{}.npy".format(data_dir, name), x)
    np.save("{}/samples/y_{}.npy".format(data_dir, name), y)
    np.save("{}/samples/theta_{}.npy".format(data_dir, name), theta)
    np.save("{}/samples/r_xz_{}.npy".format(data_dir, name), r_xz)
    np.save("{}/samples/t_xz_{}.npy".format(data_dir, name), t_xz)
    np.save("{}/samples/n_subs_{}.npy".format(data_dir, name), n_subs)
    np.save("{}/samples/avg_m_subs_{}.npy".format(data_dir, name), avg_m_subs)


def simulate_test(n=500, alpha=200, beta=-1.9, mass_base_unit=1.e9 * M_s, m_sub_min=100., resolution=64):
    sim = SubhaloSimulator(
        mass_base_unit=mass_base_unit,
        resolution=resolution,
        m_sub_min=m_sub_min,
    )

    x, latents = sim.rvs(
        alpha,
        beta,
        n
    )

    n_subs = np.array([v[0] for v in latents])
    avg_m_subs = np.array([np.mean(v[1]) for v in latents])

    return x, n_subs, avg_m_subs


def save_test(data_dir, name, x, n_subs, avg_m_subs):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/samples".format(data_dir)):
        os.mkdir("{}/samples".format(data_dir))

    np.save("{}/samples/x_{}.npy".format(data_dir, name), x)
    np.save("{}/samples/n_subs_{}.npy".format(data_dir, name), n_subs)
    np.save("{}/samples/avg_m_subs_{}.npy".format(data_dir, name), avg_m_subs)


def parse_args():
    parser = argparse.ArgumentParser(description="Strong lensing experiments: simulation")

    # Main options
    parser.add_argument("name", type=str, help='Sample name, like "train" or "test".')
    parser.add_argument("--n", type=int, default=10000, help='Number of samples to generate. Default is 10k.')
    parser.add_argument("--test", action="store_true", help="Generate test rather than train data.")

    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")

    args = parse_args()

    if args.test:
        pass
    else:
        pass

    logging.info("All done! Have a nice day!")
