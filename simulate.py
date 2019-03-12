#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

sys.path.append('../')

import logging
import argparse

from simulation.units import *
from simulation.population_sim import SubhaloSimulator

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)


def _extract_n_subs(latents):
    return np.array([v[0] for v in latents])


def _extract_heaviest_subs(latents, n=100):
    all_masses = []
    for v in latents:
        masses = np.zeros(n)
        masses[:len(masses)] = list(sorted(v[1], reverse=True))[:n]
        all_masses.append(masses)
    return np.array(all_masses)


def _extract_m_subs(latents):
    return np.array([np.sum(v[1]) for v in latents])


def simulate_train(n=1000, n_prior_samples=1000, alpha_mean=10., alpha_std=2., beta_mean=-1.9, beta_std=0.2,
                   m_sub_min=10.):
    alpha_train = np.random.normal(loc=alpha_mean, scale=alpha_std, size=n // 2)
    beta_train = np.random.normal(loc=beta_mean, scale=beta_std, size=n // 2)
    theta_train = np.vstack((alpha_train, beta_train)).T
    assert np.min(alpha_train) > 0.
    assert np.max(beta_train) < -1.

    sim = SubhaloSimulator(
        m_sub_min=m_sub_min,
        m_sub_high=m_sub_min
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
    m_subs0 = _extract_m_subs(latents0)
    n_subs0 = _extract_n_subs(latents0)

    y1 = np.ones(n // 2)
    x1, t_xz1, log_r_xz1, log_r_xz_uncertainties1, latents1 = sim.rvs_score_ratio_to_evidence_inverse(
        alpha_train,
        beta_train,
        alpha_mean,
        alpha_std,
        beta_mean,
        beta_std,
        n // 2,
        n_prior_samples
    )
    m_subs1 = _extract_m_subs(latents1)
    n_subs1 = _extract_n_subs(latents1)

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    theta = np.vstack((theta_train, theta_train))
    t_xz = np.vstack((t_xz0, t_xz1))
    log_r_xz = np.hstack((log_r_xz0, log_r_xz1))
    r_xz = np.exp(log_r_xz, dtype=np.float64)
    n_subs = np.hstack((n_subs0, n_subs1))
    m_subs = np.hstack((m_subs0, m_subs1))

    return x, y, theta, r_xz, t_xz, n_subs, m_subs


def save_train(data_dir, name, x, y, theta, r_xz, t_xz, n_subs, m_subs):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/samples".format(data_dir)):
        os.mkdir("{}/samples".format(data_dir))

    np.save("{}/data/samples/x_{}.npy".format(data_dir, name), x)
    np.save("{}/data/samples/y_{}.npy".format(data_dir, name), y)
    np.save("{}/data/samples/theta_{}.npy".format(data_dir, name), theta)
    np.save("{}/data/samples/r_xz_{}.npy".format(data_dir, name), r_xz)
    np.save("{}/data/samples/t_xz_{}.npy".format(data_dir, name), t_xz)
    np.save("{}/data/samples/n_subs_{}.npy".format(data_dir, name), n_subs)
    np.save("{}/data/samples/m_subs_{}.npy".format(data_dir, name), m_subs)


def simulate_test(n=500, alpha=200, beta=-1.9, m_sub_min=10.):
    sim = SubhaloSimulator(
        m_sub_min=m_sub_min,
        m_sub_high=m_sub_min
    )

    x, latents = sim.rvs(
        alpha,
        beta,
        n
    )
    m_subs = _extract_m_subs(latents)
    n_subs = _extract_n_subs(latents)

    return x, n_subs, m_subs


def save_test(data_dir, name, x, n_subs, m_subs):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/samples".format(data_dir)):
        os.mkdir("{}/samples".format(data_dir))

    np.save("{}/data/samples/x_{}.npy".format(data_dir, name), x)
    np.save("{}/data/samples/n_subs_{}.npy".format(data_dir, name), n_subs)
    np.save("{}/data/samples/m_subs_{}.npy".format(data_dir, name), m_subs)


def parse_args():
    parser = argparse.ArgumentParser(description="Strong lensing experiments: simulation")

    # Main options
    parser.add_argument("-n", type=int, default=10000, help='Number of samples to generate. Default is 10k.')
    parser.add_argument("--test", action="store_true", help="Generate test rather than train data.")
    parser.add_argument("--name", type=str, default= None, help='Sample name, like "train" or "test".')
    parser.add_argument("--dir", type=str, default=".", help="Directory. Results will be saved in the data/samples subfolder.")

    return parser.parse_args()


if __name__ == "__main__":
    logging.info("Hi!")

    args = parse_args()

    if args.test:
        name = "test" if args.name is None else args.name
        results = simulate_test(args.n)
        save_test(args.dir, name, *results)
    else:
        name = "train" if args.name is None else args.name
        results = simulate_train(args.n)
        save_train(args.dir, name, *results)

    logging.info("All done! Have a nice day!")
