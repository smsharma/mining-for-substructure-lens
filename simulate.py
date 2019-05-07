#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import argparse
import logging
from scipy.stats import uniform, norm

logger = logging.getLogger(__name__)
sys.path.append("./")

from simulation.units import *
from simulation.wrapper import augmented_data


def simulate_train_marginalref(n=10000, n_thetas_marginal=5000):
    logger.info("Generating training data with %s images", n)

    # Parameter points from prior
    n_calib = uniform(10.0, 400.0).rvs(size=n // 2)
    beta = uniform(-3.0, 1.9).rvs(size=n // 2)
    if n_thetas_marginal is None:
        n_calib_ref = n_calib
        beta_ref = beta
    else:
        n_calib_ref = uniform(10.0, 400.0).rvs(size=n_thetas_marginal)
        beta_ref = uniform(-3.0, 1.9).rvs(size=n_thetas_marginal)

    # Samples from numerator
    logger.info("Generating %s numerator images", n // 2)
    y0 = np.zeros(n // 2)
    theta0, x0, t_xz0, log_r_xz0, _, latents0 = augmented_data(
        n_calib=n_calib,
        beta=beta,
        n_calib_ref=n_calib_ref,
        beta_ref=beta_ref,
        n_images=n // 2,
        n_thetas_marginal=n_thetas_marginal,
        inverse=False,
        mine_gold=True,
    )

    # Samples from denominator / reference / marginal
    logger.info("Generating %s denominator images", n // 2)
    y1 = np.ones(n // 2)
    theta1, x1, t_xz1, log_r_xz1, _, latents1 = augmented_data(
        n_calib=n_calib,
        beta=beta,
        n_calib_ref=n_calib_ref,
        beta_ref=beta_ref,
        n_images=n // 2,
        n_thetas_marginal=n_thetas_marginal,
        inverse=True,
        mine_gold=True,
    )

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    theta = np.vstack((theta0, theta1))
    t_xz = np.vstack((t_xz0, t_xz1))
    log_r_xz = np.hstack((log_r_xz0, log_r_xz1))
    r_xz = np.exp(log_r_xz, dtype=np.float64)
    latents = np.vstack((latents0, latents1))

    return x, theta, y, r_xz, t_xz, latents


def simulate_train_pointref(n=10000, n_calib_ref=150., beta_ref=-1.9):
    logger.info("Generating training data with %s images", n)

    # Parameter points from prior
    n_calib = uniform(10.0, 400.0).rvs(size=n // 2)
    beta = uniform(-3.0, 1.9).rvs(size=n // 2)

    # Samples from numerator
    logger.info("Generating %s numerator images", n // 2)
    y0 = np.zeros(n // 2)
    theta0, x0, t_xz0, log_r_xz0, _, latents0 = augmented_data(
        n_calib=n_calib,
        beta=beta,
        n_calib_ref=[n_calib_ref],
        beta_ref=[beta_ref],
        n_images=n // 2,
        n_thetas_marginal=1,
        inverse=False,
        mine_gold=True,
    )

    # Samples from denominator / reference
    logger.info("Generating %s denominator images", n // 2)
    y1 = np.ones(n // 2)
    theta1, x1, t_xz1, log_r_xz1, _, latents1 = augmented_data(
        n_calib=n_calib,
        beta=beta,
        n_calib_ref=[n_calib_ref],
        beta_ref=[beta_ref],
        n_images=n // 2,
        n_thetas_marginal=1,
        inverse=True,
        mine_gold=True,
    )

    x = np.vstack((x0, x1))
    y = np.hstack((y0, y1))
    theta = np.vstack((theta0, theta1))
    t_xz = np.vstack((t_xz0, t_xz1))
    log_r_xz = np.hstack((log_r_xz0, log_r_xz1))
    r_xz = np.exp(log_r_xz, dtype=np.float64)
    latents = np.vstack((latents0, latents1))

    return x, theta, y, r_xz, t_xz, latents


def grid_point(i, alpha_min=10., alpha_max=400., beta_min=-1.1, beta_max=-3.0, resolution=25):
    alpha_test = np.linspace(alpha_min, alpha_max, resolution)
    beta_test = np.linspace(beta_min, beta_max, resolution)

    theta0, theta1 = np.meshgrid(alpha_test, beta_test)
    theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T

    return theta_grid[i]


def simulate_calibration(i_theta, n=1000):
    n_calib, beta = grid_point(i_theta)
    logger.info(
        "Generating calibration data with %s images at theta %s / 625: n_calib = %s, beta = %s",
        n,
        i_theta + 1,
        n_calib,
        beta,
    )
    theta, x, _, _, _, latents = augmented_data(
        n_calib=n_calib, beta=beta, n_images=n, mine_gold=False
    )

    return x, theta, None, None, None, latents


def simulate_calibration_reference(n=1000):
    logger.info("Generating calibration data with %s images from prior", n)

    # Parameter points from prior
    n_calib = uniform(10.0, 400.0).rvs(size=n)
    beta = uniform(-3.0, 1.9).rvs(size=n)

    theta, x, _, _, _, latents = augmented_data(
        n_calib=n_calib, beta=beta, n_images=n, inverse=False, mine_gold=False
    )

    return x, theta, None, None, None, latents


def simulate_test_point(n=1000, n_calib=150, beta=-1.9):
    logger.info(
        "Generating point test data with %s images at n_calib = %s, beta = %s",
        n,
        n_calib,
        beta,
    )
    theta, x, _, _, _, latents = augmented_data(
        n_calib=n_calib, beta=beta, n_images=n, mine_gold=False
    )

    return x, theta, None, None, None, latents


def simulate_test_prior(n=1000):
    logger.info("Generating prior test data with %s images", n)

    # Parameter points from prior
    n_calib = uniform(10.0, 400.0).rvs(size=n)
    beta = uniform(-3.0, 1.9).rvs(size=n)

    theta, x, _, _, _, latents = augmented_data(
        n_calib=n_calib, beta=beta, n_images=n, inverse=False, mine_gold=False
    )

    return x, theta, None, None, None, latents


def save(data_dir, name, x, theta, y=None, r_xz=None, t_xz=None, latents=None):
    logger.info("Saving results with name %s", name)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/data".format(data_dir)):
        os.mkdir("{}/data".format(data_dir))
    if not os.path.exists("{}/data/samples".format(data_dir)):
        os.mkdir("{}/data/samples".format(data_dir))

    np.save("{}/data/samples/x_{}.npy".format(data_dir, name), x)
    np.save("{}/data/samples/theta_{}.npy".format(data_dir, name), theta)
    if y is not None:
        np.save("{}/data/samples/y_{}.npy".format(data_dir, name), y)
    if r_xz is not None:
        np.save("{}/data/samples/r_xz_{}.npy".format(data_dir, name), r_xz)
    if t_xz is not None:
        np.save("{}/data/samples/t_xz_{}.npy".format(data_dir, name), t_xz)
    if latents is not None:
        np.save("{}/data/samples/z_{}.npy".format(data_dir, name), latents)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )

    # Main options
    parser.add_argument(
        "-n",
        type=int,
        default=10000,
        help="Number of samples to generate. Default is 10k.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Generate test rather than train data."
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Generate calibration rather than train data."
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Generate test data at specific reference model rather than sampled from the prior.",
    )
    parser.add_argument(
        "--pointref",
        action="store_true",
        help="When generating training data, use a fixed reference point rather than the full marginal model.",
    )
    parser.add_argument(
        "--calref", action="store_true", help="Generate reference sample for calibration."
    )
    parser.add_argument(
        "--name", type=str, default=None, help='Sample name, like "train" or "test".'
    )
    parser.add_argument(
        "--theta", type=int, default=None, help='Theta index for calibration (between 0 and 440)'
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Base directory. Results will be saved in the data/samples subfolder.",
    )
    parser.add_argument("--debug", action="store_true", help="Prints debug output.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    logger.info("Hi!")

    if args.test:
        name = "test" if args.name is None else args.name
        if args.point:
            results = simulate_test_point(args.n)
        else:
            results = simulate_test_prior(args.n)
    elif args.calibrate:
        assert args.theta is not None, "Please provide --theta"
        name = "calibrate_theta{}".format(args.theta) if args.name is None else args.name
        results = simulate_calibration(args.theta, args.n)
    elif args.calref:
        name = "calibrate_ref" if args.name is None else args.name
        results = simulate_calibration_ref(args.n)
    else:
        name = "train" if args.name is None else args.name
        if args.pointref:
            results = simulate_train_pointref(args.n)
        else:
            results = simulate_train_marginalref(args.n)
    save(args.dir, name, *results)

    logger.info("All done! Have a nice day!")
