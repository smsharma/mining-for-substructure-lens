#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import argparse
import logging

logger = logging.getLogger(__name__)
sys.path.append("./")

from simulation.units import *
from simulation.wrapper import augmented_data


def simulate_train(n=1000, n_thetas_marginal=1000):
    logger.info("Generating training data with %s images", n)

    logger.info("Generating %s numerator images", n // 2)
    y0 = np.zeros(n // 2)
    theta0, x0, t_xz0, log_r_xz0, _ = augmented_data(
        n_images=n // 2,
        n_thetas_marginal=n_thetas_marginal,
        inverse=False,
        mine_gold=True,
    )

    logger.info("Generating %s denominator images", n // 2)
    y1 = np.ones(n // 2)
    theta1, x1, t_xz1, log_r_xz1, _, latents = augmented_data(
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

    return x, theta, y, r_xz, t_xz, latents


def simulate_test_point(n=1000, n_calib=150, beta=-1.9):
    logger.info("Generating point test data with %s images at n_calib = %s, beta = %s", n, n_calib, beta)
    theta, x, _, _, _, latents = augmented_data(
        n_calib=n_calib,
        beta=beta,
        n_images=n,
        mine_gold=False,
    )

    return x, theta, None, None, None, latents


def simulate_test_prior(n=1000):
    logger.info("Generating prior test data with %s images", n)
    theta, x, _, _, _, latents = augmented_data(
        n_calib=None,
        beta=None,
        n_images=n,
        mine_gold=False,
    )
    z_s = latents[:,0]
    z_l = latents[:,1]
    sigma_v = latents[:,2]

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
        "--point",
        action="store_true",
        help="Generate test data at specific reference model rather than sampled from the prior.",
    )
    parser.add_argument(
        "--name", type=str, default=None, help='Sample name, like "train" or "test".'
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
    else:
        name = "train" if args.name is None else args.name
        results = simulate_train(args.n)
    save(args.dir, name, *results)

    logger.info("All done! Have a nice day!")
