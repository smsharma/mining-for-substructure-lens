#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import argparse
import logging

logger = logging.getLogger(__name__)
sys.path.append("./")

from simulation.units import *
from simulation.wrapper import augmented_data
from simulation.prior import draw_params_from_prior, get_reference_point, get_grid_point


def simulate_train(
    n=10000, n_thetas_marginal=1000, fixm=False, fixz=False, fixalign=False
):
    logger.info("Generating training data with %s images", n)

    # Parameter points from prior
    f_sub, beta = draw_params_from_prior(n)
    f_sub_alt = np.hstack((f_sub[n//2:], f_sub[:n//2]))
    beta_alt = np.hstack((beta[n//2:], beta[:n//2]))

    # Samples from numerator
    logger.info("Generating %s images", n)
    theta, theta_alt, x, t_xz, t_xz_alt, log_r_xz, log_r_xz_alt, _, z = augmented_data(
        f_sub=f_sub,
        beta=beta,
        f_sub_alt=f_sub_alt,
        beta_alt=beta_alt,
        n_images=n,
        n_thetas_marginal=n_thetas_marginal,
        mine_gold=True,
        draw_host_mass=not fixm,
        draw_host_redshift=not fixz,
        draw_alignment=not fixalign,
    )
    results = {}
    results["theta"] = theta
    results["theta_alt"] = theta_alt
    results["x"] = x
    results["t_xz"] = t_xz
    results["t_xz_alt"] = t_xz_alt
    results["log_r_xz"] = log_r_xz
    results["log_r_xz_alt"] = log_r_xz_alt
    results["z"] = z

    return results


def simulate_calibration(i_theta, n=1000, fixm=False, fixz=False, fixalign=False):
    f_sub, beta = get_grid_point(i_theta)
    logger.info(
        "Generating calibration data with %s images at theta %s / 625: f_sub = %s, beta = %s",
        n,
        i_theta + 1,
        f_sub,
        beta,
    )
    theta, x, _, _, _, z = augmented_data(
        f_sub=f_sub,
        beta=beta,
        n_images=n,
        mine_gold=False,
        draw_host_mass=not fixm,
        draw_host_redshift=not fixz,
        draw_alignment=not fixalign,
    )
    results = {}
    results["theta"] = theta
    results["x"] = x
    results["z"] = z
    return results


def simulate_calibration_ref(n=1000, fixm=False, fixz=False, fixalign=False):
    logger.info("Generating calibration data with %s images from prior", n)
    f_sub, beta = draw_params_from_prior(n)
    theta, x, _, _, _, z = augmented_data(
        f_sub=f_sub,
        beta=beta,
        n_images=n,
        mine_gold=False,
        draw_host_mass=not fixm,
        draw_host_redshift=not fixz,
        draw_alignment=not fixalign,
    )
    results = {}
    results["theta"] = theta
    results["x"] = x
    results["z"] = z
    return results


def simulate_test_point(n=1000, fixm=False, fixz=False, fixalign=False):
    f_sub, beta = get_reference_point()
    logger.info(
        "Generating point test data with %s images at f_sub = %s, beta = %s",
        n,
        f_sub,
        beta,
    )
    theta, x, _, _, _, z = augmented_data(
        f_sub=f_sub,
        beta=beta,
        n_images=n,
        mine_gold=False,
        draw_host_mass=not fixm,
        draw_host_redshift=not fixz,
        draw_alignment=not fixalign,
    )
    results = {}
    results["theta"] = theta
    results["x"] = x
    results["z"] = z
    return results


def simulate_test_prior(n=1000, fixm=False, fixz=False, fixalign=False):
    logger.info("Generating prior test data with %s images", n)
    f_sub, beta = draw_params_from_prior(n)
    theta, x, _, _, _, z = augmented_data(
        f_sub=f_sub,
        beta=beta,
        n_images=n,
        mine_gold=False,
        draw_host_mass=not fixm,
        draw_host_redshift=not fixz,
        draw_alignment=not fixalign,
    )
    results = {}
    results["theta"] = theta
    results["x"] = x
    results["z"] = z
    return results


def save(data_dir, name, data):
    logger.info("Saving results with name %s", name)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists("{}/data".format(data_dir)):
        os.mkdir("{}/data".format(data_dir))
    if not os.path.exists("{}/data/samples".format(data_dir)):
        os.mkdir("{}/data/samples".format(data_dir))

    for key, value in data.items():
        np.save("{}/data/samples/{}_{}.npy".format(data_dir, key, name), value)


def parse_args():
    parser = argparse.ArgumentParser(description="Main high-level script that starts the strong lensing simulations")

    parser.add_argument(
        "--test", action="store_true", help="Generate test rather than train data."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Generate calibration rather than train data.",
    )
    parser.add_argument(
        "--calref",
        action="store_true",
        help="Generate reference sample for calibration.",
    )
    parser.add_argument(
        "--point",
        action="store_true",
        help="Generate test data at specific reference model rather than sampled from the prior.",
    )

    parser.add_argument(
        "-n",
        type=int,
        default=10000,
        help="Number of samples to generate. Default is 10k.",
    )
    parser.add_argument("--fixm", action="store_true", help="Fix host halo mass")
    parser.add_argument("--fixz", action="store_true", help="Fix lens redshift")
    parser.add_argument(
        "--fixalign", action="store_true", help="Fix alignment between lens and source"
    )
    parser.add_argument(
        "--name", type=str, default=None, help='Sample name, like "train" or "test".'
    )
    parser.add_argument(
        "--theta",
        type=int,
        default=None,
        help="Theta index for calibration (between 0 and 440)",
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
            results = simulate_test_point(
                args.n, fixm=args.fixm, fixz=args.fixz, fixalign=args.fixalign
            )
        else:
            results = simulate_test_prior(
                args.n, fixm=args.fixm, fixz=args.fixz, fixalign=args.fixalign
            )
    elif args.calibrate:
        assert args.theta is not None, "Please provide --theta"
        name = (
            "calibrate_theta{}".format(args.theta) if args.name is None else args.name
        )
        results = simulate_calibration(
            args.theta, args.n, fixm=args.fixm, fixz=args.fixz, fixalign=args.fixalign
        )
    elif args.calref:
        name = "calibrate_ref" if args.name is None else args.name
        results = simulate_calibration_ref(
            args.n, fixm=args.fixm, fixz=args.fixz, fixalign=args.fixalign
        )
    else:
        name = "train" if args.name is None else args.name
        results = simulate_train(
            args.n, fixm=args.fixm, fixz=args.fixz, fixalign=args.fixalign
        )
    save(args.dir, name, results)

    logger.info("All done! Have a nice day!")
