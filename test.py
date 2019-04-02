#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
import argparse
import numpy as np

sys.path.append("./")

from inference.estimator import ParameterizedRatioEstimator


def make_grid(alpha_min=0., alpha_max=20., beta_min=-1., beta_max=-3., resolution=25):
    alpha_test = np.linspace(alpha_min, alpha_max, resolution)
    beta_test = np.linspace(beta_min, beta_max, resolution)


    theta0, theta1 = np.meshgrid(alpha_test, beta_test)
    theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T

    return theta_grid


def evaluate(
        data_dir, model_filename, sample_filename, result_filename, grid,
):
    estimator = ParameterizedRatioEstimator()
    estimator.load("{}/models/{}".format(data_dir, model_filename))

    if grid:
        x = np.load("{}/samples/x_{}.npy".format(data_dir, sample_filename))
        theta = make_grid()
        np.save("{}/results/theta_grid.npy")

        llr, _, grad_x = estimator.log_likelihood_ratio(
            x=x,
            theta=theta,
            test_all_combinations=False,
            evaluate_grad_x=True
        )

    else:
        x = np.load("{}/samples/x_{}.npy".format(data_dir, sample_filename))
        theta = np.load("{}/samples/theta_{}.npy".format(data_dir, sample_filename))

        llr, _, grad_x = estimator.log_likelihood_ratio(
            x=x,
            theta=theta,
            test_all_combinations=True,
            evaluate_grad_x=True
        )

    np.save("{}/results/llr_{}.npy".format(data_dir, result_filename), llr)
    np.save("{}/results/grad_x_{}.npy".format(data_dir, result_filename), grad_x)


def parse_args():
    parser = argparse.ArgumentParser(description="Strong lensing experiments: evaluation")

    # Main options
    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument("sample", type=str, help='Sample name, like "test".')
    parser.add_argument("result", type=str, help="Model name.")
    parser.add_argument("--grid", action="store_true",
                        help='Evaluates the images on a parameter grid rather than just at the original parameter points.')
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the data/models subfolder.",
    )


    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M",
                        level=logging.INFO)
    logging.info("Hi!")
    args = parse_args()
    evaluate(args.dir + "/data", args.model, args.sample, args.result, args.grid)
    logging.info("All done! Have a nice day!")
