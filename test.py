#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import logging
import argparse
import numpy as np

sys.path.append("./")

from inference.estimator import ParameterizedRatioEstimator
from inference.utils import load_and_check
from simulation.prior import (
    draw_params_from_prior,
    get_reference_point,
    get_grid,
    get_grid_point,
    get_grid_midpoint_index,
)


def evaluate(
    data_dir,
    model_filename,
    sample_filename,
    result_filename,
    aux=False,
    grid=True,
    shuffle=False,
    small=False,
    gradx=False,
    fine=False,
    i_theta_grid=None,
):
    if not os.path.exists("{}/results".format(data_dir)):
        os.mkdir("{}/results".format(data_dir))

    estimator = ParameterizedRatioEstimator()
    estimator.load("{}/models/{}".format(data_dir, model_filename))

    if grid:
        x = np.load("{}/samples/x_{}.npy".format(data_dir, sample_filename))
        aux_data, n_aux = load_aux(
            "{}/samples/z_{}.npy".format(data_dir, sample_filename), aux
        )
        if small:
            x = x[:100]
            if aux_data is not None:
                aux_data = aux_data[:100]
        theta = get_grid(fine=fine)
        grad_x_index = get_grid_midpoint_index()
        llr, _, grad_x = estimator.log_likelihood_ratio(
            x=x,
            aux=aux_data,
            theta=theta,
            test_all_combinations=True,
            evaluate_grad_x=True,
            grad_x_theta_index=grad_x_index,
        )

    else:
        x = np.load("{}/samples/x_{}.npy".format(data_dir, sample_filename))
        aux_data, n_aux = load_aux(
            "{}/samples/z_{}.npy".format(data_dir, sample_filename), aux
        )
        if i_theta_grid is not None:
            theta = np.asarray([get_grid_point(i_theta_grid) for _ in range(x.shape[0])])
            logging.info("Determined grid theta %s = %s", i_theta_grid, theta[0])
        else:
            theta = np.load("{}/samples/theta_{}.npy".format(data_dir, sample_filename))
        if shuffle:
            np.random.shuffle(theta)

        llr, _, grad_x = estimator.log_likelihood_ratio(
            x=x,
            aux=aux_data,
            theta=theta,
            test_all_combinations=False,
            evaluate_grad_x=gradx,
        )
    if shuffle:
        np.save(
            "{}/results/shuffled_theta_{}.npy".format(data_dir, result_filename), theta
        )
    np.save("{}/results/llr_{}.npy".format(data_dir, result_filename), llr)
    if gradx:
        np.save("{}/results/grad_x_{}.npy".format(data_dir, result_filename), grad_x)


def load_aux(filename, aux=False):
    if aux:
        return load_and_check(filename)[:, 2].reshape(-1, 1), 1
    else:
        return None, 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-level evaluation script"
    )

    # Main options
    parser.add_argument("model", type=str, help="Model name.")
    parser.add_argument("sample", type=str, help='Sample name, like "test".')
    parser.add_argument("result", type=str, help="File name for results.")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Evaluates the images on a parameter grid rather than just at the original parameter points.",
    )
    parser.add_argument(
        "--finegrid",
        action="store_true",
        help="If used with --grid, uses a finer grid centered on the true point.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If --grid is not used, shuffles the theta values between the images. This can be useful to make ROC curves.",
    )
    parser.add_argument(
        "-z", action="store_true", help="Provide lens redshift to the network"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the "
        "data/models subfolder.",
    )
    parser.add_argument(
        "--small", action="store_true", help="Restricts evaluation to first 100 images."
    )
    parser.add_argument("--grad", action="store_true", help="Evaluate gradients wrt x.")
    parser.add_argument(
        "--igrid",
        type=int, default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.INFO,
    )
    logging.info("Hi!")
    args = parse_args()
    evaluate(
        args.dir + "/data",
        args.model,
        args.sample,
        args.result,
        args.z,
        args.grid,
        args.shuffle,
        args.small,
        gradx=args.grad,
        fine=args.finegrid,
        i_theta_grid=args.igrid
    )
    logging.info("All done! Have a nice day!")
