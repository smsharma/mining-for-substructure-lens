#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

sys.path.append("./")

import logging
import argparse

from inference.estimator import ParameterizedRatioEstimator
from inference.utils import load_and_check


def train(
    method,
    alpha,
    data_dir,
    sample_name,
    model_filename,
    aux=None,
    architecture="resnet",
    log_input=False,
    initial_batch_size=128,
    final_batch_size=512,
    n_epochs=20,
    optimizer="adam",
    initial_lr=0.001,
    final_lr=0.0001,
    limit_samplesize=None,
):
    aux_data, n_aux = load_aux("{}/samples/z_{}.npy".format(data_dir, sample_name), aux)
    if aux_data is None:
        logging.info("%s aux variables", n_aux)
    else:
        logging.info("%s aux variables with shape %s", n_aux, aux_data.shape)

    estimator = ParameterizedRatioEstimator(
        resolution=64,
        n_parameters=2,
        n_aux=n_aux,
        architecture=architecture,
        log_input=log_input,
        rescale_inputs=True,
    )
    estimator.train(
        method,
        x="{}/samples/x_{}.npy".format(data_dir, sample_name),
        y="{}/samples/y_{}.npy".format(data_dir, sample_name),
        theta="{}/samples/theta_{}.npy".format(data_dir, sample_name),
        r_xz="{}/samples/r_xz_{}.npy".format(data_dir, sample_name),
        t_xz="{}/samples/t_xz_{}.npy".format(data_dir, sample_name),
        aux=aux_data,
        alpha=alpha,
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=initial_batch_size,
        initial_lr=initial_lr,
        final_lr=final_lr,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        limit_samplesize=limit_samplesize,
        verbose="all",
    )
    estimator.save("{}/models/{}_halftrained".format(data_dir, model_filename))
    estimator.train(
        method,
        x="{}/samples/x_{}.npy".format(data_dir, sample_name),
        y="{}/samples/y_{}.npy".format(data_dir, sample_name),
        theta="{}/samples/theta_{}.npy".format(data_dir, sample_name),
        r_xz="{}/samples/r_xz_{}.npy".format(data_dir, sample_name),
        t_xz="{}/samples/t_xz_{}.npy".format(data_dir, sample_name),
        aux=aux_data,
        alpha=alpha,
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=final_batch_size,
        initial_lr=initial_lr,
        final_lr=final_lr,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        limit_samplesize=limit_samplesize,
        verbose="all",
    )
    estimator.save("{}/models/{}".format(data_dir, model_filename))


def load_aux(filename, aux=None):
    if aux is None:
        return None, 0
    elif aux == "zs":
        return load_and_check(filename)[:, 0].reshape(-1, 1), 1
    elif aux == "zl":
        return load_and_check(filename)[:, 1].reshape(-1, 1), 1
    elif aux == "z":
        return load_and_check(filename)[:, ::2].reshape(-1, 2), 2
    elif aux == "all":
        return load_and_check(filename)[:, :].reshape(-1, 3), 3
    else:
        raise ValueError(
            "Unknown aux settings {}, please use 'zs', 'zl', 'z', or 'all'.".format(aux)
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: simulation"
    )

    # Main options
    parser.add_argument(
        "method",
        help='Inference method: "carl", "rolr", "alice", "cascal", "rascal", "alices".',
    )
    parser.add_argument(
        "--aux",
        type=str,
        default=None,
        help='Whether auxiliary information is used during training. Can be "zs" for '
        'the source redshift, "zl" for the lens redshift, "z" for both redshifts,'
        ' and "all" for both redshifts as well as sigma_v.',
    )
    parser.add_argument(
        "--sample", type=str, default="train", help='Sample name, like "train".'
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name. Defaults to the name of the method.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the "
        "data/models subfolder.",
    )

    # Training options
    parser.add_argument(
        "--vgg", action="store_true", help="Usee VGG rather than ResNet."
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use a deeper variation, i.e. ResNet-50 instead of ResNet-18.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="alpha parameter weighting the score MSE in the loss function of the SCANDAL, RASCAL, and"
        "and ALICES inference methods. Default: 0.0001",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether the log of the input is taken."
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of epochs per batch size. Default: 25."
    )
    parser.add_argument(
        "--initial_batch_size", type=int, default=256, help="Batch size during first half of training. Default: 128."
    )
    parser.add_argument(
        "--final_batch_size", type=int, default=512, help="Batch size during first half of training. Default: 512."
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        help='Optimizer. "amsgrad", "adam", and "sgd" are supported. Default: "adam".',
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=0.001,
        help="Initial learning rate. Default: 0.001.",
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=0.0001,
        help="Final learning rate. Default: 0.0001.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.3,
        help="Validation split. Default: 0.3.",
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

    if args.vgg:
        architecture = "vgg"
    elif args.deep:
        architecture = "resnet50"
    else:
        architecture = "resnet"

    train(
        method=args.method,
        aux=args.aux,
        alpha=args.alpha,
        data_dir="{}/data/".format(args.dir),
        sample_name=args.sample,
        model_filename=args.name,
        log_input=args.log,
        initial_batch_size=args.initial_batch_size,
        final_batch_size=args.final_batch_size,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        architecture=architecture,
    )

    logging.info("All done! Have a nice day!")
