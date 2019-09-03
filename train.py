#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

sys.path.append("./")

import logging
import argparse
import numpy as np

from inference.estimator import ParameterizedRatioEstimator
from inference.utils import load_and_check


def train(
    method,
    alpha,
    data_dir,
    sample_name,
    model_filename,
    aux=False,
    architecture="resnet",
    log_input=False,
    batch_size=128,
    n_epochs=50,
    optimizer="adam",
    initial_lr=1.0e-4,
    final_lr=1.0e-6,
    limit_samplesize=None,
    load=None,
    zero_bias=False,
):
    aux_data, n_aux = load_aux("{}/samples/z_{}.npy".format(data_dir, sample_name), aux)
    if aux_data is None:
        logging.info("%s aux variables", n_aux)
    else:
        logging.info("%s aux variables with shape %s", n_aux, aux_data.shape)

    logging.info("")
    logging.info("")
    logging.info("")
    logging.info("Creating estimator")
    logging.info("")
    estimator = ParameterizedRatioEstimator(
        resolution=64,
        n_parameters=2,
        n_aux=n_aux,
        architecture=architecture,
        log_input=log_input,
        rescale_inputs=True,
        zero_bias=zero_bias,
    )

    if load is not None:
        logging.info(
            "Loading pre-trained model from %s", "{}/models/{}".format(data_dir, load)
        )
        estimator.load("{}/models/{}".format(data_dir, load))

    estimator.train(
        method,
        x="{}/samples/x_{}.npy".format(data_dir, sample_name),
        theta="{}/samples/theta_{}.npy".format(data_dir, sample_name),
        theta_alt="{}/samples/theta_alt_{}.npy".format(data_dir, sample_name),
        log_r_xz="{}/samples/log_r_xz_{}.npy".format(data_dir, sample_name),
        log_r_xz_alt="{}/samples/log_r_xz_alt_{}.npy".format(data_dir, sample_name),
        t_xz="{}/samples/t_xz_{}.npy".format(data_dir, sample_name),
        t_xz_alt="{}/samples/t_xz_alt_{}.npy".format(data_dir, sample_name),
        aux=aux_data,
        alpha=alpha,
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=batch_size,
        initial_lr=initial_lr,
        final_lr=final_lr,
        nesterov_momentum=0.9,
        validation_split=0.25,
        early_stopping=True,
        limit_samplesize=limit_samplesize,
        verbose="all",
    )

    estimator.save("{}/models/{}".format(data_dir, model_filename))


def load_aux(filename, aux=False):
    if aux:
        return load_and_check(filename)[:, 2].reshape(-1, 1), 1
    else:
        return None, 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-level script for the training of the neural likelihood ratio estimators"
    )

    # Main options
    parser.add_argument(
        "method",
        help='Inference method: "carl", "rolr", "alice", "cascal", "rascal", "alices".',
    )
    parser.add_argument("sample", type=str, help='Sample name, like "train".')
    parser.add_argument(
        "name", type=str, help="Model name. Defaults to the name of the method."
    )
    parser.add_argument(
        "-z", action="store_true", help="Proivide lens redshift to the network"
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
        default=0.0002,
        help="alpha parameter weighting the score MSE in the loss function of the SCANDAL, RASCAL, and"
        "and ALICES inference methods. Default: 0.0002",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether the log of the input is taken."
    )
    parser.add_argument(
        "--load",
        default=None,
        type=str,
        help="Path of pretrained model that is loaded before training.",
    )
    parser.add_argument(
        "--zerobias", action="store_true", help="Initialize with zero bias."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs. Default: 100."
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        help='Optimizer. "amsgrad", "adam", and "sgd" are supported. Default: "adam".',
    )
    parser.add_argument(
        "--batchsize", type=int, default=128, help="Batch size. Default: 128."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
        help="Initial learning rate. Default: 0.0001",
    )
    parser.add_argument(
        "--lrdecay",
        type=float,
        default=1.0e-2,
        help="Learning rate decay (final LR / initial LR). Default: 0.01",
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
        aux=args.z,
        alpha=args.alpha,
        data_dir="{}/data/".format(args.dir),
        sample_name=args.sample,
        model_filename=args.name,
        log_input=args.log,
        batch_size=args.batchsize,
        initial_lr=args.lr,
        final_lr=args.lrdecay * args.lr,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        architecture=architecture,
        zero_bias=args.zerobias,
        load=args.load,
    )

    logging.info("All done! Have a nice day!")
