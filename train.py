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
    initial_batch_size=128,
    final_batch_size=256,
    n_epochs=50,
    optimizer="adam",
    initial_lrs=[0.0005, 0.0002, 0.0001],
    final_lrs=[0.0001, 0.00005],
    limit_samplesize=None,
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
        resolution=64, n_parameters=2, n_aux=n_aux, architecture=architecture, log_input=log_input, rescale_inputs=True
    )

    best_loss = None
    epochs_per_lr = int(round(n_epochs / (len(initial_lrs) + len(final_lrs)), 0))

    for lr in initial_lrs:
        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("Starting training with batch size %s and learning rate %s", initial_batch_size, lr)
        logging.info("")
        _, losses = estimator.train(
            method,
            x="{}/samples/x_{}.npy".format(data_dir, sample_name),
            y="{}/samples/y_{}.npy".format(data_dir, sample_name),
            theta="{}/samples/theta_{}.npy".format(data_dir, sample_name),
            r_xz="{}/samples/r_xz_{}.npy".format(data_dir, sample_name),
            t_xz="{}/samples/t_xz_{}.npy".format(data_dir, sample_name),
            aux=aux_data,
            alpha=alpha,
            optimizer=optimizer,
            n_epochs=epochs_per_lr,
            batch_size=initial_batch_size,
            initial_lr=lr,
            final_lr=lr,
            nesterov_momentum=None,
            validation_split=0.25,
            validation_split_seed=3373,
            early_stopping=True,
            limit_samplesize=limit_samplesize,
            verbose="all",
            validation_loss_before=best_loss,
        )
        all_losses = [best_loss] + list(losses) if best_loss is not None else losses
        best_loss = np.nanmin(np.asarray(all_losses))
    estimator.save("{}/models/{}_halftrained".format(data_dir, model_filename))

    best_loss = None

    for lr in final_lrs:
        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("Starting training with batch size %s and learning rate %s", final_batch_size, lr)
        logging.info("")
        _, losses = estimator.train(
            method,
            x="{}/samples/x_{}.npy".format(data_dir, sample_name),
            y="{}/samples/y_{}.npy".format(data_dir, sample_name),
            theta="{}/samples/theta_{}.npy".format(data_dir, sample_name),
            r_xz="{}/samples/r_xz_{}.npy".format(data_dir, sample_name),
            t_xz="{}/samples/t_xz_{}.npy".format(data_dir, sample_name),
            aux=aux_data,
            alpha=alpha,
            optimizer=optimizer,
            n_epochs=epochs_per_lr,
            batch_size=final_batch_size,
            initial_lr=lr,
            final_lr=lr,
            nesterov_momentum=None,
            validation_split=0.25,
            validation_split_seed=3373,
            early_stopping=True,
            limit_samplesize=limit_samplesize,
            verbose="all",
            validation_loss_before=best_loss,
        )
        all_losses = [best_loss] + list(losses)
        best_loss = np.nanmin(np.asarray(all_losses))
    estimator.save("{}/models/{}".format(data_dir, model_filename))


def load_aux(filename, aux=False):
    if aux:
        return load_and_check(filename)[:, 2].reshape(-1, 1), 1
    else:
        return None, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Strong lensing experiments: simulation")

    # Main options
    parser.add_argument("method", help='Inference method: "carl", "rolr", "alice", "cascal", "rascal", "alices".')
    parser.add_argument("sample", type=str, help='Sample name, like "train".')
    parser.add_argument("name", type=str, help="Model name. Defaults to the name of the method.")
    parser.add_argument("-z", action="store_true", help="Proivide lens redshift to the network")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the " "data/models subfolder.",
    )

    # Training options
    parser.add_argument("--vgg", action="store_true", help="Usee VGG rather than ResNet.")
    parser.add_argument("--deep", action="store_true", help="Use a deeper variation, i.e. ResNet-50 instead of ResNet-18.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="alpha parameter weighting the score MSE in the loss function of the SCANDAL, RASCAL, and"
        "and ALICES inference methods. Default: 0.0001",
    )
    parser.add_argument("--log", action="store_true", help="Whether the log of the input is taken.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs. Default: 120.")
    parser.add_argument("--optimizer", default="adam", help='Optimizer. "amsgrad", "adam", and "sgd" are supported. Default: "adam".')
    parser.add_argument("--initial_batch_size", type=int, default=128, help="Batch size during first half of training. Default: 128.")
    parser.add_argument("--final_batch_size", type=int, default=512, help="Batch size during second half of training. Default: 256.")
    parser.add_argument(
        "--initial_lrs",
        type=float,
        nargs="+",
        default=[0.003, 0.001, 0.0003],
        help="Learning rate steps during first half of training.",
    )
    parser.add_argument(
        "--final_lrs",
        type=float,
        nargs="+",
        default=[0.001, 0.0003, 0.0001],
        help="Learning rate steps during second half of training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.INFO)
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
        initial_batch_size=args.initial_batch_size,
        final_batch_size=args.final_batch_size,
        n_epochs=args.epochs,
        optimizer=args.optimizer,
        initial_lrs=args.initial_lrs,
        final_lrs=args.final_lrs,
        architecture=architecture,
    )

    logging.info("All done! Have a nice day!")
