from __future__ import absolute_import, division, print_function

import logging
import os
import json
import numpy as np
from collections import OrderedDict
import torch

from inference.models.vgg import VGGRatioEstimator
from inference.models.resnet import ResNetRatioEstimator
from inference.trainer import SingleParameterizedRatioTrainer
from inference.utils import create_missing_folders, load_and_check, get_optimizer
from inference.utils import get_loss, clean_r, clean_t
from inference.utils import restrict_samplesize

logger = logging.getLogger(__name__)


class ParameterizedRatioEstimator(object):
    theta_mean = np.array([150.0, -1.9])
    theta_std = np.array([50.0, 0.3])

    def __init__(self, resolution=64, n_parameters=2, n_aux=0, architecture="resnet", log_input=False, rescale_inputs=True, rescale_theta=True):
        self.resolution = resolution
        self.n_parameters = n_parameters
        self.n_aux = n_aux
        self.log_input = log_input
        self.rescale_inputs = rescale_inputs
        self.rescale_theta = rescale_theta
        self.architecture = architecture

        self.x_scaling_mean = None
        self.x_scaling_std = None
        self.aux_scaling_mean = None
        self.aux_scaling_std = None

        self._create_model()

    def train(
        self,
        method,
        x,
        y,
        theta,
        aux=None,
        r_xz=None,
        t_xz=None,
        alpha=1.0,
        optimizer="adam",
        n_epochs=50,
        batch_size=256,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        validation_split_seed=None,
        early_stopping=True,
        limit_samplesize=None,
        verbose="some",
        update_input_rescaling=True,
        validation_loss_before=None
    ):

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        if method in ["cascal", "rascal", "alices"]:
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)
        logger.info("  Update x rescaling:     %s", update_input_rescaling)

        # Load training data
        logger.info("Loading training data")
        theta = load_and_check(theta, memmap=False)
        x = load_and_check(x, memmap=True)
        y = load_and_check(y, memmap=False)
        r_xz = load_and_check(r_xz, memmap=False)
        t_xz = load_and_check(t_xz, memmap=False)
        aux = load_and_check(aux, memmap=False)

        self._check_required_data(method, r_xz, t_xz)
        if update_input_rescaling:
            self._initialize_input_transform(x, aux)

        # Clean up input data
        y = y.reshape((-1, 1))
        if r_xz is not None:
            r_xz = r_xz.reshape((-1, 1))
        theta = theta.reshape((-1, 2))
        r_xz = clean_r(r_xz)
        t_xz = clean_t(t_xz)

        # Rescale aux, theta, and t_xz
        aux = self._transform_aux(aux)
        theta = self._transform_theta(theta)
        if t_xz is not None:
            t_xz = self._transform_t_xz(t_xz)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_parameters = theta.shape[1]
        resolution_x = x.shape[1]
        resolution_y = x.shape[2]
        n_aux = 0 if aux is None else aux.shape[1]
        logger.info(
            "Found %s samples with %s parameters, image resolution %s x %s, and %s auxiliary parameters",
            n_samples,
            n_parameters,
            resolution_x,
            resolution_y,
            n_aux,
        )
        if resolution_x != resolution_y:
            raise RuntimeError("Currently only supports square images, but found resolution {} x {}".format(resolution_x, resolution_y))
        resolution = resolution_x
        if n_aux != self.n_aux:
            raise RuntimeError(
                "Number of auxiliary variables found in data ({}) does not match number of" "auxiliary variables in model ({})".format(n_aux, self.n_aux)
            )
        if aux is not None and aux.shape[0] != n_samples:
            raise RuntimeError("Number of samples in auxiliary variables does not match number of" "samples ({})".format(aux.shape[0], n_samples))

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, theta, y, r_xz, t_xz, aux = restrict_samplesize(limit_samplesize, x, theta, y, r_xz, t_xz, aux)

        # Check consistency of input with model
        if n_parameters != self.n_parameters:
            raise RuntimeError("Number of parameters does not match model: {} vs {}".format(n_parameters, self.n_parameters))
        if resolution != self.resolution:
            raise RuntimeError("Number of observables does not match model: {} vs {}".format(resolution, self.resolution))

        # Data
        data = self._package_training_data(method, x, theta, y, r_xz, t_xz, aux)

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = SingleParameterizedRatioTrainer(self.model, run_on_gpu=True)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            validation_split_seed=validation_split_seed,
            early_stopping=early_stopping,
            verbose=verbose,
            validation_loss_before=validation_loss_before
        )
        return result

    def log_likelihood_ratio(
        self, x, theta, aux=None, test_all_combinations=True, evaluate_score=False, evaluate_grad_x=False, batch_size=10000, grad_x_theta_index=0
    ):
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.debug("Loading evaluation data")
        x = load_and_check(x, memmap=True)
        aux = load_and_check(aux)
        theta = load_and_check(theta)

        # Rescale theta and aux
        aux = self._transform_aux(aux)
        theta = self._transform_theta(theta)

        # Evaluate
        if test_all_combinations:
            logger.debug("Starting ratio evaluation for all combinations")

            all_log_r_hat = []
            all_t_hat = []
            all_grad_x = None

            for i, this_theta in enumerate(theta):
                logger.debug("Starting ratio evaluation for thetas %s / %s: %s", i + 1, len(theta), this_theta)
                _, log_r_hat, t_hat, x_grad = self._evaluate(
                    theta0s=[this_theta], xs=x, auxs=aux, evaluate_score=evaluate_score, evaluate_grad_x=evaluate_grad_x, batch_size=batch_size
                )

                all_log_r_hat.append(log_r_hat)
                all_t_hat.append(t_hat)
                if x_grad is not None and i == grad_x_theta_index:
                    all_grad_x = x_grad

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.debug("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat, all_grad_x = self._evaluate(
                theta0s=theta, xs=x, auxs=aux, evaluate_score=evaluate_score, evaluate_grad_x=evaluate_grad_x, batch_size=batch_size
            )

        logger.debug("Evaluation done")
        return all_log_r_hat, all_t_hat, all_grad_x

    def _evaluate(self, theta0s, xs, auxs=None, evaluate_score=False, evaluate_grad_x=False, run_on_gpu=True, double_precision=False, batch_size=10000):
        # Batches
        n_xs = len(xs)
        n_batches = (n_xs - 1) // batch_size + 1

        # results
        all_s, all_log_r, all_t, all_x_grad = [], [], [], []

        for i_batch in range(n_batches):
            x_batch = np.asarray(np.copy(xs[i_batch * batch_size : (i_batch + 1) * batch_size]))
            if len(theta0s) == n_xs:
                theta_batch = np.copy(theta0s[i_batch * batch_size : (i_batch + 1) * batch_size])
            else:
                theta_batch = np.repeat(np.copy(theta0s).reshape(1,-1), x_batch.shape[0], axis=0)
            if auxs is not None:
                aux_batch = np.copy(auxs[i_batch * batch_size : (i_batch + 1) * batch_size])
            else:
                aux_batch = None

            s, log_r, t, x_grad = self._evaluate_batch(theta_batch, x_batch, aux_batch, evaluate_score, evaluate_grad_x, run_on_gpu, double_precision)

            all_s.append(s)
            all_log_r.append(log_r)
            if t is not None:
                all_t.append(t)
            if x_grad is not None:
                all_x_grad.append(x_grad)

        # mash together
        all_s = np.concatenate(all_s, 0)
        all_log_r = np.concatenate(all_log_r, 0)
        if len(all_t) > 0:
            all_t = np.concatenate(all_t, 0)
        else:
            all_t = None
        if len(all_x_grad) > 0:
            all_x_grad = np.concatenate(all_x_grad, 0)
        else:
            all_x_grad = None

        return all_s, all_log_r, all_t, all_x_grad

    def _evaluate_batch(self, theta0s, xs, auxs, evaluate_score, evaluate_grad_x, run_on_gpu, double_precision):
        # CPU or GPU?
        run_on_gpu = run_on_gpu and torch.cuda.is_available()
        device = torch.device("cuda" if run_on_gpu else "cpu")
        dtype = torch.double if double_precision else torch.float

        # Prepare data
        self.model = self.model.to(device, dtype)

        theta0s = torch.from_numpy(theta0s).to(device, dtype)
        xs = torch.from_numpy(xs).to(device, dtype)
        if auxs is not None:
            auxs = torch.from_numpy(auxs).to(device, dtype)

        # Evaluate ratio estimator with score or x gradients:
        if evaluate_score or evaluate_grad_x:
            self.model.eval()

            if evaluate_score:
                theta0s.requires_grad = True
            if evaluate_grad_x:
                xs.requires_grad = True

            s, log_r, t, x_grad = self.model(theta0s, xs, aux=auxs, track_score=evaluate_score, return_grad_x=evaluate_grad_x, create_gradient_graph=False)

            # Copy back tensors to CPU
            if run_on_gpu:
                s = s.cpu()
                log_r = log_r.cpu()
                if t is not None:
                    t = t.cpu()
                if x_grad is not None:
                    x_grad = x_grad.cpu()

            # Get data and return
            s = s.detach().numpy().flatten()
            log_r = log_r.detach().numpy().flatten()
            if t is not None:
                t = t.detach().numpy()
            if x_grad is not None:
                x_grad = x_grad.detach().numpy()

        # Evaluate ratio estimator without score:
        else:
            with torch.no_grad():
                self.model.eval()

                s, log_r, _, _ = self.model(theta0s, xs, aux=auxs, track_score=False, return_grad_x=False, create_gradient_graph=False)

                # Copy back tensors to CPU
                if run_on_gpu:
                    s = s.cpu()
                    log_r = log_r.cpu()

                # Get data and return
                s = s.detach().numpy().flatten()
                log_r = log_r.detach().numpy().flatten()
                t = None
                x_grad = None

        return s, log_r, t, x_grad

    def save(self, filename, save_model=False):
        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)
        settings = self._wrap_settings()

        with open(filename + "_settings.json", "w") as f:
            json.dump(settings, f)

        # Save state dict
        logger.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), filename + "_state_dict.pt")

        # Save model
        if save_model:
            logger.debug("Saving model to %s_model.pt", filename)
            torch.save(self.model, filename + "_model.pt")

    def load(self, filename):
        # Load settings and create model
        logger.debug("Loading settings from %s_settings.json", filename)
        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)
        self._unwrap_settings(settings)
        self._create_model()

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt", map_location="cpu"))

    def _create_model(self):
        logger.info("Creating model")
        logger.info("  Architecture:           %s", self.architecture)
        logger.info("  Log input:              %s", self.log_input)
        logger.info("  Rescale input:          %s", self.x_scaling_std is not None and self.x_scaling_mean is not None)

        if self.architecture in ["resnet", "resnet18"]:
            self.model = ResNetRatioEstimator(
                n_parameters=self.n_parameters,
                n_aux=self.n_aux,
                n_hidden=512,
                log_input=self.log_input,
                input_mean=self.x_scaling_mean,
                input_std=self.x_scaling_std,
            )

        elif self.architecture == "resnet50":
            self.model = ResNetRatioEstimator(
                n_parameters=self.n_parameters,
                n_aux=self.n_aux,
                cfg=50,
                n_hidden=1024,
                log_input=self.log_input,
                input_mean=self.x_scaling_mean,
                input_std=self.x_scaling_std,
            )

        elif self.architecture == "vgg":
            self.model = VGGRatioEstimator(
                n_parameters=self.n_parameters, log_input=self.log_input, input_mean=self.x_scaling_mean, input_std=self.x_scaling_std
            )

        else:
            raise RuntimeError("Unknown architecture {}".format(self.architecture))

        logger.info("Model has %s trainable parameters", self._count_model_parameters())

    def _count_model_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _initialize_input_transform(self, x, aux=None):
        if self.rescale_inputs:
            self.x_scaling_mean = np.mean(x)
            self.x_scaling_std = np.maximum(np.std(x), 1.0e-6)
        else:
            self.x_scaling_mean = None
            self.x_scaling_std = None

        if self.rescale_inputs and aux is not None:
            self.aux_scaling_mean = np.mean(aux, axis=0)
            self.aux_scaling_std = np.maximum(np.std(aux, axis=0), 1.0e-6)
        else:
            self.aux_scaling_mean = None
            self.aux_scaling_std = None

        self.model.input_mean = self.x_scaling_mean
        self.model.input_std = self.x_scaling_std

    def _transform_aux(self, aux):
        if aux is not None and self.aux_scaling_mean is not None and self.aux_scaling_std is not None:
            aux = aux - self.aux_scaling_mean[np.newaxis, :]
            aux = aux / self.aux_scaling_std[np.newaxis, :]
        return aux

    def _transform_theta(self, theta):
        if self.rescale_theta:
            theta = theta - self.theta_mean[np.newaxis, :]
            theta = theta / self.theta_std[np.newaxis, :]
        return theta

    def _transform_t_xz(self, t_xz):
        if self.rescale_theta:
            t_xz = t_xz * self.theta_std[np.newaxis, :]
        return t_xz

    def _wrap_settings(self):
        settings = {
            "resolution": self.resolution,
            "n_parameters": self.n_parameters,
            "n_aux": self.n_aux,
            "architecture": self.architecture,
            "log_input": self.log_input,
            "rescale_inputs": self.rescale_inputs,
            "x_scaling_mean": self.x_scaling_mean,
            "x_scaling_std": self.x_scaling_std,
            "rescale_theta": self.rescale_theta,
            "aux_scaling_mean": [] if self.aux_scaling_mean is None else list(self.aux_scaling_mean),
            "aux_scaling_std": [] if self.aux_scaling_std is None else list(self.aux_scaling_std),
        }
        return settings

    def _unwrap_settings(self, settings):
        self.resolution = int(settings["resolution"])
        self.n_parameters = int(settings["n_parameters"])
        self.n_aux = int(settings["n_aux"])
        self.architecture = str(settings["architecture"])
        self.log_input = bool(settings["log_input"])
        self.rescale_inputs = str(settings["rescale_inputs"])
        self.x_scaling_mean = float(settings["x_scaling_mean"])
        self.x_scaling_std = float(settings["x_scaling_std"])
        self.rescale_theta = bool(settings["rescale_theta"])
        self.aux_scaling_mean = list(settings["aux_scaling_mean"])
        if len(self.aux_scaling_mean) == 0:
            self.aux_scaling_mean = None
        else:
            self.aux_scaling_mean = np.array(self.aux_scaling_mean)
        self.aux_scaling_std = list(settings["aux_scaling_std"])
        if len(self.aux_scaling_std) == 0:
            self.aux_scaling_std = None
        else:
            self.aux_scaling_std = np.array(self.aux_scaling_std)

    @staticmethod
    def _check_required_data(method, r_xz, t_xz):
        if method in ["cascal", "alices", "rascal"] and t_xz is None:
            raise RuntimeError("Method {} requires joint score information".format(method))
        if method in ["rolr", "alices", "rascal"] and r_xz is None:
            raise RuntimeError("Method {} requires joint likelihood ratio information".format(method))

    @staticmethod
    def _package_training_data(method, x, theta, y, r_xz, t_xz, aux=None):
        data = OrderedDict()
        data["x"] = x
        data["theta"] = theta
        data["y"] = y
        if method in ["rolr", "alice", "alices", "rascal"]:
            data["r_xz"] = r_xz
        if method in ["cascal", "alices", "rascal"]:
            data["t_xz"] = t_xz
        if aux is not None:
            data["aux"] = aux
        return data
