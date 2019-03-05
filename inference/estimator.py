from __future__ import absolute_import, division, print_function

import six
import logging
import os
import json
import numpy as np
import torch
from torch import optim

from inference.models import VGG11RatioEstimator
from inference.trainer import SingleParameterizedRatioTrainer
from inference.utils import create_missing_folders, load_and_check, shuffle, sanitize_array
from inference.methods import get_loss, package_training_data
from inference.eval import evaluate_ratio_model

logger = logging.getLogger(__name__)


class Estimator:
    """
    Estimating likelihood ratios with machine learning.

    Each instance of this class represents one neural estimator. The most important functions are:

    * `Estimator.train()` to train an estimator.
    * `Estimator.evaluate()` to evaluate the estimated likelihood ratio.
    * `Estimator.save()` to save the trained model to files.
    * `Estimator.load()` to load the trained model from files.
    """

    def __init__(self):
        self.model = None
        self.method = None
        self.resolution = None
        self.n_parameters = None
        self.n_conv_layers = None
        self.n_dense_layers = None
        self.n_feature_maps = None
        self.kernel_size = None
        self.pooling_size = None
        self.n_hidden_dense = None
        self.activation = None

    def train(
        self,
        method,
        x_filename,
        y_filename,
        theta0_filename=None,
        r_xz_filename=None,
        t_xz0_filename=None,
        n_conv_layers=3,
        n_dense_layers=3,
        n_feature_maps=128,
        kernel_size=5,
        pooling_size=2,
        n_hidden_dense=100,
        activation="relu",
        alpha=1.0,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=None,
        early_stopping=True,
        rescale_inputs=True,
        shuffle_labels=False,
        limit_samplesize=None,
        verbose="some",
    ):

        """
        Trains a neural network to estimate the likelihood ratio.

        Parameters
        ----------
        method : str
            The inference method used. Allowed values are 'alice', 'alices', 'carl', 'cascal', 'rascal', 'rolr', the
            difference between them is the combination of terms used in the loss function.

        x_filename : str
            Path to the observations (images) of the training sample.

        y_filename : str
            Path to the class labels of the training sample.

        theta0_filename : str
            Path to the parameters used for sampling the training sample.

        r_xz_filename : str or None, optional
            Path to the joint likelihood ratios of the training sample. Required if method is "alice", "alices",
            "rascal", or "rolr". Default value: None.

        t_xz0_filename : str or None, optional
            Path to the joint score of the training sample. Required if method is "alices", "cascal", or "rascal".
            Default value: None.

        n_conv_layers : int, optional
            Number of convolutional layers. Default value: 3.

        n_dense_layers : int, optional
            Number of fully connected layers. Default value: 2.

        n_feature_maps : int, optional
            Number of feature maps. Default value: 128.

        kernel_size : int, optional
            Size of the convolutional filters. Default value: 5.

        pooling_size : int, optional
            Size of the pooling kernel. Default value: 2.

        n_hidden_dense : int, optional
            Number of units in the hidden layers between the first fully connected layer and the output layer. Default
            value: 100.

        activation : {'tanh', 'sigmoid', 'relu'}, optional
            Activation function. Default value: 'relu'.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'cascal', and 'rascal'
             methods. Default value: 1.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
            Optimization algorithm. Default value: "amsgrad".

        n_epochs : int, optional
            Number of epochs. Default value: 50.

        batch_size : int, optional
            Batch size. Default value: 128.

        initial_lr : float, optional
            Learning rate during the first epoch, after which it exponentially decays to final_lr. Default value:
            0.001.

        final_lr : float, optional
            Learning rate during the last epoch. Default value: 0.0001.

        nesterov_momentum : float or None, optional
            If optimizer is "sgd", sets the Nesterov momentum. Default value: None.

        validation_split : float or None, optional
            Fraction of samples used  for validation and early stopping (if early_stopping is True). If None, the entire
            sample is used for training and early stopping is deactivated. Default value: None.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        rescale_inputs : bool, optional
            If True, the pixel values will be rescaled by dividing by np.std(x). Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
            None

        """

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        logger.info("  Training data:          x at %s", x_filename)
        if theta0_filename is not None:
            logger.info("                          theta0 at %s", theta0_filename)
        if y_filename is not None:
            logger.info("                          y at %s", y_filename)
        if r_xz_filename is not None:
            logger.info("                          r_xz at %s", r_xz_filename)
        if t_xz0_filename is not None:
            logger.info("                          t_xz (theta0) at %s", t_xz0_filename)
        logger.info("  Method:                 %s", method)
        logger.info("  Convolutional layers:   %s", n_conv_layers)
        logger.info("  Dense layers:           %s", n_dense_layers)
        logger.info("  Feature maps:           %s", n_feature_maps)
        logger.info("  Convolutional filter:   %s", kernel_size)
        logger.info("  Pooling filter:         %s", pooling_size)
        logger.info("  Activation function:    %s", activation)
        if method in ["cascal", "rascal", "alices"]:
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Trainer:                %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info(
            "  Learning rate:          %s initially, decaying to %s",
            initial_lr,
            final_lr,
        )
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Rescale inputs:         %s", rescale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Load training data
        logger.info("Loading training data")

        theta0 = load_and_check(theta0_filename)
        x = load_and_check(x_filename)
        y = load_and_check(y_filename).astype(np.float).reshape((-1, 1))
        r_xz = load_and_check(r_xz_filename)
        t_xz0 = load_and_check(t_xz0_filename)

        # Clean up input data
        x = sanitize_array(
            x, replace_inf=1.0e6, replace_nan=1.0e6, max_value=1.0e6, min_value=0.0
        ).astype(np.float64)
        theta0 = sanitize_array(
            theta0,
            replace_inf=1.0e6,
            replace_nan=1.0e6,
            max_value=1.0e6,
            min_value=1.0e-6,
        ).astype(np.float64)
        y = (
            sanitize_array(
                y, replace_inf=0.0, replace_nan=0.0, max_value=1.0, min_value=0.0
            )
            .astype(np.float64)
            .reshape((-1, 1))
        )
        if r_xz is not None:
            r_xz = (
                sanitize_array(
                    r_xz,
                    replace_inf=1.0e6,
                    replace_nan=1.0e6,
                    max_value=1.0e6,
                    min_value=1.0e-6,
                )
                .astype(np.float64)
                .reshape((-1, 1))
            )
        if t_xz0 is not None:
            t_xz0 = sanitize_array(
                t_xz0,
                replace_inf=1.0e6,
                replace_nan=1.0e6,
                max_value=1.0e6,
                min_value=-1.0e6,
            ).astype(np.float64)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_parameters = theta0.shape[1]
        resolution_x = x.shape[1]
        resolution_y = x.shape[2]

        logger.info(
            "Found %s samples with %s parameters and resolution %s x %s",
            n_samples,
            n_parameters,
            resolution_x,
            resolution_y,
        )

        if resolution_x != resolution_y:
            raise RuntimeError(
                "Currently only supports square images, but found resolution {} x {}".format(
                    resolution_x, resolution_y
                )
            )

        resolution = resolution_x

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info(
                "Only using %s of %s training samples", limit_samplesize, n_samples
            )

            x = x[:limit_samplesize]
            if theta0 is not None:
                theta0 = theta0[:limit_samplesize]
            if y is not None:
                y = y[:limit_samplesize]
            if r_xz is not None:
                r_xz = r_xz[:limit_samplesize]
            if t_xz0 is not None:
                t_xz0 = t_xz0[:limit_samplesize]

        # Reescale inputs
        if rescale_inputs:
            self.pixel_norm = np.std(x)
            x /= self.pixel_norm

            logger.info("Rescaling pixel values by factor 1 / %s", self.pixel_norm)
        else:
            self.pixel_norm = 1.0

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            y, r_xz, t_xz0 = shuffle(y, r_xz, t_xz0)

        # Data
        data = package_training_data(method, x, theta0, y, r_xz, t_xz0)

        # Save setup
        self.method = method
        self.resolution = resolution
        self.n_parameters = n_parameters
        self.n_conv_layers = n_conv_layers
        self.n_dense_layers = n_dense_layers
        self.n_feature_maps = n_feature_maps
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.n_hidden_dense = n_hidden_dense
        self.activation = activation

        # Create model
        logger.debug("Creating model")
        self.model = VGG11RatioEstimator()

        # Loss fn
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt_kwargs = None
        if optimizer == "adam":
            opt = optim.Adam
        elif optimizer == "amsgrad":
            opt = optim.Adam
            opt_kwargs = {"amsgrad": True}
        elif optimizer == "sgd":
            opt = optim.SGD
            if nesterov_momentum is not None:
                opt_kwargs = {"momentum": nesterov_momentum}
        else:
            raise ValueError("Unknown optimizer {}".format(optimizer))

        # Train model
        logger.info("Training model")
        trainer = SingleParameterizedRatioTrainer(self.model)

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
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def evaluate_ratio(
        self,
        x,
        theta0,
        test_all_combinations=True,
        evaluate_score=False,
    ):

        """
        Evaluates a trained estimator of the log likelihood ratio.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations.

        theta0 : str or None, optional
            Sample of parameter points, or path to numpy file with parameter points.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            Whether in addition to the likelihood ratio the score is evaluated. Default value: False.

        Returns
        -------

        log_likelihood_ratio : ndarray
            The estimated log likelihood ratio. If test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`. Otherwise, it has shape `(n_samples,)`.

        score : ndarray or None
            None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load evaluation data
        logger.debug("Loading evaluation data")
        if isinstance(theta0, six.string_types):
            theta0 = load_and_check(theta0)
        if isinstance(x, six.string_types):
            x = load_and_check(x)

        # Clean up input data
        x = sanitize_array(
            x, replace_inf=1.0e6, replace_nan=1.0e6, max_value=1.0e6, min_value=0.0
        ).astype(np.float64)
        theta0 = sanitize_array(
            theta0,
            replace_inf=1.0e6,
            replace_nan=1.0e6,
            max_value=1.0e6,
            min_value=1.0e-6,
        ).astype(np.float64)

        # Rescale pixel values
        x /= self.pixel_norm

        # Evaluation for all other methods
        all_log_r_hat = []
        all_t_hat = []

        if test_all_combinations:
            logger.debug("Starting ratio evaluation for all combinations")

            for i, this_theta0 in enumerate(theta0):
                logger.debug(
                    "Starting ratio evaluation for thetas %s / %s: %s",
                    i + 1,
                    len(theta0),
                    theta0,
                )

                _, log_r_hat, t_hat = evaluate_ratio_model(
                    model=self.model,
                    theta0s=[this_theta0],
                    xs=x,
                    evaluate_score=evaluate_score,
                )

                all_log_r_hat.append(log_r_hat)
                all_t_hat.append(t_hat)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.debug("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat = evaluate_ratio_model(
                model=self.model,
                theta0s=theta0,
                xs=x,
                evaluate_score=evaluate_score,
            )

        logger.debug("Evaluation done")
        return all_log_r_hat, all_t_hat

    def evaluate_posterior(self, xs, thetas_eval, prior_thetas_eval, thetas_from_prior):
        """
        Estimates the posterior p(theta | xs) for a series of observations xs, and a list of evaluation parameter
        points theta in thetas_eval.

        Parameters
        ----------
        xs : ndarray
            Observations with shape (n_observations, resolution_x, resolution_y).

        thetas_eval : ndarray
            Parameter points to be evaluated with shape (n_evaluations, n_parameters).

        prior_thetas_eval : ndarray
            Prior evaluated at thetas_eval with shape (n_evaluations,).

        thetas_from_prior : ndarray
            Parameter points drawn from the prior with shape (n_theta_samples, n_parameters).

        Returns
        -------
        posteerior : ndarray
            Estimated posterior probabilities with shape (n_evaluations,).

        """

        # Calculate likelihood ratios
        log_r_evals = self.evaluate_ratio(xs, thetas_eval)[0]  # (n_eval, n_x)
        log_r_priors = self.evaluate_ratio(xs, thetas_from_prior)[0]  # (n_prior, n_x)

        # Calculate posteriors
        posteriors = []

        for prior_eval, log_r_eval in zip(prior_thetas_eval, log_r_evals):
            numerator = 0.0

            for log_r_prior in log_r_priors:
                log_likelihood_ratio_priorsample_eval = np.sum(log_r_prior - log_r_eval)
                numerator += np.exp(log_likelihood_ratio_priorsample_eval)

            numerator /= float(len(log_r_priors))

            posterior = prior_eval / numerator
            posteriors.append(posterior)

        return np.asarray(posteriors)

    def save(self, filename, save_model=False):
        """
        Saves the trained model to two files: a JSON file with the settings and a pickled pyTorch state dict.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with Estimator.load(), but can be useful for debugging, for instance to plot the computational graph.
            Default value: False.

        Returns
        -------
            None

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)

        settings = {
            "method": self.method,
            "resolution": self.resolution,
            "n_parameters": self.n_parameters,
            "n_conv_layers": self.n_conv_layers,
            "n_dense_layers": self.n_dense_layers,
            "n_feature_maps": self.n_feature_maps,
            "kernel_size": self.kernel_size,
            "pooling_size": self.pooling_size,
            "n_hidden_dense": self.n_hidden_dense,
            "activation": self.activation,
            "pixel_norm": self.pixel_norm,
        }

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

        """
        Loads a trained model from files.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        # Load settings
        logger.debug("Loading settings from %s_settings.json", filename)

        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)

        self.method = str(settings["method"])
        self.resolution = int(settings["resolution"])
        self.n_parameters = int(settings["n_parameters"])
        self.n_conv_layers = int(settings["n_conv_layers"])
        self.n_dense_layers = int(settings["n_dense_layers"])
        self.n_feature_maps = int(settings["n_feature_maps"])
        self.kernel_size = int(settings["kernel_size"])
        self.pooling_size = int(settings["pooling_size"])
        self.n_hidden_dense = int(settings["n_hidden_dense"])
        self.activation = str(settings["activation"])
        self.pixel_norm = float(settings["pixel_norm"])

        logger.debug(
            "  Found method %s, resolution %s, %s parameters, %s convolutional layers, %s dense layers, %s feature maps"
            ", a conv kernel of size %s, a pooling kernel of size %s, %s hidden units, and the activation function %s",
            self.method,
            self.resolution,
            self.n_parameters,
            self.n_conv_layers,
            self.n_dense_layers,
            self.n_feature_maps,
            self.kernel_size,
            self.pooling_size,
            self.n_hidden_dense,
            self.activation,
        )

        # Create model
        self.model = VGG11RatioEstimator()

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt"))