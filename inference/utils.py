from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import torch
from torch import nn, optim
import logging

from inference import losses

logger = logging.getLogger(__name__)


def get_activation_function(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Activation function %s unknown", activation)


def s_from_r(r):
    return np.clip(1.0 / (1.0 + r), 0.0, 1.0)


def r_from_s(s, epsilon=1.0e-6):
    return np.clip((1.0 - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def check_for_nans_in_parameters(model, check_gradients=True):
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True

        if check_gradients and torch.any(torch.isnan(param.grad)):
            return True

    return False


def shuffle(*arrays):
    """ Shuffles multiple arrays simultaneously """

    permutation = None
    n_samples = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        assert a.shape[0] == n_samples
        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def balance_thetas(theta_sets_types, theta_sets_values):
    """Repeats theta values such that all thetas lists have the same length """

    n_sets = max([len(thetas) for thetas in theta_sets_types])

    for i, (types, values) in enumerate(zip(theta_sets_types, theta_sets_values)):
        assert len(types) == len(values)
        n_sets_before = len(types)

        if n_sets_before != n_sets:
            theta_sets_types[i] = [types[j % n_sets_before] for j in range(n_sets)]
            theta_sets_values[i] = [values[j % n_sets_before] for j in range(n_sets)]

    return theta_sets_types, theta_sets_values


def sanitize_array(
    array,
    replace_nan=0.0,
    replace_inf=0.0,
    replace_neg_inf=0.0,
    min_value=None,
    max_value=None,
):
    array[np.isneginf(array)] = replace_neg_inf
    array[np.isinf(array)] = replace_inf
    array[np.isnan(array)] = replace_nan

    if min_value is not None or max_value is not None:
        array = np.clip(array, min_value, max_value)

    return array


def load_and_check(filename, warning_threshold=1.0e9, memmap=False):
    if filename is None:
        return None

    if memmap and "x_train.npy" in filename:
        data = np.memmap(filename, shape=(1000000, 64, 64), dtype=np.float64, mode="c")
        logger.info(
            "Loaded %s with memmap. Found shape %s, dtype %s, and first entry\n%s",
            filename,
            data.shape,
            data.dtype,
            data[0]
        )
    else:
        data = np.load(filename)

    n_nans = np.sum(np.isnan(data))
    n_infs = np.sum(np.isinf(data))
    n_finite = np.sum(np.isfinite(data))

    if n_nans + n_infs > 0:
        logger.warning(
            "Warning: file %s contains %s NaNs and %s Infs, compared to %s finite numbers!",
            filename,
            n_nans,
            n_infs,
            n_finite,
        )

    smallest = np.nanmin(data)
    largest = np.nanmax(data)

    if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
        logger.warning(
            "Warning: file %s has some large numbers, rangin from %s to %s",
            filename,
            smallest,
            largest,
        )

    return data


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """
    Calculates quantiles (similar to np.percentile), but supports weights.

    Parameters
    ----------
    values : ndarray
        Data
    quantiles : ndarray
        Which quantiles to calculate
    sample_weight : ndarray or None
        Weights
    values_sorted : bool
        If True, will avoid sorting the initial array
    old_style : bool
        If True, will correct output to be consistent with np.percentile

    Returns
    -------
    quantiles : ndarray
        Quantiles

    """

    # Input
    values = np.array(values, dtype=np.float64)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight, dtype=np.float64)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    # Sort
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    # Quantiles
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    # Postprocessing
    if old_style:
        # To be consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def approx_equal(a, b, epsilon=1.0e-6):
    return abs(a - b) < epsilon


def create_missing_folders(folders):
    if folders is None:
        return

    for folder in folders:
        if folder is None or folder == "":
            continue

        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError("Path {} exists, but is no directory!".format(folder))


def get_loss(method, alpha):
    if method in ["carl"]:
        loss_functions = [losses.ratio_xe]
        loss_weights = [1.0]
        loss_labels = ["xe"]
    elif method in ["rolr"]:
        loss_functions = [losses.ratio_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_r"]
    elif method == "cascal":
        loss_functions = [losses.ratio_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["xe", "mse_score"]
    elif method == "rascal":
        loss_functions = [losses.ratio_mse, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method in ["alice"]:
        loss_functions = [losses.ratio_augmented_xe]
        loss_weights = [1.0]
        loss_labels = ["improved_xe"]
    elif method == "alices":
        loss_functions = [losses.ratio_augmented_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    return loss_functions, loss_labels, loss_weights


def get_optimizer(optimizer, nesterov_momentum):
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
    return opt, opt_kwargs


def restrict_samplesize(n, *arrays):
    restricted_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            restricted_arrays.append(None)
            continue

        restricted_arrays.append(a[:n])

    return restricted_arrays