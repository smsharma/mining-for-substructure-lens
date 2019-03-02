from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from inference import losses


def package_training_data(method, x, theta0, y, r_xz, t_xz0):
    data = OrderedDict()
    data["x"] = x
    data["theta"] = theta0
    data["y"] = y
    if r_xz is not None:
        data["r_xz"] = r_xz
    if t_xz0 is not None:
        data["t_xz"] = t_xz0
    return data


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
