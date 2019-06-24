from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from torch.nn import BCELoss, MSELoss


def ratio_mse_num(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    inv_r_hat = torch.exp(-log_r_hat)
    inv_r_true = 1.0 / r_true
    return MSELoss()((1.0 - y_true) * inv_r_hat, (1.0 - y_true) * inv_r_true)


def ratio_mse_den(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat = torch.exp(log_r_hat)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    return ratio_mse_num(
        s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true
    ) + ratio_mse_den(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true)


def ratio_score_mse_num(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    return MSELoss()((1.0 - y_true) * t0_hat, (1.0 - y_true) * t0_true)


def ratio_xe(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat = torch.exp(log_r_hat)
    s_hat = 1.0 / (1.0 + r_hat)

    return BCELoss()(s_hat, y_true)


def ratio_augmented_xe(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat = torch.exp(log_r_hat)
    s_hat = 1.0 / (1.0 + r_hat)
    s_true = 1.0 / (1.0 + r_true)

    return BCELoss()(s_hat, s_true)
