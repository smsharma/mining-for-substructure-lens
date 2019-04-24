from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from torch.nn import BCELoss, MSELoss


def _clean_rs(log_r_hat, r_true, log_r_clip=15.):
    r_hat = torch.exp(log_r_hat)
    r_hat = torch.where(
        torch.isnan(r_true),
        torch.ones_like(r_hat),
        r_hat
    )
    r_true = torch.where(
        torch.isnan(r_true),
        torch.ones_like(r_true),
        torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    )
    return r_hat, r_true


def ratio_mse_num(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    inv_r_hat, inv_r_true = _clean_rs(1. - log_r_hat, 1. / r_true)
    return MSELoss()((1.0 - y_true) * inv_r_hat, (1.0 - y_true) * inv_r_true)


def ratio_mse_den(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat, r_true = _clean_rs(log_r_hat, r_true)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    return (ratio_mse_num(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true)
            + ratio_mse_den(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true))


def ratio_score_mse_num(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    return MSELoss()((1.0 - y_true) * t0_hat, (1.0 - y_true) * t0_true)


def ratio_xe(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat, r_true = _clean_rs(log_r_hat, r_true)
    s_hat = 1.0 / (1.0 + r_hat)

    return BCELoss()(s_hat, y_true)


def ratio_augmented_xe(s_hat, log_r_hat, t0_hat, y_true, r_true, t0_true):
    r_hat, r_true = _clean_rs(log_r_hat, r_true)
    s_hat = 1.0 / (1.0 + r_hat)
    s_true = 1.0 / (1.0 + r_true)

    return BCELoss()(s_hat, s_true)
