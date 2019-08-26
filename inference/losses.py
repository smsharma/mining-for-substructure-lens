from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.nn import BCELoss, MSELoss


def mse_r0(s_hat, log_r_hat, t_hat, y, log_r, t):
    inv_r_hat = torch.exp(-log_r_hat)
    inv_r = torch.exp(-log_r)
    return MSELoss()((1.0 - y) * inv_r_hat, (1.0 - y) * inv_r)


def mse_r1(s_hat, log_r_hat, t_hat, y, log_r, t):
    r_hat = torch.exp(log_r_hat)
    r = torch.exp(log_r)
    return MSELoss()(y * r_hat, y * r)


def mse_r(s_hat, log_r_hat, t_hat, y, log_r, t):
    return mse_r0(s_hat, log_r_hat, t_hat, y, log_r, t) + mse_r1(s_hat, log_r_hat, t_hat, y, log_r, t)


def mse_t0(s_hat, log_r_hat, t_hat, y, log_r, t):
    return MSELoss()((1.0 - y) * t_hat, (1.0 - y) * t)


def xe(s_hat, log_r_hat, t_hat, y, log_r, t):
    return BCELoss()(s_hat, y)


def augmented_xe(s_hat, log_r_hat, t_hat, y, log_r, t):
    s = 1.0 / (1.0 + torch.exp(log_r))
    return BCELoss()(s_hat, s)
