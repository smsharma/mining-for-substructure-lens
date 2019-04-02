from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
from torch import tensor

logger = logging.getLogger(__name__)


def evaluate_ratio_model(model, theta0s=None, xs=None, evaluate_score=False, evaluate_grad_x=False, run_on_gpu=True, double_precision=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % len(theta0s)], requires_grad=evaluate_score) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    theta0s = theta0s.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate ratio estimator with score or x gradients:
    if evaluate_score or evaluate_grad_x:
        model.eval()

        s_hat, log_r_hat, t_hat, x_grad = model(theta0s, xs, track_score=evaluate_score, return_grad_x=evaluate_grad_x, create_gradient_graph=False)

        # Copy back tensors to CPU
        if run_on_gpu:
            s_hat = s_hat.cpu()
            log_r_hat = log_r_hat.cpu()
            if t_hat is not None:
                t_hat = t_hat.cpu()

        # Get data and return
        s_hat = s_hat.detach().numpy().flatten()
        log_r_hat = log_r_hat.detach().numpy().flatten()
        if t_hat is not None:
            t_hat = t_hat.detach().numpy()
        if x_grad is not None:
            x_grad = x_grad.detach().numpy()

    # Evaluate ratio estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            s_hat, log_r_hat, _, _ = model(theta0s, xs, track_score=False, return_grad_x=False, create_gradient_graph=False)

            # Copy back tensors to CPU
            if run_on_gpu:
                s_hat = s_hat.cpu()
                log_r_hat = log_r_hat.cpu()

            # Get data and return
            s_hat = s_hat.detach().numpy().flatten()
            log_r_hat = log_r_hat.detach().numpy().flatten()
            t_hat = None
            x_grad = None

    return s_hat, log_r_hat, t_hat, x_grad
