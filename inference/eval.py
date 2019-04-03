from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import logging
import torch
from torch import tensor

logger = logging.getLogger(__name__)

def _evaluate_batch(model, theta0s, xs, evaluate_score, evaluate_grad_x, run_on_gpu, double_precision):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % len(theta0s)], requires_grad=evaluate_score) for i in range(n_xs)])
    xs = torch.stack([tensor(x) for x in xs])

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
            if x_grad is not None:
                x_grad = x_grad.cpu()

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


def evaluate_ratio_model(model, theta0s, xs, evaluate_score=False, evaluate_grad_x=False, run_on_gpu=True, double_precision=False, batch_size=1000):
    # Batches
    n_xs = len(xs)
    n_batches = (n_xs - 1) // batch_size + 1

    # results
    all_s, all_log_r, all_t, all_x_grad = [], [], [], []

    for i_batch in range(n_batches):
        logger.debug("Evaluating batch %s / %s", i_batch + 1, n_batches)

        theta_batch = np.copy(theta0s[i_batch * batch_size:(i_batch+1)*batch_size])
        x_batch = np.copy(xs[i_batch * batch_size:(i_batch+1)*batch_size])

        logger.debug("Batch data: x has shape %s, thetas has shape %s", x_batch.shape, theta_batch.shape)

        s, log_r, t, x_grad = _evaluate_batch(model, theta_batch, x_batch, evaluate_score, evaluate_grad_x, run_on_gpu, double_precision)

        all_s.append(s)
        all_log_r.append(log_r)
        all_t.append(t)
        all_x_grad.append(x_grad)

    # mash together
    all_s = np.concatenate(all_s, 0)
    all_log_r = np.concatenate(all_log_r, 0)
    all_t = np.concatenate(all_t, 0)
    all_x_grad = np.concatenate(all_x_grad, 0)

    return all_s, all_log_r, all_t, all_x_grad
