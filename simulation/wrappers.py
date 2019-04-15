import numpy as np
import logging

from simulation.population_sim_new import LensingObservationWithSubhalos

logger = logging.getLogger(__name__)


def augmented_data(
        n_calibs, betas,
        sim_mvgauss_mean, sim_mvgauss_cov,
        n_calib_mean, n_calib_std, beta_mean, beta_std,
        n_images=None, n_thetas_marginal=1000,
        inverse=False,
):
    # Input
    if n_images is None:
        n_images = len(n_calibs)
    n_verbose = max(1, n_images // 100)

    # Parameter from prior
    n_calib_prior = np.random.normal(loc=n_calib_mean, scale=n_calib_std, size=n_thetas_marginal)
    beta_prior = np.random.normal(loc=beta_mean, scale=beta_std, size=n_thetas_marginal)
    n_calib_prior = np.clip(n_calib_prior, 0., None)
    beta_prior = np.clip(beta_prior, None, -1.01)
    params_prior = np.vstack((n_calib_prior, beta_prior)).T

    # Output
    all_images, all_t_xz, all_log_r_xz = [], [], []

    # Main loop
    for i_sim in range(n_images):
        if (i_sim + 1) % n_verbose == 0:
            logger.info("Simulating image %s / %s", i_sim + 1, n_images)
        else:
            logger.debug("Simulating image %s / %s", i_sim + 1, n_images)

        # Prepare params
        n_calib = _pick_param(n_calibs, i_sim, n_images)
        beta = _pick_param(betas, i_sim, n_images)
        params = np.asarray([n_calib, beta])
        params_eval = np.vstack((params, params_prior))

        if inverse:
            # Choose one theta from prior that we use for sampling here
            i_sample = np.random.randint(n_images)
            n_calib, beta = params_prior[i_sample]

        # Simulate
        sim = LensingObservationWithSubhalos(
            sim_mvgauss_mean=sim_mvgauss_mean,
            sim_mvgauss_cov=sim_mvgauss_cov,
            n_calib=n_calib,
            beta=beta,
            params_eval=params_eval,
            calculate_joint_score=True
        )

        log_r_xz, uncertainty = _extract_log_r(sim, n_thetas_marginal)
        if uncertainty > 0.01:
            logger.warning("Large uncertainty: log r(x,z) = %s +/- %s", log_r_xz, uncertainty)

        all_images.append(sim.image_poiss_psf)
        all_t_xz.append(sim.joint_scores)
        all_log_r_xz.append(log_r_xz)

    all_images = np.array(all_images)
    all_t_xz = np.array(all_t_xz)
    all_log_r_xz = np.array(all_log_r_xz)

    return all_images, all_t_xz, all_log_r_xz


def _pick_param(xs, i, n):
    try:
        assert len(xs) == n
        return xs[i]
    except TypeError:
        return xs


def _extract_log_r(sim, n_thetas_marginal):
    # Evaluate likelihood ratio wrt evidence
    inverse_r_xz = 0.0
    for i_theta in range(n_thetas_marginal):
        inverse_r_xz += np.exp(sim.joint_log_probs[i_theta + 1] - sim.joint_log_probs[0])
    inverse_r_xz /= float(n_thetas_marginal)
    log_r_xz = -np.log(inverse_r_xz)

    # Estimate uncertainty of log r from MC sampling
    inverse_r_xz_uncertainty = 0.0
    for i_theta in range(n_thetas_marginal):
        inverse_r_xz_uncertainty += (np.exp(
            sim.joint_log_probs[i_theta + 1] - sim.joint_log_probs[0]) - inverse_r_xz) ** 2.0
    inverse_r_xz_uncertainty /= float(n_thetas_marginal) * (float(n_thetas_marginal) - 1.0)
    log_r_xz_uncertainty = inverse_r_xz_uncertainty / inverse_r_xz

    return log_r_xz, log_r_xz_uncertainty
