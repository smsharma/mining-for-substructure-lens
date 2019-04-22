import numpy as np
import logging
from scipy.stats import uniform, norm

from simulation.population_sim import LensingObservationWithSubhalos

logger = logging.getLogger(__name__)


def augmented_data(
        n_calib=None, beta=None,
        n_calib_prior=norm(150., 50.), beta_prior=norm(-1.9, 0.3),
        n_images=None, n_thetas_marginal=1000,
        inverse=False, mine_gold=True,
        sim_mvgauss_file="simulation/data/sim_mvgauss.npz"
):
    # Input
    if (n_calib is None or beta is None) and n_images is None:
        raise ValueError("Either n_calib and beta or n_images have to be different from None")
    if n_images is None:
        n_images = len(n_calib)
    n_verbose = max(1, n_images // 100)

    # Load fitted data
    sim_mvgauss = np.load(sim_mvgauss_file)
    sim_mvgauss_mean = sim_mvgauss["mean"]
    sim_mvgauss_cov = sim_mvgauss["cov"]

    # Test hypothesis
    if n_calib is None:
        n_calib = n_calib_prior.rvs(size=n_images)
    if beta is None:
        beta = beta_prior.rvs(size=n_images)
    n_calib = np.clip(n_calib, 0., None)
    beta = np.clip(beta, None, -1.001)

    # Reference hypothesis
    n_calib_ref = n_calib_prior.rvs(size=n_thetas_marginal)
    beta_ref = beta_prior.rvs(size=n_thetas_marginal)
    n_calib_ref = np.clip(n_calib_ref, 0., None)
    beta_ref = np.clip(beta_ref, None, -1.001)
    params_ref = np.vstack((n_calib_ref, beta_ref)).T

    # Output
    all_params, all_images, all_t_xz, all_log_r_xz, all_sub_latents, all_host_latents = [], [], [], [], [], []

    # Main loop
    for i_sim in range(n_images):
        if (i_sim + 1) % n_verbose == 0:
            logger.info("Simulating image %s / %s", i_sim + 1, n_images)
        else:
            logger.debug("Simulating image %s / %s", i_sim + 1, n_images)

        # Prepare params
        this_n_calib = _pick_param(n_calib, i_sim, n_images)
        this_beta = _pick_param(beta, i_sim, n_images)
        params = np.asarray([this_n_calib, this_beta]).reshape((1, 2))
        params_eval = np.vstack((params, params_ref)) if mine_gold else None

        logger.debug("Numerator hypothesis: n_calib = %s, beta = %s", this_n_calib, this_beta)

        if inverse:
            # Choose one theta from prior that we use for sampling here
            i_sample = np.random.randint(n_thetas_marginal)
            this_n_calib, this_beta = params_ref[i_sample]

        logger.debug("Running simulation at n_calib = %s, beta = %s", this_n_calib, this_beta)

        # Simulate
        sim = LensingObservationWithSubhalos(
            sim_mvgauss_mean=sim_mvgauss_mean,
            sim_mvgauss_cov=sim_mvgauss_cov,
            n_calib=this_n_calib,
            beta=this_beta,
            spherical_host=True,
            fix_source=True,
            params_eval=params_eval,
            calculate_joint_score=mine_gold
        )

        sub_latents = np.vstack((sim.m_subs, sim.theta_xs, sim.theta_ys)).T
        host_latents = np.asarray((sim.z_s, sim.z_l, sim.sigma_v))

        all_params.append(params)
        all_images.append(sim.image_poiss_psf)
        all_sub_latents.append(sub_latents)
        all_host_latents.append(host_latents)

        if mine_gold:
            log_r_xz, uncertainty = _extract_log_r(sim, n_thetas_marginal)
            logger.debug("log r(x,z) = %s +/- %s", log_r_xz, uncertainty)
            if uncertainty > 0.1:
                logger.warning("Large uncertainty: log r(x,z) = %s +/- %s", log_r_xz, uncertainty)
            all_t_xz.append(sim.joint_score)
            logger.debug("t(x,z) = %s", sim.joint_score)
            all_log_r_xz.append(log_r_xz)

    if mine_gold:
        return np.array(all_params).reshape((-1, 2)), np.array(all_images), np.array(all_t_xz), np.array(
            all_log_r_xz), all_sub_latents, np.array(all_host_latents)
    return np.array(all_params).reshape((-1, 2)), np.array(all_images), None, None, all_sub_latents, np.array(
        all_host_latents)


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
