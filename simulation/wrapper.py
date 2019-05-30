import numpy as np
import logging
from scipy.stats import norm, uniform
import scipy.special

from simulation.population_sim import LensingObservationWithSubhalos
from simulation.units import M_s

logger = logging.getLogger(__name__)


def augmented_data(
    f_sub=None,
    beta=None,
    f_sub_ref=None,
    beta_ref=None,
    f_sub_prior=uniform(0.001, 0.199),
    beta_prior=uniform(-2.5, 1.0),
    n_images=None,
    n_thetas_marginal=1000,
    inverse=False,
    mine_gold=True,
):
    # Input
    if (f_sub is None or beta is None) and n_images is None:
        raise ValueError("Either f_sub and beta or n_images have to be different from None")
    if n_images is None:
        n_images = len(f_sub)
    n_verbose = max(1, n_images // 100)

    # Test hypothesis
    if f_sub is None:
        f_sub = f_sub_prior.rvs(size=n_images)
    if beta is None:
        beta = beta_prior.rvs(size=n_images)
    f_sub = np.clip(f_sub, 1.0e-6, None)
    beta = np.clip(beta, None, -1.01)

    # Reference hypothesis
    if f_sub_ref is None:
        f_sub_ref = f_sub_prior.rvs(size=n_thetas_marginal)
    if beta_ref is None:
        beta_ref = beta_prior.rvs(size=n_thetas_marginal)
    f_sub_ref = np.clip(f_sub_ref, 1.0e-6, None)
    beta_ref = np.clip(beta_ref, None, -1.01)
    params_ref = np.vstack((f_sub_ref, beta_ref)).T

    # Output
    all_params, all_images, all_t_xz, all_log_r_xz, all_sub_latents, all_global_latents = [], [], [], [], [], []

    # Main loop
    for i_sim in range(n_images):
        if (i_sim + 1) % n_verbose == 0:
            logger.info("Simulating image %s / %s", i_sim + 1, n_images)
        else:
            logger.debug("Simulating image %s / %s", i_sim + 1, n_images)

        # Prepare params
        this_f_sub = _pick_param(f_sub, i_sim, n_images)
        this_beta = _pick_param(beta, i_sim, n_images)
        params = np.asarray([this_f_sub, this_beta]).reshape((1, 2))
        params_eval = np.vstack((params, params_ref)) if mine_gold else None

        logger.debug("Numerator hypothesis: f_sub = %s, beta = %s", this_f_sub, this_beta)

        if inverse:
            # Choose one theta from prior that we use for sampling here
            i_sample = np.random.randint(n_thetas_marginal)
            this_f_sub, this_beta = params_ref[i_sample]

        logger.debug("Running simulation at f_sub = %s, beta = %s", this_f_sub, this_beta)

        if mine_gold:
            logger.debug("Evaluating joint log likelihood at %s", params_eval)

        # Simulate
        sim = LensingObservationWithSubhalos(
            m_200_min_sub=1.0e7 * M_s,
            m_200_max_sub_div_M_hst=0.01,
            m_min_calib=1.0e7 * M_s,
            m_max_sub_div_M_hst_calib=0.01,
            f_sub=this_f_sub,
            beta=this_beta,
            params_eval=params_eval,
            calculate_joint_score=mine_gold,
        )

        # Store information
        sub_latents = np.vstack((sim.m_subs, sim.theta_xs, sim.theta_ys)).T
        global_latents = np.asarray(
            [
                sim.M_200_hst,  # Host mass
                sim.D_l,  # Host distance
                sim.z_l,  # Host redshift
                sim.sigma_v,
                sim.theta_x_0,  # Source offset x
                sim.theta_y_0,  # Source offset y
                sim.theta_E,  # Host Einstein radius
                sim.n_sub_roi,  # Number of subhalos
                sim.f_sub_realiz,  # Fraction of halo mass in subhalos
            ]
        )

        all_params.append(params)
        all_images.append(sim.image_poiss_psf)
        all_sub_latents.append(sub_latents)
        all_global_latents.append(global_latents)

        if mine_gold:
            log_r_xz, uncertainty = _extract_log_r(sim, n_thetas_marginal)
            logger.debug("log r(x,z) = %s +/- %s", log_r_xz, uncertainty)
            if uncertainty > 0.1:
                logger.debug("Large uncertainty: log r(x,z) = %s +/- %s", log_r_xz, uncertainty)
            all_t_xz.append(sim.joint_score)
            logger.debug("t(x,z) = %s", sim.joint_score)
            all_log_r_xz.append(log_r_xz)

    return (
        np.array(all_params).reshape((-1, 2)),
        np.array(all_images),
        np.array(all_t_xz) if mine_gold else None,
        np.array(all_log_r_xz) if mine_gold else None,
        all_sub_latents,
        np.array(all_global_latents),
    )


def _pick_param(xs, i, n):
    try:
        assert len(xs) == n
        return xs[i]
    except TypeError:
        return xs


def _extract_log_r(sim, n_thetas_marginal):
    # Just a reference point?
    if n_thetas_marginal == 1:
        log_r_xz = sim.joint_log_probs[0] - sim.joint_log_probs[1]
        return log_r_xz, 0.0

    # Evaluate likelihood ratio wrt evidence
    delta_log = np.asarray(sim.joint_log_probs[1:] - sim.joint_log_probs[0] - np.log(float(n_thetas_marginal)), dtype=np.float128)
    log_r_xz = -1.0 * scipy.special.logsumexp(delta_log)

    if not np.isfinite(log_r_xz):
        logger.warning("Infinite log r for delta_log = %s", delta_log)

    # Estimate uncertainty of log r from MC sampling
    inverse_r_xz = np.exp(-log_r_xz)
    inverse_r_xz_uncertainty = 0.0
    for i_theta in range(n_thetas_marginal):
        log_r_contribution = sim.joint_log_probs[i_theta + 1] - sim.joint_log_probs[0]
        inverse_r_contribution = np.exp(log_r_contribution)
        inverse_r_xz_uncertainty += (inverse_r_contribution - inverse_r_xz) ** 2.0
    inverse_r_xz_uncertainty /= float(n_thetas_marginal) * (float(n_thetas_marginal) - 1.0)
    log_r_xz_uncertainty = inverse_r_xz_uncertainty / inverse_r_xz

    return log_r_xz, log_r_xz_uncertainty
