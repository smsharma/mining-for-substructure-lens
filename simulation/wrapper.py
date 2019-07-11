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
    f_sub_alt=None,
    beta_alt=None,
    f_sub_ref=None,
    beta_ref=None,
    f_sub_prior=uniform(0.001, 0.199),
    beta_prior=uniform(-2.5, 1.0),
    n_images=None,
    n_thetas_marginal=1000,
    draw_host_mass=True,
    draw_host_redshift=True,
    draw_alignment=True,
    mine_gold=True,
    calculate_dx_dm=False,
    return_dx_dm=False,
    roi_size=2.,
):
    # Input
    if (f_sub is None or beta is None) and n_images is None:
        raise ValueError("Either f_sub and beta or n_images have to be different from None")
    if n_images is None:
        n_images = len(f_sub)
    n_verbose = max(1, n_images // 100)

    # Hypothesis for sampling
    beta, f_sub = _draw_params(beta, beta_prior, f_sub, f_sub_prior, n_images)

    # Alternate hypothesis (test hypothesis when swapping num - den)
    beta_alt, f_sub_alt = _draw_params(beta_alt, beta_prior, f_sub_alt, f_sub_prior, n_images)

    # Reference hypothesis
    beta_ref, f_sub_ref = _draw_params(beta_alt, beta_prior, f_sub_alt, f_sub_prior, n_thetas_marginal - 1)
    params_ref = np.vstack((f_sub_ref, beta_ref)).T

    # Output
    all_params, all_params_alt, all_images = [], [], []
    all_t_xz, all_t_xz_alt, all_log_r_xz, all_log_r_xz_alt = [], [], [], []
    all_sub_latents, all_global_latents = [], []
    all_dx_dm = []

    # Main loop
    for i_sim in range(n_images):
        if (i_sim + 1) % n_verbose == 0:
            logger.info("Simulating image %s / %s", i_sim + 1, n_images)
        else:
            logger.debug("Simulating image %s / %s", i_sim + 1, n_images)

        # Prepare params
        this_f_sub = _pick_param(f_sub, i_sim, n_images)
        this_beta = _pick_param(beta, i_sim, n_images)
        this_f_sub_alt = _pick_param(f_sub_alt, i_sim, n_images)
        this_beta_alt = _pick_param(beta_alt, i_sim, n_images)

        params = np.asarray([this_f_sub, this_beta]).reshape((1, 2))
        params_alt = np.asarray([this_f_sub_alt, this_beta_alt]).reshape((1, 2))
        params_eval = np.vstack((params, params_alt, params_ref)) if mine_gold else None

        logger.debug("Numerator hypothesis: f_sub = %s, beta = %s", this_f_sub, this_beta)

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
            draw_host_mass=draw_host_mass,
            draw_host_redshift=draw_host_redshift,
            draw_alignment=draw_alignment,
            calculate_msub_derivatives=calculate_dx_dm,
            roi_size=roi_size,
        )

        # Store information
        if calculate_dx_dm:
            sum_abs_dx_dm = np.sum(np.abs(sim.grad_msub_image).reshape(sim.grad_msub_image.shape[0], -1), axis=1)
            sub_latents = np.vstack((sim.m_subs, sim.theta_xs, sim.theta_ys, sum_abs_dx_dm)).T
            if return_dx_dm:
                all_dx_dm.append(sim.grad_msub_image)
        else:
            sub_latents = np.vstack((sim.m_subs, sim.theta_xs, sim.theta_ys)).T
        global_latents = [
                sim.M_200_hst,  # Host mass
                sim.D_l,  # Host distance
                sim.z_l,  # Host redshift
                sim.sigma_v,  # sigma_V
                sim.theta_x_0,  # Source offset x
                sim.theta_y_0,  # Source offset y
                sim.theta_E,  # Host Einstein radius
                sim.n_sub_roi,  # Number of subhalos
                sim.f_sub_realiz,  # Fraction of halo mass in subhalos
                sim.n_sub_in_ring,  # Number of subhalos with r < 90% of host Einstein radius
                sim.f_sub_in_ring,  # Fraction of halo mass in subhalos with r < 90% of host Einstein radius
                sim.n_sub_near_ring,  # Number of subhalos with r within 10% of host Einstein radius
                sim.f_sub_near_ring,  # Fraction of halo mass in subhalos with r within 10% of host Einstein radius
            ]
        global_latents = np.asarray(global_latents)

        all_params.append(params)
        all_params_alt.append(params_alt)
        all_images.append(sim.image_poiss_psf)
        all_sub_latents.append(sub_latents)
        all_global_latents.append(global_latents)

        if mine_gold:
            all_log_r_xz.append(_extract_log_r(sim, 0, n_thetas_marginal))
            all_log_r_xz_alt.append(_extract_log_r(sim, 1, n_thetas_marginal))
            all_t_xz.append(sim.joint_scores[0])
            all_t_xz_alt.append(sim.joint_scores[1])

    if calculate_dx_dm and return_dx_dm:
        return (
            np.array(all_params).reshape((-1, 2)),
            np.array(all_params_alt).reshape((-1, 2)),
            np.array(all_images),
            np.array(all_t_xz) if mine_gold else None,
            np.array(all_t_xz_alt) if mine_gold else None,
            np.array(all_log_r_xz) if mine_gold else None,
            np.array(all_log_r_xz_alt) if mine_gold else None,
            all_sub_latents,
            np.array(all_global_latents),
        )
    return (
        np.array(all_params).reshape((-1, 2)),
        np.array(all_params_alt).reshape((-1, 2)),
        np.array(all_images),
        np.array(all_t_xz) if mine_gold else None,
        np.array(all_t_xz_alt) if mine_gold else None,
        np.array(all_log_r_xz) if mine_gold else None,
        np.array(all_log_r_xz_alt) if mine_gold else None,
        all_sub_latents,
        np.array(all_global_latents),
    )


def _draw_params(beta, beta_prior, f_sub, f_sub_prior, n_images):
    if f_sub is None:
        f_sub = f_sub_prior.rvs(size=n_images)
    if beta is None:
        beta = beta_prior.rvs(size=n_images)
    f_sub = np.clip(f_sub, 1.0e-6, None)
    beta = np.clip(beta, None, -1.01)
    return beta, f_sub


def _pick_param(xs, i, n):
    try:
        assert len(xs) == n
        return xs[i]
    except TypeError:
        return xs


def _extract_log_r(sim, i, n_thetas_marginal):
    log_p_xz_from_marginal = np.delete(sim.joint_log_probs, i, axis=0)
    delta_log = np.asarray(log_p_xz_from_marginal - sim.joint_log_probs[i] - np.log(float(n_thetas_marginal)), dtype=np.float128)
    log_r_xz = -1.0 * scipy.special.logsumexp(delta_log)
    return log_r_xz
