import sys, os

sys.path.append("../")

import logging
import autograd.numpy as np
import autograd as ag

from simulation.units import (
    M_s,
    erg,
    Centimeter,
    Angstrom,
    Sec,
    radtoasc,
)  # Don't import * since that will overwrite np
from simulation.lensing_sim import LensingSim

logger = logging.getLogger(__name__)


class SubhaloSimulator:
    def __init__(
        self,
        resolution=52,
        coordinate_limit=2.0,
        mass_base_unit=1.e7*M_s,
        m_sub_min=1.,
        host_profile="sis",
        host_theta_x=0.01,
        host_theta_y=-0.01,
        host_theta_E=1.0,
        exposure=(1 / 1.8e-19) * erg ** -1 * Centimeter ** 2 * Angstrom * 1000 * Sec,
        A_iso=2e-7 * erg / Centimeter ** 2 / Sec / Angstrom / (radtoasc) ** 2,
        zs=1.0,
        zl=0.1,
        src_profile="sersic",
        src_I_gal=1e-17 * erg / Centimeter ** 2 / Sec / Angstrom,
        src_theta_e_gal=0.5,
        src_n=4,
    ):
        """ We will effectively set m_sub_min to one, so that all masses and alpha will be dimensionless"""

        self.mass_base_unit = mass_base_unit
        self.resolution = resolution
        self.coordinate_limit = coordinate_limit
        self.m_sub_min = m_sub_min

        # Host galaxy
        self.hst_param_dict = {
            "profile": host_profile,
            "theta_x": host_theta_x,
            "theta_y": host_theta_y,
            "theta_E": host_theta_E,
        }

        # Observational parameters
        self.observation_dict = {
            "nx": resolution,
            "ny": resolution,
            "xlims": (-coordinate_limit, coordinate_limit),
            "ylims": (-coordinate_limit, coordinate_limit),
            "exposure": exposure,
            "A_iso": A_iso,
        }

        # Global parameters?!
        self.global_dict = {"z_s": zs, "z_l": zl}

        # Source parameters
        self.src_param_dict = {
            "profile": src_profile,
            "I_gal": src_I_gal,
            "theta_e_gal": src_theta_e_gal,
            "n_srsc": src_n,
        }

        # Autograd
        self.d_simulate = ag.grad_and_aux(self.simulate)

    def simulate(self, params, params_eval):
        """
        Generates one observed lensed image for given parameters of the subhalo mass distribution
        dn/dm = alpha (m/M_s)^beta with m > m_min and parameters alpha > 0, beta < -1.

        Subhalo coordinates (x,y) are sampled uniformly.
        """

        # Prepare parameters and joint likelihood
        alpha = params[0]
        beta = params[1]
        alphas_eval = [alpha] + [param[0] for param in params_eval]
        betas_eval = [beta] + [param[1] for param in params_eval]
        log_p_xz_eval = [0.0 for _ in alphas_eval]

        # Number of subhalos
        n_sub = self._draw_n_sub(alpha, beta)

        # Evaluate likelihoods of numbers of subhalos
        for i_eval, (alpha_eval, beta_eval) in enumerate(zip(alphas_eval, betas_eval)):
            log_p_xz_eval[i_eval] += self._calculate_log_p_n_sub(n_sub, alpha_eval, beta_eval)
        logger.debug("Log p: %s", log_p_xz_eval)

        # Draw subhalo masses
        m_sub = self._draw_m_sub(n_sub)
        logger.debug("Subhalo masses: %s", m_sub)

        # Evaluate likelihoods of subhalo masses
        for i_eval, (alpha_eval, beta_eval) in enumerate(zip(alphas_eval, betas_eval)):
            for i_sub in range(n_sub):
                log_p_xz_eval[i_eval] += self._calculate_log_p_m_sub(m_sub[i_sub], alpha_eval, beta_eval)
        logger.debug("Log p: %s", log_p_xz_eval)

        m_sub = self._detach(m_sub)

        # Subhalo coordinates
        x_sub, y_sub = self._draw_sub_coordinates(n_sub)
        logger.debug("Subhalo x: %s", x_sub)
        logger.debug("Subhalo y: %s", y_sub)

        # Lensing simulation
        image_mean = self._lensing(n_sub, m_sub, x_sub, y_sub)
        logger.debug("Image mean: %s", image_mean)

        # Observed lensed image
        image = self._observation(image_mean)
        logger.debug("Image: %s", image)

        # Returns
        latent_variables = (n_sub, m_sub, x_sub, y_sub, image_mean, image)
        return log_p_xz_eval[0], (image, log_p_xz_eval, latent_variables)

    def _calculate_n_sub_mean(self, alpha, beta):
        return -alpha / (beta + 1.0) * (self.m_sub_min) ** (1.0 + beta)

    def _draw_n_sub(self, alpha, beta):
        n_sub_mean = self._calculate_n_sub_mean(alpha, beta)
        n_sub_mean = self._detach(n_sub_mean)
        logger.debug("Poisson mean: %s", n_sub_mean)

        # Draw number of subhalos
        n_sub = np.random.poisson(n_sub_mean)
        logger.debug("Number of subhalos: %s", n_sub)

        return n_sub

    def _calculate_log_p_n_sub(self, n_sub, alpha, beta):
        n_sub_mean_eval = self._calculate_n_sub_mean(alpha, beta)
        logger.debug("Eval subhalo mean: %s", n_sub_mean_eval)
        log_p_poisson = n_sub * np.log(n_sub_mean_eval) - n_sub_mean_eval  # Can ignore constant term
        return log_p_poisson

    def _draw_m_sub(self, n_sub, alpha, beta):
        u = np.random.uniform(0, 1, size=n_sub)
        m_sub = self.m_sub_min * (1 - u) ** (1.0 / (beta + 1.0))
        return m_sub

    def _calculate_log_p_m_sub(self, m, alpha, beta):
        log_p = np.log(-beta - 1.0) + beta * np.log(m / self.m_sub_min)
        return log_p

    def _draw_sub_coordinates(self, n_sub):
        x_sub = np.random.uniform(
            low=-self.coordinate_limit, high=self.coordinate_limit, size=n_sub
        )
        y_sub = np.random.uniform(
            low=-self.coordinate_limit, high=self.coordinate_limit, size=n_sub
        )
        return x_sub, y_sub

    def _lensing(self, n_sub, m_sub, x_sub, y_sub):
        lens_list = [self.hst_param_dict]
        for i_sub in range(n_sub):
            sub_param_dict = {
                "profile": "nfw",
                "theta_x": x_sub[i_sub],
                "theta_y": y_sub[i_sub],
                "M200": m_sub[i_sub] * self.mass_base_unit,
            }
            lens_list.append(sub_param_dict)
        lsi = LensingSim(
            lens_list, [self.src_param_dict], self.global_dict, self.observation_dict
        )
        image_mean = lsi.lensed_image()
        return image_mean

    def _observation(self, image_mean):
        return np.random.poisson(image_mean)

    @staticmethod
    def _detach(obj):
        try:
            obj = obj._value
        except AttributeError:
            pass
        return obj

    def rvs(self, alpha, beta, n_images):
        all_images = []
        all_latents = []

        n_verbose = max(1, n_images // 20)

        for i_sim in range(n_images):
            if (i_sim + 1) % n_verbose == 0:
                logger.info("Simulating image %s / %s", i_sim + 1, n_images)

            try:
                assert len(alpha) == n_images
                this_alpha = alpha[i_sim]
            except TypeError:
                this_alpha = alpha
            try:
                assert len(beta) == n_images
                this_beta = beta[i_sim]
            except TypeError:
                this_beta = beta
            params = np.array([this_alpha, this_beta])

            logger.debug(
                "Simulating image %s/%s with alpha = %s, beta = %s",
                i_sim + 1,
                n_images,
                this_alpha,
                this_beta,
            )

            _, (image, _, latents) = self.simulate(params, [])

            n_subhalos = latents[0]
            logger.debug("Image generated with %s subhalos", n_subhalos)

            all_images.append(image)
            all_latents.append(latents)

        all_images = np.array(all_images)

        return all_images, all_latents

    def rvs_score_ratio(self, alpha, beta, alpha_ref, beta_ref, n_images):
        all_images = []
        all_t_xz = []
        all_log_r_xz = []
        all_latents = []

        n_verbose = max(1, n_images // 20)

        for i_sim in range(n_images):

            if (i_sim + 1) % n_verbose == 0:
                logger.info("Simulating image %s / %s", i_sim + 1, n_images)

            # Prepare parameters
            try:
                assert len(alpha) == n_images
                this_alpha = alpha[i_sim]
            except TypeError:
                this_alpha = alpha
            try:
                assert len(beta) == n_images
                this_beta = beta[i_sim]
            except TypeError:
                this_beta = beta
            try:
                assert len(alpha_ref) == n_images
                this_alpha_ref = alpha_ref[i_sim]
            except TypeError:
                this_alpha_ref = alpha_ref
            try:
                assert len(beta_ref) == n_images
                this_beta_ref = beta_ref[i_sim]
            except TypeError:
                this_beta_ref = beta_ref
            params = np.array([this_alpha, this_beta])
            params_ref = np.array([this_alpha_ref, this_beta_ref])

            logger.debug(
                "Simulating image %s/%s with alpha = %s, beta = %s; also evaluating probability for alpha = %s, beta = %s",
                i_sim + 1,
                n_images,
                this_alpha,
                this_beta,
                this_alpha_ref,
                this_beta_ref,
            )

            t_xz, (image, log_p_xzs, latents) = self.d_simulate(params, [params_ref])
            log_r_xz = log_p_xzs[0] - log_p_xzs[1]

            try:
                t_xz = t_xz._value
            except AttributeError:
                pass
            try:
                log_r_xz = log_r_xz._value
            except AttributeError:
                pass

            n_subhalos = latents[0]
            logger.debug("Image generated with %s subhalos", n_subhalos)

            all_images.append(image)
            all_t_xz.append(t_xz)
            all_log_r_xz.append(log_r_xz)
            all_latents.append(latents)

        all_images = np.array(all_images)
        all_t_xz = np.array(all_t_xz)
        all_log_r_xz = np.array(all_log_r_xz)

        return all_images, all_t_xz, all_log_r_xz, all_latents

    def rvs_score_ratio_to_evidence(
        self,
        alpha,
        beta,
        alpha_mean,
        alpha_std,
        beta_mean,
        beta_std,
        n_images,
        n_theta_samples,
    ):
        all_images = []
        all_t_xz = []
        all_log_r_xz = []
        all_log_r_xz_uncertainties = []
        all_latents = []

        n_verbose = max(1, n_images // 20)

        for i_sim in range(n_images):

            if (i_sim + 1) % n_verbose == 0:
                logger.info("Simulating image %s / %s", i_sim + 1, n_images)

            # Prepare parameters
            try:
                assert len(alpha) == n_images
                this_alpha = alpha[i_sim]
            except TypeError:
                this_alpha = alpha
            try:
                assert len(beta) == n_images
                this_beta = beta[i_sim]
            except TypeError:
                this_beta = beta
            params = np.array([this_alpha, this_beta])

            # Draw samples from prior
            alphas = np.random.normal(
                loc=alpha_mean, scale=alpha_std, size=n_theta_samples
            )
            betas = np.random.normal(
                loc=beta_mean, scale=beta_std, size=n_theta_samples
            )

            params_prior = np.vstack((alphas, betas)).T

            # Run simulator
            logger.debug(
                "Simulating image %s/%s with alpha = %s, beta = %s; also evaluating probability for %s samples drawn "
                "from prior",
                i_sim + 1,
                n_images,
                this_alpha,
                this_beta,
                n_theta_samples,
            )

            t_xz, (image, log_p_xzs, latents) = self.d_simulate(params, params_prior)

            # Clean up
            try:
                t_xz = t_xz._value
            except AttributeError:
                pass
            for i, log_p_xz in enumerate(log_p_xzs):
                try:
                    log_p_xzs[i] = log_p_xz._value
                except AttributeError:
                    pass

            # Evaluate likelihood ratio wrt evidence
            inverse_r_xz = 0.0
            for i_theta in range(n_theta_samples):
                inverse_r_xz += np.exp(log_p_xzs[i_theta + 1] - log_p_xzs[0])
            inverse_r_xz /= float(n_theta_samples)
            log_r_xz = -np.log(inverse_r_xz)

            # Estimate uncertainty of log r from MC sampling
            inverse_r_xz_uncertainty = 0.0
            for i_theta in range(n_theta_samples):
                inverse_r_xz_uncertainty += (
                    np.exp(log_p_xzs[i_theta + 1] - log_p_xzs[0]) - inverse_r_xz
                ) ** 2.0
            inverse_r_xz_uncertainty /= float(n_theta_samples) * (float(n_theta_samples) - 1.)
            log_r_xz_uncertainty = inverse_r_xz_uncertainty / inverse_r_xz

            n_subhalos = latents[0]
            logger.debug("Image generated with %s subhalos", n_subhalos)

            all_images.append(image)
            all_t_xz.append(t_xz)
            all_log_r_xz.append(log_r_xz)
            all_log_r_xz_uncertainties.append(log_r_xz_uncertainty)
            all_latents.append(latents)

        all_images = np.array(all_images)
        all_t_xz = np.array(all_t_xz)
        all_log_r_xz = np.array(all_log_r_xz)
        all_log_r_xz_uncertainties = np.array(all_log_r_xz_uncertainties)

        return all_images, all_t_xz, all_log_r_xz, all_log_r_xz_uncertainties, all_latents
