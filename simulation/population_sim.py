import math
import logging
from simulation.units import *
from simulation.profiles import MassProfileNFW, MassProfileSIE
from simulation.lensing_sim import LensingSim
from astropy.cosmology import Planck15
from astropy.convolution import convolve, Gaussian2DKernel

logger = logging.getLogger(__name__)


class LensingObservationWithSubhalos:
    def __init__(
        self,
        mag_zero=25.5,
        mag_iso=22.5,
        exposure=1610.0,
        fwhm_psf=0.18,
        pixel_size=0.1,
        n_xy=64,
        f_sub=0.05,
        beta=-1.9,
        m_min_calib=1e6 * M_s,
        m_max_sub_div_M_hst_calib=0.01,
        m_200_min_sub=1e7 * M_s,
        m_200_max_sub_div_M_hst=0.01,
        M_200_sigma_v_scatter=False,
        params_eval=None,
        calculate_joint_score=False,
        draw_host_mass=True,
        draw_host_redshift=True,
        draw_alignment=True,
    ):
        """
        Class to simulation an observation strong lensing image, with substructure sprinkled in.

        :param mag_zero: Zero-point magnitude of observation
        :param mag_iso: Magnitude of isotropic sky brightness
        :param exposure: Exposure time of observation, in seconds (including gain)
        :param fwhm_psf: FWHM of Gaussian PSF, in arcsecs
        :param pixel_size: Pixel side size, in arcsecs
        :param n_xy: Number of pixels (along x and y) of observation

        :param f_sub: Fraction of total contained mass in substructure
        :param beta: Slope in the subhaalo mass function
        :param m_min_calib: Minimum mass above which subhalo mass fraction is `f_sub`
        :param m_max_sub_div_M_hst_calib: Maximum mass below which subhalo mass fraction is `f_sub`,
            in units of the host halo mass
        :param m_200_min_sub: Lowest mass of subhalos to draw
        :param m_200_max_sub_div_M_hst: Maximum mass of subhalos to draw, in units of host halo mass

        :param M_200_sigma_v_scatter: Whether to apply lognormal scatter in sigma_v to M_200_host mapping

        :param params_eval: Parameters (f_sub, beta) for which p(x,z|params) will be calculated
        :param calculate_joint_score: Whether grad_params log p(x,z|params) will be calculated
        """

        # Store input
        self.mag_zero = mag_zero
        self.mag_iso = mag_iso
        self.exposure = exposure
        self.fwhm_psf = fwhm_psf
        self.pixel_size = pixel_size
        self.n_xy = n_xy
        self.f_sub = f_sub
        self.beta = beta
        self.m_min_calib = m_min_calib
        self.m_max_sub_div_M_hst_calib = m_max_sub_div_M_hst_calib
        self.m_200_min_sub = m_200_min_sub
        self.m_200_max_sub_div_M_hst = m_200_max_sub_div_M_hst
        self.M_200_sigma_v_scatter = M_200_sigma_v_scatter
        self.params_eval = params_eval
        self.calculate_joint_score = calculate_joint_score
        self.draw_host_mass = draw_host_mass
        self.draw_host_redshift = draw_host_redshift
        self.draw_alignment = draw_alignment

        self.coordinate_limit = pixel_size * n_xy / 2.0

        # Draw lens properties consistent with Collett et al [1507.02657]

        # Clip lens redshift `z_l` to be less than 1; high-redshift lenses no good for our purposes!

        if self.draw_host_redshift:
            self.z_l = 2.0
            while self.z_l > 1.0:
                self.z_l = 10 ** np.random.normal(-0.25, 0.25)
        else:
            self.z_l = 10.0 ** -0.25

        if self.draw_host_mass:
            self.sigma_v = np.random.normal(225, 50)
        else:
            self.sigma_v = 225.0

        if self.draw_alignment:
            self.theta_x_0 = np.random.normal(0, 0.2)
            self.theta_y_0 = np.random.normal(0, 0.2)
        else:
            self.theta_x_0 = 0.0
            self.theta_y_0 = 0.0

        q = 1  # For now, hard-code host to be spherical

        # Fix the source properties to reasonable mean-ish values
        self.theta_s_e = 0.2
        self.z_s = 1.5
        self.mag_s = 23.0

        # Get relevant distances
        self.D_l = Planck15.angular_diameter_distance(z=self.z_l).value * Mpc
        D_s = Planck15.angular_diameter_distance(z=self.z_s).value * Mpc
        D_ls = Planck15.angular_diameter_distance_z1z2(z1=self.z_l, z2=self.z_s).value * Mpc

        # Get properties for NFW host DM halo
        if draw_host_mass:
            self.M_200_hst = self.M_200_sigma_v(self.sigma_v * Kmps, scatter=M_200_sigma_v_scatter)
        else:
            self.M_200_hst = self.M_200_sigma_v(self.sigma_v * Kmps, scatter=0.0)

        c_200_hst = MassProfileNFW.c_200_SCP(self.M_200_hst)
        r_s_hst, rho_s_hst = MassProfileNFW.get_r_s_rho_s_NFW(self.M_200_hst, c_200_hst)

        # Get properties for SIE host
        self.theta_E = MassProfileSIE.theta_E(self.sigma_v * Kmps, D_ls, D_s)

        # Don't consider configuration with subhalo fraction > 1!
        self.f_sub_realiz = 2.0
        while self.f_sub_realiz > 1.0:

            # Generate a subhalo population...
            ps = SubhaloPopulation(
                f_sub=f_sub,
                beta=beta,
                M_hst=self.M_200_hst,
                c_hst=c_200_hst,
                m_min=m_200_min_sub,
                m_max=m_200_max_sub_div_M_hst * self.M_200_hst,
                m_min_calib=m_min_calib,
                m_max_calib=m_max_sub_div_M_hst_calib * self.M_200_hst,
                theta_s=r_s_hst / self.D_l,
                theta_roi=2.0 * self.theta_E,
                theta_E=self.theta_E,
                params_eval=params_eval,
                calculate_joint_score=calculate_joint_score,
            )

            # ... and grab its properties
            self.m_subs = ps.m_sample
            self.n_sub_roi = ps.n_sub_roi
            self.theta_xs = ps.theta_x_sample
            self.theta_ys = ps.theta_y_sample
            self.f_sub_realiz = ps.f_sub_realiz
            self.n_sub_in_ring = ps.n_sub_in_ring
            self.f_sub_in_ring = ps.f_sub_in_ring
            self.n_sub_near_ring = ps.n_sub_near_ring
            self.f_sub_near_ring = ps.f_sub_near_ring

        # Convert magnitude for source and isotropic component to expected counts
        S_tot = self._mag_to_flux(self.mag_s, self.mag_zero)
        f_iso = self._mag_to_flux(self.mag_iso, self.mag_zero)

        # Set host properties. Host assumed to be at the center of the image.
        hst_param_dict = {"profile": "SIE", "theta_x_0": 0.0, "theta_y_0": 0.0, "theta_E": self.theta_E, "q": q}

        lens_list = [hst_param_dict]

        # Set subhalo properties
        for m, theta_x, theta_y in zip(self.m_subs, self.theta_xs, self.theta_ys):
            c = MassProfileNFW.c_200_SCP(m)
            r_s, rho_s = MassProfileNFW.get_r_s_rho_s_NFW(m, c)
            sub_param_dict = {"profile": "NFW", "theta_x_0": theta_x, "theta_y_0": theta_y, "M_200": m, "r_s": r_s, "rho_s": rho_s}
            lens_list.append(sub_param_dict)

        # Set source properties
        src_param_dict = {"profile": "Sersic", "theta_x_0": self.theta_x_0, "theta_y_0": self.theta_y_0, "S_tot": S_tot, "theta_e": self.theta_s_e, "n_srsc": 1}

        # Set observation and global properties
        observation_dict = {
            "n_x": n_xy,
            "n_y": n_xy,
            "theta_x_lims": (-self.coordinate_limit, self.coordinate_limit),
            "theta_y_lims": (-self.coordinate_limit, self.coordinate_limit),
            "exposure": exposure,
            "f_iso": f_iso,
        }

        global_dict = {"z_s": self.z_s, "z_l": self.z_l}

        # Inititalize lensing class and produce lensed image
        lsi = LensingSim(lens_list, [src_param_dict], global_dict, observation_dict)

        self.image = lsi.lensed_image()
        self.image_poiss = np.random.poisson(self.image)  # Poisson fluctuate
        self.image_poiss_psf = self._convolve_psf(self.image_poiss, fwhm_psf, pixel_size)  # Convolve with PSF

        # Augmented data
        self.joint_log_probs = ps.joint_log_probs
        self.joint_score = ps.joint_score

    def _convolve_psf(self, image, fwhm_psf=0.18, pixel_size=0.1):
        """
        Convolve input map of pixel_size with Gaussian PSF of with FWHM `fwhm_psf`
        """
        sigma_psf = fwhm_psf / 2 ** 1.5 * np.sqrt(np.log(2))  # Convert FWHM to standard deviation
        kernel = Gaussian2DKernel(x_stddev=1.0 * sigma_psf / pixel_size)

        return convolve(image, kernel)

    def _mag_to_flux(self, mag, mag_zp):
        """
        Returns total flux of the integrated profile corresponding to magnitude `mag`, in ADU relative to `mag_zp`
        """
        return 10 ** (-0.4 * (mag - mag_zp))

    @classmethod
    def M_200_sigma_v(cls, sigma_v, scatter=False):
        """
        Relate central velocity dispersion to halo virial mass
        From https://arxiv.org/pdf/1804.04492.pdf
        """
        a = 0.09
        b = 3.48
        if scatter:
            sigma_log10_M_200 = 0.13  # Lognormal scatter
            log10_M_200 = np.random.normal(a + b * np.log10(sigma_v / (100 * Kmps)), sigma_log10_M_200)
        else:
            log10_M_200 = a + b * np.log10(sigma_v / (100 * Kmps))
        return (10 ** log10_M_200) * 1e12 * M_s


class SubhaloPopulation:
    def __init__(
        self,
        f_sub=0.15,
        beta=-1.9,
        m_min=1e7 * M_s,
        m_max=1e11 * M_s,
        m_min_calib=1e7 * M_s,
        m_max_calib=1e11 * M_s,
        theta_roi=2.5,
        M_hst=1e14 * M_s,
        theta_s=1e-4,
        c_hst=6.0,
        theta_E=1.0,
        params_eval=None,
        calculate_joint_score=False,
    ):
        """
        Calibrate number of subhalos and generate a mass sample within lensing ROI

        SHMF is assumed to have the form
        dn/dm =  M_hst/M_0 * alpha * (m / m_0) ^ beta
        with M_0 = M_MW and m_0 = 1e9 * M_s. Note that this is slightly different
        from what's in the draft at the moment.

        :param f_sub: Fraction of mass contained in substructure
        :param beta: Slope of subhalo mass function
        :param m_min: Minimum mass of subhalos
        :param m_max: Maximum mass of subhalos
        :param theta_roi: Radius of lensing ROI, in arcsecs
        :param M_hst: Host halo mass
        :param theta_s: Angular scale radius of host halo, in rad
        :param c_hst: Concentration parameter of host halo
        :param params_eval: Parameters (f_sub, beta) for which p(x,z|params) will be calculated
        :param calculate_joint_score: Whether grad_params log p(x,z|params) will be calculated
        """

        # Store settings
        self.f_sub = f_sub
        self.beta = beta
        self.m_min = m_min
        self.m_max = m_max
        self.m_min_calib = m_min_calib
        self.m_max_calib = m_max_calib
        self.theta_roi = theta_roi
        self.M_hst = M_hst
        self.theta_s = theta_s
        self.theta_E = theta_E
        self.c_hst = c_hst

        # Alpha corresponding to calibration configuration
        self.alpha = self._alpha_f_sub(f_sub, beta, m_min_calib, m_max_calib)

        # Total expected number of subhalos within virial radius of host halo
        self.n_sub_tot = self._n_sub(m_min, m_max, M_hst, self.alpha, self.beta)

        # Fraction and number of subhalos within lensing region of interest specified by theta_roi
        self.f_sub_roi = max(MassProfileNFW.M_cyl_div_M0(self.theta_roi * asctorad / theta_s), 0.0)
        self.n_sub_roi = np.random.poisson(self.f_sub_roi * self.n_sub_tot)
        logger.debug("%s subhalos (%s expected)", self.n_sub_roi, self.f_sub_roi * self.n_sub_tot)

        # Sample of subhalo masses drawn from subhalo mass function
        self.m_sample = self._draw_m_sub(self.n_sub_roi, self.m_min, self.m_max, self.beta)

        # Fraction of halo mass in subhalos, for diagnostic purposes
        self.M_hst_roi = M_hst * MassProfileNFW.M_cyl_div_M0(self.theta_roi * asctorad / self.theta_s)
        self.f_sub_realiz = np.sum(self.m_sample) / self.M_hst_roi
        logger.debug("%s substructure fraction (%s expected)", self.f_sub_realiz, self.f_sub)

        # Sample subhalo positions uniformly within ROI
        self.theta_x_sample, self.theta_y_sample = self._draw_sub_coordinates(self.n_sub_roi, r_max=self.theta_roi)

        # For debugging: subhalos within Einstein ring and near it
        self.n_sub_in_ring, self.f_sub_in_ring = self._count_subhalos_in_radius_range(0.0, 0.9 * self.theta_E)
        self.n_sub_near_ring, self.f_sub_near_ring = self._count_subhalos_in_radius_range(0.9 * self.theta_E, 1.1 * self.theta_E)

        # Calculate augmented data
        self.joint_log_probs = self._calculate_joint_log_probs(params_eval)
        if calculate_joint_score:
            self.joint_score = self._calculate_joint_score([self.f_sub, self.beta])
        else:
            self.joint_score = None

    @staticmethod
    def _alpha_calib(m_min_calib, m_max_calib, n_calib, M_calib, beta, M_0=M_MW, m_0=1e9 * M_s):
        """
        Get normalization alpha corresponding calibration configuration specified by {n_calib, beta}
        """
        alpha = (n_calib * (-1 - beta) * M_0 / M_calib * m_0 ** beta) / (-m_max_calib ** (1.0 + beta) + m_min_calib ** (1.0 + beta))
        return alpha

    @staticmethod
    def _alpha_f_sub(f_sub, beta, m_min, m_max, M_0=M_MW, m_0=1e9 * M_s):
        """
        Get normalization alpha corresponding calibration configuration specified by {f_sub, beta}
        """
        alpha = f_sub * ((2 + beta) * M_0 * m_0 ** beta) / (m_max ** (beta + 2) - m_min ** (beta + 2))
        return alpha

    @staticmethod
    def _m_in_sub(M_hst, alpha, beta, m_min, m_max, M_0=M_MW, m_0=1e9 * M_s):
        return M_hst * alpha * (m_max ** (beta + 2) - m_min ** (beta + 2)) / ((2 + beta) * M_0 * m_0 ** beta)

    @staticmethod
    def _n_sub(m_min, m_max, M, alpha, beta, M_0=M_MW, m_0=1e9 * M_s):
        """
        Get (expected) number of subhalos between m_min, m_max
        """

        # n_sub = alpha * M * (m_max * m_min / m_0) ** beta * (m_max ** -beta * m_min - m_max * m_min ** -beta)
        # / (M_0 * (-1 + -beta))
        # ... overflows sometimes

        n_sub = alpha / (-beta - 1.0) * m_0 * M / M_0
        n_sub *= (m_min / m_0) ** (beta + 1.0) - (m_max / m_0) ** (beta + 1.0)
        return max(n_sub, 0.0)

    @staticmethod
    def _draw_m_sub(n_sub, m_sub_min, m_sub_max, beta):
        """
        Draw subhalo masses from SHMF with slope `beta` and min/max masses `m_sub_min` and `m_sub_max` . Stolen from:
        https://stackoverflow.com/questions/31114330/python-generating-random-numbers-from-a-power-law-distribution
        """
        u = np.random.uniform(0, 1, size=n_sub)
        m_low_u, m_high_u = m_sub_min ** (beta + 1), m_sub_max ** (beta + 1)
        return (m_low_u + (m_high_u - m_low_u) * u) ** (1.0 / (beta + 1.0))

    @staticmethod
    def _draw_sub_coordinates(n_sub, r_min=0.0, r_max=2.5):
        """
        Draw subhalo n_sub coordinates uniformly within a ring r_min < r < r_max
        """
        x_sub = []
        y_sub = []
        while len(x_sub) < n_sub:
            x_candidates = np.random.uniform(low=-r_max, high=r_max, size=n_sub - len(x_sub))
            y_candidates = np.random.uniform(low=-r_max, high=r_max, size=n_sub - len(x_sub))
            r2 = x_candidates ** 2 + y_candidates ** 2
            good = (r2 <= r_max ** 2) * (r2 >= r_min ** 2)
            x_sub += list(x_candidates[good])
            y_sub += list(y_candidates[good])

        return np.array(x_sub), np.array(y_sub)

    def _count_subhalos_in_radius_range(self, min_r, max_r):
        r = ((self.theta_x_sample ** 2 + self.theta_y_sample ** 2) ** 0.5).flatten()
        filter = (r >= min_r) * (r < max_r)
        n_sub = np.sum(filter.astype(np.int))
        m_sub = np.sum(self.m_sample[filter])
        f_sub = m_sub / self.M_hst_roi
        return n_sub, f_sub

    def _calculate_joint_log_probs(self, params_eval):
        """
        Calculates log p(self.n_sub_roi, self.m_sample | f_sub, beta)
        """
        if params_eval is None:
            params_eval = []

        log_probs = [0.0 for _ in params_eval]

        for i_eval, (f_sub, beta) in enumerate(params_eval):
            # Poisson term
            log_probs[i_eval] += self._log_p_n_sub(self.n_sub_roi, f_sub, beta)

            # Power law for subhalo masses
            for m_sub in self.m_sample:
                log_probs[i_eval] += self._log_p_m_sub(m_sub, beta)

        return np.array(log_probs)

    def _calculate_joint_score(self, params, eps0=1.0e-5, eps1=1.0e-3):
        """
        Calculates grad_(f_sub, beta) log p(self.n_sub_roi, self.m_sample | f_sub, beta)
        """
        eps_vec0 = np.asarray(params).flatten() + np.array([eps0, 0.0]).reshape(1, 2)
        eps_vec1 = np.asarray(params).flatten() + np.array([0.0, eps1]).reshape(1, 2)
        params = np.asarray(params).reshape(1, 2)
        all_params = np.vstack([params, eps_vec0, eps_vec1])
        log_probs = self._calculate_joint_log_probs(all_params)

        score0 = (log_probs[1] - log_probs[0]) / eps0
        score1 = (log_probs[2] - log_probs[0]) / eps1

        return np.array([score0, score1])

    def _log_p_n_sub(self, n_sub, f_sub, beta, include_constant=False, eps=1.0e-6):
        """
        Calculates log p(n_sub | f_sub, beta)
        """
        alpha = self._alpha_f_sub(f_sub, beta, self.m_min_calib, self.m_max_calib)
        expected_n_sub = self.f_sub_roi * self._n_sub(self.m_min, self.m_max, self.M_hst, alpha, beta)
        if expected_n_sub <= eps:
            logger.warning("Expected number of subs in RoI for f_sub = %s and beta = %s is %s, setting to %s", f_sub, beta, expected_n_sub, eps)
            expected_n_sub = eps

        log_p_poisson = n_sub * np.log(expected_n_sub) - expected_n_sub
        if include_constant:
            log_p_poisson = log_p_poisson - np.log(math.factorial(n_sub))
        return log_p_poisson

    def _log_p_m_sub(self, m, beta, m_0=1e9 * M_s):
        """
        Calculates log p(m_i | beta)
        """
        if m < self.m_min or m > self.m_max:
            logger.warning("Calculating probability for subhalo mass out of bounds -- this should not happen")
            return 0.0

        log_p = beta * np.log(m / m_0)
        log_p += np.log(-beta - 1.0)
        log_p -= np.log(m_0)
        log_p -= np.log((self.m_min / m_0) ** (beta + 1.0) - (self.m_max / m_0) ** (beta + 1.0))

        if not np.isfinite(log_p):
            logger.warning("Infinite log p(m_sub | beta):")
            logger.warning("  m       = %s", m)
            logger.warning("  beta    = %s", beta)
            logger.warning("  m_0     = %s", m_0)
            logger.warning("  m_min   = %s", self.m_min)
            logger.warning("  m_max   = %s", self.m_max)
            logger.warning("  term 1  = %s", beta * np.log(m / m_0))
            logger.warning("  term 2  = %s", np.log(-beta - 1.0))
            logger.warning("  term 3  = %s", -np.log(m_0))
            logger.warning("  term 4  = %s", -np.log((self.m_min / m_0) ** (beta + 1.0) - (self.m_max / m_0) ** (beta + 1.0)))
            logger.warning("  term 4a = %s", (self.m_min / m_0) ** (beta + 1.0))
            logger.warning("  term 4b = %s", -(self.m_max / m_0) ** (beta + 1.0))

        return log_p
