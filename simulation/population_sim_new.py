import math
from simulation.units import *
from simulation.profiles import MassProfileNFW
from simulation.lensing_sim import LensingSim
from astropy.cosmology import Planck15
from astropy.convolution import convolve, Gaussian2DKernel


class LensingObservationWithSubhalos:
    def __init__(self, sim_mvgauss_mean, sim_mvgauss_cov,
                 mag_zero=25.5, mag_iso=22.5, exposure=1610, fwhm_psf=0.18,
                 pixel_size=0.1, n_xy=64,
                 fix_source=True,
                 spherical_host=True,
                 m_200_min_sub=1e7 * M_s, n_calib=150, beta=1.9,
                 params_eval=None, calculate_joint_score=False,
                 ):
        """
        Class to simulation an observation strong lensing image, with substructure sprinkled in.

        Parameters corresponding to sim_mvgauss_[mean/cov] are:
        log_z_l, z_s, log_theta_E, sigma_v, q, theta_x_0, theta_y_0, log_theta_s_e, mag_s, TS

        :param sim_mvgauss_mean: Mean of drawn parameters
        :param sim_mvgauss_cov: Covariance matrix of drawn parameters
        :param mag_zero: Zero-point magnitude of observation
        :param mag_iso: Magnitude of isotropic sky brightness
        :param exposure: Exposure time of observation, in seconds
        :param fwhm_psf: FWHM of Gaussian PSF, in arcsecs
        :param pixel_size: Pixel side size, in arcsecs
        :param n_xy: Number of pixels (along x and y) of observation
        :param fix_source: Whether to fix source parameters rather than drawing, for an easier problem
        :param spherical_host: Whether to restrict to spherical hosts (q = 1), for an easier problem
        :param m_200_min_sub: Lowest mass of subhalos to draw
        :param n_calib: Number of subhalos expected between 1e8 and 1e10*M_s for a MW-sized halo, for calibration
        :param beta: Slope in the subhalo mass fn
        :param params_eval: Parameters (n_calib, beta) for which p(x,z|params) will be calculated
        :param calculate_joint_score: Whether grad_params log p(x,z|params) will be calculated
        """

        self.coordinate_limit = pixel_size * n_xy / 2.

        # Draw parameters from multivariate Gaussian
        log_z_l, z_s, log_theta_E, sigma_v, q, theta_x_0, theta_y_0, theta_s_e, mag_s, TS = \
            np.random.multivariate_normal(sim_mvgauss_mean, sim_mvgauss_cov)

        z_l = 10**log_z_l
        theta_E = 10**log_theta_E

        # If fixing the source, these are fixed to reasonable mean-ish quantities
        # and a higher-than-average brightness, for an easier problem
        if fix_source:
            theta_s_e = 0.2
            z_s = 2.
            mag_s = 23.
        else:
            theta_s_e = 10**theta_s_e
            z_s = 10**z_s

        # If only considering spherical host, set q = 1 for a simpler problem
        if spherical_host:
            q = 1
        else:
            # Clip host ellipticity to be between 0.2 and 1.0; outside values not physical
            if q > 1:
                q = 1
            if q < 0.2:
                q = 0.2

        # Get properties for NFW host
        M_200_hst = self.M_200_sigma_v(sigma_v * Kmps)
        c_200_hst = MassProfileNFW.c_200_SCP(M_200_hst)
        r_s_hst, rho_s_hst = MassProfileNFW.get_r_s_rho_s_NFW(M_200_hst, c_200_hst)

        D_l = Planck15.angular_diameter_distance(z=z_l).value * Mpc

        # Generate a subhalo population...
        ps = SubhaloPopulation(n_calib=n_calib, beta=beta, M_hst=M_200_hst, c_hst=c_200_hst,
                               m_min=m_200_min_sub, theta_s=r_s_hst / D_l, theta_roi=3 * theta_E,
                               params_eval=params_eval, calculate_joint_score=calculate_joint_score)

        # ... and grab its properties
        m_sample = ps.m_sample
        theta_x_sample = ps.theta_x_sample
        theta_y_sample = ps.theta_y_sample

        # Convert magnitude for source and isotropic component to expected counts
        S_tot = self._mag_to_flux(mag_s, mag_zero)
        f_iso = self._mag_to_flux(mag_iso, mag_zero)

        # Set host properties. Host assumed to be at the center of the image.
        hst_param_dict = {"profile": "SIE",
                          "theta_x_0": 0.0, "theta_y_0": 0.0,
                          "theta_E": theta_E,"q": q}

        lens_list = [hst_param_dict]

        # Set subhalo properties
        for m, theta_x, theta_y in zip(m_sample, theta_x_sample, theta_y_sample):
            c = MassProfileNFW.c_200_SCP(m)
            r_s, rho_s = MassProfileNFW.get_r_s_rho_s_NFW(m, c)
            sub_param_dict = {
                "profile": "NFW",
                "theta_x_0": theta_x, "theta_y_0": theta_y,
                "M_200": m,
                "r_s": r_s,
                "rho_s": rho_s
            }
            lens_list.append(sub_param_dict)

        # Set source properties
        src_param_dict = {
            "profile": "Sersic",
            "theta_x_0": theta_x_0, "theta_y_0": theta_y_0,
            "S_tot": S_tot,
            "theta_e": theta_s_e,
            "n_srsc": 1,
        }

        # Set observation and global properties
        observation_dict = {"n_x": n_xy, "n_y": n_xy,
                            "theta_x_lims": (-self.coordinate_limit, self.coordinate_limit),
                            "theta_y_lims": (-self.coordinate_limit, self.coordinate_limit),
                            "exposure": exposure,
                            "f_iso": f_iso,
        }

        global_dict = {"z_s": z_s, "z_l": z_l}

        # Inititalize lensing class and produce lensed image
        lsi = LensingSim(lens_list,
                         [src_param_dict],
                         global_dict,
                         observation_dict)

        self.image = lsi.lensed_image()
        self.image_poiss = np.random.poisson(self.image)  # Poisson fluctuate
        self.image_poiss_psf = self._convolve_psf(self.image_poiss, fwhm_psf, pixel_size)  # Convolve with PSF

        # Augmented data
        self.joint_log_probs = ps.joint_log_probs
        self.joint_scores = ps.joint_scores

    def _convolve_psf(self, image, fwhm_psf=0.18, pixel_size=0.1):
        """
        Convolve input map of pixel_size with Gaussian PSF of with FWHM fwhm_psf
        """
        sigma_psf = fwhm_psf / 2 ** 1.5 * np.sqrt(np.log(2))
        kernel = Gaussian2DKernel(x_stddev=1 * sigma_psf / pixel_size)

        return convolve(image, kernel)

    def _mag_to_flux(self, mag, mag_zp):
        """
        Returns total flux of the integrated profile, in ADU relative to mag_zp
        """
        return 10 ** (-0.4 * (mag - mag_zp))

    def M_200_sigma_v(self, sigma_v):
        """
        Relate central velocity dispersion to halo virial mass
        From https://arxiv.org/pdf/1804.04492.pdf
        """
        a = 0.09
        b = 3.48
        sigma_log10_M_200 = 0.13  # Lognormal scatter
        log10_M_200 = np.random.normal(a + b * np.log10(sigma_v / (100 * Kmps)), sigma_log10_M_200)
        return (10 ** log10_M_200) * 1e12 * M_s


class SubhaloPopulation:
    def __init__(self, n_calib=150, M_min_calib=1e8*M_s, M_max_calib=1e10*M_s,
                 beta=-1.9, m_min=1e9*M_s, theta_roi=2.5,
                 M_hst=1e14*M_s, theta_s=1e-4, c_hst=6.,
                 params_eval=None, calculate_joint_score=False):
        """
        Calibrate number of subhalos and generate a mass sample within lensing ROI

        SHMF is assumed to have the form
        dn/dm =  M_hst/M_0 * alpha * (m / m_0) ^ beta
        with M_0 = M_MW and m_0 = 1e9 * M_s. Note that this is slightly different
        from what's in the draft at the moment.

        :param n_calib: Number of subhalos expected between M_min_calib and M_max_calib for MW-sized halo
        :param beta: Slope of subhalo mass function
        :param m_min: Minimum mass of subhalos
        :param theta_roi: Radius of lensing ROI, in arcsecs
        :param M_hst: Host halo mass
        :param theta_s: Angular scale radius of host halo, in rad
        :param c_hst: Concentration parameter of host halo
        :param params_eval: Parameters (n_calib, beta) for which p(x,z|params) will be calculated
        :param calculate_joint_score: Whether grad_params log p(x,z|params) will be calculated
        """

        # Store settings
        self.n_calib = n_calib
        self.M_min_calib = M_min_calib
        self.M_max_calib = M_max_calib
        self.beta = beta
        self.m_min = m_min
        self.theta_roi = theta_roi
        self.M_hst = M_hst
        self.theta_s = theta_s
        self.c_hst = c_hst

        # Alpha corresponding to calibration configuration
        alpha = self._alpha_calib(M_min_calib, M_max_calib, n_calib, M_MW, beta)

        # Total number of subhalos within virial radius of host halo
        n_sub_tot = self._n_sub(m_min, 0.01 * M_hst, M_hst, alpha, beta)

        # Fraction and number of subhalos within lensing region of interest specified by theta_roi
        self.f_sub = MassProfileNFW.M_cyl_div_M0(theta_roi * asctorad / theta_s) \
            / MassProfileNFW.M_cyl_div_M0(c_hst * theta_s / theta_s)
        self.n_sub_roi = np.random.poisson(self.f_sub * n_sub_tot)

        # Sample of subhalo masses drawn from subhalo mass function
        self.m_sample = self._draw_m_sub(self.n_sub_roi, m_min, beta)

        # Sample subhalo positions uniformly within ROI
        self.theta_x_sample, self.theta_y_sample = self._draw_sub_coordinates(self.n_sub_roi, r_max=theta_roi)

        # Calculate augmented data
        self.joint_log_probs = self._calculate_joint_log_probs(params_eval)
        if calculate_joint_score:
            self.joint_scores = self._calculate_joint_score([n_calib, beta])
        else:
            self.joint_scores = None

    def _alpha_calib(self, m_min_calib, m_max_calib, n_calib, M_calib, beta, M_0=M_MW, m_0=1e9*M_s):
        """
        Get normalization alpha corresponding calibration configuration
        """
        return -M_0 * (m_max_calib * m_min_calib / m_0) ** -beta * n_calib * (-1 + -beta) / \
               (M_calib * (-m_max_calib ** -beta * m_min_calib + m_max_calib * m_min_calib ** -beta))

    def _n_sub(self, m_min, m_max, M, alpha, beta, M_0=M_MW, m_0=1e9*M_s):
        """
        Get (expected) number of subhalos between m_min, m_max
        """
        return alpha * M * (m_max * m_min / m_0) ** beta * \
               (m_max ** -beta * m_min - m_max * m_min ** -beta) / (M_0 * (-1 + -beta))

    def _draw_m_sub(self, n_sub, m_sub_min, beta):
        """
        Draw subhalos from SHMF
        """
        u = np.random.uniform(0, 1, size=n_sub)
        m_sub = m_sub_min * (1 - u) ** (1.0 / (beta + 1.0))
        return m_sub

    def _draw_sub_coordinates(self, n_sub, r_min=0, r_max=2.5):
        """
        Draw subhalo n_sub coordinates uniformly within a ring r_min < r < r_max
        """
        phi_sub = np.random.uniform(low=0.0, high=2.0 * np.pi, size=n_sub)
        r_sub = np.random.uniform(low=r_min, high=r_max, size=n_sub)
        x_sub = r_sub * np.cos(phi_sub)
        y_sub = r_sub * np.sin(phi_sub)
        return x_sub, y_sub

    def _calculate_joint_log_probs(self, params_eval):
        if params_eval is None:
            params_eval = []
            
        log_probs = [0.0 for _ in params_eval]

        for i_eval, (n_calib, beta) in enumerate(params_eval):
            # Poisson term
            log_probs[i_eval] += self._log_p_n_sub(self.n_sub_roi, n_calib, beta)

            # Power law for subhalo masses
            for m_sub in self.m_sample:
                log_probs[i_eval] += self._log_p_m_sub(m_sub, beta)

        return log_probs

    def _calculate_joint_score(self, params, eps=1.e-6):
        eps_vec0 = np.asarray(params).flatten() + np.array([eps, 0.0]).reshape(1, 2)
        eps_vec1 = np.asarray(params).flatten() + np.array([0.0, eps]).reshape(1, 2)
        params = params.reshape(1, 2)
        all_params = np.vstack([params, eps_vec0, eps_vec1])

        log_probs = self._calculate_joint_log_probs(all_params)

        t0 = (log_probs[1] - log_probs[0]) / eps
        t1 = (log_probs[2] - log_probs[0]) / eps

        return np.array([t0, t1])

    def _log_p_n_sub(self, n_sub, n_calib, beta, include_constant=False):
        alpha = self._alpha_calib(self.M_min_calib, self.M_max_calib, n_calib, self.M_MW, beta)
        expected_n_sub_mean = self._n_sub(self.m_min, 0.01 * self.M_hst, M_hst, alpha, beta)

        log_p_poisson = (
            n_sub * np.log(expected_n_sub_mean) - expected_n_sub_mean
        )
        if include_constant:
            log_p_poisson = log_p_poisson - np.log(math.factorial(n_sub))
        return log_p_poisson

    def _log_p_m_sub(self, m, beta):
        log_p = (
            np.log(-beta - 1.0)
            - np.log(self.m_sub_min)
            + beta * np.log(m / self.m_sub_min)
        )
        return log_p
