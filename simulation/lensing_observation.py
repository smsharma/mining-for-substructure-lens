from simulation.units import *
from simulation.profiles import MassProfileNFW
from simulation.lensing_sim import LensingSim
from astropy.cosmology import Planck15


class LensingObservation:
    def __init__(self, M_zero=25.5,
                 f_iso=24.5,
                 exposure=1610,
                 coordinate_limit=2.5
                 ):

        mean = np.array([-2.09697676e-01,  2.44516730e-01, -1.96187450e-01,  2.22271987e+02,
        6.88602039e-01,  6.32935143e-04,  3.33666896e-03, -9.58628116e-01,
        2.55134987e+01])

        cov = np.array([[ 5.80724334e-02,  2.92348948e-02, -2.30285858e-02,
         1.01103663e+00,  7.59421896e-04,  3.76096433e-04,
         1.39048957e-04, -1.61490408e-02,  1.00460878e-01],
       [ 2.92348948e-02,  4.10218447e-02,  1.24760286e-03,
         6.67292057e-02, -1.76230621e-04, -1.77442865e-04,
         2.16751619e-04, -5.56745669e-03,  6.92349750e-02],
       [-2.30285858e-02,  1.24760286e-03,  5.62146044e-02,
         8.65444476e+00,  5.20158253e-03, -7.77514797e-04,
         8.40852610e-04,  2.76005039e-02, -6.93884029e-02],
       [ 1.01103663e+00,  6.67292057e-02,  8.65444476e+00,
         2.40456731e+03,  1.53101987e+00, -1.48107863e-01,
         2.11319333e-01,  3.95506934e+00, -7.54238973e+00],
       [ 7.59421896e-04, -1.76230621e-04,  5.20158253e-03,
         1.53101987e+00,  2.55458444e-02, -5.41568216e-04,
         1.29547862e-04,  2.73287221e-03, -4.63278572e-03],
       [ 3.76096433e-04, -1.77442865e-04, -7.77514797e-04,
        -1.48107863e-01, -5.41568216e-04,  5.24698020e-02,
         5.24608013e-04,  1.28405863e-03, -8.36996897e-04],
       [ 1.39048957e-04,  2.16751619e-04,  8.40852610e-04,
         2.11319333e-01,  1.29547862e-04,  5.24608013e-04,
         9.29202582e-02, -4.92250099e-04, -1.25472433e-04],
       [-1.61490408e-02, -5.56745669e-03,  2.76005039e-02,
         3.95506934e+00,  2.73287221e-03,  1.28405863e-03,
        -4.92250099e-04,  1.53086291e-01, -3.65964687e-01],
       [ 1.00460878e-01,  6.92349750e-02, -6.93884029e-02,
        -7.54238973e+00, -4.63278572e-03, -8.36996897e-04,
        -1.25472433e-04, -3.65964687e-01,  1.35125930e+00]])

        z_l, z_s, theta_E, sigma_v, q, theta_x_0, theta_y_0, theta_s, mag = np.random.multivariate_normal(mean, cov)

        z_l = 10**z_l
        z_s = 10**z_s
        theta_E = 10**theta_E
        theta_s = 10**theta_s

        M_200_hst = self.M_200_sigma_v(sigma_v * Kmps)
        c_200_hst = MassProfileNFW.c_200_SCP(M_200_hst)
        m_200_min_sub = 1e7 * M_s
        D_s = Planck15.angular_diameter_distance(z=z_s).value * Mpc
        D_l = Planck15.angular_diameter_distance(z=z_l).value * Mpc
        r_s_hst, rho_s_hst = MassProfileNFW.get_r_s_rho_s_NFW(M_200_hst, c_200_hst)

        ps = SubhaloPopulation(N_calib=150, M_hst=M_200_hst, c_hst=c_200_hst, m_min=m_200_min_sub, theta_s=r_s_hst / D_l)

        m_sub = ps.m_sample

        S_tot = self.total_flux_adu(mag, M_zero)

        print(z_l, z_s, theta_E, sigma_v, q, theta_x_0, theta_y_0, theta_s, mag)

        hst_param_dict = {
            "profile": "SIE",
            "theta_x_0": 0.0,
            "theta_y_0": 0.0,
            "theta_E": theta_E,
            "q": q,
        }

        lens_list = [hst_param_dict]

        for m in m_sub:
            c = MassProfileNFW.c_200_SCP(m)
            r_s, rho_s = MassProfileNFW.get_r_s_rho_s_NFW(m, c)
            sub_param_dict = {
                "profile": "NFW",
                "theta_x_0": np.random.uniform(-2, 2),
                "theta_y_0": np.random.uniform(-2, 2),
                "M_200": m,
                "r_s": r_s,
                "rho_s": rho_s
            }

            lens_list.append(sub_param_dict)

        observation_dict = {
            "n_x": 64,
            "n_y": 64,
            "theta_x_lims": (-coordinate_limit, coordinate_limit),
            "theta_y_lims": (-coordinate_limit, coordinate_limit),
            "exposure": exposure,
            "f_iso": f_iso,
        }

        src_param_dict = {
            "profile": "Sersic",
            "theta_x_0": theta_x_0,
            "theta_y_0": theta_y_0,
            "S_tot": S_tot,
            "theta_e": theta_s,
            "n_srsc": 1,
        }

        global_dict = {"z_s": z_s, "z_l": z_l}

        lsi = LensingSim(lens_list,
                         [src_param_dict],
                         global_dict,
                         observation_dict)

        self.image = lsi.lensed_image()

    def total_flux_adu(self, mag, mag_zp):
        """
        Returns total flux of the integrated profile, in ADU relative to mag_zp
        """
        return 10 ** (-0.4 * (mag - mag_zp))

    def M_200_sigma_v(self, sigma_v):
        """ https://arxiv.org/pdf/1804.04492.pdf
        """
        a = 0.09
        b = 3.48
        sigma_log10_M_200 = 0.13
        log10_M_200 = np.random.normal(a + b * np.log10(sigma_v / (100 * Kmps)), sigma_log10_M_200)
        return (10 ** log10_M_200) * 1e12 * M_s


class SubhaloPopulation:
    def __init__(self, N_calib=150, beta=-1.9, m_min=1e9*M_s, r_roi=2.5,
                 M_hst=1e14*M_s, theta_s=1e-4, c_hst=6.):

        alpha = self.alpha_calib(1e8 * M_s, 1e10 * M_s, N_calib, M_MW, beta)
        self.n_sub_tot = self.n_sub(m_min, 0.01 * M_hst, M_hst, alpha, beta)

        f_sub = MassProfileNFW.M_cyl_div_M0(r_roi * asctorad / theta_s) \
            / MassProfileNFW.M_cyl_div_M0(c_hst * theta_s / theta_s)

        self.n_sub_roi = np.random.poisson(f_sub * self.n_sub_tot)
        self.m_sample = self.draw_m_sub(self.n_sub_roi, m_min, beta)

    def alpha_calib(self, m_min_calib, m_max_calib, n_calib, M_calib, beta, M_0=M_MW, m_0=1e9*M_s):
        return -M_0 * (m_max_calib * m_min_calib / m_0) ** -beta * n_calib * (-1 + -beta) / \
               (M_calib * (-m_max_calib ** -beta * m_min_calib + m_max_calib * m_min_calib ** -beta))

    def n_sub(self, m_min, m_max, M, alpha, beta, M_0=M_MW, m_0=1e9*M_s):
        return alpha * M * (m_max * m_min / m_0) ** --beta * \
               (m_max ** -beta * m_min - m_max * m_min ** -beta) / (M_0 * (-1 + -beta))

    def draw_m_sub(self, n_sub, m_sub_min, beta):
        u = np.random.uniform(0, 1, size=n_sub)
        m_sub = m_sub_min * (1 - u) ** (1.0 / (beta + 1.0))
        return m_sub
