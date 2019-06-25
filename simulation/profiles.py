from simulation.units import *
from scipy.special import gamma


class MassProfileSIE:
    def __init__(self, x_0, y_0, r_E, q):
        """
        Singular isothermal ellipsoid (SIE) mass profile class

        :param x_0: x-coordinate of center of deflector, in same units as r_E
        :param y_0: y-coordinate of center of deflector, in same units as r_E
        :param r_E: Einstein radius of deflector
        :param q: Axis-ratio of deflector
        """
        self.x_0 = x_0
        self.y_0 = y_0
        self.r_E = r_E
        self.q = q

    def deflection(self, x, y):
        """
        Calculate deflection vectors, from astro-ph/0102341
        TODO: deal with origin singularity

        :param x: x-coordinate at which deflection computed, in same units as r_E
        :param y: y-coordinate at which deflection computed, in same units as r_E
        :return: Deflections at positions specified by x, y
        """
        # Go into shifted coordinates
        x_p = x - self.x_0
        y_p = y - self.y_0

        # Compute deflection field
        psi = np.sqrt((self.q * x_p) ** 2 + y_p ** 2)

        if self.q == 1:
            x_d = self.r_E * x_p / psi
            y_d = self.r_E * y_p / psi
        else:
            x_d = self.r_E * self.q / np.sqrt(1 - self.q ** 2) * np.arctan(np.sqrt(1 - self.q ** 2) * x_p / psi)
            y_d = self.r_E * self.q / np.sqrt(1 - self.q ** 2) * np.arctanh(np.sqrt(1 - self.q ** 2) * y_p / psi)

        # Return deflection field
        return x_d, y_d

    @classmethod
    def theta_E(self, sigma_v, D_ls, D_s):
        """ Einstein radius (in arcsecs) for a SIS halo
            :param sigma_v: Central velocity dispersion of SIE halo
            :param D_ls: Angular distance between lens and source, in natural units
            :param D_s: Angular distance between observer and source, in natural units
            :return: Einstein radius of lens, in arcsecs
        """
        return 4 * np.pi * sigma_v ** 2 * D_ls / D_s * radtoasc


class MassProfileNFW:
    def __init__(self, x_0, y_0, M_200, kappa_s, r_s):
        """
        Navarro-Frenk-White (NFW) mass profile class

        :param x_0: x-coordinate of center of deflector, in same units as r_s
        :param y_0: y-coordinate of center of deflector, in same units as r_s
        :param kappa_s: Overall normalization of the DM halo (kappa_s = rho_s * r_s / Sigma_crit)
        :param r_s: Scale radius of NFW halo
        """

        self.x_0 = x_0
        self.y_0 = y_0
        self.M_200 = M_200
        self.kappa_s = kappa_s
        self.r_s = r_s

    def deflection(self, x, y):
        """
        Calculate deflection vectors, from astro-ph/0102341
        TODO: deal with origin singularity

        :param x: x-coordinate at which deflection computed, in same units as r_E
        :param y: y-coordinate at which deflection computed, in same units as r_E
        :return: Deflections at positions specified by x, y
        """

        # Go into shifted coordinates
        x_p = x - self.x_0
        y_p = y - self.y_0

        r = np.sqrt(x_p ** 2 + y_p ** 2)

        x = r / self.r_s

        # Get spherically symmetric deflection field, from astro-ph/0102341
        phi_r = 4 * self.kappa_s * self.r_s * (np.log(x / 2.0) + self.F(x)) / x

        # Get x and y coordinates of deflection
        x_d = phi_r * x_p / r
        y_d = phi_r * y_p / r

        # Convert to arcsecs and return deflection field
        return x_d, y_d

    @classmethod
    def F(self, x):
        """
        Helper function for NFW deflection, from astro-ph/0102341
        TODO: returning warnings for invalid value in sqrt for some reason
        JB: That's because all of the arguments of np.where are evaluated, including the ones with ngative arguments to
        sqrt, but only the good ones are then returned. So we can just suppress these warnings
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                x == 1.0,
                1.0,
                np.where(x <= 1.0, np.arctanh(np.sqrt(1.0 - x ** 2)) / (np.sqrt(1.0 - x ** 2)), np.arctan(np.sqrt(x ** 2 - 1.0)) / (np.sqrt(x ** 2 - 1.0))),
            )

    @classmethod
    def get_r_s_rho_s_NFW(self, M_200, c_200):
        """ Get NFW scale radius and density
        """
        r_200 = (M_200 / (4 / 3.0 * np.pi * 200 * rho_c)) ** (1 / 3.0)
        rho_s = M_200 / (4 * np.pi * (r_200 / c_200) ** 3 * (np.log(1 + c_200) - c_200 / (1 + c_200)))
        r_s = r_200 / c_200
        return r_s, rho_s

    @classmethod
    def c_200_SCP(self, M_200):
        """ Concentration-mass relation according to eq. 1 of  Sanchez-Conde & Prada 2014 (1312.1729)
            :param M_200: M_200 mass of halo
        """
        x = np.log(M_200 / (M_s / h))
        pars = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)

    @classmethod
    def M_cyl_div_M0(self, x):
        return np.log(x / 2) + self.F(x)


class LightProfileSersic:
    def __init__(self, x_0, y_0, r_e, n_srsc, S_tot):
        """
        Sersic light profile.

        :param x_0: x-coordinate of source location
        :param y_0: y-coordinate of source location
        :param r_e: The circular effective radius containing half the total light
        :param n_srsc: Sersic index controlling concentration of the light profile
        :param S_tot: Total counts or flux normalization
        """
        self.x_0 = x_0
        self.y_0 = y_0
        self.r_e = r_e
        self.n_srsc = n_srsc
        self.S_tot = S_tot

    def flux(self, x, y):
        """
        :param x: x-coordinate at which intensity computed in the same units as r_e
        :param y: y-coordinate at which intensity computed in the same units as r_e
        :return: Flux for Sersic profile at given points x, y
        """

        # Go into shifted coordinates
        x_p = x - self.x_0
        y_p = y - self.y_0

        # Radial distance for spherically symmetric profile
        r = np.sqrt(x_p ** 2 + y_p ** 2)

        # Get normalization factors
        b_n = self.b_n(self.n_srsc)
        flux_e = self.flux_e(self.S_tot, self.n_srsc, self.r_e)

        return flux_e * np.exp(-b_n * ((r / self.r_e) ** (1 / self.n_srsc) - 1))

    @classmethod
    def b_n(self, n_srsc):
        """
        Normalization parameter ensuring that the effective radius contains half of the profile's total light
        From Ciotti & Bertin 1999, A&A, 352, 447
        """
        return 2 * n_srsc - 1 / 3.0 + 4 / (405 * n_srsc) + 46 / (25515 * n_srsc ** 2) + 131 / (1148175 * n_srsc ** 3) - 2194697 / (30690717750 * n_srsc ** 4)

    @classmethod
    def flux_e(self, S_tot, n_srsc, r_e):
        """
        Compute flux at half-light radius given the total counts S_tot
        """
        if n_srsc == 1:
            return S_tot / (3.8 * np.pi * r_e ** 2)
        elif n_srsc == 4:
            return S_tot / (7.2 * np.pi * r_e ** 2)
        else:
            b_n = self.b_n(n_srsc)
            return S_tot * (b_n ** (2 * n_srsc) * np.exp(-b_n)) / (2 * n_srsc * np.pi * r_e ** 2 * gamma(2 * n_srsc))
