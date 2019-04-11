from simulation.units import *


def deflection_sie(theta_x, theta_y, theta_x0=0, theta_y0=0, theta_E=1.5, q=1):
    """ Deflection for singular isothermal ellipsoid (SIE) mass profile, from astro-ph/0102341.
#       TODO: deal with origin singularity

        :param theta_x: x-coordinate at which deflection computed, in same units as theta_E
        :param theta_y: y-coordinate at which deflection computed, in same units as theta_E
        :param theta_x0: x-coordinate of center of deflector, in same units as theta_E
        :param theta_y0: y-coordinate of center of deflector, in same units as theta_E
        :param theta_E: Einstein radius of deflector
        :param q: Axis-ratio of deflector
        :return: Deflections at positions specified by theta_x, theta_y
    """
    # Go into shifted coordinates
    theta_xp = theta_x - theta_x0
    theta_yp = theta_y - theta_y0

    # Compute deflection field

    theta_psi = np.sqrt((q * theta_xp) ** 2 + theta_yp ** 2)

    if q == 1:
        theta_xd = theta_E * theta_xp / theta_psi
        theta_yd = theta_E * theta_yp / theta_psi
    else:
        theta_xd = theta_E * q / np.sqrt(1 - q ** 2)*np.arctan(np.sqrt(1 - q ** 2) * theta_xp / theta_psi)
        theta_yd = theta_E * q / np.sqrt(1 - q ** 2)*np.arctanh(np.sqrt(1 - q ** 2) * theta_yp / theta_psi)

    # Return deflection field
    return theta_xd, theta_yd

#
def deflection_nfw(theta_x, theta_y, theta_x0=0, theta_y0=0, M=1e14 * M_s, c=20.0, D_s=1 * Mpc, D_l=0.5 * Mpc):
    """ Deflection for an NFW halo, from astro-ph/0102341
        TODO: deal with origin singularity
    """

    # Go into shifted coordinates
    theta_xp = theta_x - theta_x0
    theta_yp = theta_y - theta_y0

    theta_r = np.sqrt(theta_xp ** 2 + theta_yp ** 2)

    r_s, rho_s = get_rs_rhos_NFW(M, c)

    theta_s = r_s / D_l

    x = theta_r * asctorad / theta_s

    Sigma_crit = Sigma_cr(D_l, D_s)  # Critical lensing density

    kappa_s = rho_s * r_s / Sigma_crit

    # Get spherically symmetric deflection
    phi_theta = 4 * kappa_s * theta_s * (np.log(x / 2.0) + F(x)) / x

    # Get x and y coordinates of deflection
    xtg = phi_theta * theta_xp / theta_r
    ytg = phi_theta * theta_yp / theta_r

    # Convert to arcsecs
    return xtg * radtoasc, ytg * radtoasc


def get_rs_rhos_NFW(M200, c200):
    """ Get NFW scale radius and density
    """
    r200 = (M200/(4/3.*np.pi*200*rho_c))**(1/3.)
    rho_s = M200/(4*np.pi*(r200/c200)**3*(np.log(1 + c200) - c200/(1 + c200)))
    r_s = r200/c200
    return r_s, rho_s

def F(x):
    """ Helper function for NFW deflection, from astro-ph/0102341
    """
    # TODO: returning warnings for invalid value in sqrt for some reason...
    return np.where(x == 1, 1, np.where(x < 1, np.arctanh(np.sqrt(1 - x ** 2)) / (np.sqrt(1 - x ** 2)), \
                                        np.arctan(np.sqrt(x ** 2 - 1)) / (np.sqrt(x ** 2 - 1))))
#
# def F(x):
#     """ Helper function for NFW deflection, from astro-ph/0102341
#     """
#     if x > 1:
#         return np.arctan(np.sqrt(x ** 2 - 1)) / (np.sqrt(x ** 2 - 1))
#     elif x == 1:
#         return 1
#     elif x < 1:
#         return np.arctanh(np.sqrt(1 - x ** 2)) / (np.sqrt(1 - x ** 2))


def Sigma_cr(D_l, D_s):
    """ Critical surface density
    """
    return 1.0 / (4 * np.pi * GN) * D_s / ((D_s - D_l) * D_l)


def f_gal_sersic(
    theta_x, theta_y, theta_x0=0, theta_y0=0, theta_e_gal=1, n_srsc=4, I_gal=1e-16 * erg / Centimeter ** 2 / Sec / Angstrom):
    """ Compute the intensity of the Sersic profile at a given position.

        :param theta_x: x-coordinate at which intensity computed in the same units as theta_e_gal
        :param theta_y: y-coordinate at which intensity computed in the same units as theta_e_gal
        :param theta_e_gal: the circular effective radius containing half the total light
        :param n_srsc: Sersic index controlling concentration of the light profile (lower value -> more concentrated)
        :param I_gal: overall intensity normalization of the light profile
        :return: Intensity of specified Sersic profile at a given position.
    """

    # Go into shifted coordinates
    theta_xp = theta_x - theta_x0
    theta_yp = theta_y - theta_y0

    # Radial distance for spherically symmetric profile
    theta = np.sqrt(theta_xp ** 2 + theta_yp ** 2)

    # Normalization parameter ensuring that the effective radius contains half of the profile's total light
    b_n = 2 * n_srsc - 1 / 3.0 + 4 / (405 * n_srsc) + 46 / (25515 * n_srsc ** 2) + 131 / (1148175 * n_srsc ** 3) - \
          2194697 / (30690717750 * n_srsc ** 4)

    # Conversion between surface brightness at half-light radius and flux of galaxy
    # TODO: only valid for n_srsc = 4; fix!
    f_e_gal = I_gal / (7.2 * np.pi * theta_e_gal ** 2)

    return f_e_gal * np.exp(-b_n * ((theta / theta_e_gal) ** (1 / n_srsc) - 1))
