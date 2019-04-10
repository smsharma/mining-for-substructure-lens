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

    psi = np.sqrt((q * theta_xp) ** 2 + theta_yp ** 2)

    if q == 1:
        theta_xd = theta_E * theta_xp / psi
        theta_yd = theta_E * theta_yp / psi
    else:
        theta_xd = theta_E * q / np.sqrt(1 - q ** 2)*np.arctan(np.sqrt(1 - q ** 2) * theta_xp / psi)
        theta_yd = theta_E * q / np.sqrt(1 - q ** 2)*np.arctanh(np.sqrt(1 - q ** 2) * theta_yp / psi)

    # Return deflection field
    return theta_xd, theta_yd


# def deflection_nfw(x, y, x0=0, y0=0, M=1e14 * M_s, c=20.0, D_s=1 * Mpc, D_l=0.5 * Mpc):
#     """ Deflection for an NFW halo, from astro-ph/0102341
#         TODO: deal with origin singularity
#     """
#
#     # Coordinates in natural (not angular) units
#     xnfw = (x - x0) * D_l * asctorad
#     ynfw = (y - y0) * D_l * asctorad
#     r = np.sqrt(xnfw ** 2 + ynfw ** 2)
#
#     delta_c = (200 / 3.0) * c ** 3 / (np.log(1 + c) - c / (1 + c))
#     rho_s = rho_c * delta_c
#
#     r_s = (M / ((4 / 3.0) * np.pi * c ** 3 * 200 * rho_c)) ** (
#         1 / 3.0
#     )  # NFW scale radius
#
#     x = r / r_s
#
#     Sigma_crit = Sigma_cr(D_l, D_s)  # Critical lensing density
#
#     kappa_s = rho_s * r_s / Sigma_crit
#
#     # Get spherically symmetric deflection
#     Fvec = np.vectorize(F)
#     phitg = 4 * kappa_s * r_s * (np.log(x / 2.0) + Fvec(x)) / x
#
#     # Get x and y coordinates of deflection
#     xtg = phitg * xnfw / r
#     ytg = phitg * ynfw / r
#
#     # Convert back to rad, then arcsecs
#     return xtg / D_l * radtoasc, ytg / D_l * radtoasc
#
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
#
#
# def Sigma_cr(D_l, D_s):
#     """ Critical surface density
#     """
#     return 1.0 / (4 * np.pi * GN) * D_s / ((D_s - D_l) * D_l)


def f_gal_sersic(
    theta_x=0, theta_y=0, theta_e_gal=1, n_srsc=4, I_gal=1e-16 * erg / Centimeter ** 2 / Sec / Angstrom):
    """ Compute the intensity of the Sersic profile at a given position.

        :param theta_x: x-coordinate at which intensity computed in the same units as theta_e_gal
        :param theta_y: y-coordinate at which intensity computed in the same units as theta_e_gal
        :param theta_e_gal: the circular effective radius containing half the total light
        :param n_srsc: Sersic index controlling concentration of the light profile (lower value -> more concentrated)
        :param I_gal: overall intensity normalization of the light profile
        :return: Intensity of specified Sersic profile at a given position.
    """
    # Radial distance for spherically symmetric profile
    theta = np.sqrt(theta_x ** 2 + theta_y ** 2)

    # Normalization parameter ensuring that the effective radius contains half of the profile's total light
    # TODO: augment to fourth order.
    b_n = 2 * n_srsc - 1 / 3.0 + 4 / (405 * n_srsc) + 46 / (25515 * n_srsc ** 2)

    # Conversion between surface brightness at half-light radius and flux of galaxy
    f_e_gal = I_gal / (7.2 * np.pi * theta_e_gal ** 2)

    return f_e_gal * np.exp(-b_n * ((theta / theta_e_gal) ** (1 / n_srsc) - 1))
