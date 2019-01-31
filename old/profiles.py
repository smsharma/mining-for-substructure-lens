import numpy as np
import sys
sys.path.append("../Lensing-PowerSpectra/Simulations/")
from units import *
import tensorflow as tf

def deflection_sis(x, y, x0=0, y0=0, b=1.5):
    """ Deflection for singular isothermal ellipsoid, from astro-ph/0102341
    """ 
    # Go into shifted coordinats of the potential
    xsie = x - x0
    ysie = y - y0 
    
    # Compute potential gradient
    psi = np.sqrt(xsie**2 + ysie**2)
    xg = b * xsie / (psi + (psi == 0)) 
    yg = b * ysie / (psi + (psi == 0))

    # Return value
    return xg, yg

def f_gal_sersic(x, y, n=4, I_gal=1e-16*erg/Centimeter**2/Sec/Angstrom, theta_e_gal=1):
    """ Sersic profile surface brightness, following Daylan et al
    """
    theta = tf.sqrt(x**2 + y**2)
    b_n = 2*n - 1/3. + 4/(405*n) + 46/(25515*n**2)
    f_e_gal = I_gal/(7.2*np.pi*theta_e_gal**2)
    return f_e_gal*tf.exp(-b_n*((theta/theta_e_gal)**(1/n) - 1))
