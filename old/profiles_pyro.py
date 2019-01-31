import numpy as np
import sys
sys.path.append("../Lensing-PowerSpectra/Simulations/")
from units import *
import torch

def deflection_sis(x, y, x0=0, y0=0, b=1.5):
    """ Deflection for singular isothermal ellipsoid, from astro-ph/0102341
    """ 
    # Go into shifted coordinats of the potential
    xsie = x - x0
    ysie = y - y0 

    # Compute potential gradient
    psi = torch.sqrt(xsie**2 + ysie**2)
    xg = b * xsie / psi
    yg = b * ysie / psi

    # Return value
    return xg.type(torch.double), yg.type(torch.double)

def deflection_nfw(x, y, x0=0, y0=0, M=1e14*M_s, c=20., D_s=1*Mpc, D_l=0.5*Mpc):
    """ Deflection for an NFW halo, from astro-ph/0102341
        TODO: deal with origin singularity as in SIE case
    """ 
    D_ls = D_s - D_l
    
    # Coordinates in natural (not angular) units
    xnfw = (x - x0)*D_l*asctorad
    ynfw = (y - y0)*D_l*asctorad
    r = torch.sqrt(xnfw.type(torch.double)**2 + ynfw.type(torch.double)**2)

    delta_c = (200/3.)*c**3/(np.log(1+c) - c/(1+c)) 
    rho_s = rho_c*delta_c 
        
    r_s = (M/((4/3.)*np.pi*c**3*200*rho_c))**(1/3.) # NFW scale radius
    
    x = r.type(torch.double)/r_s.type(torch.double)

    Sigma_crit = Sigma_cr(D_l, D_s) # Critical lensing density
    
    kappa_s = rho_s*r_s/Sigma_crit
    
    # Get spherically symmetric deflection
    # Fvec =  tt.switch(tt.eq(x,1), 1, tt.switch(x > 1, F1(x), F2(x)))
    x = x.type(torch.double)
    Fvec =  torch.where(x > 1, F1(x), F2(x)).type(torch.double)
    print(Fvec)
    # print(F2(x))
    phitg = 4*kappa_s*r_s*(torch.log(x/2.) + Fvec)/x
    
    # Get x and y coordinates of deflection
    xtg = phitg*xnfw.type(torch.double)/r.type(torch.double)
    ytg = phitg*ynfw.type(torch.double)/r.type(torch.double)
    
    # Convert back to rad, then arcsecs
    return xtg/D_l*radtoasc, ytg/D_l*radtoasc

# def F(x):
#     """ Helper function for NFW deflection, from astro-ph/0102341
#     """
#     if x > 1:
#         return torch.atan(np.sqrt(x**2-1))/(np.sqrt(x**2 - 1))
#     elif x == 1:
#         return 1
#     elif x < 1:
#         return arctanh(np.sqrt(1-x**2))/(np.sqrt(1-x**2))

def F1(x):
    """ Helper function for NFW deflection, from astro-ph/0102341
    """
    return x
    # return torch.atan(torch.sqrt(x**2-1))/(torch.sqrt(x**2 - 1))

def F2(x):
    """ Helper function for NFW deflection, from astro-ph/0102341
    """
    return arctanh(torch.sqrt(1-x**2))/(torch.sqrt(1-x**2))

def arctanh(x):
    return 0.5*torch.log((1+x)/(1-x))


def Sigma_cr(D_l, D_s):
    return 1./(4*np.pi*GN)*D_s/((D_s - D_l)*D_l)


def f_gal_sersic(x, y, n=4, I_gal=1e-16*erg/Centimeter**2/Sec/Angstrom, theta_e_gal=1):
    """ Sersic profile surface brightness, following Daylan et al
    """
    theta = torch.sqrt(x**2 + y**2)
    b_n = 2*n - 1/3. + 4/(405*n) + 46/(25515*n**2)
    f_e_gal = I_gal/(7.2*np.pi*theta_e_gal**2)
    out = (f_e_gal*torch.exp(-b_n*((theta/theta_e_gal)**(1/n) - 1))).type(torch.double)

    return out