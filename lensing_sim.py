import numpy as np
from astropy.cosmology import Planck15

from units import *

import sys, os
from profiles import *

class LensingSim():
    def __init__(self, lenses_list=[{}], 
                    sources_list=[{}],
                    global_dict={},
                    observation_dict={}):
        """
        Class for simulation of strong lensing images
        """        

        self.lenses_list = lenses_list
        self.sources_list = sources_list

        self.global_dict = global_dict
        self.observation_dict = observation_dict

        self.set_up_observation()
        self.set_up_global()

    def set_up_global(self):
        """ Set some global variables so don't need to recompute each time
        """
        self.z_s = self.global_dict['z_s'] 
        self.z_l = self.global_dict['z_l'] 

        self.D_s = Planck15.angular_diameter_distance(z=self.z_s).value*Mpc
        self.D_l = Planck15.angular_diameter_distance(z=self.z_l).value*Mpc

    def set_up_observation(self):
        """ Set up observational grid and parameters
        """
        # Coordinate limits (in arcsecs)
        self.xlims = self.observation_dict['xlims'] 
        self.ylims = self.observation_dict['ylims']

        # Size of grid
        self.nx = self.observation_dict['nx'] 
        self.ny = self.observation_dict['ny'] 

        # Exposure and background noise level
        self.exposure = self.observation_dict['exposure'] 
        self.A_iso = self.observation_dict['A_iso'] 

        # x/y-coordinates of grid and pixel area in arcsec**2

        self.y_coords, self.x_coords = torch.meshgrid([torch.linspace(self.xlims[0],self.xlims[1], self.nx, dtype=torch.double), torch.linspace(self.ylims[0],self.ylims[1], self.ny, dtype=torch.double)])

        self.pixarea = ((self.xlims[1] - self.xlims[0])/self.nx)*((self.ylims[1] - self.ylims[0])/self.ny)

    def lensed_image(self):
        """ Get strongly lensed image
        """

        # Get lensing potential gradients

        xg, yg = torch.zeros((self.nx, self.ny), dtype=torch.double), torch.zeros((self.nx, self.ny), dtype=torch.double)
        
        for lens_dict in self.lenses_list:
            if lens_dict['profile'] == 'sis':
                self.theta_x_hst = lens_dict['theta_x'] 
                self.theta_y_hst = lens_dict['theta_y'] 
                self.theta_E_hst = lens_dict['theta_E'] 
                _xg, _yg = deflection_sis(self.x_coords, self.y_coords, x0=self.theta_x_hst, y0=self.theta_y_hst, b=self.theta_E_hst)
                xg += _xg
                yg += _yg
            else:
                raise Exception('Unknown lens profile specification!')

        # Get lensed image

        self.i_lens = torch.zeros((self.nx, self.ny), dtype=torch.double) 
    
        for source_dict in self.sources_list:
            if source_dict['profile'] == 'sersic':
                self.I_gal = source_dict['I_gal']
                self.n_srsc = source_dict['n_srsc']
                self.theta_e_gal = source_dict['theta_e_gal']
                self.i_lens += f_gal_sersic(self.x_coords-xg, self.y_coords-yg, self.n_srsc, self.I_gal, self.theta_e_gal)
            else:
                raise Exception('Unknown source profile specification!')

        self.i_iso = self.A_iso*torch.ones((self.nx, self.ny), dtype=torch.double) # Isotropic background
        self.i_tot = (self.i_lens + self.i_iso)*self.exposure*self.pixarea # Total lensed image

        return (self.i_tot.type(torch.FloatTensor))