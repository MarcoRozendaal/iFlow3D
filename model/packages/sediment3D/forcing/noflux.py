"""
In this file, the forcing components of the no-flux forcing are presented.


# TODO test functions
"""

import warnings
import numpy as np

import ngsolve


from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponents, conditional_conjugate, conditional_real_part
from model.packages.hydrodynamics3D.classes.hydrodynamics import Hydrodynamics
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder
from model.packages.sediment3D.classes.capacity import SedimentCapacity
from model.packages.sediment3D.classes.capacity_forcing_components import SedimentCapacityForcingComponents
from model.packages.sediment3D.classes.capacity_order import SedimentCapacityOrder
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters


class SedimentCapacityForcingNoFlux():
    """"
       Class to generate the no-flux forcing components
    """

    def __init__(self,
                 hydro: Hydrodynamics,
                 sedcaplead: SedimentCapacityOrder,
                 sed_phys_params: SedimentPhysicalParameters):
        """
        Initialization
        Args:
            hydro: Hydrodynamics object
            sedcaplead: SedimentCapacityOrder object
            sed_phys_params: SedimentPhysicalParameters object
        """
        # general parameters
        self.hydro = hydro
        self.sedcaplead = sedcaplead
        self.params = sed_phys_params



    # Core functionality of this class:
    def generate_forcing_components(self, k, n, f, alpha):
        # Create instance of SedimentCapacityForcingComponents class and return it
        # TODO check late bind
        return SedimentCapacityForcingComponents(lambda z, n=n, f=f, alpha=alpha: self.hatC(k, n, f, alpha, z), self.hatC_DA(k, n, f, alpha))



    def hatChi(self, k, n, f, alpha):
        """
        No-flux forcing at the free surface
        """

        # We compute the Fourier coefficient
        chi = None

        # We have hatChi
        if k==1 and n==1 and f=='noflux' and alpha=='a':
            chi = 1j * self.params.omega * ngsolve.Conj(self.hydro.Z[0][1]['tide']) * self.sedcaplead.hatC[2]['etide']['a'](self.params.R)

        # Check if no valid combination is found
        if chi is None:
            raise("Invalid parameters supplied to hatchi forcing")

        # Return forcing
        return chi


    ### The sediment capacity ###
    def hatC(self, k, n, f, alpha, z):
        """ The sediment capacity due to no-flux forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        return -self.hatChi(k, n, f, alpha) * self.params.K(n) * self.params._expzR(z) \
               * (self.params.sigma * self.params._sinhzH(n, z)
                  + self.params.lambda0(n) * self.params._coshzH(n, z))


    def hatC_DA(self, k, n, f, alpha):
        """ The depth_num-averaged sediment capacity due to noflux forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        # The depth_num-averaged sediment capacity simplifies for n=0:
        return -self.hatChi(k, n, f, alpha) * 1 / (n * 1j * self.params.omega * self.params.D) * (1 - self.params.w_s * self.params.lambda0(n) * self.params.K(n) * ngsolve.exp(self.params.sigma * self.params.D))


