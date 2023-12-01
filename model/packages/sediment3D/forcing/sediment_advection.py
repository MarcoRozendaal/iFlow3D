"""
In this file, the forcing components of the advection of sediment forcing are presented.

We make sediment advection fully discretized
# TODO test functions
"""

import ngsolve

import numpy as np

from model.general.classes.discretized.numerical_depth_quadrature import NumericalDepthQuadrature
from model.general.classes.discretized.disc_collection import DiscretizedCollection
from model.general.classes.discretized.function import innerproduct
from model.packages.hydrodynamics3D.classes.hydrodynamics import Hydrodynamics
from model.packages.sediment3D.classes.capacity import SedimentCapacity
from model.packages.sediment3D.classes.capacity_forcing_components import SedimentCapacityForcingComponents
from model.packages.sediment3D.classes.capacity_order import SedimentCapacityOrder
from model.packages.sediment3D.classes.numerical_parameters import SedimentNumericalParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters


class SedimentCapacityForcingSedimentAdvection():
    """"
       Class to generate the sediment advection forcing components
    """

    def __init__(self,
                 disccol: DiscretizedCollection,
                 sed_phys_params: SedimentPhysicalParameters,
                 sed_num_params: SedimentNumericalParameters):
        """
        Previously depended on hydro and sedcaplead, now depends on disccol
        Args:
            disccol: DiscretizedCollection object
            sed_phys_params: SedimentPhysicalParameters object
            sed_num_params: SedimentNumericalParameters object
        """
        # general parameters
        self.disccol = disccol
        self.numquad: NumericalDepthQuadrature = sed_num_params.numquad
        self.params = sed_phys_params




    # Core functionality of this class:
    def generate_forcing_components(self, k, n, f, alpha):
        # Create instance of SedimentCapacityForcingComponents class and return it
        hatC_disc = self.hatC_disc(k, n, f, alpha)
        hatC_DA_disc = self.numquad.depth_average(hatC_disc)
        return SedimentCapacityForcingComponents(hatC_disc, hatC_DA_disc)



    def hateta_disc(self, k, n, f, alpha):
        """ Advection of sediment forcing """

        # We compute the Fourier coefficient
        hateta_disc = None

        # We need first order, M_2, sediment advection
        if k==1 and n==1 and f=='sedadv':
            if alpha=='a':
                # Use the custom defined innerproduct for disc functions
                hateta_disc = innerproduct(self.disccol.U_V[0][1]['tide'], self.disccol.nablahatC[0][0]['etide']['a']) \
                         + self.disccol.W[0][1]['tide'] * self.disccol.hatC_z[0][0]['etide']['a'] \
                         + 1/2 * innerproduct(np.conj(self.disccol.U_V[0][1]['tide']), self.disccol.nablahatC[0][2]['etide']['a']) \
                         + 1/2 * np.conj(self.disccol.W[0][1]['tide']) * self.disccol.hatC_z[0][2]['etide']['a']
            else:
                # We use pythons automatic broadcasting to make vector scalar products work
                hateta_V = self.disccol.U_V[0][1]['tide'] * self.disccol.hatC[0][0]['etide']['a'] \
                         + 1/2 * np.conj(self.disccol.U_V[0][1]['tide']) * self.disccol.hatC[0][2]['etide']['a']
                if alpha=='a_x':
                    hateta_disc = hateta_V[0]
                elif alpha=='a_y':
                    hateta_disc = hateta_V[1]

        # Check if no valid combination is found
        if hateta_disc is None:
            raise("Invalid parameters supplied to advection of sediment forcing")

        # Return erosion forcing
        return hateta_disc



    ### The sediment capacity ###
    def hatC_disc(self, k, n, f, alpha):
        """ The sediment capacity due to erosion forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """


        # The first and second homogenous solutions discretized
        C111_disc = self.numquad.discretize_function3D(lambda z, n=n: self.params._expzH(z) * self.params._sinhzR(n, z))
        C112_disc = self.numquad.discretize_function3D(lambda z, n=n: self.params._expzH(z) * self.params._coshzR(n, z))


        # We build the variation of parameters numerical integrals
        hateta_disk = self.hateta_disc(k, n, f, alpha)

        partial_integrand_A_disc = self.numquad.discretize_function3D(lambda z, n=n: 1 / (self.params.Kv * self.params.lambda0(n)) *
                                                                                1 / self.params._expzH(z) *
                                                                                self.params._coshzR(n, z))

        partial_integrand_B_disc = self.numquad.discretize_function3D(lambda z, n=n: -1 / (self.params.Kv * self.params.lambda0(n)) *
                                                                                1 / self.params._expzH(z) *
                                                                                self.params._sinhzR(n, z))

        integrand_A_disc = partial_integrand_A_disc * hateta_disk
        integrand_B_disc = partial_integrand_B_disc * hateta_disk





        # Compute the cumulative depth integral
        A_disc = self.numquad.cumulative_depth_integral(integrand_A_disc)
        B_disc = self.numquad.cumulative_depth_integral(integrand_B_disc)



        # Compute the sediment capacity
        hatC_disc = A_disc * C111_disc + B_disc * C112_disc

        return hatC_disc


    def hatC_DA_disc(self, k, n, f, alpha):
        """ The depth_num-averaged sediment capacity due to erosion forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        return self.numquad.depth_average(self.hatC_disc(k, n, f, alpha))

