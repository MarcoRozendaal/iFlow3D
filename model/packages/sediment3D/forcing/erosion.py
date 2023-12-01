"""
In this file, the forcing components of the erosion forcing are presented.

Child of SedimentCapacityForcingComponents

# TODO test functions

"""

import warnings
import numpy as np

import ngsolve

from model.general.classes.spatial_parameter import SpatialParameterFromCoefficientFunction
from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponents, conditional_conjugate, conditional_real_part
from model.packages.hydrodynamics3D.classes.hydrodynamics import Hydrodynamics
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder
from model.packages.sediment3D.classes.capacity_forcing_components import SedimentCapacityForcingComponents
from model.packages.sediment3D.classes.numerical_parameters import SedimentNumericalParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters


class SedimentCapacityForcingErosion():
    """"
       Class to generate the erosion forcing components
    """
    # define epsilon numeric
    eps_n = 1e-3
    erosion_forcing_names = ['etide', 'eriver', 'ebaroc', 'ereference', 'eadv', 'estokes', 'enostress', 'epslip']

    def __init__(self, hydro: Hydrodynamics,
                 sed_phys_params: SedimentPhysicalParameters,
                 sed_num_params: SedimentNumericalParameters):
        """
        Initialization
        Args:
            hydro: Hydrodynamics object
            sed_phys_params: SedimentPhysicalParameters object
            sed_num_params: SedimentNumericalParameters object
        """
        # general parameters
        self.hydro = hydro
        self.physparams = sed_phys_params
        self.numparams = sed_num_params



    # Core functionality of this class:
    def generate_forcing_components(self, k, n, f, alpha):
        # Create instance of SedimentCapacityForcingComponents class and return it
        # TODO check late bind
        return SedimentCapacityForcingComponents(lambda z: self.hatC(k, n, f, alpha, z), self.hatC_DA(k, n, f, alpha))




    def hatE(self, k, n, f, alpha):
        """ Erosion forcing at the bed """

        # We compute the Fourier coefficient
        fourier_coefficient = None

        # hatE scales with a only
        if alpha=='a':

            # Precompute parameters
            U01b_V = self.hydro.U_V[0][1]['tide'](-self.physparams.H)
            alpha_V = U01b_V / ngsolve.Norm(U01b_V)
            a = self.innerproduct(alpha_V, alpha_V)
            r = ngsolve.Norm(a)
            theta = ngsolve.atan2(a.imag, a.real)

            # Leading order
            if k==0 and f=='etide':
                # M0
                if n==0:
                    fourier_coefficient = self.f0(r) * ngsolve.Norm(U01b_V)
                #M4
                elif n==2:
                    fourier_coefficient = self.f2(r) * ngsolve.Norm(U01b_V) * ngsolve.exp(1j*theta)

            # First-order, M2
            elif k==1 and n==1 and f in self.erosion_forcing_names:
                # Remove the e to obtain the hydro forcing symbol
                hydro_symbol = f[1:]

                D_1 = 2/np.pi * (self.I0(r) * alpha_V + self.I2(r) * ngsolve.Conj(alpha_V) * ngsolve.exp(1j*theta))
                D_3 = 2/np.pi * (self.I2(r) * alpha_V * ngsolve.exp(1j*theta) + self.I4(r) * ngsolve.Conj(alpha_V) * ngsolve.exp(1j*2*theta))

                # Not all forcing mechansisms are defined for all frequency components, if it is not defined, return zero vector
                def get_at_bed(dict, key):
                    if key in dict.keys():
                        return dict[key](-self.physparams.H)
                    else:
                        return ngsolve.CoefficientFunction((0, 0), dims=(2, 1))

                U10b_V = get_at_bed(self.hydro.U_V[1][0], hydro_symbol)
                U12b_V = get_at_bed(self.hydro.U_V[1][2], hydro_symbol)

                fourier_coefficient = D_1*U10b_V + 1/2*ngsolve.Conj(D_1)*U12b_V + 1/2*D_3*ngsolve.Conj(U12b_V)

        # Check if no valid combination is found
        if fourier_coefficient is None:
            raise("Invalid parameters supplied to erosion hatE forcing")

        # Return erosion forcing
        return self.physparams.rho0 * self.physparams.sf0 * self.physparams.M * fourier_coefficient

    ### The sediment capacity ###
    def hatC(self, k, n, f, alpha, z):
        """ The sediment capacity due to erosion forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        # The sediment capacity simplifies for n=0:
        if n==0:
            return self.hatE(k, 0, f, alpha) * 1 / self.physparams.w_s * ngsolve.exp(-2 * self.physparams.sigma * (z + self.physparams.H))
        else:
            return self.hatE(k, n, f, alpha) * self.physparams.K(n) * self.physparams._expzH(z) \
                   * (-self.physparams.sigma * self.physparams._sinhzR(n, z)
                      + self.physparams.lambda0(n) * self.physparams._coshzR(n, z))


    def hatC_DA(self, k, n, f, alpha):
        """ The depth_num-averaged sediment capacity due to erosion forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        # The depth_num-averaged sediment capacity simplifies for n=0:
        if n==0:
            return self.hatE(k, 0, f, alpha) * 1 / (2 * self.physparams.w_s * self.physparams.sigma * self.physparams.D) * (1 - ngsolve.exp(-2 * self.physparams.sigma * self.physparams.D))
        else:
            return self.hatE(k, n, f, alpha) * self.physparams.K(n) * self.physparams._sinhD(n) / self.physparams.D


    ### Other functions needed for other forcing mechanisms ###
    def hatC_z(self, k, n, f, alpha, z):
        """ The z-derivative of the sediment capacity due to erosion forcing. Order k, frequency n, forcing f, scaling alpha, vertical position z. """
        return self.hatE(k, n, f, alpha) * self.physparams.K(n) * self.physparams._expzH(z) \
               * ((self.physparams.lambda0(n) ** 2 + self.physparams.sigma ** 2) * self.physparams._sinhzR(n, z)
                  - 2 * self.physparams.sigma * self.physparams.lambda0(n) * self.physparams._coshzR(n, z))

    def nablahatC(self, k, n, f, alpha, z):
        """ Nabla of C"""
        return self.nabla_hatEK(k, n, f, alpha) * self.physparams._expzH(z) \
               * (-self.physparams.sigma * self.physparams._sinhzR(n, z)
                  + self.physparams.lambda0(n) * self.physparams._coshzR(n, z)) \
               + self.hatE(k, n, f, alpha) * self.physparams.K(n) * self.physparams._expzH(z) \
               * ((-self.physparams.lambda0(n) ** 2 * self.physparams.Gradient_R + self.physparams.sigma ** 2 * self.physparams.Gradient_H)
                  * self.physparams._sinhzR(n, z)
                  + self.physparams.sigma * self.physparams.lambda0(n) * (self.physparams.Gradient_R - self.physparams.Gradient_H)
                  * self.physparams._coshzR(n, z))


    def nabla_hatEK(self, k, n, f, alpha):
        """ Nabla of hatE K
        We numerically compute the gradient of this quantity
        We project onto an NGSolve GridFunction and take the gradient of that
        """
        hatEK_sp = SpatialParameterFromCoefficientFunction(self.hatE(k, n, f, alpha) * self.physparams.K(n), "linear", self.numparams.mesh)

        # Return the gradient of the spatial coefficient function
        return hatEK_sp.gradient_cf


    # Some utility functions
    def innerproduct(self, cf_1, cf_2):
        """ Computes the innerproduct without conjugation on the second argument """
        return cf_1[0]*cf_2[0] + cf_1[1]*cf_2[1]


    def f0(self, r):
        """ Function used to approximate the fourier coefficients """
        return (1.319675833 - 1.241923267*r)/(1.865818996 + (-1.740781287 + (0.0304939468 + (0.0583267878 - 0.09176029038*r)*r)*r)*r)

    def f2(self, r):
        """ Function used to approximate the fourier coefficients """
        return (0.0011553994 + ( 0.6392220857 - 0.5711563408*r)*r)/(1.876856269 + (-1.874289441 + 0.1607692043*r)*r)

    def I0(self, r):
        """ Function used to approximate the fourier coefficients """
        return (3.694249569 + (-3.343402152 + 0.1256704315*r)*r)/(1.663642695 + (-1.148513254 - 0.2403747084*r)*r) - 1/2*ngsolve.log(1-r + self.eps_n)

    def I2(self, r):
        """ Function used to approximate the fourier coefficients """
        return (- 0.00060244784 + (-0.0837643764 + (0.4442844276 - 0.3240069772*r)*r)*r)/(1.927203145 + (-2.111941818 + 0.3216217540*r)*r) + 1/2*ngsolve.log(1 - r + self.eps_n)

    def I4(self, r):
        """ Function used to approximate the fourier coefficients """
        return (-0.00059672002 + (-1.175876043 + (1.790378347 - 0.6310292261*r)*r)*r)/(2.397004301 + (-4.101506231 + 1.722929722*r)*r) - 1/2*ngsolve.log(1 - r + self.eps_n)

