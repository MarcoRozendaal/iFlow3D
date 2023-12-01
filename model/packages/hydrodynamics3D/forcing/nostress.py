"""
In this file, the forcing components of the no-stress forcing are collected and returned.

"""

import warnings
import numpy as np

import ngsolve


from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponents, conditional_conjugate, conditional_real_part
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder


class HydrodynamicForcingNoStress():
    """"
       Class to generate the no-stress forcing components
    """

    def __init__(self, hydrolead: HydrodynamicsOrder, hydrodynamic_physical_parameters: HydrodynamicPhysicalParameters):
        # general parameters
        self.hydrolead = hydrolead
        self.params = hydrodynamic_physical_parameters



    # Core functionality of this class:
    def generate_forcing_components(self, k, n):
        # Create instance of HydrodynamicForcingComponents class and return it
        return HydrodynamicForcingComponents(lambda z, n=n: self.Rcheck1chi_V(n, z),
                                                     lambda z, n=n: self.Rcalcheck1chi_V(n, z),
                                                     hydrophysicalparameters=self.params)



    def chi1_V(self, n):
        """ Generate chi1_V forcing [unittested]"""

        # Check if the forcing frequency contains 1 and "tide":
        if 1 not in self.hydrolead.freqcomp_list:
            warnings.warn("The no-stress forcing is only implemented when the leading-order hydrodynamics consists of an n=1 component")
        if "tide" not in self.hydrolead.forcing_mechanism_nest_dict[1]:
            warnings.warn("The no-stress forcing is only implemented if for the leading-order amplitude forcing")

        # Check if the requested frequency component is in the allowed range
        if n not in [0, 2]:
            warnings.warn("The no-stress forcing is only implemented to generate a n=0 and a n=2 response")

        return conditional_real_part(
            1 / 2 * conditional_conjugate(self.hydrolead.Z[1]["tide"], n) * self.params.Av0 *
            self.hydrolead.dU_dzz_V[1]["tide"](self.params.R), n)


    def A0chi(self, n, i):
        """ Parameter used in the no-stress solution. Frequency component n, index i (=1,2). [unittested] """
        return self.params.Av0 * self.params.alpha0(n, i) * self.params._coshH(n, i) + self.params.sf0 * self.params._sinhH(n, i)


    def c1chi(self, n, i, z):
        """ Vertical structure of no-stress forcing. Frequency component n, index i (=1,2), vertical position z. [unittested 2/2] """
        # Test for small alpha0(n, i):
        if np.abs(n * self.params.omega + (-1) ** (i + 1) * self.params.f) < 1e-8:
            c1chi = - (z + self.params.H) / self.params.Av0 - 1 / self.params.sf0
        else:
            c1chi = - 1 / (self.params.Av0 * self.params.alpha0(n, i)) * (
                    self.params._sinh(n, i, z) + self.A0chi(n, i) * self.params.beta0(n, i) * self.params._cosh(n, i,
                                                                                                                z))
        return c1chi


    def C1chi(self, n, i, z):
        """ Vertical structure of integrated no-stress forcing. Frequency component n, index i (=1,2), vertical position z. [unittested 2/2] """
        if np.abs(n * self.params.omega + (-1) ** (i + 1) * self.params.f) < 1e-8:
            C1chi = -(z + self.params.H) * ((z + self.params.H) / (2 * self.params.Av0) + 1 / self.params.sf0)
        else:
            C1chi = - 1 / (self.params.Av0 * self.params.alpha0(n, i) ** 2) * (
                    self.A0chi(n, i) * self.params.beta0(n, i) * (
                    self.params._sinh(n, i, z) + self.params._sinhH(n, i)) +
                    self.params._cosh(n, i, z) - self.params._coshH(n, i))
        return C1chi



    ### The matrices ###
    def c1chi_M(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return ngsolve.CoefficientFunction((self.c1chi(n, 1, z), 0, 0, self.c1chi(n, 2, z)), dims=(2, 2))


    def C1chi_M(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return ngsolve.CoefficientFunction((self.C1chi(n, 1, z), 0, 0, self.C1chi(n, 2, z)), dims=(2, 2))


    ### The rotating flow variables ###
    def Rcheck1chi_V(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return self.c1chi_M(n, z) * self.params.P_H_M() * self.chi1_V(n)


    def Rcalcheck1chi_V(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return self.C1chi_M(n, z) * self.params.P_H_M() * self.chi1_V(n)
