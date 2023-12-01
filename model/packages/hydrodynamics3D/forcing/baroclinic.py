"""
In this file, the forcing components of the baroclinic forcing are collected and returned.
"""

import numpy as np

import ngsolve

from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponents
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters


class HydrodynamicForcingBaroclinic:
    """"
       Class to generate the baroclinic forcing components
    """

    def __init__(self, rho0, nabla_rho0n_V, hydrodynamic_physical_parameters: HydrodynamicPhysicalParameters):
        # forcingsparameters
        self.rho0 = rho0
        self.nabla_rho0n_V = nabla_rho0n_V

        # general parameters
        self.params = hydrodynamic_physical_parameters


    # Core functionality of this class:
    def generate_forcing_components(self, k, n):
        # Create instance of HydrodynamicForcingComponents class and return it
        return HydrodynamicForcingComponents(lambda z, n=n: self.Rcheck1varsigma_V(n, z), lambda z, n=n: self.Rcalcheck1varsigma_V(n, z), hydrophysicalparameters=self.params)



    def A0varsigma(self, n, i):
        """ Parameter used in the baroclinc solution. Frequency component n, index i (=1,2). [unittested] """
        return self.params.Av0 * self.params.alpha0(n, i) * (self.params._coshH(n, i) - 1) + self.params.sf0 * (
                    self.params._sinhH(n, i) - self.params.alpha0(n, i) * self.params.D)


    def c1varsigma(self, n, i, z):
        """ Vertical structure of baroclinic forcing. Frequency component n, index i (=1,2), vertical position z. [unittested] """
        # Test for small alpha0(n, i):
        if np.abs(n * self.params.omega + (-1) ** (i + 1) * self.params.f) < 1e-8:
            c1varsigma = - self.params.g / (6 * self.params.Av0 * self.rho0) * (
                        (z + self.params.H) * (z - self.params.R) * (z - self.params.H - 2 * self.params.R) + (
                            z + self.params.H + 3 * self.params.Av0 / self.params.sf0) * (
                                    self.params.H + self.params.R) ** 2)
        else:
            c1varsigma = -self.params.g / (self.params.Av0 * self.rho0 * self.params.alpha0(n, i) ** 3) * (
                    self.params._sinh(n, i, z) + self.A0varsigma(n, i) * self.params.beta0(n, i) * self.params._cosh(n,
                                                                                                                     i,
                                                                                                                     z) - self.params.alpha0(
                n, i) * (z - self.params.R))
        return c1varsigma



    def C1varsigma(self, n, i, z):
        """ Vertical structure of baroclinic forcing. Frequency component n, index i (=1,2), vertical position z. [unittested] """
        if np.abs(n * self.params.omega + (-1) ** (i + 1) * self.params.f) < 1e-8:
            C1varsigma = - self.params.g / (24 * self.params.Av0 * self.rho0) * (z + self.params.H) * (
                    (z + self.params.H) * (2 * (z - self.params.R) ** 2 - (z + self.params.H) ** 2) + 4 * (
                    z + self.params.H + 3 * self.params.Av0 / self.params.sf0) * self.params.D ** 2)
        else:
            C1varsigma = -self.params.g / (self.params.Av0 * self.rho0 * self.params.alpha0(n, i) ** 3) * (
                    (self.A0varsigma(n, i) * self.params.beta0(n, i) / self.params.alpha0(n, i)) * (
                    self.params._sinh(n, i, z) + self.params._sinhH(n, i)) + 1 / self.params.alpha0(n, i) * (
                            self.params._cosh(n, i, z) - self.params._coshH(n,
                                                                            i)) - self.params.alpha0(
                n, i) / 2 * ((z - self.params.R) ** 2 - self.params.D ** 2))
        return C1varsigma



    ### The matrices ###
    def c1varsigma_M(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return ngsolve.CoefficientFunction((self.c1varsigma(n, 1, z), 0, 0, self.c1varsigma(n, 2, z)), dims=(2, 2))


    def C1varsigma_M(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return ngsolve.CoefficientFunction((self.C1varsigma(n, 1, z), 0, 0, self.C1varsigma(n, 2, z)), dims=(2, 2))


    ### The rotating flow variables ###
    def Rcheck1varsigma_V(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return self.c1varsigma_M(n, z) * self.params.P_H_M() * self.nabla_rho0n_V


    def Rcalcheck1varsigma_V(self, n, z):
        """Order 1, frequency component n, vertical position z. [unittested] """
        return self.C1varsigma_M(n, z) * self.params.P_H_M() * self.nabla_rho0n_V