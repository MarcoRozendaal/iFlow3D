"""
In this file, the (leading-order) physical parameters are collected in a single class for the sediment dynamics.
The parameters that depend on these physical parameters are also included.
"""



import ngsolve
import numpy as np


class SedimentPhysicalParameters():
    """
    Class containing the parameters occurring in the leading-order equations and boundary conditions at the free surface
    and bed.

    NGSolve handles the horizontal (x,y) grid, and we manually handle the vertical z dimension
    """

    def __init__(self, g, f, omega, rho0, Kv, Kh, w_s, M, Av0_sp, sf0_sp, H_sp, R_sp):
        # The spatially constant parameters
        self.g = g
        self.f = f
        self.omega = omega
        self.rho0 = rho0
        self.Kv = Kv
        self.Kh = Kh
        self.w_s = w_s
        self.M = M

        # The horizontally variable parameters
        # We omit cf for brevity in the later expressions
        self.Av0 = Av0_sp.cf
        self.sf0 = sf0_sp.cf
        self.H = H_sp.cf
        self.R = R_sp.cf
        self.D = H_sp.cf + R_sp.cf

        # The gradients of the horizontally variable parameters
        self.Gradient_Av = Av0_sp.gradient_cf
        self.Gradient_sf = sf0_sp.gradient_cf
        self.Gradient_H = H_sp.gradient_cf
        self.Gradient_R = R_sp.gradient_cf
        self.Gradient_D = H_sp.gradient_cf + R_sp.gradient_cf


        # Combination of physical parameters
        self.sigma = self.w_s / (2 * self.Kv)

    #-------------------------------------------------------------------------------------------------------------------#
    # Help functions to define the physical parameters

    # z-dependent functions defined relative to z=R
    def _sinhzR(self, n, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, vertical position z """
        return ngsolve.sinh(self.lambda0(n) * (z - self.R))

    def _coshzR(self, n, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, vertical position z """
        return ngsolve.cosh(self.lambda0(n) * (z - self.R))

    def _expzR(self, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, vertical position z """
        return ngsolve.exp(-self.sigma * (z - self.R))

    # z-dependent functions defined relative to z=-H
    def _sinhzH(self, n, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, vertical position z """
        return ngsolve.sinh(self.lambda0(n) * (z + self.H))

    def _coshzH(self, n, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, vertical position z """
        return ngsolve.cosh(self.lambda0(n) * (z + self.H))

    def _expzH(self, z):
        """ Shortcut for a frequently used vertical function. Vertical position z """
        return ngsolve.exp(-self.sigma * (z + self.H))

    # Functions dependent on the depth_num D
    def _sinhD(self, n):
        """ Shortcut for a frequently used function. Frequency component n """
        return ngsolve.sinh(self.lambda0(n) * self.D)

    def _coshD(self, n):
        """ Shortcut for a frequently used function. Frequency component n """
        return ngsolve.cosh(self.lambda0(n) * self.D)

    # The vector with the horizontally variable parameters
    def phi0_V(self):
        return ngsolve.CoefficientFunction((self.Av0, self.sf0, self.H, self.R), dims=(4, 1))

    # ------------------------------------------------------------------------------------------------------------------#
    # The physical coefficients
    def lambda0(self, n):
        """ Order 0, frequency component n """
        return 1 / (2 * self.Kv) * ngsolve.sqrt(self.w_s ** 2 + 4 * n * 1j * self.omega * self.Kv)

    def K(self, n):
        """ Frequency component n """
        return 1 / (self.Kv*(self.lambda0(n) ** 2 + self.sigma ** 2) * self._sinhD(
            n) + self.w_s * self.lambda0(n) * self._coshD(n))
