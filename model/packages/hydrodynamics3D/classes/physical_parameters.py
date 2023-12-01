"""
In this file, the (leading-order) physical parameters are collected in a single class for the hydrodynamics.
The parameters that depend on these physical parameters are also included.

# TODO maybe in future:
TODO: Maybe move the definitions of in this file to a separate file in hydrolead. And call it something like
TODO    forced_response / forced_hydrodynamics?

TODO: Maybe remove all the functions/methods from this object and put them into a dedicated object,
TODO        maybe into a hydrodynamic_forcing_components leading order tide file. Maybe I do not know
"""

import ngsolve
import numpy as np


class HydrodynamicPhysicalParameters():
    """
    Class containing the parameters occurring in the leading-order equations and boundary conditions at the free surface
    and bed.

    NGSolve handles the horizontal (x,y) grid, and we manually handle the vertical z dimension
    """

    def __init__(self, g, f, omega, Av0_sp, sf0_sp, H_sp, R_sp):
        # The spatially constant parameters
        self.g = g
        self.f = f
        self.omega = omega

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

    #-------------------------------------------------------------------------------------------------------------------#
    # Help functions to define the physical parameters

    def _sinh(self, n, i, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, index i (=1,2), vertical position z """
        return ngsolve.sinh(self.alpha0(n, i) * (z - self.R))

    def _cosh(self, n, i, z):
        """ Shortcut for a frequently used vertical function. Frequency component n, index i (=1,2), vertical position z """
        return ngsolve.cosh(self.alpha0(n, i) * (z - self.R))

    def _sinhH(self, n, i):
        """ Shortcut for a frequently used function. Frequency component n, index i (=1,2) """
        return ngsolve.sinh(self.alpha0(n, i) * self.D)

    def _coshH(self, n, i):
        """ Shortcut for a frequently used function. Frequency component n, index i (=1,2) """
        return ngsolve.cosh(self.alpha0(n, i) * self.D)

    # The vector with the horizontally variable parameters
    def phi0_V(self):
        return ngsolve.CoefficientFunction((self.Av0, self.sf0, self.H, self.R), dims=(4, 1))

    # ------------------------------------------------------------------------------------------------------------------#
    # The physical coefficients, vectors and matrices needed to construct Z, R_V, U_V, Rcal_V, q_V, R_DA_V, U_DA_V

    # The leading-order spatially dependent coefficients (or scalars)
    def alpha0(self, n, i):
        """Order 0, frequency component n, matrix index i (for i = 1,2, not zero-based)"""
        return ngsolve.sqrt(1j * (n * self.omega + (-1) ** (i + 1) * self.f) / self.Av0)

    def beta0(self, n, i):
        """Order 0, frequency component n, matrix index i (for i = 1,2, not zero-based)"""
        return 1 / (self.Av0 * self.alpha0(n, i) * self._sinhH(n, i) + self.sf0 * self._coshH(n, i))

    def c0(self, n, i, z):
        """Order 0, frequency component n, matrix index i (for i = 1,2, not zero-based), vertical position z"""
        # Test for small alpha0(n, i):
        if np.abs(n * self.omega + (-1) ** (i + 1) * self.f) < 1e-8:
            c0 = self.g / (2 * self.Av0) * (
                        (z - self.R) ** 2 - (self.D + 2 * self.Av0 / self.sf0) * self.D)
        else:
            c0 = self.g / (self.Av0 * self.alpha0(n, i) ** 2) * (
                self.sf0 * self.beta0(n, i) * self._cosh(n, i, z) - 1)
        return c0

    def C0(self, n, i, z):
        """Order 0, frequency component n, matrix index i (for i = 1,2, not zero-based), vertical position z"""
        # Test for small alpha0(n, i):
        if np.abs(n * self.omega + (-1) ** (i + 1) * self.f) < 1e-8:
            C0 = self.g / (6 * self.Av0) * (
                        (z - self.R) ** 3 + self.D ** 3 - 3 * (self.D + 2 * self.Av0 / self.sf0) * self.D * (z + self.H))
        else:
            C0 = self.g / (self.Av0 * self.alpha0(n, i) ** 2) * (self.sf0 * self.beta0(n, i) / self.alpha0(n, i) * (
                    self._sinh(n, i, z) + self._sinhH(n, i)) - (z + self.H))
        return C0

    # The leading-order constant matrices (denoted with the _M)
    def P_M(self):
        return 1 / np.sqrt(2) * ngsolve.CoefficientFunction((1, 1, -1j, 1j), dims=(2, 2))

    def P_H_M(self):
        return 1 / np.sqrt(2) * ngsolve.CoefficientFunction((1, 1j, 1, -1j), dims=(2, 2))

    # The leading-order spatially dependent coefficient matrices (denoted with the _M)
    def c0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        return ngsolve.CoefficientFunction((self.c0(n, 1, z), 0, 0, self.c0(n, 2, z)), dims=(2, 2))

    def C0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        return ngsolve.CoefficientFunction((self.C0(n, 1, z), 0, 0, self.C0(n, 2, z)), dims=(2, 2))

    def C0_R_M(self, n):
        """Order 0, frequency component n, evaluated at z=R"""
        return self.C0_M(n, self.R)

    # The leading-order spatially dependent composite coefficient matrices (denoted with the _M)
    def d0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        return self.P_M() * self.c0_M(n, z) * self.P_H_M()

    def D0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        return self.P_M() * self.C0_M(n, z) * self.P_H_M()

    def D0_R_M(self, n):
        """Order 0, frequency component n, evaluated at z=R"""
        return self.D0_M(n, self.R)

    # ------------------------------------------------------------------------------------------------------------------#
    # The physical coefficients, vectors and matrices needed to construct W, Wcal, W_DA

    # The constant vectors (the vector part is denoted with _V)
    def P_breve_V(self):
        return 1 / np.sqrt(2) * ngsolve.CoefficientFunction((1, 1j), dims=(2, 1))

    def P_breve_H_V(self):
        return 1 / np.sqrt(2) * ngsolve.CoefficientFunction((1, -1j), dims=(1, 2))


    # The vector with the components of the gradient of the leading-order spatially variable parameters
    def partialphi0_partialxj_V(self, j):
        """Partial derivative of phi0 with respect to x_j for j=1,2"""
        component = j - 1
        return ngsolve.CoefficientFunction(
            (self.Gradient_Av[component], self.Gradient_sf[component], self.Gradient_H[component],
             self.Gradient_R[component]), dims=(4, 1))

    # ..................................................................................................................#
    # The leading-order spatially dependent derivative matrices to construct dC0_dxj, nablahatC0_M, KcalC0_M

    def partialC0_partialphi0_V(self, n, i, z):
        """Order 0, frequency component n, index i, vertical position z"""
        # Derivatives of C0 with respect to phi0:
        partialC0_partialAv0 = -1 / self.Av0 * self.C0(n, i, z)
        partialC0_partialsf0 = self.g * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i) ** 3) * (
                self._sinh(n, i, z) + self._sinhH(n, i))
        partialC0_partialH = self.g / (self.Av0 * self.alpha0(n, i) ** 2) * (self.sf0 * self.beta0(n, i) *
                                                                             self._coshH(n, i) - 1)
        partialC0_partialR = self.g * self.sf0 * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i) ** 2) * (
                -self._cosh(n, i, z) + self._coshH(n, i))
        return ngsolve.CoefficientFunction(
            (partialC0_partialAv0, partialC0_partialsf0, partialC0_partialH, partialC0_partialR), dims=(1, 4))


    def partialC0_partialvartheta0_V(self, n, i, z):
        """Order 0, frequency component n, index i, vertical position z"""
        # Derivatives of C0 with respect to vartheta0:
        partialC0_partialalpha0 = -2 / self.alpha0(n, i) * self.C0(n, i, z) + self.g * self.sf0 * self.beta0(n, i) / (
                self.Av0 * self.alpha0(n, i) ** 3) * (-1 / self.alpha0(n, i) * (
                self._sinh(n, i, z) + self._sinhH(n, i)) + (z - self.R) * self._cosh(n, i, z) + self.D * self._coshH(n, i))
        partialC0_partialbeta0 = self.g * self.sf0 / (self.Av0 * self.alpha0(n, i) ** 3) * (
                self._sinh(n, i, z) + self._sinhH(n, i))
        return ngsolve.CoefficientFunction((partialC0_partialalpha0, partialC0_partialbeta0), dims=(1, 2))


    def dvartheta0_dphi0_M(self, n, i):
        """Order 0, frequency component n, index i"""
        # Derivatives of alpha0 with respect to phi0
        partialalpha0_partialAv0 = -1 / (2 * self.Av0) * self.alpha0(n, i)
        partialalpha0_partialsf0 = 0
        partialalpha0_partialH = 0
        partialalpha0_partialR = 0

        # Chainrule for dbeta0_dAv
        partialbeta0_partialAv0 = -self.alpha0(n, i) * self.beta0(n, i) ** 2 * self._sinhH(n, i)
        partialbeta0_partialalpha0 = -self.beta0(n, i) ** 2 * (self.Av0 * (
                self._sinhH(n, i) + self.D * self.alpha0(n, i) * self._coshH(n, i)) + self.sf0 * self.D * self._sinhH(n,
                                                                                                                      i))

        # Derivatives of beta0 with respect to phi0
        dbeta0_dAv0 = partialbeta0_partialAv0 + partialbeta0_partialalpha0 * partialalpha0_partialAv0
        partialbeta0_partialsf0 = - self.beta0(n, i) ** 2 * self._coshH(n, i)
        partialbeta0_partialH = - self.alpha0(n, i) * self.beta0(n, i) ** 2 * (
                self.Av0 * self.alpha0(n, i) * self._coshH(n, i) + self.sf0 * self._sinhH(n, i))
        partialbeta0_partialR = partialbeta0_partialH

        return ngsolve.CoefficientFunction((partialalpha0_partialAv0, partialalpha0_partialsf0, partialalpha0_partialH,
                                            partialalpha0_partialR, dbeta0_dAv0, partialbeta0_partialsf0,
                                            partialbeta0_partialH, partialbeta0_partialR), dims=(2, 4))


    # The above matrices combine to give the derivative of C0 with respect to x_j:
    def dC0_dxj(self, n, i, j, z):
        """Order 0, frequency component n, matrix index i (i=1,2), derivative to x_j (j=1,2), vertical position z"""
        dC0_dphi0_V = self.partialC0_partialphi0_V(n, i, z) + self.partialC0_partialvartheta0_V(n, i,
                                                                                          z) * self.dvartheta0_dphi0_M(
            n, i)
        return dC0_dphi0_V * self.partialphi0_partialxj_V(j)

    def nablahatC0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        dC0dx_11 = self.dC0_dxj(n, 1, 1, z)
        dC0dx_22 = self.dC0_dxj(n, 2, 1, z)
        dC0dy_11 = self.dC0_dxj(n, 1, 2, z)
        dC0dy_22 = self.dC0_dxj(n, 2, 2, z)
        return ngsolve.CoefficientFunction((dC0dx_11, dC0dx_22, dC0dy_11, -dC0dy_22), dims=(2, 2))

    def KcalC0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        C0_11 = self.C0(n, 1, z)
        C0_22 = self.C0(n, 2, z)
        nablahatC0_M = self.nablahatC0_M(n, z)
        return ngsolve.CoefficientFunction((C0_11, 0, C0_22, 0, nablahatC0_M[0, 0], nablahatC0_M[0, 1],
                                            0, C0_11, 0, C0_22, nablahatC0_M[1, 0], nablahatC0_M[1, 1]), dims=(2, 6))


    # ..................................................................................................................#
    # The leading-order spatially dependent derivative matrices to construct int_dC0_dxj

    def int_C0(self, n, i, z):
        """Order 0, frequency component n, matrix index i (i=1,2), vertical position z"""
        return self.g / (self.Av0 * self.alpha0(n, i) ** 2) * (self.sf0 * self.beta0(n, i) / self.alpha0(n, i) * (
                1 / self.alpha0(n, i) * (self._cosh(n, i, z) - self._coshH(n, i)) + (z + self.H) * self._sinhH(n,
                                                                                                               i)) - 1 / 2 * (
                                                                       z + self.H) ** 2)

    def int_partialC0_partialphi0_V(self, n, i, z):
        """Order 0, frequency component n, index i (i=1,2), vertical position z"""
        # Derivatives of C0 with respect to phi0:
        int_partialC0_partialAv0 = -1 / self.Av0 * self.int_C0(n, i, z)
        int_partialC0_partialsf0 = self.g * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i) ** 3) * (
                1 / self.alpha0(n, i) * (self._cosh(n, i, z) - self._coshH(n, i)) + (z + self.H) * self._sinhH(n, i))
        int_partialC0_partialH = self.g / (self.Av0 * self.alpha0(n, i) ** 2) * (z + self.H) * (
                self.sf0 * self.beta0(n, i) *
                self._coshH(n, i) - 1)
        int_partialC0_partialR = self.g * self.sf0 * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i) ** 2) * (
                -1 / self.alpha0(n, i) * (self._sinh(n, i, z) + self._sinhH(n, i)) + (z + self.H) * self._coshH(n, i))
        return ngsolve.CoefficientFunction(
            (int_partialC0_partialAv0, int_partialC0_partialsf0, int_partialC0_partialH, int_partialC0_partialR),
            dims=(1, 4))

    def int_partialC0_partialvartheta0_V(self, n, i, z):
        """Order 0, frequency component n, index i, vertical position z"""
        # Derivatives of C0 with respect to vartheta0:
        int_partialC0_partialalpha0 = -2 / self.alpha0(n, i) * self.int_C0(n, i, z) + self.g * self.sf0 * self.beta0(n, i) / (
                self.Av0 * self.alpha0(n, i) ** 3) * (-2 / self.alpha0(n, i) ** 2  * (self._cosh(n,i,z)-self._coshH(n,i)) + 1/self.alpha0(n,i) *
                                                      ((z-self.R)*self._sinh(n, i, z) - (z+self.H + self.D)*self._sinhH(n, i)) + (z + self.H) * self.D * self._coshH(n, i))
        int_partialC0_partialbeta0 = self.g * self.sf0 / (self.Av0 * self.alpha0(n, i) ** 3) * (
                1/self.alpha0(n,i)*(self._cosh(n, i, z) - self._coshH(n,i)) + (z + self.H)*self._sinhH(n, i))
        return ngsolve.CoefficientFunction((int_partialC0_partialalpha0, int_partialC0_partialbeta0), dims=(1, 2))


    def int_dC0_dxj(self, n, i, j, z):
        """Order 0, frequency component n, matrix index i (i=1,2), derivative to x_j (j=1,2), vertical position z"""
        int_dC0_dphi0_V = self.int_partialC0_partialphi0_V(n, i, z) + self.int_partialC0_partialvartheta0_V(n, i,
                                                                                          z) * self.dvartheta0_dphi0_M(
            n, i)
        return int_dC0_dphi0_V * self.partialphi0_partialxj_V(j)

    def int_nablahatC0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        int_dC0dx_11 = self.int_dC0_dxj(n, 1, 1, z)
        int_dC0dx_22 = self.int_dC0_dxj(n, 2, 1, z)
        int_dC0dy_11 = self.int_dC0_dxj(n, 1, 2, z)
        int_dC0dy_22 = self.int_dC0_dxj(n, 2, 2, z)
        return ngsolve.CoefficientFunction((int_dC0dx_11, int_dC0dx_22, int_dC0dy_11, -int_dC0dy_22), dims=(2, 2))

    def KcalhatC0_M(self, n, z):
        """Order 0, frequency component n, vertical position z"""
        int_C0_11 = self.int_C0(n, 1, z)
        int_C0_22 = self.int_C0(n, 2, z)
        int_nablaC0_M = self.int_nablahatC0_M(n, z)
        return ngsolve.CoefficientFunction((int_C0_11, 0, int_C0_22, 0, int_nablaC0_M[0, 0], int_nablaC0_M[0, 1],
                                            0, int_C0_11, 0, int_C0_22, int_nablaC0_M[1, 0], int_nablaC0_M[1, 1]), dims=(2, 6))

    def KcalhatC0_R_M(self, n):
        """Order 0, frequency component n, evaluated at z=R"""
        return self.KcalhatC0_M(n, self.R)

    # ..................................................................................................................#
    # The leading-order spatially dependent derivative matrices to construct dc0_dxj

    def partialc0_partialphi0_V(self, n, i, z):
        """Order 0, frequency component n, index i, vertical position z"""
        # Derivatives of c0 with respect to phi0:
        partialc0_partialAv0 = -1 / self.Av0 * self.c0(n, i, z)
        partialc0_partialsf0 = self.g * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i) ** 2) * self._cosh(n, i, z)
        partialc0_partialH = 0
        partialc0_partialR = -self.g * self.sf0 * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i)) * self._sinh(n, i, z)
        return ngsolve.CoefficientFunction(
            (partialc0_partialAv0, partialc0_partialsf0, partialc0_partialH, partialc0_partialR), dims=(1, 4))

    def partialc0_partialvartheta0_V(self, n, i, z):
        """Order 0, frequency component n, index i, vertical position z"""
        # Derivatives of c0 with respect to vartheta0:
        partialc0_partialalpha0 = -2 / self.alpha0(n, i) * self.c0(n, i, z) + self.g * self.sf0 * self.beta0(n, i) / (
                self.Av0 * self.alpha0(n, i) ** 2) * (z - self.R) * self._sinh(n, i, z)
        partialc0_partialbeta0 = self.g * self.sf0 / (self.Av0 * self.alpha0(n, i) ** 2) * self._cosh(n, i, z)
        return ngsolve.CoefficientFunction((partialc0_partialalpha0, partialc0_partialbeta0), dims=(1, 2))

    def dc0_dxj(self, n, i, j, z):
        """Order 0, frequency component n, matrix index i (i=1,2), derivative to x_j (j=1,2), vertical position z"""
        dc0_dphi0_V = self.partialc0_partialphi0_V(n, i, z) + self.partialc0_partialvartheta0_V(n, i,
                                                                                                z) * self.dvartheta0_dphi0_M(
            n, i)
        return dc0_dphi0_V * self.partialphi0_partialxj_V(j)


    # The leading-order spatially dependent nablacheckc matrix used to construct the vorticity
    def nablacheckc0_M(self, n, z):
        """Order 0, frequency component n, vertical position z. [unittested]"""
        dc0dx_11 = self.dc0_dxj(n, 1, 1, z)
        dc0dx_22 = self.dc0_dxj(n, 2, 1, z)
        dc0dy_11 = self.dc0_dxj(n, 1, 2, z)
        dc0dy_22 = self.dc0_dxj(n, 2, 2, z)
        return ngsolve.CoefficientFunction((dc0dx_11, -dc0dx_22, dc0dy_11, dc0dy_22), dims=(2, 2))


    # ..................................................................................................................#
    # The leading-order spatially dependent vertical derivative matrices to construct u_z and u_zz
    def dc0_dz(self, n, i, z):
        """Order 0, frequency component n, matrix index i (i=1,2), vertical position z. [unittested]"""
        return self.g * self.sf0 * self.beta0(n, i) / (self.Av0 * self.alpha0(n, i)) * self._sinh(n, i, z)

    def dc0_dzz(self, n, i, z):
        """Order 0, frequency component n, matrix index i (i=1,2), vertical position z. [unittested]"""
        return self.g * self.sf0 * self.beta0(n, i) / self.Av0 * self._cosh(n, i, z)

    def dc0_dzz_M(self, n, z):
        """Order 0, frequency component n, vertical position z. [unittested]"""
        return ngsolve.CoefficientFunction((self.dc0_dzz(n, 1, z), 0, 0, self.dc0_dzz(n, 2, z)), dims=(2, 2))



