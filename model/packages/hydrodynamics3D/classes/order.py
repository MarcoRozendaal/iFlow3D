"""
In this file, the results of the kth-order hydrodynamic computation are stored

24-11-2022: Updated this file to include the forcing contributions.
26-11-2022: Removed the 0 from the Z variables as it also works for other orders
6-12-2022:  Added hydrodynamic_forcing_components to keyword arguments
12-12-2022: Added dU_dzz
14-12-2022: Added summation option using 'all'
23-12-2022: Added U_xi and U_eta
12-02-2023: Changed tau to psi
14-02-2023: Added eps, changed m to signed m
17-02-2023: Added vorticity omega (Assuming forced component only)
20-02-2023: Updated definition of W and Wcal
20-02-2023: Changed ang from radians to degrees
24-03-2023: changed H+R to D for readability

# TODO implement Wcheck and Wcalcheck in hydrodynamic_forcing_components
"""

import numpy as np
import ngsolve

from model.general.boundary_fitted_coordinates import BoundaryFittedCoordinates
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.hydrodynamics3D.classes.numerical_parameters import HydrodynamicNumericalParameters
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters



class HydrodynamicsOrder():
    """
    Class containing the kth-order hydrodynamic results
    # Assumptions:
    1.  We neglect the forcing Wcheck and Wcalcheck components for now
    2.  dU_dzz_V and Omega assume forced component only

    Can be used as follows:
        hydrolead.Z[1]['tide']
    We also include an all option:
        hydro_order.Z[n]['all'] sums all forcing mechanisms for a given n
    """

    sum_symbol = "all"
    todegree = 180/np.pi


    def __init__(self, Z_gf_nest_dict, finite_element_space,
                 hydrodynamic_numerical_parameters: HydrodynamicNumericalParameters,
                 hydrodynamic_physical_parameters: HydrodynamicPhysicalParameters,
                 forcing_mechanism_collection: HydrodynamicForcingMechanismCollection):

        # Numerical property
        self.finite_element_space = finite_element_space

        # forcing properties
        self.k = forcing_mechanism_collection.k
        self.freqcomp_list = forcing_mechanism_collection.frequency_component_list
        self.forcing_mechanism_nest_dict = forcing_mechanism_collection.forcing_mechanism_nest_dict

        # kth-order free surface
        self.Z = Z_gf_nest_dict

        # kth-order flow variables
        self._set_hydro_flow_variables(Z_gf_nest_dict,
                                       hydrodynamic_numerical_parameters,
                                       hydrodynamic_physical_parameters,
                                       forcing_mechanism_collection)

        # Sum over all forcing mechanisms for a given frequency component
        self._set_hydro_flow_variables_sum(hydrodynamic_physical_parameters)



    ## Operators working of Z
    def nablahatTZ_M(self, n, symbol):
        # Order kth, frequency component n, forcing mechanism symbol
        Znf = self.Z[n][symbol]
        return ngsolve.CoefficientFunction(
            (ngsolve.Grad(Znf)[0], ngsolve.Grad(Znf)[1], ngsolve.Grad(Znf)[0], -ngsolve.Grad(Znf)[1]), dims=(2, 2))

    def LcalZ_M(self, n, symbol):
        # Order kth, frequency component n, forcing mechanism symbol
        Znf = self.Z[n][symbol]
        HZnf_M = Znf.Operator("hesse")
        nablahatTZnf_M = self.nablahatTZ_M(n, symbol)
        return ngsolve.CoefficientFunction((HZnf_M[0, 0], HZnf_M[0, 1],
                                            HZnf_M[1, 0], HZnf_M[1, 1],
                                            HZnf_M[0, 0], -HZnf_M[0, 1],
                                            -HZnf_M[1, 0], HZnf_M[1, 1],
                                            nablahatTZnf_M[0, 0], nablahatTZnf_M[0, 1],
                                            nablahatTZnf_M[1, 0], nablahatTZnf_M[1, 1]), dims=(6, 2))

    def LaplacianZ(self, n, symbol):
        """ Order kth, frequency component n, forcing mechanism symbol """
        Znf = self.Z[n][symbol]
        HZnf_M = Znf.Operator("hesse")
        return HZnf_M[0, 0] + HZnf_M[1, 1]



    ## Tidal ellipse parameters
    def abs(self, cf):
        """Return the absolute value of cf"""
        return ngsolve.Norm(cf)

    def arg(self, cf):
        """Return the argument of cf in degree"""
        return ngsolve.atan2(cf.imag, cf.real) * self.todegree

    def _M(self, cf_V):
        """Magnitude of the major-axis tidal ellipse"""
        return 1/np.sqrt(2) * (self.abs(cf_V[0]) + self.abs(cf_V[1]))

    def _m(self, cf_V):
        """Signed magnitude of the minor-axis tidal ellipse"""
        return 1/np.sqrt(2) * (self.abs(cf_V[0]) - self.abs(cf_V[1]))

    def _theta(self, cf_V):
        """Orientation of tidal ellipse in degrees"""
        phi_1 = self.arg(cf_V[0])
        phi_2 = - self.arg(cf_V[1])
        return 1 / 2 * (phi_1 + phi_2)

    def _psi(self, cf_V):
        """Phase of tidal ellipse in degrees"""
        phi_1 = self.arg(cf_V[0])
        phi_2 = - self.arg(cf_V[1])
        return -1 / 2 * (phi_1 - phi_2)

    def _eps(self, cf_V):
        """Ellipticity of tidal ellipse"""
        m = self._m(cf_V)
        M = self._M(cf_V)
        return m/M


    ## Setting the leading-order flow variables
    def _set_hydro_flow_variables(self, Z_gf_dict_dict,
                                  hydro_numerical_parameters,
                                  hydrophysparams,
                                  forcing_mechanism_collection):
        """This function sets the kth-order flow variables as dict dicts [unittested]"""

        ## kth-order flow variables
        self.R_V = {n: {} for n in self.freqcomp_list}
        self.U_V = {n: {} for n in self.freqcomp_list}
        self.W = {n: {} for n in self.freqcomp_list}

        # Their components
        self.R_1 = {n: {} for n in self.freqcomp_list}
        self.R_2 = {n: {} for n in self.freqcomp_list}
        self.U = {n: {} for n in self.freqcomp_list}
        self.V = {n: {} for n in self.freqcomp_list}

        ## kth-order depth_num-integrated flow variables
        self.Rcal_V = {n: {} for n in self.freqcomp_list}
        self.q_V = {n: {} for n in self.freqcomp_list}
        self.Wcal = {n: {} for n in self.freqcomp_list}

        # Their components
        self.Rcal_1 = {n: {} for n in self.freqcomp_list}
        self.Rcal_2 = {n: {} for n in self.freqcomp_list}
        self.q_1 = {n: {} for n in self.freqcomp_list}
        self.q_2 = {n: {} for n in self.freqcomp_list}

        ## kth-order depth_num-integrated flow variables evaluated at R
        self.Rcal_R_V = {n: {} for n in self.freqcomp_list}
        self.q_R_V = {n: {} for n in self.freqcomp_list}
        self.Wcal_R = {n: {} for n in self.freqcomp_list}

        # Their components
        self.Rcal_R_1 = {n: {} for n in self.freqcomp_list}
        self.Rcal_R_2 = {n: {} for n in self.freqcomp_list}
        self.q_R_1 = {n: {} for n in self.freqcomp_list}
        self.q_R_2 = {n: {} for n in self.freqcomp_list}

        ## Depth-averaged flow variables
        self.R_DA_V = {n: {} for n in self.freqcomp_list}
        self.U_DA_V = {n: {} for n in self.freqcomp_list}
        self.W_DA = {n: {} for n in self.freqcomp_list}

        # Their components
        self.R_DA_1 = {n: {} for n in self.freqcomp_list}
        self.R_DA_2 = {n: {} for n in self.freqcomp_list}
        self.U_DA = {n: {} for n in self.freqcomp_list}
        self.V_DA = {n: {} for n in self.freqcomp_list}

        # Double vertical derivative of horizontal velocity (Assumed forced component only)
        self.dU_dzz_V = {n: {} for n in self.freqcomp_list}

        ## Vorticity (Assuming forced component only)
        self.Omega = {n: {} for n in self.freqcomp_list}

        ## Tidal ellipse parameters
        self.M = {n: {} for n in self.freqcomp_list}
        self.m = {n: {} for n in self.freqcomp_list}
        self.theta = {n: {} for n in self.freqcomp_list}
        self.psi = {n: {} for n in self.freqcomp_list}
        self.eps = {n: {} for n in self.freqcomp_list}

        ## Tidal ellipse parameters of the depth_num-integrated flow
        self.Mhat = {n: {} for n in self.freqcomp_list}
        self.mhat = {n: {} for n in self.freqcomp_list}
        self.thetahat = {n: {} for n in self.freqcomp_list}
        self.psihat = {n: {} for n in self.freqcomp_list}
        self.epshat = {n: {} for n in self.freqcomp_list}

        ## Tidal ellipse parameters of the depth_num-averaged flow
        self.M_DA = {n: {} for n in self.freqcomp_list}
        self.m_DA = {n: {} for n in self.freqcomp_list}
        self.theta_DA = {n: {} for n in self.freqcomp_list}
        self.psi_DA = {n: {} for n in self.freqcomp_list}
        self.eps_DA = {n: {} for n in self.freqcomp_list}


        for n in self.freqcomp_list:
            for symbol, Znf_gf in Z_gf_dict_dict[n].items():
                # Get the forcing components per n and forcing mechanism f
                forcing_components = forcing_mechanism_collection.forcing_mechanism_nest_dict[n][symbol].hydrodynamic_forcing_components


                # Assuming forced component only
                LcalZnf_M = self.LcalZ_M(n, symbol)
                DeltaZnf = self.LaplacianZ(n, symbol)

                ## Leading-order flow variables
                # The n=n type of arguments of the lambda function is to force an early bind, else python will use late binding
                # and only check the value of e.g. n at run time instead of at the time it is defined
                self.R_V[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf, forcing_components=forcing_components: hydrophysparams.c0_M(n,
                                                                                                                                z) * hydrophysparams.P_H_M() * ngsolve.Grad(
                    Znf_gf) + forcing_components.Rcheck_V(z)
                self.U_V[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf, forcing_components=forcing_components: hydrophysparams.d0_M(n, z) * ngsolve.Grad(
                    Znf_gf) + forcing_components.Ucheck_V(z)
                self.W[n][symbol] = lambda \
                        z, n=n, Znf_gf=Znf_gf, forcing_components=forcing_components: -1 / 2 * (
                        hydrophysparams.C0(n, 1, z) + hydrophysparams.C0(n, 2,
                                                                         z)) * DeltaZnf - hydrophysparams.P_breve_H_V() * hydrophysparams.nablahatC0_M(
                    n, z) * hydrophysparams.P_H_M() * ngsolve.Grad(
                    Znf_gf) + forcing_components.Wcheck(z)



                # Their components
                self.R_1[n][symbol] = lambda z, n=n, symbol=symbol: self.R_V[n][symbol](z)[0]
                self.R_2[n][symbol] = lambda z, n=n, symbol=symbol: self.R_V[n][symbol](z)[1]
                self.U[n][symbol] = lambda z, n=n, symbol=symbol: self.U_V[n][symbol](z)[0]
                self.V[n][symbol] = lambda z, n=n, symbol=symbol: self.U_V[n][symbol](z)[1]


                ## Leading-order depth_num-integrated flow variables
                self.Rcal_V[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf, forcing_components=forcing_components: hydrophysparams.C0_M(n,
                                                                                                                                   z) * hydrophysparams.P_H_M() * ngsolve.Grad(
                    Znf_gf) + forcing_components.Rcalcheck_V(z)
                self.q_V[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf, forcing_components=forcing_components: hydrophysparams.D0_M(n,
                                                                                                                                z) * ngsolve.Grad(
                    Znf_gf) + forcing_components.qcheck_V(z)
                self.Wcal[n][symbol] = lambda \
                        z, n=n, LcalZnf_M=LcalZnf_M, forcing_components=forcing_components: -1 / 2 * (
                        hydrophysparams.int_C0(n, 1, z) + hydrophysparams.int_C0(n, 2,
                                                                                 z)) * DeltaZnf - hydrophysparams.P_breve_H_V() * hydrophysparams.int_nablahatC0_M(
                    n, z) * hydrophysparams.P_H_M() * ngsolve.Grad(
                    Znf_gf) + forcing_components.Wcalcheck(z)

                # Their components
                self.Rcal_1[n][symbol] = lambda z, n=n, symbol=symbol: self.Rcal_V[n][symbol](z)[0]
                self.Rcal_2[n][symbol] = lambda z, n=n, symbol=symbol: self.Rcal_V[n][symbol](z)[1]
                self.q_1[n][symbol] = lambda z, n=n, symbol=symbol: self.q_V[n][symbol](z)[0]
                self.q_2[n][symbol] = lambda z, n=n, symbol=symbol: self.q_V[n][symbol](z)[1]


                ## Leading-order depth_num-integrated flow variables evaluated at z=R
                self.Rcal_R_V[n][symbol] = self.Rcal_V[n][symbol](hydrophysparams.R)
                self.q_R_V[n][symbol] = self.q_V[n][symbol](hydrophysparams.R)
                self.Wcal_R[n][symbol] = self.Wcal[n][symbol](hydrophysparams.R)

                # Their components
                self.Rcal_R_1[n][symbol] = self.Rcal_R_V[n][symbol][0]
                self.Rcal_R_2[n][symbol] = self.Rcal_R_V[n][symbol][1]
                self.q_R_1[n][symbol] = self.q_R_V[n][symbol][0]
                self.q_R_2[n][symbol] = self.q_R_V[n][symbol][1]


                ## Leading-order depth_num-averaged flow variables
                self.R_DA_V[n][symbol] = 1 / hydrophysparams.D * self.Rcal_R_V[n][
                    symbol]
                self.U_DA_V[n][symbol] = 1 / hydrophysparams.D * self.q_R_V[n][symbol]
                self.W_DA[n][symbol] = 1 / hydrophysparams.D * self.Wcal_R[n][symbol]

                # Their components
                self.R_DA_1[n][symbol] = self.R_DA_V[n][symbol][0]
                self.R_DA_2[n][symbol] = self.R_DA_V[n][symbol][1]
                self.U_DA[n][symbol] = self.U_DA_V[n][symbol][0]
                self.V_DA[n][symbol] = self.U_DA_V[n][symbol][1]


                # Double vertical derivative of horizontal velocity (Assumed forced component only)
                self.dU_dzz_V[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf: hydrophysparams.P_M() * hydrophysparams.dc0_dzz_M(n,
                                                                                                                           z) * hydrophysparams.P_H_M() * ngsolve.Grad(
                    Znf_gf)


                # Vorticity (Assuming forced component only)
                self.Omega[n][symbol] = lambda z, n=n, Znf_gf=Znf_gf: -1j / 2 * (
                        hydrophysparams.c0(n, 1, z) - hydrophysparams.c0(n, 2, z)) * DeltaZnf \
                                                                      - 1j * hydrophysparams.P_breve_H_V() * hydrophysparams.nablacheckc0_M(
                    n, z) * self.nablahatTZ_M(n, symbol) * hydrophysparams.P_breve_V()


                # Set tidal ellipse parameters
                self._set_tidal_ellipse_parameters(n, symbol, hydrophysparams)




    def _set_tidal_ellipse_parameters(self, n, symbol, hydrophysparams):
        """
        Function that sets the tidal ellipse parameters for a given n and symbol
        Args:
            n: frequency component
            symbol: symbol of forcing

        """
        ## Tidal ellipse parameters
        self.M[n][symbol] = lambda z, n=n, symbol=symbol: self._M(self.R_V[n][symbol](z))
        self.m[n][symbol] = lambda z, n=n, symbol=symbol: self._m(self.R_V[n][symbol](z))
        self.theta[n][symbol] = lambda z, n=n, symbol=symbol: self._theta(self.R_V[n][symbol](z))
        self.psi[n][symbol] = lambda z, n=n, symbol=symbol: self._psi(self.R_V[n][symbol](z))
        self.eps[n][symbol] = lambda z, n=n, symbol=symbol: self._eps(self.R_V[n][symbol](z))


        ## Tidal ellipse parameters of the depth_num-integrated flow
        self.Mhat[n][symbol] = lambda z, n=n, symbol=symbol: self._M(self.Rcal_V[n][symbol](z))
        self.mhat[n][symbol] = lambda z, n=n, symbol=symbol: self._m(self.Rcal_V[n][symbol](z))
        self.thetahat[n][symbol] = lambda z, n=n, symbol=symbol: self._theta(self.Rcal_V[n][symbol](z))
        self.psihat[n][symbol] = lambda z, n=n, symbol=symbol: self._psi(self.Rcal_V[n][symbol](z))
        self.epshat[n][symbol] = lambda z, n=n, symbol=symbol: self._eps(self.Rcal_V[n][symbol](z))


        ## Tidal ellipse parameters of the depth_num-averaged flow
        self.M_DA[n][symbol] = 1 / hydrophysparams.D * self.Mhat[n][symbol](
            hydrophysparams.R)
        self.m_DA[n][symbol] = 1 / hydrophysparams.D * self.mhat[n][symbol](
            hydrophysparams.R)
        self.theta_DA[n][symbol] = self.thetahat[n][symbol](hydrophysparams.R)
        self.psi_DA[n][symbol] = self.psihat[n][symbol](hydrophysparams.R)
        self.eps_DA[n][symbol] = self.epshat[n][symbol](hydrophysparams.R)




    def _set_hydro_flow_variables_sum(self, hydrophysparams):
        """ Function that extracts the dict_dict variables and creates
            Z[n]["all"] [unittested 1/2]
            The tidal ellipse flow variables are non-linear, i.e., we cannot sum them together, these are computed explicitly afterwards
        """
        tidalellipse_flow_variable_list = ["M", "m", "theta", "psi", "eps",  "Mhat", "mhat", "thetahat", "psihat", "epshat",  "M_DA", "m_DA", "theta_DA", "psi_DA", "eps_DA"]

        def islambda(v):
            LAMBDA = lambda: 0
            return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


        # Creating a shallow copy for iteration, so the original can be modified during the loop
        for name_flow_variable, value in self.__dict__.copy().items():
            # Filter out non flow variables:
            # Check if it is a dictionary and not forcing_mechanism_nest_dict
            if isinstance(value, dict) and value is not self.forcing_mechanism_nest_dict and name_flow_variable not in tidalellipse_flow_variable_list:
                # True flow variable
                # The symbol indices are summed
                for n in self.freqcomp_list:
                    # Two sums: summation for z dependent and not z dependent
                    sum_z = lambda z: ngsolve.CoefficientFunction(0)
                    sum = ngsolve.CoefficientFunction(0)
                    isfunctionofz = False

                    for symbol, obj in value[n].items():
                        # Testing for z dependence:
                        if islambda(obj):
                            isfunctionofz = True
                            # Pass objects that change with each iteration early using keyword arguments
                            sum_z = lambda z, sum_z=sum_z, obj=obj: sum_z(z) + obj(z)
                        else:
                            sum += obj


                    # Set the sum index
                    if isfunctionofz:
                        getattr(self, name_flow_variable)[n][self.sum_symbol] = lambda z, sum_z=sum_z: sum_z(z)
                    else:
                        getattr(self, name_flow_variable)[n][self.sum_symbol] = sum

        # Next, for each nonlinear term the nonlinear operator of the sum is computed
        # i.e, the tidal ellipse parameters are computed using the previously calculated sum
        for n in self.freqcomp_list:
            self._set_tidal_ellipse_parameters(n, self.sum_symbol, hydrophysparams)


    # Extra method to set U_xi U_eta
    def set_U_xi_U_eta(self, bfc: BoundaryFittedCoordinates):
        """
        Function that sets the velocity in the xi and eta direction
        Args:
            bfc: boundary fitted coordinates object
        """
        xi_gf = bfc.xi_gf
        eta_gf = bfc.eta_gf

        # The unit normals in the xi and eta direction
        n_xi = ngsolve.Grad(xi_gf) / ngsolve.Norm(ngsolve.Grad(xi_gf))
        n_eta = ngsolve.Grad(eta_gf) / ngsolve.Norm(ngsolve.Grad(eta_gf))

        # Define flow variables
        self.U_xi = {n: {} for n in self.freqcomp_list}
        self.U_eta = {n: {} for n in self.freqcomp_list}

        self.U_xi_DA = {n: {} for n in self.freqcomp_list}
        self.U_eta_DA = {n: {} for n in self.freqcomp_list}

        # For each frequency component and forcing mechanisms compute U_xi, U_eta
        for n in self.freqcomp_list:
            for symbol in self.Z[n]:
                # U_xi(z) and U_eta(z)
                self.U_xi[n][symbol] = lambda z, n=n, symbol=symbol: ngsolve.InnerProduct(self.U_V[n][symbol](z), n_xi)
                self.U_eta[n][symbol] = lambda z, n=n, symbol=symbol: ngsolve.InnerProduct(self.U_V[n][symbol](z), n_eta)

                # U_xi_DA and U_eta_DA
                self.U_xi_DA[n][symbol] = ngsolve.InnerProduct(self.U_DA_V[n][symbol], n_xi)
                self.U_eta_DA[n][symbol] = ngsolve.InnerProduct(self.U_DA_V[n][symbol], n_eta)
