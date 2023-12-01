"""
In this file, the forcing components of a hydrodynamic forcing mechanism are collected.
Note: the Wcheck and Wcalcheck are not implemented properly yet.

# TODO implement Wcheck and Wcalcheck
"""
import ngsolve
import warnings


zero_vector = ngsolve.CoefficientFunction((0, 0), dims=(2, 1))
zero_vector_z = lambda z: zero_vector
zero_scalar = ngsolve.CoefficientFunction(0)


class HydrodynamicForcingComponents():
    """"
    Class for hydrodynamic forcing components of a single forcing mechanisms
    """

    def __init__(self, Rcheck_V, Rcalcheck_V, hydrophysicalparameters, qcheck_R_V=None):
        """
        forcing components of a single forcing mechanism
        Args:
            Rcheck_V: The forcing component of R
            Rcalcheck_V: The forcing component of Rcal
            hydrophysicalparameters: A hydrodynamic physical parameter of object
            qcheck_R_V: The forcing component of q evaluated at z=R
        """

        self.Rcheck_V = Rcheck_V
        self.Rcalcheck_V = Rcalcheck_V


        self.Ucheck_V = lambda z: hydrophysicalparameters.P_M() * Rcheck_V(z)
        self.qcheck_V = lambda z: hydrophysicalparameters.P_M() * Rcalcheck_V(z)

        if qcheck_R_V is None:
            self.qcheck_R_V = self.qcheck_V(hydrophysicalparameters.R)
        else:
            self.qcheck_R_V = qcheck_R_V

        # TODO implement these quantities
        def Wcheck(z):
            """Raise warning to notify the user that the specified method is not ready yet"""
            warnings.warn("Wcheck is not implemented yet. For simplicity, it is assumed to be zero.")
            return zero_scalar

        def Wcalcheck(z):
            """Raise warning to notify the user that the specified method is not ready yet"""
            warnings.warn("Wcalcheck is not implemented yet. For simplicity, it is assumed to be zero.")
            return zero_scalar

        self.Wcheck = Wcheck
        self.Wcalcheck = Wcalcheck


    # Overwrite the print command
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)



class PhysicalParamsMockup():
    """Simple class that contains the method P_M"""
    def P_M(self):
        return ngsolve.CoefficientFunction((0, 0, 0, 0), dims=(2, 2))


class HydrodynamicForcingComponentsZero(HydrodynamicForcingComponents):
    """
    Class that initializes the HydrodynamicForcingComponents class with zeros only
    """
    def __init__(self, qcheck_R_V=zero_vector):
        physical_params = PhysicalParamsMockup()
        super().__init__(zero_vector_z, zero_vector_z, physical_params, qcheck_R_V=qcheck_R_V)



#### Help functions for forcing components #####
def conditional_conjugate(func, n):
    """ For n=0: take the conjugate of func
        For n=2: return func
    """
    if n == 0:
        return ngsolve.Conj(func)
    elif n == 2:
        return func
    else:
        raise Exception("Conditional conjugate not defined for n="+str(n))


def conditional_real_part(forcing, n):
    """ For n=0, take the real part of the forcing before the computation.
    Explanation:
    For n=0, the coefficients of the linear equations in the Fourier domain are real.
    Taking the real part on both sides of the linear equations then shows that only the real part of the forcing
    generates a real response, which is the part we are after for n=0.
    """
    if n == 0:
        # Compute the real part of forcing
        return (forcing + ngsolve.Conj(forcing))/2
    else:
        return forcing

