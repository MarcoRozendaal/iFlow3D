"""
In this file, a single hydrodynamic forcing mechanism class is created

15-12-2022: Made useonboundary default to true and adjusted code to reflect this
07-09-2023: Changed sign of q_R_river, to better reflect the direction of the river flow
"""
import ngsolve

from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponentsZero


class HydrodynamicForcingMechanism():
    """
    Class for single hydrodynamic forcing mechanisms

    Added flag for use of forcing on boundary
    """

    def __init__(self, k, n, symbol, hydrodynamic_forcing=None,
                 A_sea=0, q_R_walldown=0, q_R_wallup=0, q_R_river=0, useonboundary=True):
        """
        A single forcing mechanism
        Args:
             k: order of the forcing mechanism
             n: frequency component of the forcing
             symbol:  shorthand symbol to denote the forcing mechanism, e.g., "A" for amplitude forcing at the seaward boundary
        Keyword Args (all initialized to zero):
            hydrodynamic_forcing: object to generate hydrodynamic forcing components
            A_sea: Amplitude forcing at the seaward boundary
            q_R_walldown: transport of water forcing at the walldown boundary
            q_R_wallup: transport of water forcing at the wallup boundary
            q_R_river: transport of water forcing at the river boundary. Note: here positive implies into the domain and
                                                                                negative out of the domain.
            useonboundary: Flag to indicate if forcing components should also be evaluated at the Neumann boundaries
        """
        # Indices
        self.k = k
        self.n = n
        self.symbol = symbol

        if hydrodynamic_forcing is not None:
            self.hydrodynamic_forcing_components = hydrodynamic_forcing.generate_forcing_components(k, n)
        else:
            # Hydrodynamic forcing components
            self.hydrodynamic_forcing_components = HydrodynamicForcingComponentsZero()


        # forcing mechanisms at the water column
        self.q_R_body_V = self.hydrodynamic_forcing_components.qcheck_R_V

        # forcing mechanisms at the boundaries
        self.A_sea = A_sea
        self.q_R_walldown = q_R_walldown
        self.q_R_wallup = q_R_wallup
        self.q_R_river = -q_R_river # Note the minus sign here, reflecting that the river typically flows into the domain

        # Also use q_R_body_V at the boundaries, if useonboundary:
        if useonboundary:
            self.q_R_walldown += ngsolve.InnerProduct(self.q_R_body_V, ngsolve.specialcf.normal(2))
            self.q_R_wallup += ngsolve.InnerProduct(self.q_R_body_V, ngsolve.specialcf.normal(2))
            self.q_R_river += ngsolve.InnerProduct(self.q_R_body_V, ngsolve.specialcf.normal(2))


    # Overwrite the print command
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


