"""
In this file, a single sediment capacity forcing mechanism class is created
"""

import ngsolve

from model.packages.sediment3D.classes.capacity_forcing_components import SedimentCapacityForcingComponentsZero


class SedimentCapacityForcingMechanism():
    """
    Class for single sediment capacity forcing mechanisms
    """

    def __init__(self, k, n, symbol, alpha, sediment_capacity_forcing=None):
        """
        A single forcing mechanism
        Args:
             k: order of the forcing mechanism
             n: frequency component of the forcing
             symbol: shorthand symbol to denote the forcing mechanism
             alpha: how the forcing mechanism scales with Phi(a). Can be a, a_x or a_y.
        Keyword Args (initialized to None):
            sediment_capacity_forcing: instance of SedimentCapacityForcingComponentsZero class related to the specific forcing mechanism
        """
        # Indices
        self.k = k
        self.n = n
        self.symbol = symbol
        self.alpha = alpha
        self.sediment_capacity_forcing = sediment_capacity_forcing


        # Sediment capacity forcing components
        if sediment_capacity_forcing is not None:
            self.sediment_capacity_forcing_components = sediment_capacity_forcing.generate_forcing_components(k, n, symbol, alpha)
        else:
            self.sediment_capacity_forcing_components = SedimentCapacityForcingComponentsZero()



    # Overwrite the print command
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


