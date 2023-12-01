"""
In this file, the forcing components of a sediment capacity forcing mechanism are collected.
# TODO test
"""

import ngsolve
import warnings


zero_scalar = ngsolve.CoefficientFunction(0)
zero_scalar_z = lambda z: zero_scalar


class SedimentCapacityForcingComponents():
    """"
    Class for sediment capacity forcing components of a single forcing mechanisms
    """

    def __init__(self, hatC, hatC_DA):
        """
        forcing components of a single forcing mechanism
        Args:
            hatC: The sediment capacity of a certain forcing mechanism
            hatC_DA: The depth_num-averaged sediment capacity
        """
        # Sediment capacity TODO dictionary with indices, since I need k,n,f,alpha?
        self.hatC = hatC

        # Depth-averaged sediment capacity
        self.hatC_DA = hatC_DA



    # Overwrite the print command
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)




class SedimentCapacityForcingComponentsZero(SedimentCapacityForcingComponents):
    """
    Class that initializes the SedimentCapacityForcingComponents class with zero entries only
    """
    def __init__(self):
        super().__init__(zero_scalar_z, hatC_DA=zero_scalar_z)
