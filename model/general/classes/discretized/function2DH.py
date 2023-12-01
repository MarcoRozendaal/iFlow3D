"""
In this file, we define a 2DH numerical function class
"""

import numpy as np
import sympy

import ngsolve

from model.general.classes.numerical.function import NumericalFunction


class NumericalFunction2DH(NumericalFunction):
    """
    Class which represents a numerical function in two horizontal spatial dimensions.
    We store a numpy array and tell python how to handle it.
    """

    def __init__(self, data, mesh):
        """
       Initialization
       Args:
           data: 2DH numpy array
           mesh: ngsolve mesh object
       """
        self.data = data
        self.mesh = mesh



    def __add__(self, other):
        """ method overriding to specify how to add NumericalFunction2DH """
        if isinstance(other, NumericalFunction2DH) and other.mesh == self.mesh:
            return NumericalFunction2DH(self.data + other.data, self.mesh)
        else:
            # Raise an exception if the other object is not the same
            raise TypeError("Unsupported operand type")



    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return
