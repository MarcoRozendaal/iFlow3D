"""
In this file, we define a 3D numerical function class
"""

import numpy as np
import sympy

import ngsolve

from model.general.classes.numerical.function import NumericalFunction


class NumericalFunction3D(NumericalFunction):
    """
    Class numerical function in three spatial dimensions.
    We store a numpy array and tell python how to handle it.
    """

    def __init__(self, data, mesh):
        """
        Initialization
        Args:
            data: 3D numpy array
            mesh: ngsolve mesh object
        """
        self.data = data
        self.mesh = mesh



    def __add__(self, other):
        """ method overriding to specify how to add NumericalFunction3D """
        if isinstance(other, NumericalFunction3D) and other.mesh == self.mesh:
            return NumericalFunction3D(self.data + other.data, self.mesh)
        else:
            # Raise an exception if the other object is not the same object
            raise TypeError("Unsupported operand type for +")


    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return

