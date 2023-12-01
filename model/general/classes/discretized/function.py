"""
In this file, we define a general numerical function class
# TODO test
"""

import numpy as np
import sympy

import ngsolve

# TODO maybe use np.ndarray as super class


class DiscretizedFunction:
    """
    Class which represents a numerical function that is based on a numpy array
    We store a numpy array and tell python how to handle it.
    """

    # TODO do we want this?
    # __array_ufunc__ = None

    def __init__(self, data, mesh):
        """
       Initialization
       Args:
           data: numpy array
           mesh: ngsolve mesh object
       """
        self.data = data
        self.mesh = mesh

    def conjugate(self):
        """ How to take the complex conjugate of NumericalFunctions """
        return DiscretizedFunction(np.conj(self.data), self.mesh)

    def __getitem__(self, item):
        """ How to get an item of NumericalFunctions """
        return DiscretizedFunction(self.data[item], self.mesh)


    def __add__(self, other):
        """ How to add NumericalFunctions """
        if self._isinstance_and_samemesh(other):
            # Perform addition and return a new object
            return DiscretizedFunction(self.data + other.data, self.mesh)
        elif isinstance(other, (int, float)):
            return DiscretizedFunction(self.data + other, self.mesh)
        else:
            # Raise an exception if the other object is not
            raise TypeError("Unsupported operand type")

    def __sub__(self, other):
        """ How to subtract NumericalFunctions """
        if self._isinstance_and_samemesh(other):
            return DiscretizedFunction(self.data - other.data, self.mesh)
        elif isinstance(other, (int, float)):
            return DiscretizedFunction(self.data - other, self.mesh)
        elif isinstance(other, np.ndarray):
            return DiscretizedFunction(self.data - other, self.mesh)
        else:
            raise ValueError("Unsupported operand type")

    def __mul__(self, other):
        """ How to multiply NumericalFunctions """
        if self._isinstance_and_samemesh(other):
            return DiscretizedFunction(self.data * other.data, self.mesh)
        elif isinstance(other, (int, float)):
            return DiscretizedFunction(self.data * other, self.mesh)
        elif isinstance(other, np.ndarray):
            return DiscretizedFunction(self.data * other, self.mesh)
        else:
            raise ValueError("Unsupported operand type")

    def __truediv__(self, other):
        """ How to divide NumericalFunctions """
        if self._isinstance_and_samemesh(other):
            return DiscretizedFunction(self.data / other.data, self.mesh)
        elif isinstance(other, (int, float)):
            return DiscretizedFunction(self.data / other, self.mesh)
        elif isinstance(other, np.ndarray):
            return DiscretizedFunction(self.data / other, self.mesh)
        else:
            raise ValueError("Unsupported operand type")

    def __neg__(self):
        # Negation (unary minus)
        return DiscretizedFunction(-self.data, self.mesh)

    # Reverse argument operatorations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if self._isinstance_and_samemesh(other):
            return DiscretizedFunction(other.data / self.data, self.mesh)
        elif isinstance(other, (int, float)):
            return DiscretizedFunction(other / self.data, self.mesh)
        elif isinstance(other, np.ndarray):
            return DiscretizedFunction(other / self.data, self.mesh)
        else:
            raise ValueError("Unsupported operand type")

    def __repr__(self):
        return str(self.data)

    # Utility functions
    def _isinstance_and_samemesh(self, other):
        """ Function to check if other is the same instance of DiscretizedFunction object and if they use the same mesh """
        if isinstance(other, DiscretizedFunction):
            if np.array_equal(other.mesh, self.mesh):
                return True
        return False

    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return


class DiscretizedFunction2DH(DiscretizedFunction):
    """
    Class which represents a DiscretizedFunction in two horizontal spatial dimensions.
    We store a numpy array and tell python how to handle it.

    For scalars, the data is stored according to:            n_xy
    For vectors, the vector dimension is prepended as: n_v * n_xy
    Python broadcasting knows how to add and multiply these objects elementwise (or point wise)

    Here n_v is the vector length, n_xy are the number of xy degrees of freedom and n_z the number of z layers.
    """

    def __init__(self, data, vertices_2DHmesh):
        """
       Initialization
       Args:
           data: 2DH numpy array
           vertices_2DHmesh: ngsolve mesh object
       """
        super().__init__(data, vertices_2DHmesh)
        self.data = data
        self.mesh = vertices_2DHmesh

    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return



class DiscretizedMatrixFunction2DH(DiscretizedFunction):
    """
    Class which represents a matrix DiscretizedFunction in two horizontal spatial dimensions.
    We store a numpy array and tell python how to handle it.

    For n by m matrices, the matrix dimension is prepended as: n * m * n_xy
    Python broadcasting knows how to add and multiply these objects elementwise (or point wise)

    """

    def __init__(self, data, vertices_2DHmesh):
        """
       Initialization
       Args:
           data: 2DH numpy array
           vertices_2DHmesh: ngsolve mesh object
       """
        super().__init__(data, vertices_2DHmesh)
        self.data = data
        self.mesh = vertices_2DHmesh

    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return




class DiscretizedFunction3D(DiscretizedFunction):
    """
    Class discretized function in three spatial dimensions.
    We store a numpy array and tell python how to handle it.


    For scalars, the data is stored according to:            n_xy * n_z
    For vectors, the vector dimension is prepended as: n_v * n_xy * n_z
    Python broadcasting knows how to add and multiply these objects elementwise (or point wise)

    Here n_v is the vector length, n_xy are the number of xy degrees of freedom and n_z the number of z layers.
    """

    def __init__(self, data, vertices_2DHmesh):
        """
        Initialization
        Args:
            data: 3D numpy array
            vertices_2DHmesh: ngsolve mesh object
        """
        super().__init__(data, vertices_2DHmesh)
        self.data = data
        self.mesh = vertices_2DHmesh



    def __call__(self, *args, **kwargs):
        """TODO return ngsolve coefficient function"""
        return







########################################
##     We also define some functions  ##
########################################
def innerproduct(discfunc1: DiscretizedFunction, discfunc2: DiscretizedFunction):
    """ Function to add two DiscretizedFunction along the vector dimension """
    # Point-wise multiplication
    discprod = discfunc1 * discfunc2

    # Sum or contact along vector dimension
    data_summed = np.sum(discprod.data, axis=0)

    return DiscretizedFunction(data_summed, discprod.mesh)

def real(discfunc: DiscretizedFunction):
    """ Function to take the real part of a discretized function """
    return DiscretizedFunction(np.real(discfunc.data), discfunc.mesh)

def make_matrix(discfunc1_V: DiscretizedFunction, discfunc2_V: DiscretizedFunction):
    """ Function to combine two discretized function vectors 2DH into a single matrix by appending them together """
    data = np.stack((discfunc1_V.data, discfunc2_V.data), axis=1)
    return DiscretizedMatrixFunction2DH(data, discfunc1_V.mesh)


def make_diagonal_matrix(discfunc: DiscretizedFunction):
    """ Function to promote a scalar2DH to a diagonal matrix """
    V = discfunc.data
    Z = np.zeros_like(V)

    data = np.squeeze(np.array([[V, Z], [Z, V]]))
    return DiscretizedMatrixFunction2DH(data, discfunc.mesh)



