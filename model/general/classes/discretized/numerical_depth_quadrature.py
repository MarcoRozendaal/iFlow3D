"""
In this file, a class is defined that can numerically integrate over the depth
# TODO maybe add dimensions to DiscretizedFunction objects?
"""

import numpy as np
import scipy.integrate as integrate

import ngsolve

from model.general.classes.discretized.function import DiscretizedFunction2DH, DiscretizedFunction3D, \
    DiscretizedFunction
from model.general.classes.discretized.depth_parameters import NumericalDepthParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters


class NumericalDepthQuadrature:
    """
    Class to numerically integrate over the depth and return discretized function objects

    """

    def __init__(self, numerical_depth_parameters: NumericalDepthParameters,
                 sediment_physical_parameters: SedimentPhysicalParameters, mesh):
        """
        Initialization
        Args:
            numerical_depth_parameters: NumericalDepthParameter object
            sediment_physical_parameters: SedimentPhysicalParameters object
            mesh: ngsolve mesh object
        """
        # Initialization
        self.params = numerical_depth_parameters
        self.phys_params = sediment_physical_parameters
        self.mesh = mesh
        self.mesh_vertices = self._compute_mesh_vertices(self.mesh)
        self.sigmalayer_z_cf_list = self._compute_sigma_layer_z_cf_list()
        self.z_grid_3D = self._compute_z_grid_3D()
        self.disc_depth = self.discretize_cf(sediment_physical_parameters.D)


    def _compute_mesh_vertices(self, mesh):
        """ Computes the mesh vertices """
        mesh_vertices = np.array([vertex.point for vertex in mesh.vertices])
        return mesh_vertices

    def _compute_sigma_layer_z_cf_list(self):
        """ Computes the coefficient function z at each sigma layer """
        sigma = np.linspace(-1, 0, self.params.number_sigma_layers)
        sigma_layer_z_cf_list = [self.phys_params.D * sigma_layer + self.phys_params.R for sigma_layer in sigma]
        return sigma_layer_z_cf_list

    def _compute_z_grid_3D(self):
        """ Computes the z values at the 3D grid """
        z_sigma_layer_vector = self.eval_funccomp_at_sigmalayers(lambda z: z)
        z_grid_3D = self.discretize_cf(z_sigma_layer_vector)
        return z_grid_3D

    def eval_funccomp_at_sigmalayers(self, function, index=0):
        """ Evaluates component index of function at the z CoefficientFunction sigma layers """
        # TODO Added compile, it does not seem to slow down or speed up much, maybe remove?
        # TODO removed .Compile()

        funccomp_at_sigmalayers_list = [function(sigmalayer_z_cf)[index] for sigmalayer_z_cf in self.sigmalayer_z_cf_list]

        funccomp_at_sigmalayers_tuple = tuple(funccomp_at_sigmalayers_list)

        CF_vector_at_sigmalayers = ngsolve.CoefficientFunction(funccomp_at_sigmalayers_tuple)
        return CF_vector_at_sigmalayers

    def discretize_cf(self, coefficient_function):
        """ Function to eval coefficient function at the mesh vertices """
        pnts = self.mesh_vertices
        discretized_coefficient_function = coefficient_function(self.mesh(pnts[:, 0], pnts[:, 1]))
        return discretized_coefficient_function



    def discretize_function3D(self, function):
        """
        Function to convert a function handle to a DiscretizedFunction3D
        Args:
            function: function of z

        Returns:
            DiscretizedFunction3D
        """
        # NGSolve CoefficientFunctions do not keep their dimensions after evaluating at a point or a list of points.
        # To keep track of the dimensions, we evaluate each element of a vector CoefficientFunction individually and
        # manually combine them

        # We retrieve the length of the CoefficientFuntion vector
        n_v = function(0).dim

        # For each component of the vector, we evaluate it at the z sigma layers:
        funccomp_at_sigmalayer_list = [self.eval_funccomp_at_sigmalayers(function, i) for i in range(n_v)]

        # For each component of the vector, we discretize it
        funccomp_discretized_list = [self.discretize_cf(funccomp) for funccomp in funccomp_at_sigmalayer_list]

        # The dimensions of the vectors are combined
        data = np.array(funccomp_discretized_list)

        return DiscretizedFunction3D(data, self.mesh_vertices)

    def discretize_function2DH(self, coefficient_function):
        """
        Function to convert a function handle to a DiscretizedFunction2DH
        Args:
            coefficient_function: CoefficientFunction object

        Returns:
            DiscretizedFunction2DH
        """
        # NGSolve CoefficientFunctions do not keep their dimensions after evaluating at a point or a list of points
        # To keep track of the dimensions, we evaluate each element of a vector CoefficientFunction individually and manually combine them


        # If function is a constant of float, return early
        if isinstance(coefficient_function, (float, int)):
            data = np.full(len(self.mesh_vertices[:, 0]), coefficient_function)
            return DiscretizedFunction2DH(data, self.mesh_vertices)

        # We retrieve the length of the CoefficientFuntion vector
        n_v = coefficient_function.dim

        # For each component of the vector, we discretize it
        funccomp_discretized_list = [self.discretize_cf(coefficient_function[i]) for i in range(n_v)]

        # The dimensions of the vectors are combined
        data = np.squeeze(np.array(funccomp_discretized_list))

        return DiscretizedFunction2DH(data, self.mesh_vertices)


    ############################# Depth integrals ########################
    def cumulative_depth_integral(self, integrand):
        """
        Computes the cumulative integral from z=-H to z=R
        The parameters specified by numerical_depth_parameters object
        Args:
            integrand: Function to be integrated, can be a function f(z) or a DiscretizedFunction3D object

        Returns:
            DiscretizedFunction3D
        """

        # Determine the type of integrand
        disc_func_3D = None
        # TODO make DiscretizedFunction3D class persistent
        if isinstance(integrand, DiscretizedFunction3D) or isinstance(integrand, DiscretizedFunction):
            disc_func_3D = integrand
        else:
            disc_func_3D = self.discretize_function3D(integrand)

        # We integrate over the last dimension
        # Numpy does not like singleton dimensions, so we remove them # TODO kijken hoe dit werkt voor vectoren
        # Use the cumulative trapezoidal method
        cum_depth_integral_disc_func_3D = integrate.cumulative_trapezoid(np.squeeze(disc_func_3D.data), self.z_grid_3D, initial=0)

        return DiscretizedFunction3D(cum_depth_integral_disc_func_3D, self.mesh_vertices)



    def depth_integal(self, integrand):
        """
        Computes the integral from z=-H to z=R using sigma layers
        The parameters specified by numerical_depth_parameters object
        Args:
            integrand: Function to be integrated, can be a function f(z) or a DiscretizedFunction3D object

        Returns:
            DiscretizedFunction2DH
        """

        # Determine the type of integrand
        disc_func_3D = None
        # TODO make DiscretizedFunction3D class persistent
        if isinstance(integrand, DiscretizedFunction3D) or isinstance(integrand, DiscretizedFunction):
            disc_func_3D = integrand
        else:
            disc_func_3D = self.discretize_function3D(integrand)


        # We integrate over the last dimension
        # Use the trapezoid method to compute the depth_num-integral and return a DiscretizedFunction2DH object
        depth_integral_disc_func = integrate.trapezoid(disc_func_3D.data, self.z_grid_3D)


        return DiscretizedFunction2DH(depth_integral_disc_func, self.mesh_vertices)


    def depth_average(self, integrand):
        """
        Computes the depth_num-average of the integrand
        We integrate from z=-H to z=R using sigma layers
        The parameters specified by numerical_depth_parameters object
        Args:
            integrand: Function to be integrated, can be a function f(z) or a DiscretizedFunction3D object

        Returns:
            DiscretizedFunction2DH
        """
        return self.depth_integal(integrand) / self.disc_depth




