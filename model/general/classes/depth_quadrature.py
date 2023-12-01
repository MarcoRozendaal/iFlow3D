"""
In this file, a class is defined that can numerically integrate NGSolve objects over the depth
"""

import numpy as np
import scipy.integrate as integrate
from itertools import pairwise

import ngsolve

from model.general.classes.discretized.function import DiscretizedFunction2DH, DiscretizedFunction3D, \
    DiscretizedFunction
from model.general.classes.discretized.depth_parameters import NumericalDepthParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters

# TODO make this class work
class DepthQuadrature:
    """
    Class to numerically integrate over the depth and return a NGSolve object

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
        # self.mesh_vertices = self._compute_mesh_vertices(self.mesh) # TODO remove?
        self.z_at_sigmalayer_list = self.generate_z_at_sigmalayer_list()
        self.dz = self.generate_dz()
        self.dz_c = self.generate_dz_c()


        # TODO
        # TODO remove? self.z_grid_3D = self._compute_z_grid_3D()
        # TODO REMOVE? self.disc_depth = self.discretize_cf(sediment_physical_parameters.D)


    def generate_z_at_sigmalayer_list(self):
        """
        Function to compute the z coordinate of the sigma layers as a CoefficientFunction

        Note: sigma ranges from -1 to 0, where sigma=-1 corresponds to z=-H and sigma=0 to z=R, i.e.,
        we start at the bed and move upward
        """
        sigma = np.linspace(-1, 0, self.params.number_sigma_layers)

        # For each value of sigma, we compute the corresponding z value as a CoefficientFunction
        z_at_sigmalayer_list = [self.phys_params.D * sigma_layer + self.phys_params.R for sigma_layer in sigma]

        return z_at_sigmalayer_list


    def generate_dz(self):
        """
        Function to compute the backwards difference between two sigma layers
        Returns: dc
        """
        # We put the initial sigmalayer list into a computationally more friendly form
        z_at_sigmalayer_pre = self.z_at_sigmalayer_list

        # Compute central difference between z sigmalayers with same length as sigmalayers
        # At the boundaries use the backwards difference
        dz = [z_layer_1-z_layer_0 for (z_layer_0, z_layer_1) in zip(z_at_sigmalayer_pre[:-1], z_at_sigmalayer_pre[1:])]

        return dz

    def generate_dz_c(self):
        """
        Function to compute the central difference between two sigma layers
        Returns: dc_z
        """

        # We extend the z_at_sigmalayer to a more computationally friendly form
        z_at_sigmalayer_ext = [self.z_at_sigmalayer_list[0], *self.z_at_sigmalayer_list, self.z_at_sigmalayer_list[-1]]

        # Compute central difference between z sigmalayers with same length as sigmalayers
        # At the boundaries use the backwards difference
        dz_c = [z_layer_1-z_layer_0 for (z_layer_0, z_layer_1) in zip(z_at_sigmalayer_ext[:-2], z_at_sigmalayer_ext[2:])]

        return dz_c

    def generate_dz_s(self):
        """
        Function to compute the sum difference between two sigma layers
        Returns: dc_s
        """

        # We extend the z_at_sigmalayer to a more computationally friendly form
        z_at_sigmalayer_pre = [self.z_at_sigmalayer_list[0], *self.z_at_sigmalayer_list]

        z_end = z_at_sigmalayer_pre[-1]
        # Compute sum difference between z sigmalayers with same length as sigmalayers
        dz_s = [2*z_end - z_layer_0 - z_layer_1 for (z_layer_0, z_layer_1) in zip(z_at_sigmalayer_pre[:-1], z_at_sigmalayer_pre[1:])]

        return dz_s




    def eval_at_sigmalayers(self, integrand):
        """
        Evaluates integrand at the z value of the sigmalayers
        Args:
            integrand: function of z

        Returns:
            list of f(z) evaluated at the sigma layers

        # TODO check vectors here, maybe add index argument here?
        """
        integrand_at_sigmalayers = [integrand(z_at_sigmalayer) for z_at_sigmalayer in self.z_at_sigmalayer_list]

        return integrand_at_sigmalayers


    def trapezoid(self, integrand_at_sigmalayer_list):
        """
        Integrates an integrand_at_sigmalayers over the depth using the trapezoidal method
        Args:
            integrand_at_sigmalayers:

        Returns:
            depth-integral of the integrand as determined by the trapezoidal method
        """
        # We use the function centric trapezoidal formulation
        sum = ngsolve.CoefficientFunction(0)
        for integrand_at_sigmalayer, dz_sigmalayer in zip(integrand_at_sigmalayer_list, self.dz_c):
            sum += integrand_at_sigmalayer * dz_sigmalayer

        trapezoid = sum/2

        return trapezoid

    def cumulative_trapezoid(self, integrand_at_sigmalayer_list):
        """
        Cumulatively integrates integrand_at_sigmalayer_list over the depth using the trapezoidal method
        Args:
            integrand_at_sigmalayers:

        Returns:
            Cumalative depth-integral of the integrand as determined by the trapezoidal method
        """
        # TODO think about which method to use here and why
        # TODO maybe define a special class that knows how to evaluate at a given point or grid
        # TODO But that can also take a depth integral in a fast manner.

        # TODO I want some function that knows when to evaluate the iterated integral
        # We use the interval centric trapezoidal formulation, i.e., standard
        sum = ngsolve.CoefficientFunction(0)
        cumsum = []
        for integrand_at_sigmalayer, dz_sigmalayer in zip(integrand_at_sigmalayer_list, self.dz):
            sum += integrand_at_sigmalayer * dz_sigmalayer / 2
            cumsum.append()
            # TODO Can we use the current method for the cum trapz??



        return trapezoid


    ##### Main functions #####
    def depth_integral(self, integrand):
        """
        Computes the integral from z=-H to z=R using sigma layers
        The parameters specified by depth_parameters object
        Args:
            integrand: Function to be integrated, it can be a function of z: f(z), for a given z it is a
            CoefficientFunction, or it can be a list of CoefficientFunctions evaluated at the sigma layers

        Returns:
            NGSolve Coefficient Function
        """

        # Eval integrand at sigmalayers, if needed
        integrand_at_sigmalayers = None
        if isinstance(integrand, list):
            integrand_at_sigmalayers = integrand
        else:
            integrand_at_sigmalayers = self.eval_at_sigmalayers(integrand)

        # Integrate over the depth
        depth_integral = self.trapezoid(integrand_at_sigmalayers)

        return depth_integral

    def depth_average(self, integrand):
        """
        Computes the depth-average using sigma layers
        The parameters specified by depth_parameters object
        Args:
            integrand: Function to be integrated, it can be a function of z: f(z), for a given z it is a
            CoefficientFunction, or it can be a list of CoefficientFunctions evaluated at the sigma layers

        Returns:
            NGSolve Coefficient Function
        """

        return self.depth_integal(integrand) / self.phys_params.D

    def cumulative_depth_integral(self, integrand):
        """
        Computes the cumulative integral from z=-H to z=R using sigma layers
        The parameters specified by depth parameters object
        Args:
            integrand: Function to be integrated, it can be a function of z: f(z), for a given z it is a
            CoefficientFunction, or it can be a list of CoefficientFunctions evaluated at the sigma layers

        Returns:
            List of NGSolve CoefficientFunctions
        """

        # Eval integrand at sigmalayers, if needed
        integrand_at_sigmalayers = None
        if isinstance(integrand, list):
            integrand_at_sigmalayers = integrand
        else:
            integrand_at_sigmalayers = self.eval_at_sigmalayers(integrand)

       # TODO write cum trapz method
        cum_depth_integral = self.cumulative_trapezoid(integrand_at_sigmalayers)

        return cum_depth_integral


    # TODO remove before ##################################################################################
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




