"""
In this file, the suspended sediment concentration is solved
"""

import ngsolve
import numpy as np

from model.general.classes.discretized.function import DiscretizedFunction
from model.general.classes.spatial_parameter import SpatialParameterFromData
from model.general.create_geometry import BOUNDARY_DICT, SEA
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.sediment3D.classes.capacity import SedimentCapacity
from model.packages.sediment3D.classes.numerical_parameters import SedimentNumericalParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters
from model.packages.sediment3D.classes.sediment import Sediment
from model.packages.sediment3D.classes.transport_capacity import TransportCapacity

# TODO define forcing mechansisms for this class
def compute_sediment(sediment_numerical_parameters: SedimentNumericalParameters,
                     sediment_physical_parameters: SedimentPhysicalParameters,
                     sediment_capacity: SedimentCapacity,
                     transport_capacity: TransportCapacity,
                     Phi_sea) -> Sediment:
    """
    Args:
        sediment_numerical_parameters: SedimentNumericalParameters object
        sediment_physical_parameters: SedimentPhysicalParameters object
        transport_capacity: TransportCapacity object
        Phi_sea: value at sea
    Returns:
        Sediment: Sediment object
    """

    # We create the finite element space considered
    finite_element_space = ngsolve.H1(sediment_numerical_parameters.mesh,
                                      order=sediment_numerical_parameters.order_basisfunctions,
                                      dirichlet=BOUNDARY_DICT[SEA])  # TODO changed to real valued vector space

    psi = finite_element_space.TestFunction()
    Phi = finite_element_space.TrialFunction()

    # Create the bilinear form

    D_Na_M_disc = transport_capacity.D_M['all']
    T_a_V_disc = transport_capacity.T_V['all']



    # We project the discretized objects on NGSolve CoefficientFunctions
    # TODO check coeff functions
    D_Na_M = project_matrix_on_ngsolve(D_Na_M_disc, sediment_numerical_parameters.mesh)
    T_a_V = project_vector_on_ngsolve(T_a_V_disc, sediment_numerical_parameters.mesh)

    a = ngsolve.BilinearForm(finite_element_space)
    a += - ngsolve.InnerProduct(D_Na_M * ngsolve.Grad(Phi) + T_a_V*Phi, ngsolve.Grad(psi)) * ngsolve.dx

    # Create linear form
    b = ngsolve.LinearForm(finite_element_space)


    # Add amplitude forcing
    Phi_gf = ngsolve.GridFunction(finite_element_space, "Phi")
    Phi_gf.Set(Phi_sea, ngsolve.BND)

    ngsolve.solvers.BVP(bf=a, lf=b, gf=Phi_gf, inverse="pardiso")


    # TODO thinabout discretized varibales too, these are not in sedcap
    sediment = Sediment(sediment_capacity, Phi_gf)


    return sediment



def project_matrix_on_ngsolve(discfunc: DiscretizedFunction, mesh):
    """ Function to project a discrete vector on a NGSolve grid """
    mesh_vertices = discfunc.mesh
    data = discfunc.data

    sp_list = []

    # We loop over the small dimensions of data and do a scalar projection
    for i in range(2):
        for j in range(2):
            scalar_data = np.expand_dims(data[i, j], axis=1)
            # TODO check this

            pointclouddata = np.concatenate((mesh_vertices, scalar_data), axis=1)
            cf = SpatialParameterFromData(pointclouddata, 'griddata', 'linear', mesh).cf
            sp_list.append(cf)
    cf_M = ngsolve.CoefficientFunction(tuple(sp_list), dims=(2, 2))

    return cf_M


def project_vector_on_ngsolve(discfunc: DiscretizedFunction, mesh):
    """ Function to project a discrete vector on a NGSolve grid """
    # TODO check
    mesh_vertices = discfunc.mesh
    data = discfunc.data

    sp_list = []

    # We loop over the small dimensions of data and do a scalar projection
    for i in range(2):
        scalar_data = np.expand_dims(data[i], axis=1)

        pointclouddata = np.concatenate((mesh_vertices, scalar_data ), axis=1)
        cf = SpatialParameterFromData(pointclouddata, 'griddata', 'linear', mesh).cf
        sp_list.append(cf)
    cf_V = ngsolve.CoefficientFunction(tuple(sp_list), dims=(2, 1))

    return cf_V


