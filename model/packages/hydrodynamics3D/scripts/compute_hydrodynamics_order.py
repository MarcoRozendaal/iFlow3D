"""
In this file, the leading-order hydrodynamics under the standard forcing conditions are solved.
"""

import numpy as np
import scipy.sparse as sp

import ngsolve

from model.general.create_geometry import BOUNDARY_DICT, SEA, WALLUP, WALLDOWN, RIVER
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder
from model.packages.hydrodynamics3D.classes.numerical_parameters import HydrodynamicNumericalParameters
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters


def compute_hydrodynamics_order(hydrodynamic_numerical_parameters: HydrodynamicNumericalParameters,
                                hydrodynamic_physical_parameters: HydrodynamicPhysicalParameters,
                                forcing_mechanism_collection: HydrodynamicForcingMechanismCollection) -> HydrodynamicsOrder:
    """
    Args:
        hydrodynamic_numerical_parameters: Object containing the kth-order hydrodynamic numerical parameters
        hydrodynamic_physical_parameters: Object containing the leading-order hydrodynamic physical parameters
        forcing_mechanism_collection: Object containing the kth-order forcing mechanisms
    Returns:
        hydrodynamics_order: Object containing the kth-order hydrodynamic variables
    """

    # We create the finite element space considered
    finite_element_space = ngsolve.H1(hydrodynamic_numerical_parameters.mesh,
                                      order=hydrodynamic_numerical_parameters.order_basisfunctions,
                                      dirichlet=BOUNDARY_DICT[SEA], complex=True)

    phi = finite_element_space.TestFunction()
    Znf = finite_element_space.TrialFunction()

    # Store the gridfunction solution in a nested dictionairy of frequency component n and forcing symbol f, e.g., Z[n][f]
    Z_gf_dict_dict = {n: {} for n in forcing_mechanism_collection.frequency_component_list}

    # We loop over the frequency components of the forcing
    for n in forcing_mechanism_collection.frequency_component_list:

        # Create the bilinear form for n
        D0n_R_M = hydrodynamic_physical_parameters.D0_R_M(n)
        omega = hydrodynamic_physical_parameters.omega

        a = ngsolve.BilinearForm(finite_element_space)
        a += - ngsolve.InnerProduct(D0n_R_M * ngsolve.Grad(Znf), ngsolve.Grad(phi)) * ngsolve.dx
        a += n * 1j * omega * Znf * phi * ngsolve.dx


        # We loop over the forcing mechanamisms
        for symbol, forcing_mechanism in forcing_mechanism_collection[n].items():

            # Create linear form
            b = ngsolve.LinearForm(finite_element_space)
            # Add body and boundary forcing mechanisms at wallup, walldown and river
            b += ngsolve.InnerProduct(forcing_mechanism.q_R_body_V, ngsolve.Grad(phi)) * ngsolve.dx
            b += - phi * forcing_mechanism.q_R_wallup * ngsolve.ds(BOUNDARY_DICT[WALLUP])
            b += - phi * forcing_mechanism.q_R_walldown * ngsolve.ds(BOUNDARY_DICT[WALLDOWN])
            b += - phi * forcing_mechanism.q_R_river * ngsolve.ds(BOUNDARY_DICT[RIVER])


            # Add amplitude forcing
            Z_gf_dict_dict[n][symbol] = ngsolve.GridFunction(finite_element_space, "Z" + str(n) + "_" + symbol)
            Z_gf_dict_dict[n][symbol].Set(forcing_mechanism.A_sea, ngsolve.BND)

            ngsolve.solvers.BVP(bf=a, lf=b, gf=Z_gf_dict_dict[n][symbol], inverse="pardiso")

    hydrodynamics_order = HydrodynamicsOrder(Z_gf_dict_dict, finite_element_space,
                                             hydrodynamic_numerical_parameters,
                                             hydrodynamic_physical_parameters,
                                             forcing_mechanism_collection)

    return hydrodynamics_order



