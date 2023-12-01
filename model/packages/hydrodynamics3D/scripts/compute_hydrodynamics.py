""" In this file, we define a function that returns the hydrodynamic object
under the standard forcing conditions

We neglect advection"""
from model.general.boundary_fitted_coordinates import BoundaryFittedCoordinates
from model.packages.hydrodynamics3D.classes.forcing_mechanism import HydrodynamicForcingMechanism
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.hydrodynamics3D.classes.hydrodynamics import Hydrodynamics
from model.packages.hydrodynamics3D.classes.numerical_parameters import HydrodynamicNumericalParameters
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.forcing.baroclinic import HydrodynamicForcingBaroclinic
from model.packages.hydrodynamics3D.forcing.nostress import HydrodynamicForcingNoStress
from model.packages.hydrodynamics3D.forcing.stokes import HydrodynamicForcingStokes
from model.packages.hydrodynamics3D.scripts.compute_hydrodynamics_order import compute_hydrodynamics_order


def compute_hydrodynamics(hydro_num_params: HydrodynamicNumericalParameters, hydro_phys_params: HydrodynamicPhysicalParameters,
                          bfc: BoundaryFittedCoordinates, hydrolead_extformech_list, hydrofirst_extformech_list):
    """ Function to compute the hydrodynamic variables

    Args:
        bfc: object with xi_gf and eta_gf
    """
    # Generate forcing mechanisms classes
    hydrolead_formechcol = HydrodynamicForcingMechanismCollection(hydrolead_extformech_list)

    # Solve the leading-order water motion
    hydrolead: HydrodynamicsOrder = compute_hydrodynamics_order(hydro_num_params,
                                                                hydro_phys_params,
                                                                hydrolead_formechcol)
    # Include the along- and across-channel velocities
    hydrolead.set_U_xi_U_eta(bfc.xi_gf)

    # The first-order forcing mechanisms: baroclinic forcing, nostress, tidal return flow
    hydrofornostress = HydrodynamicForcingNoStress(hydrolead, hydro_phys_params)
    hydroforstokes = HydrodynamicForcingStokes(hydrolead, hydro_phys_params)

    # The first-order forcing mechanisms for n=0:
    hydrofirst_formech_nostress0 = HydrodynamicForcingMechanism(1, 0, "nostress", hydrodynamic_forcing=hydrofornostress)
    hydrofirst_formech_stokes0 = HydrodynamicForcingMechanism(1, 0, "stokes", hydrodynamic_forcing=hydroforstokes,
                                                              useonboundary=False)

    # The first-order forcing components for n=2:
    hydrofirst_formech_nostress2 = HydrodynamicForcingMechanism(1, 2, "nostress", hydrodynamic_forcing=hydrofornostress)
    hydrofirst_formesh_stokes2 = HydrodynamicForcingMechanism(1, 2, "stokes", hydrodynamic_forcing=hydroforstokes,
                                                              useonboundary=False)

    # Generate forcing mechanism class for first-order
    hydrofirst_formech_list = [hydrofirst_formech_nostress0, hydrofirst_formech_stokes0,
                               hydrofirst_formech_nostress2, hydrofirst_formesh_stokes2]
    hydrofirst_formech_list.extend(hydrofirst_extformech_list)
    hydrofirst_formechcol = HydrodynamicForcingMechanismCollection(hydrofirst_formech_list)

    # Solve first-order water motion
    hydrofirst: HydrodynamicsOrder = compute_hydrodynamics_order(hydro_num_params,
                                                                 hydro_phys_params,
                                                                 hydrofirst_formechcol)
    # Include the along and across-channel velocities
    hydrofirst.set_U_xi_U_eta(bfc.xi_gf)

    # We create a general hydrodynamics object
    hydro = Hydrodynamics([hydrolead, hydrofirst])

    return hydro

