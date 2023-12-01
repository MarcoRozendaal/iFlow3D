"""
In this file, the forcing components of the stokes forcing are collected and returned.

"""

import warnings

from model.packages.hydrodynamics3D.classes.forcing_components import HydrodynamicForcingComponentsZero, \
    conditional_conjugate, conditional_real_part, HydrodynamicForcingComponents
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder



class HydrodynamicForcingStokes():
    """"
       Class to generate the stokes (or tidal return flow) forcing components
    """

    def __init__(self, hydrolead: HydrodynamicsOrder, hydrodynamic_physical_parameters: HydrodynamicPhysicalParameters):
        # general parameters
        self.hydrolead = hydrolead
        self.params = hydrodynamic_physical_parameters


    # Core functionality of this class:
    def generate_forcing_components(self, k, n):
        # Create instance of HydrodynamicForcingComponentsZero class and return it
        return HydrodynamicForcingComponentsZero(qcheck_R_V=self.gamma1_V(n))



    def gamma1_V(self, n):
        """ Generate gamma1_V forcing [Tested against iFlow2DV]"""

        # Check if the forcing frequency contains 1 and "tide":
        if 1 not in self.hydrolead.freqcomp_list:
            warnings.warn("The stokes forcing is only implemented when the leading-order hydrodynamics consists of an n=1 component")
        if "tide" not in self.hydrolead.forcing_mechanism_nest_dict[1]:
            warnings.warn("The stokes forcing is only implemented for leading-order amplitude forcing")

        # Check if the requested frequency component is in the allowed range
        if n not in [0, 2]:
            warnings.warn("The stokes forcing is only implemented to generate a n=0 and a n=2 response")


        return conditional_real_part(
            1 / 2 * conditional_conjugate(self.hydrolead.Z[1]["tide"], n) * self.hydrolead.U_V[1]["tide"](
                self.params.R), n)
