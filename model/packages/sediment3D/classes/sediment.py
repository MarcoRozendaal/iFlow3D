"""
In this file, a class is defined that stores the suspended sediment concentrations



24-09-2023: Removed alpha index from sediment C
"""

import ngsolve

from model.general.classes.nested_default_dict import NestedDefaultDict
from .capacity import SedimentCapacity



class Sediment():
    """
    Class containing the suspended sediment concentrations

    Can be used as follows:
        sed.C[k][n][f][alpha] for order k, frequency component n, forcing mechanisms f, scaling alpha.
    """
    sum_symbol = 'all'


    def __init__(self, sedcap: SedimentCapacity, Phi_gf):
        """
        Class that contains the suspended sediment concentration and erodability Phi

        Args:
            sedcap: An instance of SedimentCapacity
        """

        self.Phi = Phi_gf



        # Convert each sediment capacity to a suspended sediment concentration
        C = NestedDefaultDict()

        # Fill dictionary
        for k, hatC_k in sedcap.hatC.items():
            for n, hatC_kn in hatC_k.items():
                for symbol, hatC_knf in hatC_kn.items():
                    for alpha, hatC_knfalpha in hatC_knf.items():
                        # Detemine value due to scaling:
                        value = None
                        if alpha == "a":
                            value = lambda z, hatC_knfalpha=hatC_knfalpha: hatC_knfalpha(z) * self.Phi
                        elif alpha == "a_x":
                            value = lambda z, hatC_knfalpha=hatC_knfalpha: hatC_knfalpha(z) * ngsolve.Grad(self.Phi)[0]
                        elif alpha == "a_y":
                            value = lambda z, hatC_knfalpha=hatC_knfalpha: hatC_knfalpha(z) * ngsolve.Grad(self.Phi)[1]
                        else:
                            raise("Scaling of alpha is not a valid option")

                        # Set value
                        C[k][n][symbol] = value

        # Set the property
        self.C = C.get_dict()



        # Convert each depth-averaged sediment capacity to a depth-averaged suspended sediment concentration
        C_DA = NestedDefaultDict()

        # Fill dictionary
        for k, hatC_DA_k in sedcap.hatC_DA.items():
            for n, hatC_DA_kn in hatC_DA_k.items():
                for symbol, hatC_DA_knf in hatC_DA_kn.items():
                    for alpha, hatC_DA_knfalpha in hatC_DA_knf.items():
                        # Detemine value due to scaling:
                        value = None
                        if alpha == "a":
                            value = hatC_DA_knfalpha * self.Phi
                        elif alpha == "a_x":
                            value = hatC_DA_knfalpha * ngsolve.Grad(self.Phi)[0]
                        elif alpha == "a_y":
                            value = hatC_DA_knfalpha * ngsolve.Grad(self.Phi)[1]
                        else:
                            raise ("Scaling of alpha is not a valid option")

                        # Set value
                        C_DA[k][n][symbol] = value

        # Set the property
        self.C_DA = C_DA.get_dict()

        # Create the all index
        self._set_sum()

        # TODO Maybe add more objects to sediment?

        # TODO we create a method to add the objects



    def _set_sum(self):
        """ Function to create the sum variants of the ssc
        """
        # TODO check
        property_name_list = ["C", "C_DA"]

        def islambda(v):
            LAMBDA = lambda: 0
            return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


        # Creating a shallow copy for iteration, so the original can be modified during the loop
        for property_name in property_name_list:
            # The symbol indices are summed
            # Loop over order
            for k, property_k in getattr(self, property_name).items():
                # Loop over frequency component
                for n, property_kn in property_k.items():
                    # We sum the forcing mechanisms
                    # Two sums: summation for z dependent and not z dependent
                    sum_z = lambda z: ngsolve.CoefficientFunction(0)
                    sum = ngsolve.CoefficientFunction(0)
                    isfunctionofz = False

                    for symbol, property_knf in property_kn.items():
                        # Testing for z dependence:
                        if islambda(property_knf):
                            isfunctionofz = True
                            # Pass objects that change with each iteration early using keyword arguments
                            sum_z = lambda z, sum_z=sum_z, property_knf=property_knf: sum_z(z) + property_knf(z)
                        else:
                            sum += property_knf


                # Set the sum index
                if isfunctionofz:
                    getattr(self, property_name)[k][n][self.sum_symbol] = lambda z, sum_z=sum_z: sum_z(z)
                else:
                    getattr(self, property_name)[k][n][self.sum_symbol] = sum




