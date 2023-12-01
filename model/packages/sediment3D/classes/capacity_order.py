"""
In this file, the results of the kth-order sediment capacity computation are stored
# TODO test
"""

import ngsolve

from model.general.classes.nested_default_dict import NestedDefaultDict
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.sediment3D.classes.forcing_mechanism_collection import SedimentCapacityForcingMechanismCollection
from model.packages.sediment3D.classes.numerical_parameters import SedimentNumericalParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters



class SedimentCapacityOrder():
    """
    Class containing the kth-order sediment capacity

    Can be used as follows:
        sedcaplead.hatC[n][f][alpha]
    # TODO
    We also include an all option:
        sedcaplead.hatC[n][f]['all'] sums all alpha's for a given n and f
    # TODO maybe also
        sedcaplead.hatC[n]['all']['all'] sums all alpha's for a given n and f
    """

    sum_symbol = "all"

    # TODO test alpha index

    # TODO Think about the case if z is already discretized such as by advection of sediment

    def __init__(self,
                 sediment_numerical_parameters: SedimentNumericalParameters,
                 sediment_physical_parameters: SedimentPhysicalParameters,
                 forcing_mechanism_collection: SedimentCapacityForcingMechanismCollection):

        # forcing properties
        self.k = forcing_mechanism_collection.k
        self.freqcomp_list = forcing_mechanism_collection.frequency_component_list
        self.forcing_mechanism_nest_dict = forcing_mechanism_collection.forcing_mechanism_nest_dict



        # variables
        self._set_sediment_capacity_variables(sediment_numerical_parameters,
                                              sediment_physical_parameters,
                                              forcing_mechanism_collection)

        # Sum over all forcing mechanisms for a given frequency component
        # TODO
        #self._set_sediment_capacity_variables_sum()


    ## Setting the leading-order flow variables
    def _set_sediment_capacity_variables(self,
                                         sediment_numerical_parameters,
                                         sediment_physical_parameters,
                                         forcing_mechanism_collection):
        """ This function sets the order variables as nested dicts [] """

        # Set the sediment capacity. We use nested default dicts
        hatC = NestedDefaultDict()
        hatC_DA = NestedDefaultDict()
        nablahatC = NestedDefaultDict()
        hatC_z = NestedDefaultDict()

        for n, forcing_mechanism_dict_dict in self.forcing_mechanism_nest_dict.items():
            for symbol, forcing_mechanism_dict in forcing_mechanism_dict_dict.items():
                for alpha, forcing_mechanism in forcing_mechanism_dict.items():
                    # Get the forcing components per n, per forcing mechanism f, per scaling alpha
                    sediment_capacity_forcing_components = forcing_mechanism.sediment_capacity_forcing_components

                    # The sediment capacity per forcing, per n, per alpha:
                    hatC[n][symbol][alpha] = lambda z, sediment_capacity_forcing_components=sediment_capacity_forcing_components: \
                        sediment_capacity_forcing_components.hatC(z)

                    # The depth-averaged sediment capacity per forcing, per n, per alpha:
                    hatC_DA[n][symbol][alpha] = sediment_capacity_forcing_components.hatC_DA

                    # If leading-order and the forcing is 'etide' add another property
                    # TODO _added leading-order check
                    if self.k==0 and symbol=='etide':
                        # TODO check late binding
                        nablahatC[n][symbol][alpha] = lambda z, n=n, forcing_mechanism=forcing_mechanism: \
                            forcing_mechanism.sediment_capacity_forcing.nablahatC(self.k, n, 'etide', 'a', z)
                        hatC_z[n][symbol][alpha] = lambda z, n=n, forcing_mechanism=forcing_mechanism: \
                            forcing_mechanism.sediment_capacity_forcing.hatC_z(self.k, n, 'etide', 'a', z)


        # Set the self properties
        self.hatC = hatC.get_dict()
        self.hatC_DA = hatC_DA.get_dict()
        self.nablahatC = nablahatC.get_dict()
        self.hatC_z = hatC_z.get_dict()


    # TODO implement the the sum functionallility
    def _set_sediment_capacity_variables_sum(self):
        """ Function that extracts the dict_dict variables and creates
            hatC[n][f]["all"]
            []
        """
        # TODO check if this works for triple dicts

        def islambda(v):
            LAMBDA = lambda: 0
            return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


        # Creating a shallow copy for iteration, so the original can be modified during the loop
        for variable_name, value in self.__dict__.copy().items():
            # Filter out non variables:
            # Check if it is a dictionary and not forcing_mechanism_nest_dict
            if isinstance(value, dict) and value is not self.forcing_mechanism_nest_dict:
                # True variable

                # TODO think about this or maybe temporary disable, it is not needed.
                # The symbol indices are summed
                for n in self.freqcomp_list:
                    # Two sums: summation for z dependent and not z dependent
                    sum_z = lambda z: ngsolve.CoefficientFunction(0)
                    sum = ngsolve.CoefficientFunction(0)
                    isfunctionofz = False

                    for symbol, obj in value[n].items():
                        # Testing for z dependence:
                        if islambda(obj):
                            isfunctionofz = True
                            # Pass objects that change with each iteration early using keyword arguments
                            sum_z = lambda z, sum_z=sum_z, obj=obj: sum_z(z) + obj(z)
                        else:
                            sum += obj


                    # Set the sum index
                    if isfunctionofz:
                        getattr(self, variable_name)[n][self.sum_symbol] = lambda z, sum_z=sum_z: sum_z(z)
                    else:
                        getattr(self, variable_name)[n][self.sum_symbol] = sum