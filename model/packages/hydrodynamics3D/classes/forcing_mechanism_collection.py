"""
In this file, a general forcing mechanism class is defined, which collects the forcing mechanisms of a certain order.
# TODO test
"""
from model.general.classes.nested_default_dict import NestedDefaultDict


class HydrodynamicForcingMechanismCollection():
    """
    Class containing the hydrodynamic forcing mechanisms of a certain order

    This collection may be seen as a sort of glorified list / dictionary
    """
    def __init__(self, forcing_mechanism_list: list):
        """
        Class containing all leading-order forcing mechanisms
        Args:
            forcing_mechanism_list: list of (Type)ForcingMechanism objects
        """
        self.k = self._get_single_order(forcing_mechanism_list)
        self.frequency_component_list = self._generate_frequency_component_list(forcing_mechanism_list)
        self.forcing_symbol_list = [forcing_mechanism.symbol for forcing_mechanism in forcing_mechanism_list]

        # Creates a nested dict first sorted on frequency component and then on forcing mechanisms:
        self.forcing_mechanism_nest_dict = self._generate_forcing_mechanism_nest_dict(forcing_mechanism_list)


    # We make this class subscriptable such that we can call ForcingMechanismsHydroLead[frequency_component]
    # and get the corresponding forcing mechanisms. Similar to a dictionary in python

    def __getitem__(self, frequency_component):
        return self.forcing_mechanism_nest_dict[frequency_component]

    # private methods
    def _get_single_order(self, forcing_mechanism_list):
        """ Check if all forcing mechanisms have the same order and return it, else error """
        k_list = []

        # Collect all orders k of the forcing mechanisms
        for forcing_mechanism in forcing_mechanism_list:
            k = forcing_mechanism.k
            k_list.append(k)

        # Check if k_list consists of a one value
        k_single = k_list[0]
        if k_list.count(k_single) == len(k_list) and k_list:
            return k_single
        else:
            raise Exception("The order of the forcing mechanisms supplied to ForcingMechanismsCollection are not all the same."
                            "Please check the order of the forcing mechanisms.")



    def _generate_frequency_component_list(self, forcing_mechanism_list):
        """ Generate unique list of frequency components of the forcing """
        frequency_component_list = []
        for forcing_mechanism in forcing_mechanism_list:
            frequency_component = forcing_mechanism.n
            if frequency_component not in frequency_component_list:
                frequency_component_list.append(frequency_component)

        frequency_component_list.sort()
        return frequency_component_list

    def _generate_forcing_mechanism_nest_dict(self, forcing_mechanism_list):
        """ Generates a nested dict first sorted on frequency component and then on forcing mechanisms symbol:
            i.e. {1: {"A": forcing_mechanism}, 0: {"q": forcing_mechanism, "gamma": forcing mechanism}} """

        # We start with a default dict and then convert it to a dict
        forcing_mechanism_nest_ddict = NestedDefaultDict()

        for forcing_mechanism in forcing_mechanism_list:
            # Indices for readability
            n = forcing_mechanism.n
            symbol = forcing_mechanism.symbol

            # Using default dict behaviour
            forcing_mechanism_nest_ddict[n][symbol] = forcing_mechanism

        forcing_mechanism_nest_dict = forcing_mechanism_nest_ddict.get_dict()
        return forcing_mechanism_nest_dict