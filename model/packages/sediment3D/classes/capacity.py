"""
In this file, a class is defined that stores the sediment capacity results of all orders

# TODO unit test object
"""


class SedimentCapacity():
    """
    Class containing all ordered sediment capacity results

    Can be used as follows:
        sedcap.hatC[k][n][f][alpha] for order k, frequency component n, forcing mechanisms f, scaling alpha.
    """


    def __init__(self, sedcap_order_list):
        """TODO: maybe also set the same numerical properties as sedcap_order"""
        #self.finite_element_space = finite_element_space
        #self.freqcomp_list = forcing_mechanism_collection.frequency_component_list
        #self.forcing_mechanism_nest_dict = forcing_mechanism_collection.forcing_mechanism_nest_dict

        self.sedcap_variable_name_list = self._get_sediment_capacity_variable_name_list(sedcap_order_list)


        is_firstiterate = True
        # We create a loop to set the dictionairy keys
        for sedcap_order in sedcap_order_list:
            # Order
            k = sedcap_order.k

            # For each sediment capacity variable, create new leading level of order index k:
            for sedcap_variable_name in self.sedcap_variable_name_list:
                # create initial dictionairy
                if is_firstiterate:
                    setattr(self, sedcap_variable_name, {})

                # Get child dict
                child_dict = getattr(sedcap_order, sedcap_variable_name)

                # Set parent dict
                getattr(self, sedcap_variable_name)[k] = child_dict

            is_firstiterate = False




    def _get_sediment_capacity_variable_name_list(self, sedcap_order_list):
        """ creates a list of the hydrodynamic variables of hydro_order """
        sedcap_variable_name_list = []
        sedcap_order = sedcap_order_list[0]

        # Looping
        for variable_name, value in sedcap_order.__dict__.items():
            # Filter out non-hydrodynamic variables:
            # Check if it is a dictionary and not forcing_mechanism_nest_dict
            if isinstance(value, dict) and value is not sedcap_order.forcing_mechanism_nest_dict:
                # True hydrodynamic variable
                sedcap_variable_name_list.append(variable_name)

        return sedcap_variable_name_list