"""
In this file, a class is defined that stores the hydrodynamic results of all orders

# TODO unit test object
"""


class Hydrodynamics():
    """
    Class containing all ordered hydrodynamic results
    # Assumptions:
    1.  We neglect the forcing Wcheck and Wcalcheck components for now
    2.  dU_dzz_V and Omega assume forced component only

    Can be used as follows:
        hydro.Z[0][1]['tide'] for leading-order M2 forced by the tide
    In general:
        hydro.Z[k][n][f] for order k, frequency component n, forcing mechanisms f.
    """


    def __init__(self, hydro_order_list):
        """TODO: maybe also set the same numerical properties as hydro_order"""
        #self.finite_element_space = finite_element_space
        #self.freqcomp_list = forcing_mechanism_collection.frequency_component_list
        #self.forcing_mechanism_nest_dict = forcing_mechanism_collection.forcing_mechanism_nest_dict

        self.hydrodynamic_variable_name_list = self._get_hydrodynamic_variable_name_list(hydro_order_list)


        is_firstiterate = True
        # We create a loop to set the dictionairy keys
        for hydro_order in hydro_order_list:
            # Order
            k = hydro_order.k

            # For each hydrodynamic variable, create new leading level of order index k:
            for hydrodynamic_variable_name in self.hydrodynamic_variable_name_list:
                # create initial dictionairy
                if is_firstiterate:
                    setattr(self, hydrodynamic_variable_name, {})

                # Get child dict
                child_dict = getattr(hydro_order, hydrodynamic_variable_name)

                # Set parent dict
                # Lets try this way
                getattr(self, hydrodynamic_variable_name)[k] = child_dict

                # Before
                #setattr(self, hydrodynamic_variable_name[k], child_dict)

            is_firstiterate = False




    def _get_hydrodynamic_variable_name_list(self, hydro_order_list):
        """ creates a list of the hydrodynamic variables of hydro_order """
        hydrodynamic_variable_name_list = []
        hydro_order = hydro_order_list[0]

        # Looping
        for hydrodynamic_variable_name, value in hydro_order.__dict__.items():
            # Filter out non-hydrodynamic variables:
            # Check if it is a dictionary and not forcing_mechanism_nest_dict
            if isinstance(value, dict) and value is not hydro_order.forcing_mechanism_nest_dict:
                # True hydrodynamic variable
                hydrodynamic_variable_name_list.append(hydrodynamic_variable_name)

        return hydrodynamic_variable_name_list