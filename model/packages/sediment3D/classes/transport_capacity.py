"""
In this file, a closs is defined that computes the transport capacity of sediment
#TODO test this class

"""

import numpy as np

import ngsolve
from numpy.core.defchararray import lstrip

from model.general.classes.discretized.numerical_depth_quadrature import NumericalDepthQuadrature
from model.general.classes.discretized.disc_collection import DiscretizedCollection
from model.general.classes.discretized.function import real, make_matrix, make_diagonal_matrix
from model.packages.sediment3D.classes.numerical_parameters import SedimentNumericalParameters
from model.packages.sediment3D.classes.physical_parameters import SedimentPhysicalParameters


class TransportCapacity:
    """
    Class containing the transport capacity terms

    The transport terms T[f] and diffusive transport D[f] are computed

    Can be used as follows:
        transcap.T[f] for forcing mechanisms f
        transcap.D[f] for forcing mechanisms f
    """
    transport_zero_V = np.zeros((2, 1))


    def __init__(self, disccol: DiscretizedCollection,
                 sediment_physical_parameters: SedimentPhysicalParameters,
                 sediment_numerical_parameters: SedimentNumericalParameters):
        """ We use the hydrodynamic and sediment capacity terms to numerically compute the transport capacity

        """
        self.disccol = disccol

        self.numquad: NumericalDepthQuadrature = sediment_numerical_parameters.numquad
        self.sediment_physical_parameters = sediment_physical_parameters

        self._set_transport_capacity_terms(disccol)

        self._set_diffusive_transport_capacity_terms(disccol)




    def _get_symbol_list(self):
        """ Function to obtain the total forcing symbol list """
        symbol_list = []
        symbol_list.extend(self.T_M2_V.keys())
        symbol_list.extend(self.T_M0_V.keys())
        symbol_list.extend(self.T_M4_V.keys())

        # Only retain unique list
        unique_symbol_list = list(set(symbol_list))

        return unique_symbol_list

    def _set_transport_capacity_terms(self, disccol):
        """ Function to compute the transport capacity terms """

        # Initialize an empty dictionary
        # We fill it with the corresponding indices
        self.T_V = {}

        # We consider each contribution to the transport capacity individually

        # We consider T_M2
        self.T_M2_V = {}
        for hatC_symbol, hatC_forcing in disccol.hatC[1][1].items():
            for alpha, hatC_alpha in hatC_forcing.items():
                if alpha=='a' and hatC_symbol != 'all':
                    value = self.numquad.depth_integal(1/2*real( disccol.U_V[0][1]['tide'] * np.conj(hatC_alpha) ) )
                    self._add_value_to_dict(self.T_M2_V, hatC_symbol, value)



        # We consider T_M0
        self.T_M0_V = {}
        for U_symbol, U_forcing_V in disccol.U_V[1][0].items():
            if U_symbol != 'all':
                # Added real part here to cast complex to real variable
                value = self.numquad.depth_integal(real(U_forcing_V * disccol.hatC[0][0]['etide']['a']))
                self._add_value_to_dict(self.T_M0_V, U_symbol, value)


        # We consider T_M4
        self.T_M4_V = {}
        for U_symbol, U_forcing_V in disccol.U_V[1][2].items():
            if U_symbol != 'all':
                value = self.numquad.depth_integal(1/2 * real(U_forcing_V * np.conj(disccol.hatC[0][2]['etide']['a'])))
                self._add_value_to_dict(self.T_M4_V, U_symbol, value)


        # We consider T_diff
        # Cast to real
        self.T_diff_V = - self.numquad.discretize_function2DH(self.sediment_physical_parameters.Kh) * self.numquad.depth_integal(real(disccol.nablahatC[0][0]['etide']['a']))


        # We consider T_stokes, which is evaluated at the free surface z=R:
        # TODO check if eval at surface
        # TODO check what happens when we do not squeeze a dim e.g. hatC near the surface
        self.T_stokes_V = real(
                            1/2*np.conj(disccol.Z[0][1]['tide'])
                            * disccol.U_V[0][1]['tide'][:, :, -1]
                            * disccol.hatC[0][0]['etide']['a'][:, :, -1]
                            +
                            1/4 * disccol.Z[0][1]['tide']
                            * disccol.U_V[0][1]['tide'][:, :, -1]
                            * np.conj(disccol.hatC[0][2]['etide']['a'][:, :, -1])
        )


        # Build symbol list
        self.symbol_list = self._get_symbol_list()

        # Next, we compute the forcing decomposition of the advective terms
        for symbol in self.symbol_list:
            self.T_V[symbol] = self.T_M2_V.get(symbol, 0) + self.T_M0_V.get(symbol, 0) + self.T_M4_V.get(symbol, 0)

        # We also input the other transports
        self.T_V['diff'] = self.T_diff_V
        self.T_V['stokes'] = self.T_stokes_V

        # We sum the indices
        self.T_V['all'] = sum(self.T_V.values())


    def _set_diffusive_transport_capacity_terms(self, disccol):
        """ Function to set the transport capacity terms """

        # Initialize an empty dictionary.
        # We fill it with the corresponding indices
        self.D_M = {}

        # We consider D_M2_M
        # TODO check if the depth integral works for vectors
        self.D_M2_ax_V = self.numquad.depth_integal(1/2*real(disccol.U_V[0][1]['tide'] * np.conj(disccol.hatC[1][1]['sedadv']['a_x'])))
        self.D_M2_ay_V = self.numquad.depth_integal(1/2*real(disccol.U_V[0][1]['tide'] * np.conj(disccol.hatC[1][1]['sedadv']['a_y'])))

        self.D_M2_M = make_matrix(self.D_M2_ax_V, self.D_M2_ay_V)

        # We consider D_diff
        # TODO Check dims
        # Cast to real
        self.D_diff_M = make_diagonal_matrix(-self.numquad.discretize_function2DH(self.sediment_physical_parameters.Kv) * self.numquad.depth_integal(real(disccol.hatC[0][0]['etide']['a'])))

        # We collect the contributions of the diffusive sediment transport
        self.D_M['M2'] = self.D_M2_M
        self.D_M['diff'] = self.D_diff_M

        # We sum the indices
        self.D_M['all'] = sum(self.D_M.values())




    def _add_value_to_dict(self, dict_obj, key, value):
        """ Function to add the contributions of a dictionairy if they share the same key
        NB We remove the leading e """

        # Some keys contain an e for erosion, for consistency we remove any leading e's
        key_new = key.lstrip('e')

        # If two items have the same key_new we add them:
        dict_obj[key_new] = dict_obj.get(key_new, 0) + value
        return None