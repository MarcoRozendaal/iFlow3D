"""
In this file, a class is defined that collects all the discretized quantities,
so they can be used in the depth integrals in the transport capacity
"""
from model.general.classes.discretized.numerical_depth_quadrature import NumericalDepthQuadrature
from model.general.classes.nested_default_dict import NestedDefaultDict
from model.packages.hydrodynamics3D.classes.hydrodynamics import Hydrodynamics


class DiscretizedCollection:
    """
    Class to collect discretized quantities neededfor the transport capacity computation

    """

    # TODO Think If I need initialization
    def __init__(self, numquad: NumericalDepthQuadrature):
        """
        Initialization
        Args:
        """
        # Initialization
        self.numquad = numquad



    def set_hydro_variables(self, hydro: Hydrodynamics):
        """ Set the discretized hydrodynamic variables """

        print("Discretizing hydrodynamic variables")


        # TODO denken wat hier gebeurt, we discretizen een vector, gaat dat automatisch goed of moeten we nog wat extras doen?
        # Set the discretized U_V components
        U_V = NestedDefaultDict()
        for k, U_kV in hydro.U_V.items():
            for n, U_knV in U_kV.items():
                for f, U_knfV in U_knV.items():
                    print(" n:" + str(n) + " f:" + f)
                    U_V[k][n][f] = self.numquad.discretize_function3D(U_knfV)

        # Set make it a property
        self.U_V = U_V.get_dict()


        # Set W01A
        W = NestedDefaultDict()
        W[0][1]['tide'] = self.numquad.discretize_function3D(hydro.W[0][1]['tide'])
        self.W = W.get_dict()


        # Set Z01A
        Z = NestedDefaultDict()
        Z[0][1]['tide'] = self.numquad.discretize_function2DH(hydro.Z[0][1]['tide'])
        self.Z = Z.get_dict()


        print("Finished discretizing hydrodynamic variables")




    def set_sedcaplead_variables(self, sedcaplead):
        """ Set the discretized sediment capacity leading-order variables for the transport capacity and the advection of sediment foricng """

        print("Discretizing leading-order sediment capacity variables")

        # leading-order hatC
        hatC = NestedDefaultDict()
        for n, hatCn in sedcaplead.hatC.items():
            for f, hatCf in hatCn.items():
                for alpha, hatCfalpha in hatCf.items():
                    print(" n:" + str(n) + " f:" + f + " alpha:" + alpha)
                    hatC[0][n][f][alpha] = self.numquad.discretize_function3D(hatCfalpha)
        # Set dict
        self.hatC = hatC.get_dict()


        # leading-order nablahatC
        nablahatC = NestedDefaultDict()
        for n, nablahatCn in sedcaplead.nablahatC.items():
            for f, nablahatCnf in nablahatCn.items():
                for alpha, nablahatCnfalpha in nablahatCnf.items():
                    nablahatC[0][n][f][alpha] = self.numquad.discretize_function3D(nablahatCnfalpha)
        # Set dict
        self.nablahatC = nablahatC.get_dict()


        # leading-order hatC_z
        hatC_z = NestedDefaultDict()
        for n, hatC_zn in sedcaplead.hatC_z.items():
            for f, hatC_znf in hatC_zn.items():
                for alpha, hatC_znfalpha in hatC_znf.items():
                    hatC_z[0][n][f][alpha] = self.numquad.discretize_function3D(hatC_znfalpha)
        # Set dict
        self.hatC_z = hatC_z.get_dict()

        print("Finished discretizing leading-order sediment capacity variables")

    def set_sedcapfirst_variables(self, sedcapfirst):
        """ Set the discretized sediment capacity first-order variables for the transport capacity """

        print("Discretizing first-order sediment capacity variables")

        # first-order hatC
        hatC = NestedDefaultDict()
        for n, hatCn in sedcapfirst.hatC.items():
            for f, hatCf in hatCn.items():
                for alpha, hatCfalpha in hatCf.items():
                    print(" n:" + str(n) + " f:" + f + " alpha:" + alpha)
                    hatC[n][f][alpha] = self.numquad.discretize_function3D(hatCfalpha)
        # Set dict
        self.hatC[1] = hatC.get_dict()

    def set_sedadv_variables(self, list):
        """ Function to manually set the discretized advection of sediment forcings """

        print("Adding the discretized first-order sediment capacity variables")

        # We extend the first-order sediment capacity with the sediment advection contributions
        self.hatC[1][1]['sedadv'] = {}

        # first-order hatC
        for formech_sedadv in list:
            n = formech_sedadv.n
            f = formech_sedadv.symbol
            alpha = formech_sedadv.alpha
            hatC_sedadv = formech_sedadv.sediment_capacity_forcing_components.hatC

            print(" n:" + str(n) + " f:" + f + " alpha:" + alpha)
            self.hatC[1][n][f][alpha] = hatC_sedadv

        print("Finished adding the discretized first-order sediment capacity variables")


