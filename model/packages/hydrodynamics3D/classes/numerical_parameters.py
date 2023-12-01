"""
In this file, a general hydrodynamic numerical parameter object is defined, which collects the numerical parameters in a single class.
"""

class HydrodynamicNumericalParameters():
    """"
    Class containing the numerical parameters required to solve the hydrodynamics
    """
    def __init__(self, geometry, mesh, global_maxh, geometry_spline_degree, curved_elements_degree, order_basisfunctions):
       # Initialize numerical parameter object
       self.geometry = geometry
       self.mesh = mesh
       self.global_maxh = global_maxh
       self.geometry_spline_degree = geometry_spline_degree
       self.curved_elements_degree = curved_elements_degree
       self.order_basisfunctions = order_basisfunctions