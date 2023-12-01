"""
In this file, a general sediment numerical parameter object is defined, which collects the numerical parameters in a single class.
"""

class SedimentNumericalParameters():
    """"
    Class containing the numerical parameters required to solve the sediment dynamics
    """
    def __init__(self, geometry, mesh, global_maxh, geometry_spline_degree, curved_elements_degree, order_basisfunctions,
                 numerical_depth_quadrature):
       # Initialize numerical parameter object
       self.geometry = geometry
       self.mesh = mesh
       self.global_maxh = global_maxh
       self.geometry_spline_degree = geometry_spline_degree
       self.curved_elements_degree = curved_elements_degree
       self.order_basisfunctions = order_basisfunctions
       self.numquad = numerical_depth_quadrature

       # TODO maybe add some parameters regarding the vertical numerical procedure used for the sediment dynamics