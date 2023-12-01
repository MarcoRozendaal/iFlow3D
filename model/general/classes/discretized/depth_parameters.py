"""
In this file, a numerical depth parameter object is defined that collects the numerical depth parameters in a single class.
"""

class NumericalDepthParameters():
    """"
    Class containing the numerical parameters required for the depth integration
    """

    def __init__(self, number_sigma_layers):
       # Initialize numerical parameter object
       self.number_sigma_layers = number_sigma_layers
