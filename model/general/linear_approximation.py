"""In this file a NGSolve gridfunction is generated from a point cloud

The point cloud input data is provided as
pointclouddata = np.array([[x1,y1,s1],[x2,y2,s2],[x3,y3,s3]...])
"""

import numpy as np
from scipy.interpolate import Rbf

import ngsolve


def generate_pointclouddata(eta, mesh):
    """
    Function used to generate sample point cloud data.
    Parameters:
    - eta : ngsolve grid function
    - mesh: computational mesh
    Returns:
    - pointclouddata: Pointclouddata in the form np.array([[x1,y1,s1],[x2,y2,s2],...])
    """
    # Evaluate the gridfunction xi in the mesh vertices
    pointclouddata = []
    for v in mesh.vertices:
        x,y = v.point
        s = eta(mesh(x,y))
        pointclouddata.append([x, y, s])
    return np.array(pointclouddata)

def rbf_fit(pointclouddata):
    """
       Function generates a radial basis function fit of the given point cloud data
       Parameters:
           - pointclouddata: Pointclouddata in the form np.array([[x1,y1,s1],[x2,y2,s2],...])
       Returns:
           - s_rbf_fit: fit object from numpy
       """
    s_rbf_fit = Rbf(pointclouddata[:, 0], pointclouddata[:, 1], pointclouddata[:, 2], smooth=0, function='gaussian')
    return s_rbf_fit


def creategridfunction(s_fit, mesh):
    """
    Function used to create a linear NGSolve gridfunction from a given fitting object
    Parameters:
        - s_fit : fit object from numpy
        - mesh: computational mesh
    Returns:
        - s_gf: NGSolve grid function that linearly approximates the fit
    """

    # Generate a linear approximation space on the mesh
    fes_approximation = ngsolve.H1(mesh, order=1)
    s_gf = ngsolve.GridFunction(fes_approximation)

    # Evaluate s_fg on the mesh vertices
    s_meshvertices = []
    for v in mesh.vertices:
        s_meshvertices.append(s_fit(*v.point))

    # Set the coefficients of s_gf equal to s_meshvertices at the mesh vertices
    s_gf.vec.FV().NumPy()[:] = s_meshvertices
    return s_gf
