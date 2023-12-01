""""
File that contains the bathymetry profiles

Each bathymetry profile is an instance of the BottomProfile class.
"""

import math
import ngsolve
import numpy as np
from model.general import post_processing as pp
from scipy.interpolate import Rbf


class BottomProfile():
    """"
    The BottomProfile class containing:
         h -- the bathymetry
         hx -- the x derivative of h
         hy -- the y derivative of h
    """
    def __init__(self, h, hx, hy):
        self.h = h
        self.hx = hx
        self.hy = hy


def gauss_profile(Hmax, H0, psi, B):
    """"
    A Gaussian profile for a rectangular channel

    The Gauss profile is defined as
        h = Hmax *  exp(-C * Y ^ 2)
    with
        C = log(Hmax/H0)
        Y = 1-sqrt(1 + psi ^ 2 - 2 * psi * y / B)/psi
    Arguments:
        Hmax    -- maximal depth_num of the main channel
        H0      -- minimal depth_num at the boundary of the channel
        psi     -- skewness of the channel -1<psi<1
        B       -- HALF width of the channel
    """
    C = math.log(Hmax/H0)
    if psi == 0:
        Y = ngsolve.y/B
    else:
        Y = (1 - ngsolve.sqrt(1 + psi ** 2 - 2 * psi * ngsolve.y / B))/psi
    Hmaxx = 0

    h = Hmax * ngsolve.exp(-C * Y ** 2)
    hx = (- Y ** 2 + 1) * Hmaxx * ngsolve.exp(-C * Y ** 2)
    hy = -2 * Hmax*ngsolve.exp(- C * Y ** 2) * Y * C / (ngsolve.sqrt(1 + psi ** 2 + 2 * psi * ngsolve.y / B) * B)
    return BottomProfile(h, hx, hy)


def flat_profile(Hmax):
    """"
    A flat bottom profile for any geometry

    Arguments:
        Hmax    -- uniform depth_num of the flat profile
    """
    h = ngsolve.CoefficientFunction(Hmax)
    hx = ngsolve.CoefficientFunction(0)
    hy = ngsolve.CoefficientFunction(0)
    return BottomProfile(h, hx, hy)


def step_profile(Hmax, Hmin, shift=0):
    """"
    A step bottom profile for a rectangular channel

    Arguments:
        Hmin    -- lowest level of step
        Hmax    -- highest level of step
    """
    h = ngsolve.IfPos(ngsolve.y-shift, ngsolve.CoefficientFunction(Hmin), ngsolve.CoefficientFunction(Hmax))
    hx = ngsolve.CoefficientFunction(0)
    hy = ngsolve.CoefficientFunction(0)
    return BottomProfile(h, hx, hy)


def linear_profile(Hmax, Hmin,  B):
    """"
    A linear bottom profile for a rectangular channel

    Arguments:
        Hmin    -- lowest level of linear incline
        Hmax    -- highest level of linear incline
        B       -- HALF width of the channel
    """
    h = Hmin + (ngsolve.y + B) / (2*B) * (Hmax-Hmin)
    hx = ngsolve.CoefficientFunction(0)
    hy = ngsolve.CoefficientFunction( 1 / (2*B) * (Hmax-Hmin))
    return BottomProfile(h, hx, hy)



def quadratic_profile(Hmax, Hmin,  B):
    """"
    A quadratic bottom profile for a rectangular channel

    Arguments:
        Hmin    -- lowest level of linear incline
        Hmax    -- highest level of linear incline
        B       -- HALF width of the channel
    """
    h = Hmin + (Hmax-Hmin) * (1 - (ngsolve.y/B) ** 2 )
    hx = ngsolve.CoefficientFunction(0)
    hy = ngsolve.CoefficientFunction( - (Hmax-Hmin) * 2 * ngsolve.y/B**2 ) # TODO check
    return BottomProfile(h, hx, hy)




## The code below was used to test different methods to generate more complex bathymetric profiles
def linear_gauss_profile(Hmax, H0, psi, B, L, maxh, mesh, geo):
    """
    Creates a BottomProfile based on a linear approximation of the gauss_profile

    An gauss_profile is created. Then the gauss_profile is approximated by linear triangular elements
    on a finer mesh than the computational mesh and returned.
    Arguments:
        Hmax    -- maximal depth_num of the main channel
        H0      -- minimal depth_num at the boundary of the channel
        psi     -- skewness of the channel -1<psi<1
        B       -- Half width of the channel
        L       -- length of the channel
        maxh    -- maximal distance between two nodes in a triangle in the computational mesh
        mesh    -- the computational mesh
        geo     -- geometry of the computational mesh
    """
    h_gauss_profile = gauss_profile(Hmax, H0, psi, B)
    h, hx, hy = linearize_bathymetry(h_gauss_profile.h, maxh, geo)
    return BottomProfile(h, hx, hy)


def linear_fitted_gauss_profile(Hmax, H0, psi, B, L, maxh, mesh, geo):
    """
    Creates a BottomProfile based on a linear approximation of a fit of the gauss_profile

    An gauss_profile is created. The gauss_profile is sampled at random points from which an radial basis function fit
    is created. The fit is approximated by linear triangular elements on a finer mesh than the computational mesh
    and returned.
    Arguments:
        Hmax    -- maximal depth_num of the main channel
        H0      -- minimal depth_num at the boundary of the channel
        psi     -- skewness of the channel -1<psi<1
        B       -- Half width of the channel
        L       -- length of the channel
        maxh    -- maximal distance between two nodes in a triangle in the computational mesh
        mesh    -- the computational mesh
        geo     -- geometry of the computational mesh
    """
    # Parameters
    np.random.seed(100) # Set the seed for consistency
    number_samples = 12000
    smoothness = 0

    # Create gauss_profile
    h_gauss_profile = gauss_profile(Hmax, H0, psi, B)

    # Generate sampled h_data
    x_sample = L * np.random.rand(number_samples)
    y_sample = B * (2 * np.random.rand(number_samples) - 1)
    h_sample = pp.evaluateV(h_gauss_profile.h, mesh, np.column_stack([x_sample, y_sample]))
    h_data =  np.column_stack([x_sample, y_sample, h_sample])

    # Create a bathymetry fit and linearly approximate it
    h_fit = fit_bathymetry_data(h_data, smoothness)
    h, hx, hy = linearize_bathymetry(h_fit, maxh, geo)
    return BottomProfile(h, hx, hy)


def fit_bathymetry_data(h_data, s):
    """"
    Creates a fit of the bathymetry data

    The bathymetry data is fitted using radial basis functions.
    Arguments:
        h_data   -- bathymetry data. Numpy array structered as [x,y,h]
        s       -- smoothness_spline_fit of radial basis function fit
    """
    h_radial_fit = Rbf(h_data[:, 0], h_data[:, 1], h_data[:, 2], smooth=s, function='gaussian')
    return h_radial_fit


def linearize_bathymetry(h, maxh, geo):
    """"
    Creates a linear approximation of the bathymetry

    A refined bathymetry mesh is created. A linear bathymetry gridfunction is created such that
    it equals the profided bottom h at the refined bathymetry mesh vertices.
    Arguments:
        h       -- bottom profile to be linearized. Can be a coefficient function or radial basis function fit
        maxh    -- maximal distance between two nodes in a triangle in the computational mesh
        mesh    -- the computational mesh
    """
    refinement = 1

    # Generate bathymetry mesh and linear bathymetry gridfunction
    mesh_bed = ngsolve.Mesh(geo.GenerateMesh(maxh=maxh/refinement))
    fes_bed = ngsolve.H1(mesh_bed, order=1)
    h_bed = ngsolve.GridFunction(fes_bed)

    # Evaluate h on bathymetry mesh vertices
    h_mesh = []
    for v in mesh_bed.vertices:
        if isinstance(h, ngsolve.fem.CoefficientFunction):
            h_mesh.append(h(mesh_bed(*v.point)))
        else:
            h_mesh.append(h(*v.point))

    # Set bathymetry basis functions coefficients equal to h at the bathymetry vertices
    h_bed.vec.FV().NumPy()[:] = h_mesh

    # Compute horizontal derivatives
    hx_bed = ngsolve.GridFunction(fes_bed)
    hy_bed = ngsolve.GridFunction(fes_bed)
    hx_bed.Set(ngsolve.Grad(h_bed)[0])
    hy_bed.Set(ngsolve.Grad(h_bed)[1])

    #hx_bed = ngsolve.Grad(h_bed)[0] # TODO zombie code
    #hy_bed = ngsolve.Grad(h_bed)[1]

    return h_bed, hx_bed, hy_bed


def linearize_bathymetry_exact_derivatives(hbp, maxh, geo):
    """"
    Creates a linear approximation of the bathymetry

    A refined bathymetry mesh is created. A linear bathymetry gridfunction is created such that
    it equals the profided bottom h at the refined bathymetry mesh vertices.
    Arguments:
        h       -- bottom profile to be linearized. Can be a coefficient function or radial basis function fit
        maxh    -- maximal distance between two nodes in a triangle in the computational mesh
        mesh    -- the computational mesh
    """
    h = hbp.h
    refinement = 1

    # Generate bathymetry mesh and linear bathymetry gridfunction
    mesh_bed = ngsolve.Mesh(geo.GenerateMesh(maxh=maxh/refinement))
    fes_bed = ngsolve.H1(mesh_bed, order=1)
    h_bed = ngsolve.GridFunction(fes_bed)

    # Evaluate h on bathymetry mesh vertices
    h_mesh = []
    for v in mesh_bed.vertices:
        if isinstance(h, ngsolve.fem.CoefficientFunction):
            h_mesh.append(h(mesh_bed(*v.point)))
        else:
            h_mesh.append(h(*v.point))

    # Set bathymetry basis functions coefficients equal to h at the bathymetry vertices
    h_bed.vec.FV().NumPy()[:] = h_mesh

    # Compute horizontal derivatives
    hx_bed = hbp.hx
    hy_bed = hbp.hy

    return h_bed, hx_bed, hy_bed


def project_bathymetry(hbp, fesh):
    h_smooth = ngsolve.GridFunction(fesh)
    h_smooth.Set(hbp.h)
    hbp.h = h_smooth
    hbp.hx = ngsolve.Grad(h_smooth)[0]
    hbp.hy = ngsolve.Grad(h_smooth)[1]
    return hbp


def rbf_fitted_gauss_profile(Hmax, H0, psi, B, L, maxh, mesh, geo):
    """
    Creates a BottomProfile based on a radial basis function fit of the gauss_profile

    An gauss_profile is created. The gauss_profile is sampled at random points from which an radial basis function fit
    is created. The radial basis function fit is then converted to an NGSolve rbf object.
    Arguments:
        Hmax    -- maximal depth_num of the main channel
        H0      -- minimal depth_num at the boundary of the channel
        psi     -- skewness of the channel -1<psi<1
        B       -- Half width of the channel
        L       -- length of the channel
        maxh    -- maximal distance between two nodes in a triangle in the computational mesh
        mesh    -- the computational mesh
        geo     -- geometry of the computational mesh
    """
    # Parameters
    np.random.seed(100) # Set the seed for consistency
    number_samples = 1000
    smoothness = 0

    # Create gauss_profile
    h_gauss_profile = gauss_profile(Hmax, H0, psi, B)

    # Generate sampled h_data
    x_sample = L * np.random.rand(number_samples)
    y_sample = B * (2 * np.random.rand(number_samples) - 1)
    h_sample = pp.evaluateV(h_gauss_profile.h, mesh, np.column_stack([x_sample, y_sample]))
    h_data =  np.column_stack([x_sample, y_sample, h_sample])

    # Create rbf bathymetry fit
    h_fit = fit_bathymetry_data(h_data, smoothness)

    h, hx, hy = rbf_bathymetry(h_fit, maxh, geo)
    return BottomProfile(h, hx, hy)


def rbf_bathymetry(h_fit, maxh, geo):
    epsilon = h_fit.epsilon
    xi = h_fit.xi
    di = h_fit.di

    rbf_ngsolve = ngsolve.CoefficientFunction(0)
    for i in range(len(di)):
        r = ngsolve.sqrt((ngsolve.x-xi[0, i])**2 + (ngsolve.y-xi[1, i])**2)
        rbf_ngsolve = rbf_ngsolve + di[i]*ngsolve.exp(-(r/epsilon)**2)

    return rbf_ngsolve, rbf_ngsolve.Diff(ngsolve.x), rbf_ngsolve.Diff(ngsolve.y)