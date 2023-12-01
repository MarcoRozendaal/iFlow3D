"""In this file, the leading- and first-order hydrodynamic equations are solved under the standard forcing conditions.
The spatial parameters defined in terms of the along-channel coordinate xi and the across-channel coordinate eta.

Specialised for the NCK Days 2023

"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import importlib
import time
import sympy
import ngsolve


from model.general.boundary_fitted_coordinates import generate_bfc
from model.packages.hydrodynamics3D.classes.order import HydrodynamicsOrder
from model.general.classes.spatial_parameter import SpatialParameter
from model.packages.hydrodynamics3D.classes.forcing_mechanism import HydrodynamicForcingMechanism
from model.packages.hydrodynamics3D.classes.forcing_mechanism_collection import HydrodynamicForcingMechanismCollection
from model.packages.hydrodynamics3D.classes.physical_parameters import HydrodynamicPhysicalParameters
from model.packages.hydrodynamics3D.classes.numerical_parameters import HydrodynamicNumericalParameters
from model.packages.hydrodynamics3D.scripts.compute_hydrodynamics_order import compute_hydrodynamics_order

from model.packages.hydrodynamics3D.forcing.baroclinic import HydrodynamicForcingBaroclinic
from model.packages.hydrodynamics3D.forcing.nostress import HydrodynamicForcingNoStress
from model.packages.hydrodynamics3D.forcing.stokes import HydrodynamicForcingStokes

from model.general.create_geometry import parametric_geometry, WALLDOWN, WALLUP
from model.general import post_processing as pp
from model.general import geometries


# Function to plot a scalar over the cross-section
def plot_line_scalar(gfu, p1, p2, post, **kwargs):
    """Function to plot scalar over the cross-section"""
    nxy = 200
    nudging_parameter = 1 / 1000

    tangential_vector = p2 - p1

    # Nudge points inside domain
    p1 = post.nudge_point_inside_domain(p1, tangential_vector, nudging_parameter)
    p2 = post.nudge_point_inside_domain(p2, -tangential_vector, nudging_parameter)

    # Generate display grids
    xline = np.linspace(p1[0], p2[0], nxy)
    yline = np.linspace(p1[1], p2[1], nxy)
    xtilde = np.sqrt((xline - xline[0]) ** 2 + (yline - yline[0]) ** 2)


    gfu_e = gfu(post.mesh(xline, yline))


    # Plot the solution
    fig = plt.figure()
    plt.plot(xtilde, gfu_e, **kwargs)
    return fig

def arg(gfu):
    """Function to compute the argument of complex number"""
    rad2deg = 57.2957795
    return ngsolve.atan2(gfu.imag, gfu.real) * rad2deg



#################### Input ####################
### Geometric parameters
# Exponential Rational
C1 = [-0.02742e-3, 1.8973]
C2 = [4.9788e-11, -9.213e-6, 1]
B0 = 6667.866875
L = 161e3
geometrycurves = geometries.exponential_rational(C1, C2, B0, L)


# Geometry parameters
degree_spline_geometry = 3
degree_curved_geometry = 3
smoothness_spline_geometry = 0

### Numeric parameters
# Maximal global mesh size
maxh_global = 1e3/1.4

# Maximal mesh size along the partitioned boundary
boundary_parameter_partitition_dict = {WALLDOWN: [0, 0.2, 0.8, 1], WALLUP: [0, 0.2, 0.8, 1]}
boundary_maxh_dict = {WALLDOWN: [maxh_global, maxh_global/4, maxh_global/5], WALLUP: [maxh_global/5, maxh_global/4, maxh_global]}



# FEM parameter
order_basisfunctions_lead = 15

### Physical parameters
# Spatially constant parameters
g = 9.81
f = 1e-4
omega = 1.4e-4

# Spatially varying parameters in terms of xi and eta using sympy functions
"""
def H(xi, eta):
    # Bathymetry
    Hmax = 10
    H0 = 2
    C = sympy.log(Hmax/H0)
    return Hmax * sympy.exp(-C * eta ** 2)

"""
def H(xi, eta):
    # Double channel bathymetry
    Hmax = 11 # Determined to obtain approximately the same average
    Hmin = 2
    Havg = (Hmax+Hmin)/2
    Hdiff = (Hmax-Hmin)/2
    return Havg - Hdiff * sympy.cos(2*sympy.pi*eta)





def R(xi, eta):
    """Subtidal reference level"""
    return 0

def Av0(xi, eta):
    """Leading-order Av0"""
    return 0.01

def sf0(xi, eta):
    """Leading-order sf0"""
    return 0.003





# Parameters for the leading-order forcing
A01 = 1  # Tidal M2 amplitude forcing

# Parameters for the first-order forcing
# For n=0:
q10 = 1.2  # River discharge forcing

# Parameters for the first-order hydrodynamics
rho0 = 1020

def rho(xi, eta):
    """ Spatially varying density field. We use a hyperbolic tangent profile from Talke2009 """
    beta = 7.6e-4  # per psu
    Ssea = 30  # psu
    xi_L = 0.235 # [-]
    xi_c = 0.5 # [-]
    S = Ssea/2 * (1 - sympy.tanh( (xi-xi_c)/xi_L ))
    return rho0 * (1 + beta * S)

# For n=2:
A12 = 0.1  # Tidal M4 amplitude forcing



################ Computation ################
initial_time = time.time()

# NGSolve debug parameter
# ngsglobals.msg_level = 10

# Create geometry
#geometry = general_spline_geometry(geometrydata, geometry_spline_degree, geometry_spline_smoothness,
#                                      boundary_parameter_partitition_dict, boundary_maxh_dict)
geometry = parametric_geometry(geometrycurves, boundary_parameter_partitition_dict, boundary_maxh_dict)



# Boundary fitted coordinate mesh
mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_global))
mesh.Curve(degree_curved_geometry)
print("Number of vertices:", mesh.nv, ", Number of elements:", mesh.ne)

# Generate boundary fitted coordinates
bfc = generate_bfc(mesh, order_basisfunctions_lead, "diffusion", alpha=1)

# Define spatial parameters as coefficient functions
Av0_sp = SpatialParameter(Av0, bfc)
sf0_sp = SpatialParameter(sf0, bfc)
H_sp = SpatialParameter(H, bfc)
R_sp = SpatialParameter(R, bfc)

# Baroclinic forcing sp
rho_sp = SpatialParameter(rho, bfc)

# The leading-order forcing mechanisms: tidal forcing
tide_forcing01 = HydrodynamicForcingMechanism(0, 1, "tide", A_sea=A01)

# Generate leading-order numerical, physical parameter and forcing mechanisms classes
numparams_hydrolead = HydrodynamicNumericalParameters(geometry, mesh, maxh_global, degree_spline_geometry,
                                                      degree_curved_geometry, order_basisfunctions_lead)
physparams_hydrolead = HydrodynamicPhysicalParameters(g, f, omega, Av0_sp, sf0_sp, H_sp, R_sp)
formechs_hydrolead = HydrodynamicForcingMechanismCollection([tide_forcing01])


# Solve the leading-order water motion
res_hydrolead: HydrodynamicsOrder = compute_hydrodynamics_order(numparams_hydrolead, physparams_hydrolead, formechs_hydrolead)
# Include the along and across-channel velocities
res_hydrolead.set_U_xi_U_eta(bfc)


# The external first-order forcing mechanisms: baroclinic forcing, nostress, tidal return flow
# The first-order forcing mechanisms for n=0:
river_forcing10 = HydrodynamicForcingMechanism(1, 0, "river", q_R_river=q10)

baroc_forcing_components10 = HydrodynamicForcingBaroclinic(rho0, rho_sp.gradient_cf, physparams_hydrolead)
baroc_forcing10 = HydrodynamicForcingMechanism(1, 0, "baroc",
                                               hydrodynamic_forcing=baroc_forcing_components10)

nostress_forcing_components10 = HydrodynamicForcingNoStress(res_hydrolead, physparams_hydrolead)
nostress_forcing10 = HydrodynamicForcingMechanism(1, 0, "nostress",
                                                  hydrodynamic_forcing=nostress_forcing_components10)

stokes_forcing_components10 = HydrodynamicForcingStokes(res_hydrolead, physparams_hydrolead)
stokes_forcing10 = HydrodynamicForcingMechanism(1, 0, "stokes",
                                                hydrodynamic_forcing=stokes_forcing_components10,
                                                useonboundary=False)

# The first-order forcing components for n=2:
tide_forcing12 = HydrodynamicForcingMechanism(1, 2, "tide", A_sea=A12)

nostress_forcing_components12 = HydrodynamicForcingNoStress(res_hydrolead, physparams_hydrolead)
nostress_forcing12 = HydrodynamicForcingMechanism(1, 2, "nostress",
                                                  hydrodynamic_forcing=nostress_forcing_components12)

stokes_forcing_components12 = HydrodynamicForcingStokes(res_hydrolead, physparams_hydrolead)
stokes_forcing12 = HydrodynamicForcingMechanism(1, 2, "stokes",
                                                hydrodynamic_forcing=stokes_forcing_components12,
                                                useonboundary=False)

# Generate forcing mechanism class for first-order
formechs_hydrofirst = HydrodynamicForcingMechanismCollection(
    [river_forcing10, baroc_forcing10, nostress_forcing10, stokes_forcing10, tide_forcing12, nostress_forcing12,
     stokes_forcing12])

# TODO we removed the first-order water motion since we do not use it here
# Solve first-order water motion
#res_hydrofirst: HydrodynamicsOrder = compute_hydrodynamics_order(numparams_hydrolead, physparams_hydrolead,
#                                                                 formechs_hydrofirst)
# Include the along and across-channel velocities
#res_hydrofirst.set_U_xi_U_eta(bfc)




print("Done computing in {:.5f} seconds".format(time.time() - initial_time))
t1 = time.time()



############## Output ##############

"""
# Below are examples of how to plot the flow variables using NGSolve:
ngsolve.Draw(-hydrophysparams.H, mesh, "-H")

# Leading-order free surface
ngsolve.Draw(ngsolve.Norm(hydrolead.Z[1]['all']), mesh, "Z01_all")

# Leading-order velocity (tidal ellipse)
ngsolve.Draw(hydrolead.M_DA[1]['all'], mesh, "M01_DA_all")
ngsolve.Draw(hydrolead.m_DA[1]['all'], mesh, "m01_DA_all")
ngsolve.Draw(hydrolead.theta_DA[1]['all'], mesh, "theta01_DA_all")
ngsolve.Draw(hydrolead.psi_DA[1]['all'], mesh, "psi01_DA_all")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrolead.U_xi_DA[1]['all']), mesh, "U01_xi_DA_all")
ngsolve.Draw(ngsolve.Norm(hydrolead.U_eta_DA[1]['all']), mesh, "U01_eta_DA_all")


# The depth_num-averaged vertical velocity
#ngsolve.Draw(hydrolead.W_DA[1]['all'], mesh, "W01_DA_all")


# First-order M0:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['all']), mesh, "Z10_all")

# First-order velocity (tidal ellipse)
ngsolve.Draw(hydrofirst.M_DA[0]['all'], mesh, "M10_DA_all")
ngsolve.Draw(hydrofirst.m_DA[0]['all'], mesh, "m10_DA_all")
ngsolve.Draw(hydrofirst.theta_DA[0]['all'], mesh, "theta10_DA_all")
ngsolve.Draw(hydrofirst.psi_DA[0]['all'], mesh, "psi10_DA_all")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['all']), mesh, "U10_xi_DA_all")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[0]['all']), mesh, "U10_eta_DA_all")



# Baroc
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['baroc']), mesh, "Z10_baroc")
ngsolve.Draw(hydrofirst.M_DA[0]['baroc'], mesh, "M10_DA_baroc")
ngsolve.Draw(hydrofirst.m_DA[0]['baroc'], mesh, "m10_DA_baroc")
ngsolve.Draw(hydrofirst.theta_DA[0]['baroc'], mesh, "theta10_DA_baroc")
ngsolve.Draw(hydrofirst.psi_DA[0]['baroc'], mesh, "psi10_DA_baroc")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['baroc']), mesh, "U10_xi_DA_baroc")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[0]['baroc']), mesh, "U10_eta_DA_baroc")


# Stokes
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['stokes']), mesh, "Z10_stokes")
ngsolve.Draw(hydrofirst.M_DA[0]['stokes'], mesh, "M10_DA_stokes")
ngsolve.Draw(hydrofirst.m_DA[0]['stokes'], mesh, "m10_DA_stokes")
ngsolve.Draw(hydrofirst.theta_DA[0]['stokes'], mesh, "theta10_DA_stokes")
ngsolve.Draw(hydrofirst.psi_DA[0]['stokes'], mesh, "psi10_DA_stokes")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['stokes']), mesh, "U10_xi_DA_stokes")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[0]['stokes']), mesh, "U10_eta_DA_stokes")

# river
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['river']), mesh, "Z10_river")
ngsolve.Draw(hydrofirst.M_DA[0]['river'], mesh, "M10_DA_river")
ngsolve.Draw(hydrofirst.m_DA[0]['river'], mesh, "m10_DA_river")
ngsolve.Draw(hydrofirst.theta_DA[0]['river'], mesh, "theta10_DA_river")
ngsolve.Draw(hydrofirst.psi_DA[0]['river'], mesh, "psi10_DA_river")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['river']), mesh, "U10_xi_DA_river")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[0]['river']), mesh, "U10_eta_DA_river")


# No-stress
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['nostress']), mesh, "Z10_nostress")
ngsolve.Draw(hydrofirst.M_DA[0]['nostress'], mesh, "M10_DA_nostress")
ngsolve.Draw(hydrofirst.m_DA[0]['nostress'], mesh, "m10_DA_nostress")
ngsolve.Draw(hydrofirst.theta_DA[0]['nostress'], mesh, "theta10_DA_nostress")
ngsolve.Draw(hydrofirst.psi_DA[0]['nostress'], mesh, "psi10_DA_nostress")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['nostress']), mesh, "U10_xi_DA_nostress")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[0]['nostress']), mesh, "U10_eta_DA_nostress")


# First-order M4:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[2]['all']), mesh, "Z12_all")
# First-order velocities (tidal ellipse)
ngsolve.Draw(hydrofirst.M_DA[2]['all'], mesh, "M12_DA_all")
ngsolve.Draw(hydrofirst.m_DA[2]['all'], mesh, "m12_DA_all")
ngsolve.Draw(hydrofirst.theta_DA[2]['all'], mesh, "theta12_DA_all")
ngsolve.Draw(hydrofirst.psi_DA[2]['all'], mesh, "psi12_DA_all")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['all']), mesh, "U12_xi_DA_all")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[2]['all']), mesh, "U12_eta_DA_all")



# M4 tide:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[2]['tide']), mesh, "Z12_tide")
ngsolve.Draw(hydrofirst.M_DA[2]['tide'], mesh, "M12_DA_tide")
ngsolve.Draw(hydrofirst.m_DA[2]['tide'], mesh, "m12_DA_tide")
ngsolve.Draw(hydrofirst.theta_DA[2]['tide'], mesh, "theta12_DA_tide")
ngsolve.Draw(hydrofirst.psi_DA[2]['tide'], mesh, "psi12_DA_tide")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['tide']), mesh, "U12_xi_DA_tide")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[2]['tide']), mesh, "U12_eta_DA_tide")


# M4 stokes:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[2]['stokes']), mesh, "Z12_stokes")
ngsolve.Draw(hydrofirst.M_DA[2]['stokes'], mesh, "M12_DA_stokes")
ngsolve.Draw(hydrofirst.m_DA[2]['stokes'], mesh, "m12_DA_stokes")
ngsolve.Draw(hydrofirst.theta_DA[2]['stokes'], mesh, "theta12_DA_stokes")
ngsolve.Draw(hydrofirst.psi_DA[2]['stokes'], mesh, "psi12_DA_stokes")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['stokes']), mesh, "U12_xi_DA_stokes")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[2]['stokes']), mesh, "U12_eta_DA_stokes")


# M4 nostress:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[2]['nostress']), mesh, "Z12_nostress")
ngsolve.Draw(hydrofirst.M_DA[2]['nostress'], mesh, "M12_DA_nostress")
ngsolve.Draw(hydrofirst.m_DA[2]['nostress'], mesh, "m12_DA_nostress")
ngsolve.Draw(hydrofirst.theta_DA[2]['nostress'], mesh, "theta12_DA_nostress")
ngsolve.Draw(hydrofirst.psi_DA[2]['nostress'], mesh, "psi12_DA_nostress")

# DA along and across channel velocities
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['nostress']), mesh, "U12_xi_DA_nostress")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[2]['nostress']), mesh, "U12_eta_DA_nostress")


# Measures of tidal distortion:
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[2]['all']/hydrolead.Z[1]['all']), mesh, "MeasureTidalDistortion")
ngsolve.Draw(hydrofirst.M_DA[2]['all']/hydrolead.M_DA[1]['all'], mesh, "M4M2_M_MeasureTidalDistortion")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['all']/hydrolead.U_xi_DA[1]['all']), mesh, "M4M2_U_xi_MeasureTidalDistortion")
ngsolve.Draw(ngsolve.Norm(hydrofirst.U_eta_DA[2]['all']/hydrolead.U_eta_DA[1]['all']), mesh, "M4M2_U_eta_MeasureTidalDistortion")


# M0/M2
ngsolve.Draw(ngsolve.Norm(hydrofirst.Z[0]['all']/hydrolead.Z[1]['all']), mesh, "M0M2_Z_MeasureTidalDistortion")
ngsolve.Draw(hydrofirst.M_DA[0]['all']/hydrolead.M_DA[1]['all'], mesh, "M0M2_M_MeasureTidalDistortion")


# Example of the along and acorss channel velocities at the bed
ngsolve.Draw(ngsolve.Norm(hydrolead.U_xi[1]['all'](-hydrophysparams.H)), mesh, "U01_xi_-H_all")
ngsolve.Draw(ngsolve.Norm(hydrolead.U_eta[1]['all'](-hydrophysparams.H)), mesh, "U01_eta_-H_all")

#ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[0]['all']/hydrolead.U_xi_DA[1]['all']), mesh, "M0M2_U_xi_MeasureTidalDistortion")
#ngsolve.Draw(ngsolve.Norm(hydrofirst.U_xi_DA[2]['all']/hydrolead.U_xi_DA[1]['all']), mesh, "M4M2_U_xi_MeasureTidalDistortion")
#ngsolve.Draw(ngsolve.Norm(hydrofirst.M_DA[2]['all']/hydrolead.m_DA[1]['all']), mesh, "M4M2_Mm_MeasureTidalDistortion")


"""


importlib.reload(pp)
#importlib.reload(cg)
post = pp.PostProcessing(mesh, physparams_hydrolead, subdiv=1) #subdiv=3 is more accurate but slower



"""

# Plots using matplotlib
fig = post.plot_comp_and_display_mesh()


# Note contour always plots the abs
fig = post.contour(hydrophysparams.H, title="bathymetry", phasetimes=None, subphase_lines=0, showmesh=False, colormap='viridis', isLabel=False)
fig.gca().axis('equal')
fig.set_size_inches(4, 3)
fig.subplots_adjust(bottom=0.17)


# Coordinates
fig = post.contour(bfc.xi_gf, title="xi", showmesh=False, colormap='viridis', isLabel=False, amplitude_lines=40)
plt.gca().axis('equal')
fig = post.contour(bfc.eta_gf, title="eta", showmesh=False, colormap='viridis', isLabel=False, amplitude_lines=10)
plt.gca().axis('equal')


# Plot speed proxy
post.plot_speed_proxy(hydrolead.Z[1]['all'], vmin=9.95, vmax=10)

post.plot_amplification_proxy(hydrolead.Z[1]['all'])


fig = post.contour(res_hydrolead.Z[1]['all'], title="Free surface", phasetimes=None, subphase_lines=0, showmesh=False, colormap='viridis')
fig = post.contour(hydrolead.U_xi_DA[1]['all'], title="U_xi_DA", phasetimes=None, subphase_lines=0, showmesh=False, colormap='viridis')
fig = post.contour(hydrolead.m_DA[1]['all'], title="m_DA M2", phasetimes=None, showmesh=False, colormap='jet', amplitude_range=[-0.01, 0.01], extend_colormap="both")
fig = post.contour(hydrolead.eps_DA[1]['all'], title="eps_DA M2", phasetimes=None, showmesh=False, colormap='jet', amplitude_range=[-0.01, 0.01], extend_colormap="both")
fig = post.contour(hydrolead.theta_DA[1]['all'], title="theta_DA M2", phasetimes=None, showmesh=False, colormap='jet')
fig = post.contour(hydrolead.psi_DA[1]['all'], title="psi_DA M2", phasetimes=None, showmesh=False, colormap='jet')


# Vorticity at bed
fig = post.contour(hydrolead.Omega[1]['all'](-hydrophysparams.H), title="Omega M2 at z=-H", phasetimes=None, showmesh=False, colormap='jet', isLabel=False)
fig = post.contour(hydrolead.Omega[1]['all'](hydrophysparams.R), title="Omega M2 at z=R", phasetimes=None, showmesh=False, colormap='jet')

#
fig = post.contour(hydrolead.U_DA[1]['all'], title="U_DA", phasetimes=None, subphase_lines=0, showmesh=False, subamplitude_lines=0, colormap='viridis')
# TODO
#fig = post.pcolor(ngsolve.Norm(hydrolead.U_xi_DA[1]['all']))


#fig = post.contour(hydrolead.U_eta_DA[1]['all'], title="U_eta_DA", phasetimes=None, subphase_lines=0, showmesh=False, colormap='viridis')
#fig = post.contour(hydrofirst.U_eta_DA[2]['all'], title="U_eta_DA_M4", phasetimes=None, subphase_lines=0, showmesh=True, colormap='viridis')

#fig = post.contour(ngsolve.Norm(hydrophysparams.Gradient_H), title="|grad H|", showmesh=False, colormap='viridis')



# Tidal ellipse parameters of the bed wrt to the depth_num average
fig = post.contour(hydrolead.theta[1]['all'](-hydrophysparams.H) - hydrolead.theta_DA[1]['all'], title="theta_b-theta_DA M2", phasetimes=None, showmesh=False,colormap='jet', isLabel=False)
fig = post.contour(hydrolead.psi[1]['all'](-hydrophysparams.H) - hydrolead.psi_DA[1]['all'], title="psi_b-psi_DA M2", phasetimes=None, subphase_lines=0, showmesh=False,colormap='jet', isLabel=False)

# Tidal ellipse parameters of the bed wrt to the depth_num average M4
fig = post.contour(hydrofirst.theta[2]['all'](-hydrophysparams.H) - hydrofirst.theta_DA[2]['all'], title="theta_b-theta_DA M4", phasetimes=None, showmesh=False, colormap='jet', amplitude_range=[-5, 5], extend_colormap='both', isLabel=False)
fig = post.contour(hydrofirst.psi[2]['all'](-hydrophysparams.H) - hydrofirst.psi_DA[2]['all'], title="psi_b-psi_DA M4", phasetimes=None, showmesh=False,colormap='jet', amplitude_range=[-10, 5], extend_colormap='both', isLabel=False)



"""
## Cross-section plots
eps = 100
L_cross = 25e3
p1 = np.array([L_cross, -2276.79+eps])
p2 = np.array([L_cross, 2276.79-eps])



fig = post.sectionnormal(res_hydrolead.U[1]['all'], res_hydrolead.V[1]['all'], p1, p2, title="U normal", speed="sigma", phasetimes=None, isLabel=False)
fig.set_size_inches(4, 3)
fig.subplots_adjust(bottom=0.17, left=0.17)


fig = post.sectiontangential(res_hydrolead.U[1]['all'], res_hydrolead.V[1]['all'], p1, p2, title="U tangential", speed="sigma", phasetimes=None, isLabel=False)
fig.set_size_inches(4, 3)
fig.subplots_adjust(bottom=0.17, left=0.17)

fig = post.cross_section_scalar(res_hydrolead.W[1]['all'], p1, p2, title="W M2", isLabel=False)
fig.set_size_inches(4, 3)
fig.subplots_adjust(bottom=0.17, left=0.17)

print(" p = ", order_basisfunctions_lead)
plt.show()










"""


# Plot zeta over cross-section
fig = plot_line_scalar(ngsolve.Norm(hydrolead.Z[1]['all']), p1, p2, post)
fig.gca().set_title("abs zeta M2")

fig = plot_line_scalar(arg(hydrolead.Z[1]['all']), p1, p2, post)
fig.gca().set_title("arg zeta M2")


fig = plot_line_scalar(ngsolve.Norm(ngsolve.Grad(hydrolead.Z[1]['all'])[1]), p1, p2, post)
fig.gca().set_title("abs zeta_y M2")

fig = plot_line_scalar(arg(ngsolve.Grad(hydrolead.Z[1]['all'])[1]), p1, p2, post)
fig.gca().set_title("arg zeta_y M2")



fig = plot_line_scalar(ngsolve.Grad(hydrolead.Z[1]['all'])[1].real, p1, p2, post)
fig.gca().set_title("real zeta_y M2")



# M0 cross-sectional plots
fig = post.sectionnormal(hydrofirst.U[0]['all'], hydrofirst.V[0]['all'], p1, p2, title="U_normal_0", speed="sigma", phasetimes=None)
fig = post.sectiontangential(hydrofirst.U[0]['all'], hydrofirst.V[0]['all'], p1, p2, title="U_tangential_0", speed="sigma", phasetimes=None)


# M4 cross-sectional plots
fig = post.sectionnormal(hydrofirst.U[2]['all'], hydrofirst.V[2]['all'], p1, p2, title="U_normal_2", speed="sigma", phasetimes=None)
fig = post.sectiontangential(hydrofirst.U[2]['all'], hydrofirst.V[2]['all'], p1, p2, title="U_tangential_2", speed="sigma", phasetimes=None)


# TODO testing
# Leading-order M2
#fig = post.cross_section_scalar(hydrolead.W[1]['all'], p1, p2, title="W M2")
fig = post.cross_section_scalar(hydrolead.R_1[1]['all'], p1, p2, title="R1 M2")
fig = post.cross_section_scalar(hydrolead.R_2[1]['all'], p1, p2, title="R2 M2")

# Tidal ellipse parameters
fig = post.cross_section_scalar(hydrolead.M[1]['all'], p1, p2, title="M M2")
fig = post.cross_section_scalar(hydrolead.m[1]['all'], p1, p2, title="m M2", colormap="coolwarm")
fig = post.cross_section_scalar(hydrolead.eps[1]['all'], p1, p2, title="eps M2",  colormap="coolwarm")
fig = post.cross_section_scalar(hydrolead.theta[1]['all'], p1, p2, title="theta M2")
fig = post.cross_section_scalar(hydrolead.psi[1]['all'], p1, p2, title="psi M2")

# Vorticity
fig = post.cross_section_scalar(hydrolead.Omega[1]['all'], p1, p2, title="vorticity M2",colormap="jet")



# First-order M0
fig = post.cross_section_scalar(hydrofirst.W[0]['all'], p1, p2, title="W M0")
fig = post.cross_section_scalar(hydrofirst.R_1[0]['all'], p1, p2, title="R1 M0")
fig = post.cross_section_scalar(hydrofirst.R_2[0]['all'], p1, p2, title="R2 M0")
"""
print("Done plotting in {:.5f} seconds".format(time.time() - t1))
