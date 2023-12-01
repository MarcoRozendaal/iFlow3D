""""
File containing post-processing utilities

"""
import warnings

import numpy as np
import math
import time
from matplotlib import pyplot as plt, ticker
from matplotlib import tri, _tri
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import cmath
from ngsolve import *
import scipy.special
import itertools

import ngsolve
from model.general import create_geometry as cg


############### Set some properties of matplotlib figures ###################
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#plt.rcParams['font.size'] = 10
#plt.rcParams['axes.labelsize'] = 10
#plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['axes.titlesize'] = 10
#plt.rcParams['xtick.labelsize'] = 10.5
#plt.rcParams['ytick.labelsize'] = 10.5
#plt.rcParams['legend.fontsize'] = 10
#plt.rcParams['figure.titlesize'] = 12
#plt.rcParams['font.size'] = 11
#plt.rcParams['savefig.dpi'] = 300

# Reset the user defined properties
#plt.rcdefaults()



############## Extract mesh data ###################
def vertices(mesh):
    return np.array([v.point for v in mesh.vertices])


def elmat(mesh):
    # Connectivity of mesh nodes only, not all dofs
    return np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])


def count_free_dofs(fes):
    i = 0
    for isFree in fes.FreeDofs():
        i = i + isFree
    return i


########## Evaluate grid functions ##############
def evaluateV(gfu, mesh, pnts):
    # Evaluate grid function gfu for the vector pnts
    # OLD np.array([gfu(mesh(*p)) for p in pnts])
    return gfu(mesh(pnts[:, 0], pnts[:, 1])).flatten()


def evaluateM(gfu, mesh, X, Y):
    # Evaluate grid function gfu for the matrices X and Y
    return np.array([gfu(mesh(*p)) for p in np.nditer([X, Y])]).reshape(X.shape)


def evaluate(gfu, mesh, pnts):
    # Evaluate grid function gfu for the vector pnts
    return gfu(mesh(pnts[:, 0], pnts[:, 1]))


def evalLine(gfu, mesh, xp, yp):
    # Evaluate the amplitude of grid function gfu for vector x and y
    return np.array([gfu(mesh(x,y)) for y in yp for x in xp]).reshape((len(yp), len(xp))).T

def evalAmpLine(gfu, mesh, xp, yp):
    # Evaluate the amplitude of grid function gfu for vector x and y
    return np.array([amp(gfu)(mesh(x,y)) for y in yp for x in xp]).reshape((len(yp), len(xp))).T


def evaluateXYZ(u, X, Y, Z, mesh):
    # z component analytic, x y gridfunction, matrices X, Y, Z
    return np.array([u(z)(mesh(x, y)) for x, y, z in np.nditer([X, Y, Z])]) #.reshape(X.shape)


def amp(gfu):
    # 29-7-2020: Old sqrt(gfu.real ** 2 + gfu.imag ** 2)
    return sqrt(gfu*Conj(gfu)).real


def phaselag(gfu, p):
    """"
    Returns the phase lag in hours from 0 to 2*pi/(p.omega * 60 * 60)
    i.e. with branch cut along the positive axis
    """
    angle = -atan2(gfu.imag, gfu.real)
    return IfPos(angle, angle, CoefficientFunction(2*np.pi) + angle)/(p.omega * 60 * 60)


def phaselag2(gfu, p):
    """"
    Returns the phase lag in hours from (-pi to pi)/(p.omega * 60 * 60).
    i.e. with branch cut along the negative axis
    """
    return -atan2(gfu.imag, gfu.real)/(p.omega * 60 * 60)



##################### Convergence ######################
def L2norm(gfu, mesh, order=10):
    """ grid function gfu, computes ||gfu||_2 """
    return sqrt( Integrate(gfu*Conj(gfu), mesh, order=order) ).real


def relL2error(gfu, gfu_ex, mesh, order=5):
    """ Computes ||gfu-gfu_ex||_2/||gfu_ex||_2 """
    return L2norm(gfu-gfu_ex, mesh, order)/L2norm(gfu_ex, mesh, order)


def orderP(err2, h):
    err2 = np.array(err2)
    h = np.array(h)
    return np.log(err2[:-1]/err2[1:])/np.log(h[:-1]/h[1:])



#################### Other #######################
def averageAbs(gfu, x_a, mesh, p):
    y_a = np.linspace(-p.B2/2, p.B2/2, 100)
    return np.average(evalAmpLine(gfu, mesh, [x_a], y_a))


def average(gfu, x_a, mesh, p):
    y_a = np.linspace(-p.B2/2, p.B2/2, 100)
    return np.average(evalLine(gfu, mesh, [x_a], y_a))


def addPeriodicTimeDependency(Zmesh, Nframes, p):
    tv = np.linspace(0, 2 * np.pi / p.omega, Nframes, endpoint=False)
    zeta = np.zeros((Nframes, len(Zmesh)))
    for i in range(Nframes):
        zeta[i, :] = (Zmesh * np.exp(1j * p.omega * tv[i])).real
    return zeta





#################### plot functions ##################
def plotStreamlinesXZ(u, w, y, mesh, p):
    #  Grid function v
    n = 100
    xmin, xmax = 0, p.L
    zmin, zmax = -p.h, 0

    X, Z = np.mgrid[xmin:xmax:n*1j, zmin:zmax:n*1j]
    Y = y*np.ones(X.shape)
    U, W = evaluateXYZ(u, mesh, X, Y, Z), evaluateXYZ(w, mesh, X, Y, Z)
    plt.streamplot(X[:, 0], Z[0, :], U, W)
    return plt.axis('equal')


def plotQuiverXZ(u, w, y, mesh, p):
    #  Grid function v
    n = 20
    xmin, xmax = 0, p.L
    zmin, zmax = -p.h, 0

    X, Z = np.mgrid[xmin:xmax:n*1j, zmin:zmax:n*1j]
    Y = y*np.ones(X.shape)
    U, W = evaluateXYZ(u, mesh, X, Y, Z), evaluateXYZ(w, mesh, X, Y, Z)
    return plt.quiver(X, Z, U, W)


def plotAmpXZ(u, y, mesh, p):
    #  Grid function u
    n = 100
    xmin, xmax = 0, p.L
    zmin, zmax = -p.h, 0

    X, Z = np.mgrid[xmin:xmax:n*1j, zmin:zmax:n*1j]
    Y = y*np.ones(X.shape)
    U = evaluateXYZ(lambda z : amp(u(z)), mesh, X, Y, Z)
    plt.contourf(X, Z, U, 10)
    return plt.colorbar()


#################### post processing object #####################
class PostProcessing():
    """
    Object used for post-processing

    This object can be used for creating:
    - plots of the triangular grid: plot_trianglesmesh()
    - plots of contour lines of real and complex quantities: contour(...)
    - plots of the velocities through a cross section: cross_section(...), sectionnormal(...), sectiontangential(...)

    """
    def __init__(self, mesh, physical_parameters, subdiv=2):
        """Initialize post-processing object

        Params:
            mesh                        - the NGSolve mesh
            physical_parameters         - the physical_parameters object
            subdiv                      -  the number of times the triangluar mesh of NGSolve needs to be refined (subdivided) into smaller
                     triangles for plotting
        """
        self.mesh = mesh
        self.physical_parameters = physical_parameters
        self.vertices_comp_mesh = vertices(mesh)
        self.elmat_comp_mesh = elmat(mesh)
        self.triangulation_comp_mesh = tri.Triangulation(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], self.elmat_comp_mesh)

        # Uniformly refine mesh for smoother output figures
        refiner = tri.UniformTriRefiner(self.triangulation_comp_mesh)
        self.triangulation_display_mesh, self.parent_triangle_index = refiner.refine_triangulation(return_tri_index=True, subdiv=subdiv)

        self.vertices_display_mesh = np.column_stack([self.triangulation_display_mesh.x, self.triangulation_display_mesh.y])

        # Nudge vertices on the boundary inside the domain
        # This is needed due to curvature of boundary and the uniform mesh refinement
        self.nudge_concave_boundary_vertices()

        # TODO maybe make some kind of reverse function to also bloat the geometry for the convex vertices




    def nudge_concave_boundary_vertices(self):
        """Function that (slowly) nudges mesh vertices outside the domain into the computational domain"""
        nudging_parameter = 0.002


        # Initialization.
        # 1) Find the indices of the vertices that are outside the domain
        self.index_vertices_outside = self.get_index_of_vertices_outside_domain()
        # 1.5) Break if there are no vertices outside
        if len(self.index_vertices_outside) == 0:
            return
        # 2) Find parent triangle of these indices
        parent_triangle_indices_outside = self.parent_triangle_index[self.index_vertices_outside]
        # 3) Compute nudging vector using the parent triangle, and freeze it. E.g. we do not update it, we will index it instead, similar to a look up table
        self.parent_nudging_vector = self.compute_parent_nudging_vector(parent_triangle_indices_outside)
        # 4) We create a mapping between the parent nudging vector index and the global vertex index table
        self.global_index_to_nudging_index = {global_index: nudging_index for nudging_index, global_index in enumerate(self.index_vertices_outside)}


        counter = 0
        while True:
            counter += 1

            # 1) Nudge the vertices outside using the nudging vector
            self.vertices_nudged = self.nudge(nudging_parameter)

            # 2) Update the vertices
            self.triangulation_display_mesh.x[self.index_vertices_outside] = self.vertices_nudged[:, 0]
            self.triangulation_display_mesh.y[self.index_vertices_outside] = self.vertices_nudged[:, 1]
            self.vertices_display_mesh[self.index_vertices_outside, :] = self.vertices_nudged

            # 3) Check if vertices outside are still outside:
            self.index_vertices_outside = self.update_index_vertices_outside()

            # TODO For debugging
            #print("Counter", counter, "Outside", len(self.index_vertices_outside))
            if len(self.index_vertices_outside) == 0:
                break


    # Functions required for the nudging
    # 1) Find which display_mesh points are outside the domain
    def get_index_of_vertices_outside_domain(self):
        """Find which display_mesh points are outside the domain"""
        index_vertices_outside = []
        for index, vertex in enumerate(self.vertices_display_mesh):
            if not self.mesh.Contains(*vertex):
                index_vertices_outside.append(index)
        index_vertices_outside = np.array(index_vertices_outside)
        return index_vertices_outside


    # 3) Compute the vector between the furthest away vertex and middle of the other two vertices
    def compute_parent_nudging_vector(self, parent_triangle_indices_outside):
        """Compute the vector between the furthest away vertex point and centre of the other two vertex points
        Return the (x,y) of the vector """

        # Get the indices and x,y coordinates of the parent vertices of the vertex that is outside of the domain
        parent_vertices_indices_outside = self.triangulation_comp_mesh.triangles[parent_triangle_indices_outside, :]
        parent_vertices_x_outside = self.triangulation_comp_mesh.x[parent_vertices_indices_outside]
        parent_vertices_y_outside = self.triangulation_comp_mesh.y[parent_vertices_indices_outside]

        # We compute on which side the vertex outside originally started out
        # Do this using the triangle inequality

        # We compute the distance between the vertex outside and the 3 parent vertices
        vertices_outside = self.vertices_display_mesh[self.index_vertices_outside, :]
        diff_x_per_vertex_outside = parent_vertices_x_outside - vertices_outside[:, 0].reshape(-1, 1)
        diff_y_per_vertex_outside = parent_vertices_y_outside - vertices_outside[:, 1].reshape(-1, 1)
        distance_to_parent_vertices_outside = np.sqrt(np.square(diff_x_per_vertex_outside) + np.square(
            diff_y_per_vertex_outside))

        # We compute the distance between the 3 parent vertices directly
        diff_x_parent_vertices = parent_vertices_x_outside - np.roll(parent_vertices_x_outside, -1, axis=1)
        diff_y_parent_vertices = parent_vertices_y_outside - np.roll(parent_vertices_y_outside, -1, axis=1)
        distance_parent_vertices = np.sqrt(np.square(diff_x_parent_vertices) + np.square(
            diff_y_parent_vertices))

        # Using the triangle inequality we can find on which edge the vertex outside lives
        eps = 1e-10
        sum_matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])

        # We determine for the vertices_outside: on which edge it is on and if its on an edge:
        onwhichedge_is_vertexoutsideo = np.abs(
            np.matmul(distance_to_parent_vertices_outside, sum_matrix) - distance_parent_vertices) < eps

        # Determine all vectices outside that are on an edge
        is_vertexoutside_onedge = np.any(onwhichedge_is_vertexoutsideo, axis=1)

        # The other vertices are internal vertices
        is_vertexoutside_internal = np.invert(is_vertexoutside_onedge)


        ## For the verticesoutside on edge the nudging vector is computed:

        # Using the edge information to exact the vertices of this edge and the other vertex, called edge_1, edge_2 and not_edge
        parent_vertex_x_edge_1 = parent_vertices_x_outside[onwhichedge_is_vertexoutsideo]
        parent_vertex_y_edge_1 = parent_vertices_y_outside[onwhichedge_is_vertexoutsideo]
        parent_vertex_x_edge_2 = np.roll(parent_vertices_x_outside, -1, axis=1)[onwhichedge_is_vertexoutsideo]
        parent_vertex_y_edge_2 = np.roll(parent_vertices_y_outside, -1, axis=1)[onwhichedge_is_vertexoutsideo]
        parent_vertex_x_not_edge = np.roll(parent_vertices_x_outside, 1, axis=1)[onwhichedge_is_vertexoutsideo]
        parent_vertex_y_not_edge = np.roll(parent_vertices_y_outside, 1, axis=1)[onwhichedge_is_vertexoutsideo]

        # Using these coordinates and ordering we can compute the point on the centre of the two edge vertices
        parent_vertices_x_edge = np.column_stack([parent_vertex_x_edge_1, parent_vertex_x_edge_2])
        parent_vertices_y_edge = np.column_stack([parent_vertex_y_edge_1, parent_vertex_y_edge_2])
        parent_x_avg = np.mean(parent_vertices_x_edge, 1)
        parent_y_avg = np.mean(parent_vertices_y_edge, 1)

        # Using this average point, we compute the nudging vector
        parent_vector_x = parent_vertex_x_not_edge - parent_x_avg
        parent_vector_y = parent_vertex_y_not_edge - parent_y_avg
        parent_nudging_vector_onedge = np.column_stack([parent_vector_x, parent_vector_y])


        ## The parent nudging vector for the internal triangles is computed
        parent_nudging_vector_internal = []

        # We determine the parent triangle indices for the internal vertices outside
        parent_triangle_indices_internal = parent_triangle_indices_outside[is_vertexoutside_internal]

        # We create the parent triangle index for the vertices outside on the edges
        parent_triangle_indices_onedge = parent_triangle_indices_outside[is_vertexoutside_onedge]

        # We determine the average of the nudging vectors of the other vertices outside on the edge
        for parent_triangle_index in parent_triangle_indices_internal:
            is_same_parent_index = parent_triangle_indices_onedge==parent_triangle_index
            parent_nudging_vectors_same_parent = parent_nudging_vector_onedge[is_same_parent_index]
            parent_nudging_vector_avg = np.mean(parent_nudging_vectors_same_parent, axis=0)
            parent_nudging_vector_internal.append(parent_nudging_vector_avg)
        parent_nudging_vector_internal = np.array(parent_nudging_vector_internal).reshape(-1, 2)


        ## The total parent nuding vector is constructed
        parent_nudging_vector = np.zeros_like(vertices_outside)
        parent_nudging_vector[is_vertexoutside_onedge] = parent_nudging_vector_onedge
        parent_nudging_vector[is_vertexoutside_internal] = parent_nudging_vector_internal

        return parent_nudging_vector


    # 4) Iteratively nudge the vertices outside using the parent vector
    def nudge(self, nudging_parameter):
        """Function that iteratively nudges the child points using the parent vector"""
        vertices_outside = self.vertices_display_mesh[self.index_vertices_outside, :]
        nudging_index_outside = np.array([self.global_index_to_nudging_index[index_outside] for index_outside in self.index_vertices_outside])
        nudging_vector_outside = self.parent_nudging_vector[nudging_index_outside]

        vertices_nudged = vertices_outside + nudging_vector_outside * nudging_parameter
        return vertices_nudged

    # 5) Update index vertices outside
    def update_index_vertices_outside(self):
        """Function that checks the index_vertices_outside if they are still ouside of the domain and updates this list"""
        index_vertices_outside = []
        #self.vertices_nudged
        #self.index_vertices_outside
        for index, vertex in zip(self.index_vertices_outside, self.vertices_nudged):
            if not self.mesh.Contains(*vertex):
                index_vertices_outside.append(index)
        index_vertices_outside = np.array(index_vertices_outside)
        return index_vertices_outside

    # TODO
    def plot_debug_2(self):
        """"
        Debugging the nudging
        """

        fig, ax = plt.subplots()
        if len(self.index_vertices_outside) > 0:
            vertices_outside = self.vertices_display_mesh[self.index_vertices_outside, :]
            plt.plot(vertices_outside[:, 0], vertices_outside[:, 1], 'or')


        plt.triplot(self.triangulation_display_mesh, '-', color='orange', linewidth=1)
        plt.triplot(self.triangulation_comp_mesh, '-b')
        plt.plot(self.vertices_nudged[:, 0], self.vertices_nudged[:, 1], 'dg')
        plt.gca().axis('equal')
        plt.title("Mesh")
        self.axis_to_km(plt)
        return fig




    def plot_comp_and_display_mesh(self):
        """"
        Method to display the computational and display triangular meshes
        """
        fig, ax = plt.subplots()
        plt.triplot(self.triangulation_display_mesh, '-', color='orange', linewidth=1)
        plt.triplot(self.triangulation_comp_mesh, '-b')
        plt.gca().axis('equal')
        plt.title("Mesh")

        # meter to km
        self.axis_to_km(ax)

        return fig


    def plot_trianglesmesh(self):
        """"
        Method to display the triangular mesh
        """
        fig, ax = plt.subplots()
        plt.triplot(self.triangulation_comp_mesh)
        plt.gca().axis('equal')
        plt.title("Mesh")
        self.axis_to_km(ax)
        return fig


    def contour(self, Z, title="",  amplitude_range=None, amplitude_lines=10, subamplitude_lines=2,
                 phasetimes="hours", subphase_lines=0, colormap='Blues', extend_colormap='neither', tol=9e-1, tol2=1e-7,
                 eval_function_at_cursor=True, showmesh=None, geometrydata=None, isLabel=True):
        """"
        Contour plot of Z and phase lines

        Quick method to plot complex solution Z. The phase is in between 0 and 2 pi.
        There is a branch cut along the positive axis. The triangles corresponding to the zero contourline of the phase
        are masked. This contour line is instead created by the phase2 function with -pi to pi. For which pi/-pi is masked.
        Arguments:
            Z -- complex or real function to be plotted
            title -- title of plot
            amplitude_range -- manually select range for contour plot
            amplitude_lines -- number of amplitude lines with label
            subamplitude_lines -- number of amplitude lines without label in between the amplitude_lines
            phasetimes -- string indicating phase lines with label: 'hours', 'halfhours', 'quarters', 'eighths'
            subphase_lines -- number of phase lines without label in between the phasetime lines
            colormap -- choose the colormap used to create the countour plot
            extend_colormap -- If the amplitude_range is selected manually, some amplitudes may fall outside of the specified range. What to do with those amplitudes: 'neither', 'both', 'min', 'max'"
            tol -- the tolerance used to determine the masked regions, i.e., the branch cuts
            tol2 -- tolerance used to mask triangles with small amplitudes
            geometrydata -- geometrydata object for plotting the geometry
            eval_function_at_cursor -- boolean to specify if mpl should eval the plotted function at the cursor. Handy for interpretation of figures. Might be slow or buggy.
            showmesh -- display mesh on top of contour plot
            geometrydata -- plot the given geometrydata on top of the contour

        2021-09-16 (Marco) - fixed extend default argument
        """

        def first_n_digits(num, n):
            num = abs(num)
            num_str = str(num)

            if n >= len(num_str):
                return num

            return int(num_str[:n])

        # To be plotted
        # If Z is complex, plot abs(Z) else plot Z:
        if Z.is_complex:
            Zamp = evaluateV(amp(Z), self.mesh, self.vertices_display_mesh)
        else:
            Zamp = evaluateV(Z, self.mesh, self.vertices_display_mesh)


        if Z.is_complex and phasetimes is not None:
            Zphase = evaluateV(phaselag(Z, self.physical_parameters), self.mesh, self.vertices_display_mesh)
            Zphase2 = evaluateV(phaselag2(Z, self.physical_parameters), self.mesh, self.vertices_display_mesh)

            levels_phase, phase_decimals = self._divide_phase(phasetimes, subphase_lines)

            # Masks phase
            mask = self.mask_triangles_branchcut_positive_axis(Zphase, tol)
            mask2 = self.mask_triangles_branchcut_negative_axis(Zphase2, tol)

        # Contour levels amplitude
        if amplitude_range is None:
            levels_amp = np.linspace(np.min(Zamp), np.max(Zamp), amplitude_lines*(subamplitude_lines+1))
        else:
            levels_amp = np.linspace(np.min(amplitude_range), np.max(amplitude_range), amplitude_lines * (subamplitude_lines + 1))

        # Masks amplitude
        mask_small_amp = self.mask_triangles_small_amplitude(Zamp, tol2)


        ### The plotting ###
        fig, ax = plt.subplots()

        # Amplitude
        tcf_amp = ax.tricontourf(self.triangulation_display_mesh, Zamp, levels_amp, cmap=plt.cm.get_cmap(name=colormap, lut=None), extend=extend_colormap)
        tc_amp = ax.tricontour(self.triangulation_display_mesh, Zamp, levels_amp, colors=['k'] + ["0.4"] * subamplitude_lines,
                               linewidths=[0.7]+[0.1]*subamplitude_lines)
        if isLabel:
            ax.clabel(tc_amp, levels_amp[0::subamplitude_lines+1], inline=1, fontsize=10, fmt="%1.2f")

        # Phase
        if Z.is_complex and phasetimes is not None:
            # If there is large enough phase variation
            if np.max(Zphase) - np.min(Zphase) > levels_phase[1] - levels_phase[0]:
                self.triangulation_display_mesh.set_mask(np.any([mask, mask_small_amp], axis=0))
                tc_phase = ax.tricontour(self.triangulation_display_mesh, Zphase, levels_phase, colors=['k'] + ["darkred"] * subphase_lines,
                                         linewidths=[1.3]+[1.3]*subphase_lines, linestyles='dashed')
                ax.clabel(tc_phase, levels_phase[0::subphase_lines+1], inline=1, fontsize=10, fmt="%1." + str(phase_decimals) + "f")
                self.triangulation_display_mesh.set_mask(None)

            # Zero phase
            if np.max(Zphase2) > 0 > np.min(Zphase2):
                self.triangulation_display_mesh.set_mask(np.any([mask2, mask_small_amp], axis=0))
                tc_phase2 = ax.tricontour(self.triangulation_display_mesh, Zphase2, [0], colors='k', linewidths=[1.3], linestyles='dashed')
                ax.clabel(tc_phase2, inline=1, fontsize=10, fmt="%1." + str(phase_decimals) + "f")
                self.triangulation_display_mesh.set_mask(None)

        # Overlay mesh
        if showmesh is not None and showmesh:
            ax.triplot(self.triangulation_comp_mesh, lw=1)

        # Geometry
        # Assuming curves only and linear points only
        if geometrydata is not None:
            # Curves
            if callable(geometrydata[0][0]):
                t = np.linspace(0, 1, 500)
                for curve, boundary_condition in geometrydata:
                    if int(boundary_condition) == cg.SEA:
                        color = "darkblue"
                    elif int(boundary_condition) == cg.WALL:
                        color = "k"
                    elif int(boundary_condition) == cg.RIVER:
                        color = "royalblue"
                    curve_display = np.array([curve(t_) for t_ in t])
                    plt.plot(curve_display[:, 0], curve_display[:, 1], color=color)

            # linear points
            else:
                points_segment_list, boundary_conditions_list = cg.split_on_boundary_condition_type(geometrydata)
                for points_segment, boundary_condition in zip(points_segment_list, boundary_conditions_list):
                    if int(boundary_condition) == cg.SEA:
                        color = "darkblue"
                    elif int(boundary_condition) == cg.WALL:
                        color = "k"
                    elif int(boundary_condition) == cg.RIVER:
                        color = "royalblue"
                    plt.plot(points_segment[:, 0], points_segment[:, 1], color=color, linewidth=1)

        if eval_function_at_cursor:
            # Add the value Z(x,y) to the bottom of figure. Can be slow.
            if Z.is_complex:
                def fmt(x, y):
                    if self.mesh.Contains(x, y):
                        Zamp_display = amp(Z)(self.mesh(x, y))
                        Zphase_display = phaselag(Z, self.physical_parameters)(self.mesh(x, y))
                    else:
                        Zamp_display = np.NAN
                        Zphase_display = np.NAN
                    return 'x={x:.5f}  y={y:.5f}  |Z|={z:.5f}  Zphaselag={p:.5f}'.format(x=x, y=y, z=Zamp_display, p=Zphase_display)
            else:
                # Real
                def fmt(x, y):
                    if self.mesh.Contains(x, y):
                        Zamp_display = Z(self.mesh(x, y))
                    else:
                        Zamp_display = np.NAN
                    return 'x={x:.5f}  y={y:.5f}  Z={z:.5f}'.format(x=x, y=y, z=Zamp_display)
            ax.format_coord = fmt

        # Other figure options
        ax.set_title(title)
        self.axis_to_km(ax)
        fig.colorbar(tcf_amp)
        return fig


    def pcolor(self, Z, axis='equal', isColorbar=True, **kwargs):
        """
        Psuedo color plot of Z

            Z -- real function to be plotted
        """
        # To be plotted
        Zplt = evaluateV(Z, self.mesh, self.vertices_display_mesh)

        fig, ax = plt.subplots()
        tc = ax.tripcolor(self.triangulation_display_mesh, Zplt, **kwargs)
        # TODO ax.axis(axis)
        if isColorbar:
            fig.colorbar(tc)

        self.axis_to_km(ax)

        # Add z value to figure. Can be slow.
        def fmt(x, y):
            if self.mesh.Contains(x, y):
                z = Z(self.mesh(x, y))
            else:
                z = np.NAN
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

        ax.format_coord = fmt
        return fig


    def sectionnormal(self, U, V, p1, p2, title="U_normal", **kw):
        return self.cross_section(U, V, p1, p2, type="normal", title=title, **kw)


    def sectiontangential(self, U, V, p1, p2, title="U_tangential", **kw):
        return self.cross_section(U, V, p1, p2, type="tangential", title=title, **kw)

    def nudge_point_inside_domain(self, point, tangential_vector, nudging_parameter):
        """ Function to nudge points into domain """

        max_iter = round(1 / nudging_parameter)
        for i in range(max_iter):
            # Check if vertices of cross section are in mesh
            if self.mesh.Contains(*point):
                break

            point = point + nudging_parameter * tangential_vector
        return point

    def cross_section(self, U, V, p1, p2, type="normal", title="", speed="fast", amplitude_lines=10, subamplitude_lines=2,
                      phasetimes="hours", subphase_lines=0, tol=4e-1, tol2=1e-7, isLabel=True):
        """Cross-section of the normal or tangential velocities between the points p1 and p2.

        The fast method evaluetes the velocity profile on an uniform grid and then truncates the profile.
        The slow method uses sigma layers.

        Arguments:
            U -- the complex x velocity component
            V -- the complex y velocity component
            p1 -- point one, start of cross section. numpy array.
            p2 -- point two, end of cross-section. numpy array.
            type -- string specifying the velocity components to be plotted relative to the cross-section: 'tangential', 'normal'
            title -- title of plot
            speed -- string specifying evaluation method: 'slow', 'fast', 'sigma'
            amplitude_lines -- number of amplitude lines with label
            subamplitude_lines -- number of amplitude lines without label in between the amplitude_lines
            phasetimes -- string indicating phase lines with label: 'hours', 'halfhours', 'quarters', 'eighths'
            subphase_lines -- number of phase lines without label in between the phasetime lines
            tol -- the tolerance used to determine the masked regions
        """



        nz = 100
        nxy = 200

        # Compute tangential and normal flow components w.r.t. cross-section
        tangential_vector = p2-p1
        tangential_unit_vector = tangential_vector/np.linalg.norm(p2-p1)
        normal_unit_vector = np.array([tangential_unit_vector[1], -tangential_unit_vector[0]])

        def U_tangential(z):
            return np.dot([U(z), V(z)], tangential_unit_vector)
        def U_normal(z):
            return np.dot([U(z), V(z)], normal_unit_vector)

        if type == "tangential":
            def Uflow(z):
                return U_tangential(z)
        elif type == "normal":
            def Uflow(z):
                return U_normal(z)
        else:
            raise("Wrong type given. Type can be 'normal' or 'tangential'")

        # TODO we make them contract to the computational geometry
        nudging_parameter = 1 / 1000
        p1 = self.nudge_point_inside_domain(p1, tangential_vector, nudging_parameter)
        p2 = self.nudge_point_inside_domain(p2, -tangential_vector, nudging_parameter)
        print("Nudged points")

        # Generate display grids
        xline = np.linspace(p1[0], p2[0], nxy)
        yline = np.linspace(p1[1], p2[1], nxy)
        hline = -self.physical_parameters.H(self.mesh(xline, yline)).flatten()
        zline = np.linspace(0, min(hline), nz)
        xtilde = np.sqrt((xline - xline[0]) ** 2 + (yline - yline[0]) ** 2)



        # Detemine display speed
        if speed == "fast":
            # Uniform grid
            zgrid = zline
            xtildegrid = xtilde
            mask_below = self.mask_below_depth(zline, hline, nz, nxy)

            def eval(gfu, xline, yline):
                return self.evaluate_uniform_grid(gfu, xline, yline, zline, nxy, nz)
        elif speed == "slow":
            # Sigma layer grid, but inefficient
            zgrid = np.array([np.linspace(0, -self.physical_parameters.H(self.mesh(x, y)), nz) for x, y in zip(xline, yline)]).T
            xtildegrid = np.tile(xtilde, (nz, 1))
            mask_below = None
            def eval(gfu, xline, yline):
                return self.evaluate_sigma_grid(gfu, xline, yline, zline, nxy, nz)

        elif speed == "sigma":
            # Sigma layer grid
            # We create nz grid functions evaluated at the sigma layers
            zgrid = np.array(
                [np.linspace(self.physical_parameters.R(self.mesh(x, y)), -self.physical_parameters.H(self.mesh(x, y)), nz) for x, y in zip(xline, yline)]).T
            xtildegrid = np.tile(xtilde, (nz, 1))
            mask_below = None

            def eval(gfu, xline, yline):
                sigma = np.linspace(0, -1, nz)
                gfu_grid = np.zeros((nz, nxy))
                for index, sigma_layer in enumerate(sigma):
                    sigma_layer_cf = self.physical_parameters.R + (
                            self.physical_parameters.R + self.physical_parameters.H) * sigma_layer
                    gfu_grid[index, :] = gfu(sigma_layer_cf)(self.mesh(xline, yline)).flatten()
                return gfu_grid

        else:
            raise("Wrong speed given. Type can be 'fast', 'slow' or 'sigma'")

        Uamp = eval(lambda z: amp(Uflow(z)), xline, yline)
        Uamp = np.ma.array(Uamp, mask=mask_below)


        # Options for contour plot
        levels_amp = np.linspace(np.min(Uamp), np.max(Uamp), amplitude_lines * (subamplitude_lines + 1))

        if Uflow(0).is_complex and phasetimes is not None:
            Uphase = eval(lambda z: phaselag(Uflow(z), self.physical_parameters), xline, yline)
            Uphase2 = eval(lambda z: phaselag2(Uflow(z), self.physical_parameters), xline, yline)

            is_zero1  = (self.physical_parameters.omega * 60 * 60) * Uphase < tol
            is_pi2 = np.pi - (self.physical_parameters.omega * 60 * 60) * Uphase2 < tol

            mask1 = np.any([mask_below, is_zero1], axis=0)
            mask2 = np.any([mask_below, is_pi2], axis=0)

            Uphase = np.ma.array(Uphase, mask=mask1)
            Uphase2 = np.ma.array(Uphase2, mask=mask2)

            levels_phase, phase_decimals = self._divide_phase(phasetimes, subphase_lines)

        ### Plotting ###
        fig, ax = plt.subplots()


        def remove_close_contours(c_amp):
            """The absolute value of a function can results in two contour lines that are very close together if the function goes through zero.
            This function removes these smallest contours.

            It is assumed that there are only two such contours """

            # Since the levels are arranged from smallest to highest we only need to check the first one
            level = c_amp.collections[0]
            points = []
            for kp, path in reversed(list(enumerate(level.get_paths()))):
                points.append(path.vertices)

            # Some measure to determine if the two curves are too close
            if len(points) >= 2:
                error1 = points[0] - points[1]
                error2 = points[0] - np.flip(points[1], axis=0)
                length = np.linalg.norm(points[0][0] - points[0][-1])

                if np.sum(np.abs(error1)) < 1e-10 * length or np.sum(np.abs(error2)) < 1e-10 * length:
                    del(level.get_paths()[1])

            return c_amp



        # Amplitude
        cf_amp = ax.contourf(xtildegrid, zgrid, Uamp, levels_amp, cmap=plt.cm.get_cmap(name='Blues', lut=None))
        c_amp = ax.contour(xtildegrid, zgrid, Uamp, levels_amp,
                           colors=['k'] + ["0.4"] * subamplitude_lines,
                           linewidths=[0.7] + [0.1] * subamplitude_lines)
        c_amp = remove_close_contours(c_amp)
        if isLabel:
            ax.clabel(c_amp, levels_amp[0::subamplitude_lines+1], inline=True, fontsize=10, fmt="%1.2f")


        # Phase
        if Uflow(0).is_complex and phasetimes is not None:
            # If there is large enough phase variation
            if np.max(Uphase)-np.min(Uphase) > levels_phase[1]-levels_phase[0]:
                c_phase = ax.contour(xtildegrid, zgrid, Uphase, levels_phase, colors=['k']+["darkred"]*subphase_lines,
                                     linewidths=[1.3]+[1.3]*subphase_lines, linestyles='dashed')
                ax.clabel(c_phase, levels_phase[0::subphase_lines+1], inline=1, fontsize=10, fmt="%1." + str(phase_decimals) + "f")

            # Zero phase
            if np.max(Uphase2) > 0 > np.min(Uphase2):
                c_phase2 = ax.contour(xtildegrid, zgrid, Uphase2, [0], colors='k', linewidths=1.3, linestyles='dashed')
                ax.clabel(c_phase2, inline=1, fontsize=10, fmt="%1." + str(phase_decimals) + "f")

        # Fill bottom with a silver color
        ax.fill_between(xtilde, min(hline), hline, color="silver", zorder=2)
        ax.plot(xtilde, hline, color="k", linewidth=0.7, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("distance along cross section (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(cf_amp)
        return fig



    def cross_section_scalar(self, W, p1, p2, title="", amplitude_lines=10, subamplitude_lines=2, tol=4e-1, tol2=1e-7, colormap='Blues', isLabel=True):
        """Cross-section of a scalar quantity between the points p1 and p2.


        Arguments:
            W -- the complex scalar
            p1 -- point one, start of cross section. numpy array.
            p2 -- point two, end of cross-section. numpy array.
            title -- title of plot
            amplitude_lines -- number of amplitude lines with label
            subamplitude_lines -- number of amplitude lines without label in between the amplitude_lines
            tol -- the tolerance used to determine the masked regions
        """

        nz = 100
        nxy = 200
        nudging_parameter = 1 / 1000

        tangential_vector = p2 - p1

        # Nudge points inside domain
        p1 = self.nudge_point_inside_domain(p1, tangential_vector, nudging_parameter)
        p2 = self.nudge_point_inside_domain(p2, -tangential_vector, nudging_parameter)

        # Generate display grids
        xline = np.linspace(p1[0], p2[0], nxy)
        yline = np.linspace(p1[1], p2[1], nxy)
        hline = -self.physical_parameters.H(self.mesh(xline, yline)).flatten()
        zline = np.linspace(0, min(hline), nz)
        xtilde = np.sqrt((xline - xline[0]) ** 2 + (yline - yline[0]) ** 2)



        # Sigma layer grid
        # We create nz grid functions evaluated at the sigma layers
        zgrid = np.array(
            [np.linspace(self.physical_parameters.R(self.mesh(x, y)), -self.physical_parameters.H(self.mesh(x, y)), nz) for x, y in zip(xline, yline)]).T
        xtildegrid = np.tile(xtilde, (nz, 1))

        def eval(gfu, xline, yline):
            sigma = np.linspace(0, -1, nz)
            gfu_grid = np.zeros((nz, nxy))
            for index, sigma_layer in enumerate(sigma):
                sigma_layer_cf = self.physical_parameters.R + (
                        self.physical_parameters.R + self.physical_parameters.H) * sigma_layer
                gfu_grid[index, :] = gfu(sigma_layer_cf)(self.mesh(xline, yline)).flatten()
            return gfu_grid

        # if complex plot abs
        if W(0).is_complex:
            Wamp = eval(lambda z: amp(W(z)), xline, yline)
        else:
            Wamp = eval(W, xline, yline)


        # Options for contour plot
        levels_amp = np.linspace(np.min(Wamp), np.max(Wamp), amplitude_lines * (subamplitude_lines + 1))


        ### Plotting ###
        fig, ax = plt.subplots()


        def remove_close_contours(c_amp):
            """The absolute value of a function can results in two contour lines that are very close together if the function goes through zero.
            This function removes these smallest contours.

            It is assumed that there are only two such contours """

            # Since the levels are arranged from smallest to highest we only need to check the first one
            level = c_amp.collections[0]
            points = []
            for kp, path in reversed(list(enumerate(level.get_paths()))):
                points.append(path.vertices)

            # Some measure to determine if the two curves are too close
            if len(points) >= 2:
                error1 = points[0] - points[1]
                error2 = points[0] - np.flip(points[1], axis=0)
                length = np.linalg.norm(points[0][0] - points[0][-1])

                if np.sum(np.abs(error1)) < 1e-10 * length or np.sum(np.abs(error2)) < 1e-10 * length:
                    del(level.get_paths()[1])

            return c_amp



        # Amplitude
        cf_amp = ax.contourf(xtildegrid, zgrid, Wamp, levels_amp, cmap=plt.cm.get_cmap(name=colormap, lut=None))
        c_amp = ax.contour(xtildegrid, zgrid, Wamp, levels_amp,
                           colors=['k'] + ["0.4"] * subamplitude_lines,
                           linewidths=[0.7] + [0.1] * subamplitude_lines)
        c_amp = remove_close_contours(c_amp)
        if isLabel:
            ax.clabel(c_amp, levels_amp[0::subamplitude_lines+1], inline=True, fontsize=10, fmt="%1.2f")


        # Fill bottom with a silver color
        ax.fill_between(xtilde, min(hline), hline, color="silver", zorder=2)
        ax.plot(xtilde, hline, color="k", linewidth=0.7, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("distance along cross section (m)")
        ax.set_ylabel("z (m)")
        fig.colorbar(cf_amp)
        return fig



    def cross_section_scalar_comparison(self, Z1, Z2, p1, p2, title="", labels=None):
        """"
        Comparison between numeric NGSolve solution Z1 and analytic function Z2
        """

        if labels is None:
            labels = ["", ""]

        nxy = 100
        xline = np.linspace(p1[0], p2[0], nxy)
        yline = np.linspace(p1[1], p2[1], nxy)
        rtilde = np.sqrt((xline) ** 2 + (yline) ** 2)

        Z1amp = evaluateV(amp(Z1), self.mesh, np.column_stack((xline,yline)))


        Z2amp = np.array([np.abs(Z2(r)) for r in rtilde])

        ### Plotting ###
        fig, ax = plt.subplots()

        plt.plot(rtilde[::3], Z2amp[::3], "o", label=labels[1])
        plt.plot(rtilde, Z1amp, label=labels[0])
        plt.legend()
        ax.set_title(title)
        ax.set_xlabel("r (_m)")
        ax.set_ylabel("Z (_m)")
        return fig


    def cross_section_pcolor(self, Z, p1, p2):
        """"
        Cross-section psuedo color plot of Z

        Arguments:
            Z -- real function to be plotted
            p1 -- point one, start of cross section. numpy array.
            p2 -- point two, end of cross-section. numpy array.
        """
        nz = 100
        nxy = 100

        # Generate display grids
        xline = np.linspace(p1[0], p2[0], nxy)
        yline = np.linspace(p1[1], p2[1], nxy)
        hline = -self.physical_parameters.H(self.mesh(xline, yline)).flatten()
        zline = np.linspace(0, min(hline), nz)
        xtilde = np.sqrt((xline - xline[0]) ** 2 + (yline - yline[0]) ** 2)

        # To be plotted
        Zgrid = self.evaluate_uniform_grid(Z, xline, yline, zline, nxy, nz)

        fig, ax = plt.subplots()
        pc = ax.pcolormesh(xtilde, zline, Zgrid)
        fig.colorbar(pc)

        ax.set_xlabel("tilde{x} (_m)")
        ax.set_ylabel("z (_m)")

        # Add z value to figure. Can be slow.
        def fmt(xtilde, z):
            x, y = p1 + (p2-p1)*xtilde/np.linalg.norm(p2-p1)
            Zp = Z(z)(self.mesh(x,y))
            return 'x={x:.5f}  y={y:.5f}  z={z:.5f}, Z={Zp:.5f}'.format(x=x, y=y, z=z, Zp=Zp)

        ax.format_coord = fmt
        return fig


    ## Plots tidal ellipses # TODO
    def plot_tidal_ellipses(self, res_hydro, n, symbol, color=None, isDA=True, scale=300, cmap='bwr', isCentercolormap=False, maxvalue_colormap=None):
        """ Function to plot tidal ellipse
         Args:
             color =  name of property used to color ellipses
         """

        # xy: Center of ellipse, width: total length of horizontal axis, height: total length of vertical axis, angle: angle in degrees anti-clockwise
        xy = self.vertices_comp_mesh


        # Choose between depth_num-averaged and depth_num-integrated tidal ellipses
        if isDA:
            width = scale * 2 * res_hydro.M_DA[n][symbol]
            height = scale * 2 * res_hydro.m_DA[n][symbol]
            angle = res_hydro.theta_DA[n][symbol]
            color = getattr(res_hydro, color)[n][symbol]
        else:
            # Assuming Depth-integrated
            # TODO
            width = scale * 2 * res_hydro.Mhat[n][symbol](self.physical_parameters.R)
            height = scale * 2 * res_hydro.mhat[n][symbol](self.physical_parameters.R)
            angle = res_hydro.theta[n][symbol](self.physical_parameters.R)
            color = getattr(res_hydro, color)[n][symbol](self.physical_parameters.R)

        # TODO computation mesh?



        # We precompute the ellipse parameters
        width_xy = evaluateV(width, self.mesh, xy)
        height_xy = evaluateV(height, self.mesh, xy)
        angle_xy = evaluateV(angle, self.mesh, xy)
        color_xy = evaluateV(color, self.mesh, xy)

        # Loop over points and draw ellipses
        ells = [Ellipse(xy=xy[i, :],
                        width=width_xy[i],
                        height=height_xy[i],
                        angle=angle_xy[i])
                for i in range(len(xy))]

        # We call the self pcolor method
        fig = self.pcolor(self.physical_parameters.H, isColorbar=False)

        ax = fig.gca()


        # Collection method
        p = PatchCollection(ells, cmap=plt.get_cmap(cmap), alpha=1)
        p.set_array(color_xy)
        if isCentercolormap:
            if maxvalue_colormap is None:
                maxvalue_colormap = np.max(np.abs(color_xy))
            p.set_clim([-maxvalue_colormap, maxvalue_colormap])
        ax.add_collection(p)
        fig.colorbar(p)

        # TODO loop method
        """
        for i, e in enumerate(ells):
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor(color_xy[i])
            # TODO np.random.rand(3)
        """

        plt.show()
        return fig




# TODo check.
    # TODO move
    def Re(self, gfu):
        """Returns the real part of a complex number"""
        return 1 / 2 * (gfu + Conj(gfu))

    def Im(self, gfu):
        """Returns the imaginairy part of a complex number"""
        return 1 / (2 * 1j) * (gfu - Conj(gfu))



    def plot_speed_proxy(self, gfu, isPlotvectors=True, **kwargs):
        """ Local speed of the tidal wave (amplitude).
        Args:
            gfu - complex valued gridfunction
        """

        speed = self.physical_parameters.omega / Norm( self.Im( Grad(gfu)/gfu ) )
        fig = self.pcolor(speed, **kwargs)

        if isPlotvectors:
            phase_velocity = - self.physical_parameters.omega * self.Im( Grad(gfu)/gfu ) / Norm( self.Im( Grad(gfu)/gfu ) )**2
            # The imaginairy part is zero. Hence, we only consider the real part
            phase_velocity_e = evaluate(phase_velocity, self.mesh, self.vertices_comp_mesh).real

            # We manually remove too large vectors
            max_vector = 20 # OR kwargs['vmax']
            magnitude_phase_velocity = np.linalg.norm(phase_velocity_e, axis=1)
            indices = magnitude_phase_velocity > max_vector
            mag_double = np.column_stack([magnitude_phase_velocity[indices], magnitude_phase_velocity[indices]])

            phase_velocity_e[indices, :] = np.divide(phase_velocity_e[indices, :], mag_double) * max_vector

            scale = 2000/5
            width = 0.001
            color = "tab:orange"
            plt.quiver(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], phase_velocity_e[:, 0], phase_velocity_e[:, 1], scale=scale, color=color)

        return fig

    # TODO remove
    def plot_speed_anomaly(self, gfu, isPlotvectors=True, **kwargs):
        """ Local speed of the tidal wave (amplitude) minus the frictionless tidal wave
        Args:
            gfu - complex valued gridfunction
        """

        speed = self.physical_parameters.omega / Norm( self.Im( Grad(gfu)/gfu ) )
        speed_frictionless = sqrt(self.physical_parameters.g *self.physical_parameters.H)

        # We compute the speed anomaly
        speed_anomaly = speed - speed_frictionless

        fig = self.pcolor(speed_anomaly, **kwargs)

        # TODO remove
        """
        if isPlotvectors:
            phase_velocity = self.physical_parameters.omega * self.Im( Grad(gfu)/gfu ) / Norm( self.Im( Grad(gfu)/gfu ) )**2
            # The imaginairy part is zero. Hence, we only consider the real part
            phase_velocity_e = evaluate(phase_velocity, self.mesh, self.vertices_comp_mesh).real

            # We manually remove too large vectors
            max_vector = 20 # OR kwargs['vmax']
            magnitude_phase_velocity = np.linalg.norm(phase_velocity_e, axis=1)
            indices = magnitude_phase_velocity > max_vector
            mag_double = np.column_stack([magnitude_phase_velocity[indices], magnitude_phase_velocity[indices]])

            phase_velocity_e[indices, :] = np.divide(phase_velocity_e[indices, :], mag_double) * max_vector

            # We multiply the vectors with -1
            phase_velocity_e = phase_velocity_e * (-1)

            scale = 2000
            plt.quiver(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], phase_velocity_e[:, 0], phase_velocity_e[:, 1], scale=scale)
        """
        return fig

    # TODO
    def plot_M0_vectors(self, res_hydro, isPlotvectors=True, **kwargs):
        """ Local speed of the tidal wave (amplitude).
        Args:
            gfu - complex valued gridfunction
        """

        velocity_vector = res_hydro.U_DA_V[0]['all']


        fig = self.pcolor(Norm(velocity_vector), **kwargs)

        if isPlotvectors:
            # The imaginairy part is zero. Hence, we only consider the real part
            velocity_e = evaluate(velocity_vector, self.mesh, self.vertices_comp_mesh).real

            """"# We manually remove too large vectors
            max_vector = 20 # OR kwargs['vmax']
            magnitude_phase_velocity = np.linalg.norm(velocity_e, axis=1)
            indices = magnitude_phase_velocity > max_vector
            mag_double = np.column_stack([magnitude_phase_velocity[indices], magnitude_phase_velocity[indices]])

            velocity_e[indices, :] = np.divide(velocity_e[indices, :], mag_double) * max_vector

            # We multiply the vectors with -1
            velocity_e = velocity_e * (-1)
            """

            scale = 2000/400
            width = 0.001
            color = "tab:orange"
            plt.quiver(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], velocity_e[:, 0], velocity_e[:, 1], scale=scale, color=color)

        return fig


    def plot_amplification_proxy(self, gfu, **kwargs):
        """ Local amplification of the tidal wave (amplitude).
        Args:
            gfu - complex valued gridfunction
        """

        # TODO before amplification = self.physical_parameters.omega / Norm( self.Re( Grad(gfu)/gfu ) )
        #fig = self.pcolor(amplification, **kwargs)

        # The amplification in the direction of travel
        log_derivative = Grad(gfu)/gfu
        log_derivative_im = self.Im(log_derivative)
        n_unit_wave = - log_derivative_im / Norm(log_derivative_im)

        local_amplification = InnerProduct(self.Re(log_derivative), n_unit_wave)

        # Plot
        fig = self.pcolor(local_amplification.real, **kwargs)

        # TODO remove
        # phase_velocity = self.physical_parameters.omega * self.Im( Grad(gfu)/gfu ) / Norm( self.Im( Grad(gfu)/gfu ) )**2
        # The imaginairy part is zero. Hence, we only consider the real part
        #phase_velocity_e = evaluate(phase_velocity, self.mesh, self.vertices_comp_mesh).real

        # We manually remove too large vectors
        #max_vector = 20 # OR kwargs['vmax']
        #magnitude_phase_velocity = np.linalg.norm(phase_velocity_e, axis=1)
        #indices = magnitude_phase_velocity > max_vector
        #mag_double = np.column_stack([magnitude_phase_velocity[indices], magnitude_phase_velocity[indices]])

        #phase_velocity_e[indices, :] = np.divide(phase_velocity_e[indices, :], mag_double) * max_vector

        # We multiply the vectors with -1
        #phase_velocity_e = phase_velocity_e * (-1)

        #scale = 2000
        #plt.quiver(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], phase_velocity_e[:, 0], phase_velocity_e[:, 1], scale=scale)

        return fig

    # TODO
    def plot_Poynting_vector(self, res_hydro, n, symbol, isPlotvectors=True, **kwargs):
        """ Plot the direction of the tidally averaged engery flow

        We assume leading-oder M2 only

        Args:

        """

        Poynting_vector = self.physical_parameters.g * (self.physical_parameters.H + self.physical_parameters.R) * 1/2 * self.Re(res_hydro.U_DA_V[n][symbol] * Conj(res_hydro.Z[n][symbol]))



        Poynting_magnitude = Norm(Poynting_vector)
        fig = self.pcolor(Poynting_magnitude, **kwargs)

        # The velocity vector are plotted on top
        if isPlotvectors:
            Poynting_vector_e = evaluate(Poynting_vector, self.mesh, self.vertices_comp_mesh).real
            width = 0.001
            color = "tab:orange"
            plt.quiver(self.vertices_comp_mesh[:, 0], self.vertices_comp_mesh[:, 1], Poynting_vector_e[:, 0],
                       Poynting_vector_e[:, 1], width=width, color=color)

        return fig

        # TODO
    def plot_dissipation(self, res_hydrolead, **kwargs):
        """ Plot the tidally averaged dissipation

        We assume leading-oder M2 only

        Args:

        """

        # Note: here we implicitly use that NGsolves automatically applies the complex conjugate to the second argument of the Innerproduct function
        dissipation = self.physical_parameters.sf0 * 1 / 2 * self.Re( InnerProduct(
            res_hydrolead.U_DA_V[1]['all'], res_hydrolead.U_V[1]['all'](-self.physical_parameters.H)))


        fig = self.pcolor(dissipation.real, **kwargs)
        return fig

    def plot_energy(self, res_hydrolead, **kwargs):
        """ Plot the tidally averaged energy

        We assume leading-oder M2 only

        Args:

        """

        # TODO maybe replace the norm squared with the complex number times its conjugate
        energy = 1 / 4 * self.physical_parameters.g * Norm(res_hydrolead.Z[1]['all'])**2 + 1 / 4 * (
                    self.physical_parameters.H + self.physical_parameters.R) * Norm(res_hydrolead.U_DA_V[1]['all'])**2

        fig = self.pcolor(energy, **kwargs)
        return fig

    def plot_phase(self, gfu, **kwargs):
        """ Plot the phase of a complex field
        Args:
            gfu - complex valued gridfunction
        """
        # TODO check. DO we want minus the phase such that the phase is mainly positive?
        def phase(gfu):
            """Returns the pahse of a complex number in degrees"""
            todegree = 180/np.pi
            return atan2(self.Im(gfu).real, self.Re(gfu).real) * todegree

        fig = self.pcolor(phase(gfu), **kwargs)
        return fig






    # TODO
    # Think about the grid spacing of the ellipses and maybe their color
    # TODO maybe refined the grid to start at -H and go until R
    def plot_vectors_cross_section(self, res_hydrolead, p1, p2):
        """Function to plot the velocity vectors in the cross-section"""

        def shifted_linspace(begin, endpoint, n):
            """Function that shifts the linspace points dx/2 to the right"""
            output, dx = np.linspace(begin, endpoint, n, endpoint=False, retstep=True)
            return output + dx/2


        # 1) We compute the along-cross-section velocity and the vertical velocity
        nz = 7
        nxy = 15

        # Compute tangential and normal flow components w.r.t. cross-section
        tangential_vector = p2 - p1
        tangential_unit_vector = tangential_vector / np.linalg.norm(tangential_vector)
        normal_unit_vector = np.array([tangential_unit_vector[1], -tangential_unit_vector[0]])

        def U_cross(z):
            unit_ngsolve = ngsolve.CoefficientFunction(tuple(tangential_unit_vector), dims=(2, 1))
            return ngsolve.InnerProduct(res_hydrolead.U_V[1]['all'](z), unit_ngsolve)



        nudging_parameter = 1 / 1000
        p1 = self.nudge_point_inside_domain(p1, tangential_vector, nudging_parameter)
        p2 = self.nudge_point_inside_domain(p2, -tangential_vector, nudging_parameter)
        print("Nudged points")

        # Generate display grids
        # We omit the first and last index
        xline = shifted_linspace(p1[0], p2[0], nxy)
        yline = shifted_linspace(p1[1], p2[1], nxy)
        hline = -self.physical_parameters.H(self.mesh(xline, yline)).flatten()
        zline = np.linspace(0, min(hline), nz)
        xtilde = np.sqrt((xline - p1[0]) ** 2 + (yline - p1[1]) ** 2)



        # 2) The eval funtion
        # Sigma layer grid
        # We create nz grid functions evaluated at the sigma layers
        zgrid = np.array(
            [shifted_linspace(self.physical_parameters.R(self.mesh(x, y)), -self.physical_parameters.H(self.mesh(x, y)), nz)
             for x, y in zip(xline, yline)]).T
        xtildegrid = np.tile(xtilde, (nz, 1))

        def eval(gfu, xline, yline):
            # We omit the first and last index
            sigma = shifted_linspace(0, -1, nz)
            gfu_grid = np.zeros((nz, nxy), dtype=np.complex)
            for index, sigma_layer in enumerate(sigma):
                sigma_layer_cf = self.physical_parameters.R + (
                        self.physical_parameters.R + self.physical_parameters.H) * sigma_layer
                gfu_grid[index, :] = gfu(sigma_layer_cf)(self.mesh(xline, yline)).flatten()
            return gfu_grid


        # 2) We evaluate these components at different times over the tidal cycle
        # The vector field we want to plot
        U_cross_e = eval(U_cross, xline, yline)
        W_cross_e = eval(res_hydrolead.W[1]['all'], xline, yline)



        #    We plot the computed vectors
        # We plot a background state
        self.cross_section_scalar(lambda z: ngsolve.Norm(U_cross(z)), p1, p2, isLabel=False)


        steps = 300
        phase_v = np.linspace(0, 2*np.pi, steps, endpoint=False)
        phase_v = np.flip(phase_v)

        cmap = plt.get_cmap("hsv")

        # We walk through phase_v backwards to be able to plot the orange vector at t=0 on top
        for phase in phase_v:
            velocity_1_e_phase = np.real(U_cross_e * np.exp(1j*phase))
            velocity_2_e_phase = np.real(W_cross_e * np.exp(1j*phase))



            # Differentiate between colours
            # TODO before
            #if phase == phase_v[-1]:
            #    color = "tab:orange"
            #else:
            #    color = "grey"

            # TODO new colorscale
            colorscale = round(phase / (2 * np.pi) * 255)
            color = cmap(colorscale)

            scale = 2/10000

            # TODO remove
            headwidth = 1
            headlength = 0
            plt.quiver(xtildegrid.flatten(), zgrid.flatten(), velocity_1_e_phase.flatten(), velocity_2_e_phase.flatten(),
                       angles='xy', scale_units='xy', color=color, scale=scale, headwidth=headwidth,headlength=headlength)




    ################# plot along lines ##################
    # We generate a method to get countour lines of a given contour value
    def get_contour(self, scalar_fied, value):
        """
        Args:
            scalar_fied: coefficient function from which the contourline must be found
            value: value of the function along contour line

        Returns:
            vector of x,y points of the contour
        """
        # TODO For now we assume that the scalar_field is real valued
        scalar_fied_on_disp_mesh = evaluateV(scalar_fied, self.mesh, self.vertices_display_mesh)

        # Method without creating a figure to get the contour
        contour_generator = _tri.TriContourGenerator(self.triangulation_display_mesh.get_cpp_triangulation(), scalar_fied_on_disp_mesh)
        contour = contour_generator.create_contour(value)[0][0]

        # TODO remove
        # Method with figure to get contour
        #fig, ax = plt.subplots()
        #contour_plot = ax.tricontour(self.triangulation_display_mesh, scalar_fied_on_disp_mesh, [value])
        #contour = contour_plot.allsegs[0][0]
        #plt.plot(contour[:, 0], contour[:, 1], '*-')
        return contour

    def plot_scalar_along_contour(self, Z_cf, contour_cf, cf_lvl):
        """

        Args:
            Z_cf: Scalar coefficient function to be plotted along contour
            contour_cf: (Real-valued) scalar function to compute the contour level from
            cf_lvl: value of contour level

        Returns:
            fig
        """
        fig, ax = plt.subplots()
        contour = self.get_contour(contour_cf, cf_lvl)

        # We evaluate the given function along the contour
        Z_c = evaluateV(Z_cf, self.mesh, contour)

        # We compute the distance between the given contour points
        # 1) We compute the vector line segment as the difference between two consecutive points
        # 2) We compute the magnitude of the line segments
        # 3) We cumulatively add the line-segments to get the along contour line distance
        dl_v = np.diff(contour, axis=0)
        dl = np.linalg.norm(dl_v, axis=1)
        x_c = np.zeros(len(contour))
        x_c[1:] = np.cumsum(dl)

        plt.plot(x_c, Z_c, '-*')
        return plt


    ### Conversion method ####
    def axis_to_km(self, ax):
        """
        Function that converts ax in meters to km.
        """
        scale = 1000
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)

        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        return plt


    ########## Evaluate functions ##################
    def evaluate_sigma_grid(self, gfu, xline, yline, zline, nxy, nz):
        """Evaluate function on sigma layers"""
        Zgrid = np.array([gfu(z)(self.mesh(x, y)) for x, y in zip(xline, yline)
                          for z in np.linspace(0, -self.physical_parameters.H(self.mesh(x, y)), nz)]).reshape(nxy, nz).T
        return Zgrid


    def evaluate_uniform_grid(self, gfu, xline, yline, zline, nxy, nz):
        """Evaluate function on uniform grid. Can be 10x faster.
        2020-7-29: Changed nxy and nz"""
        Zgrid = np.array([gfu(z)(self.mesh(xline, yline)) for z in zline]).reshape(nz, nxy)
        return Zgrid


    ################## Maskes ###############
    def mask_triangles_branchcut_positive_axis(self, Zphase, tol):
        """Masks triangles that intersect the branch cut along the positive real axis"""
        Zphase_triangle = (self.physical_parameters.omega * 60 * 60) * Zphase[self.triangulation_display_mesh.triangles]
        n_zeros = np.count_nonzero(Zphase_triangle < tol, axis=1)
        n_2pi = np.count_nonzero(2 * np.pi - Zphase_triangle < tol, axis=1)
        mask = np.any([np.all([n_zeros == 2, n_2pi == 1], axis=0), np.all([n_zeros == 1, n_2pi == 2], axis=0)], axis=0)
        return mask


    def mask_triangles_branchcut_negative_axis(self, Zphase2, tol):
        """Masks triangles that intersect the branch cut along the negative real axis"""
        Zphase2_triangle = (self.physical_parameters.omega * 60 * 60) * Zphase2[self.triangulation_display_mesh.triangles]
        n_mpi = np.count_nonzero(np.pi + Zphase2_triangle  < tol, axis=1)
        n_pi = np.count_nonzero(np.pi - Zphase2_triangle < tol, axis=1)
        mask2 = np.any([np.all([n_mpi == 2, n_pi == 1], axis=0), np.all([n_mpi == 1, n_pi == 2], axis=0)], axis=0)
        return mask2


    def mask_triangles_small_amplitude(self, Zamp, tol2):
        """Masks triangles with very small amplitude. The phase of these complex numbers can become unstable
        and more importantly for linear elements a single nearly zero vertex pollutes the whole triangle"""
        Zamp_triangle = Zamp[self.triangulation_display_mesh.triangles]
        mask_small_amp = np.any([Zamp_triangle<tol2], axis=2).flatten()
        return mask_small_amp


    def mask_below_depth(self, zline, hline, nz, nxy):
        mask_below = np.zeros((nz, nxy), dtype=bool)
        for i in range(nxy):
            mask_below[:, i] = zline < hline[i]
        return mask_below


    def _divide_phase(self, phasetimes, subphase_lines):
        """""Function to divide the phase into nice intervals for the contour plot"""
        period_hours = 2*np.pi/(self.physical_parameters.omega*60*60)
        period_wholehours = int(period_hours)

        if phasetimes == "hours":
            levels_phase = list(np.linspace(0, period_wholehours, period_wholehours * (1 + subphase_lines) + 1))
            phase_decimals = 0
        elif phasetimes == "halfhours":
            levels_phase = np.linspace(0, period_wholehours, 2 * period_wholehours * (1 + subphase_lines) + 1)
            phase_decimals = 1
        elif phasetimes == "quarters":
            levels_phase = np.linspace(0, period_wholehours, 4 * period_wholehours * (1 + subphase_lines) + 1)
            phase_decimals = 2
        elif phasetimes == "eighths":
            levels_phase = np.linspace(0, period_wholehours, 8 * period_wholehours * (1 + subphase_lines) + 1)
            phase_decimals = 3
        else:
            raise("Wrong phasetimes supplied. Possibilities are: hours, halfhours, quarters, eights")

        # Append the sub intervals between wholehours and period_in_hours
        step = levels_phase[1]-levels_phase[0]
        times_residue = int((period_hours-period_wholehours)/step)
        levels_phase_tail = np.linspace(period_wholehours + step, period_wholehours + times_residue*step, times_residue)
        levels_phase = np.concatenate((levels_phase, levels_phase_tail))

        return levels_phase, phase_decimals







