""" File with functions to save output """
import numpy as np

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkWedge, VtkTriangle


from model.general.post_processing import evaluateV
from model.packages.sediment3D.classes.sediment import Sediment
from model.packages.sediment3D.classes.transport_capacity import TransportCapacity


def save_freesurface(res_hydrolead, post_processing):
    """ Function that saves the element matrix containing the connection between faces and vertices """

    vertices_mesh = post_processing.vertices_display_mesh
    faces = post_processing.triangulation_display_mesh.triangles

    bathymetry = evaluateV(post_processing.physical_parameters.H, post_processing.mesh, vertices_mesh)
    Z = evaluateV(res_hydrolead.Z[1]['all'], post_processing.mesh, vertices_mesh)

    bathymetry_vertices = np.column_stack([vertices_mesh, bathymetry])
    Zphasor_vertices = np.column_stack([vertices_mesh, Z])


    # Save data
    np.savetxt("bathymetry_vertices.txt", bathymetry_vertices)
    np.savetxt("Zphasor_vertices.txt", Zphasor_vertices)
    np.savetxt("faces.txt", faces,  fmt="%d")

    print("Done saving")
    return




### TODO a function to export a 3D grid consisting of wedges
## Can be read into paraview

def save_3D_grid(post, res_hydrolead):
    """Function to create a 3D grid consisting of wedges (triangle prisms)"""
    # The vertical dimension is discretized using sigma layers
    VERTICES_PER_WEDGE = 6


    nz = 10  # Number of sigma layers
    nmesh = len(post.vertices_display_mesh) # number of mesh points


    # We put the location of all the 3D vertices into x,y,z
    x = np.zeros(nmesh * nz)
    y = np.zeros(nmesh * nz)
    z = np.zeros(nmesh * nz)

    U_amp = np.zeros(nmesh * nz)
    V_amp = np.zeros(nmesh * nz)
    W_amp = np.zeros(nmesh * nz)
    U_phase = np.zeros(nmesh * nz)
    V_phase = np.zeros(nmesh * nz)
    W_phase = np.zeros(nmesh * nz)

    # Get z values at the sigma layers
    sigma = np.linspace(-1, 0, nz)
    for i, sigma_layer in enumerate(sigma):
        z_sigma_layer_cf = post.physical_parameters.R + post.physical_parameters.D * sigma_layer
        z_sigma_layer_e = z_sigma_layer_cf(post.mesh(post.vertices_display_mesh[:, 0], post.vertices_display_mesh[:, 1]))

        # We evaluate U_V,W
        U_V_layer_e = res_hydrolead.U_V[1]['all'](z_sigma_layer_cf)(post.mesh(post.vertices_display_mesh[:, 0], post.vertices_display_mesh[:, 1]))
        W_layer_e = res_hydrolead.W[1]['all'](z_sigma_layer_cf)(post.mesh(post.vertices_display_mesh[:, 0], post.vertices_display_mesh[:, 1]))


        # Build the matrices
        indices = np.arange(i*nmesh, (i+1)*nmesh)
        x[indices] = post.vertices_display_mesh[:, 0]
        y[indices] = post.vertices_display_mesh[:, 1]
        z[indices] = z_sigma_layer_e.ravel()

        U_amp[indices] = np.abs(U_V_layer_e[:, 0])
        V_amp[indices] = np.abs(U_V_layer_e[:, 1])
        W_amp[indices] = np.abs(W_layer_e.ravel())

        U_phase[indices] = np.angle(U_V_layer_e[:, 0])
        V_phase[indices] = np.angle(U_V_layer_e[:, 1])
        W_phase[indices] = np.angle(W_layer_e.ravel())

    # TODO check
    # We also evaluate point data
    point_data = {"U_amp": U_amp, "V_amp": V_amp, "W_amp": W_amp, "U_phase": U_phase, "V_phase": V_phase, "W_phase": W_phase}

    # Define connectivity or vertices that belongs to each element
    connectivity_layer = post.triangulation_display_mesh.triangles
    ntriangles = len(connectivity_layer)
    nwedges = (nz-1) * ntriangles


    # We use a little convoluted method compute the connectivity matrix
    for i in range(nz-1):
        connectivity_wedge = np.hstack([connectivity_layer + nmesh*i, connectivity_layer + nmesh*(i+1)])
        if i == 0:
            connectivity_matrix = connectivity_wedge
        else:
            connectivity_matrix = np.hstack([connectivity_matrix, connectivity_wedge])

    connectivity = connectivity_matrix.flatten()


    # Define offset of last vertex of each element
    offset = np.cumsum(np.ones(nwedges)*VERTICES_PER_WEDGE)


    # Define cell types
    ctype = np.ones(nwedges)*VtkWedge.tid

    unstructuredGridToVTK(
        "unstructured_wedges",
        x,
        y,
        z,
        connectivity=connectivity,
        offsets=offset,
        cell_types=ctype,
        pointData=point_data
    )



# Used to save transports for Paraview
def save_sed_all(sed: Sediment, transcap: TransportCapacity, post):
    """ Saving the transcap and sediment all enties """

    # The vertical dimension is discretized using sigma layers
    VERTICES_PER_TRIANGLE = 3
    comp_mesh_vert = post.vertices_comp_mesh


    nmesh = len(post.vertices_comp_mesh)  # number of mesh points

    # We put the location of all the 2D vertices into x,y. Note the comp mesh here
    x = np.array(comp_mesh_vert[:, 0])
    y = np.array(comp_mesh_vert[:, 1])
    z = np.zeros_like(x)

    T_V = transcap.T_V['all'].data

    # TODO discretize data
    C00EA_DA = sed.C_DA[0][0]['etide'](post.mesh(x, y)).flatten()

    # Amplitude phase
    C02EA_DA_e = sed.C_DA[0][2]['etide'](post.mesh(x, y)).flatten()
    C02EA_DA_amp = np.abs(C02EA_DA_e)
    C02EA_DA_phase = np.angle(C02EA_DA_e)

    # Amplitude phase
    C11all_DA_e = sed.C_DA[1][1]['all'](post.mesh(x, y)).flatten()
    C11all_DA_amp = np.abs(C11all_DA_e)
    C11all_DA_phase = np.angle(C11all_DA_e)

    # TODO check
    # We also evaluate point data
    # TODO maybe make all
    point_data = {"C00EA": C00EA_DA, "C02EA_amp": C02EA_DA_amp, "C02EA_phase": C02EA_DA_phase,
                                    "C11all_amp": C11all_DA_amp, "C11all_phase": C11all_DA_phase,
                    "T_0": T_V[0], "T_1": T_V[1], "zeros": z}

    # Define connectivity or vertices that belongs to each element
    connectivity_matrix = post.triangulation_comp_mesh.triangles # Note comp mesh
    ntriangles = len(connectivity_matrix)

    # TODO check this
    connectivity = connectivity_matrix.flatten()

    # Define offset of last vertex of each element
    offset = np.cumsum(np.ones(ntriangles) * VERTICES_PER_TRIANGLE)

    # Define cell types
    ctype = np.ones(ntriangles) * VtkTriangle.tid

    unstructuredGridToVTK(
        "unstructured_triangles",
        x,
        y,
        z,
        connectivity=connectivity,
        offsets=offset,
        cell_types=ctype,
        pointData=point_data
    )
