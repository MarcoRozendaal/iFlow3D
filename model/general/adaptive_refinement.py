"""
    This file contains a module to automatically refine the grid based on an error-estimator.

    For normal FEM, the adaptive refinement is guided by the ZZ-estimator.
    For mixed FEM, I created my own error estimator. So its properties are unknown.
"""

from ngsolve import *
import numpy as np
import time

from model.backup import weak_formulations as wf
from model.general import create_geometry as cg


def adaptive_refinement(order, mesh, maxdofs, minH1error, p, degree_curved_geometry=1, isCalculateLastError=False):
    """ Adaptive refinement for normal FEM
    Solves -Grad(v)*D(0)*Grad(Z) + j*omega*v*Z = 0 for Z in mesh

    We could use a preconditioner c = Preconditioner(a, "direct") and then , pre=c in the solver
     """
    fes = H1(mesh, order=order, dirichlet=cg.BOUNDARY_DICT[cg.SEA], complex=True, autoupdate=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes, symmetric=False, condense=True)
    a += - Grad(v) * (p.D(0) * Grad(u)) * dx + 1j * p.omega * v * u * dx

    f = LinearForm(fes)

    Zi = GridFunction(fes, "free_surface", autoupdate=True)

    Draw(Zi)


    # Finite element space and gridfunction to represent the flux:
    space_flux = HDiv(mesh, order=order-1, complex=True, autoupdate=True)
    gf_flux = GridFunction(space_flux, "flux", autoupdate=True)


    H1_error_list = [(fes.ndof, 1)]

    def SolveBVP():
        Zi.Set(p.A, definedon=mesh.Boundaries(cg.BOUNDARY_DICT[cg.SEA]))
        Draw(Zi)
        solvers.BVP(bf=a, lf=f, gf=Zi)
        #Redraw(blocking=True) #TODO commenting this line since it appears to crash here


    def CalcError():
        flux = p.D(0) * Grad(Zi)

        # Interpolate finite element flux into H(div) space:
        gf_flux.Set(flux)

        # Gradient-recovery error estimator
        err = (flux - gf_flux) * Conj(flux - gf_flux)
        elerr = np.array(Integrate(err, mesh, VOL, element_wise=True)).real

        max_el_err = np.max(elerr)
        H1err = np.sqrt(np.sum(elerr))
        H1_error_list.append((fes.ndof, H1err))
        print("H1_error =",  H1err, ", max_el_err =", max_el_err, ", dofs =", fes.ndof)

        for el in mesh.Elements():
            mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25 * max_el_err)


    with TaskManager():
        while fes.ndof < maxdofs and H1_error_list[-1][1] > minH1error:
            SolveBVP()
            CalcError()
            mesh.Refine()
            mesh.Curve(degree_curved_geometry)

    start1 = time.time()
    SolveBVP()
    print("Last solve {:.5f} seconds".format(time.time() - start1))


    if isCalculateLastError:
        CalcError()

    Zdraw = Conj(Zi)


    Draw(Zdraw, mesh, "correct_animation")
    print(H1_error_list)
    return Zi, fes, H1_error_list


def adaptive_refinement_mixedRT(order_flux, mesh, maxdofs, minH2error, p):
    """Adaptive refinement for the mixed methods using Raviart Thomas elements"""
    fs = 0

    V = HDiv(mesh, order=order_flux, complex=True, dirichlet="{}|{}|{}".format(cg.BOUNDARY_DICT[cg.WALLDOWN], cg.BOUNDARY_DICT[cg.WALLUP], cg.BOUNDARY_DICT[cg.RIVER]), RT=True, autoupdate=True)
    Q = L2(mesh, order=order_flux, complex=True, autoupdate=True)

    fesm = FESpace([V, Q])

    sigma, u = fesm.TrialFunction()
    tau, v = fesm.TestFunction()

    normal = specialcf.normal(mesh.dim)

    am = BilinearForm(fesm, symmetric=False, condense=True)
    am += (tau * (p.Dinv(0) * sigma) + v * div(sigma) + 1j * p.omega * v * u + div(
        tau) * u) * dx

    fm = LinearForm(fesm)
    fm += fs * v * dx + p.A * (tau.Trace() * normal) * ds(cg.BOUNDARY_DICT[cg.SEA])

    gfm = GridFunction(fesm, autoupdate=True)
    gfsigmai, Zi = gfm.components


    Draw(Zi)

    H2_error_list = [(fesm.ndof, 1)]

    def SolveBVP():
        fesm.Update()
        gfsigmai.Update()
        Zi.Update()
        gfsigmai.Set(0 * normal, BND)
        solvers.BVP(bf=am, lf=fm, gf=gfm)
        Redraw(blocking=True)

    def CalcError():
        HZ = wf.fast_Hessian(gfsigmai, p)

        # My error estimator
        sigmoid = 1/(1 + exp(-x/(1.1e2)))
        err = (HZ[0, 1] - HZ[1, 0]) * Conj(HZ[0, 1] - HZ[1, 0]) * sigmoid
        elerr = np.array(Integrate(err, mesh, VOL, element_wise=True)).real

        max_el_err = np.max(elerr)
        H2err = np.sqrt(np.sum(max_el_err))
        H2_error_list.append((fesm.ndof, H2err))
        print("H2_error =", H2err, ", max_el_err =", max_el_err, ", dofs =", fesm.ndof)

        for el in mesh.Elements():
            mesh.SetRefinementFlag(el, elerr[el.nr] > 0.25 * max_el_err)


    with TaskManager():
        while fesm.ndof < maxdofs and H2_error_list[-1][1] > minH2error:
            SolveBVP()
            CalcError()
            mesh.Refine()

    SolveBVP()
    CalcError()

    print(H2_error_list)
    return gfsigmai, Zi, fesm, H2_error_list













