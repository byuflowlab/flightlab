"""The steady analysis driver.

Ported from VortexLattice.jl (``src/analyses.jl``), steady portion only.  The
unsteady solver, the free wake, the nonlinear (viscous-coupled) analysis, the
rotor helpers and the VTK writers are deliberately not ported -- the course
never needs them.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.linalg as sla

from .freestream import (
    Freestream,
    freestream_velocity_derivatives,
    rotational_velocity_derivatives,
)
from .geometry import grid_to_surface_panels
from .induced import influence_coefficients
from .nearfield import DERIVS, near_field_forces
from .panel import Surface
from .reference import Reference
from .system import System

__all__ = ["steady_analysis"]


def _as_list(value, n, name, default):
    """Broadcast a scalar flag to one value per surface."""
    if value is None:
        value = default
    if isinstance(value, (bool, int, np.bool_, np.integer)) and not isinstance(
        value, (list, tuple, np.ndarray)
    ):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} must have one entry per surface (got {len(value)})")
    return value


def steady_analysis(
    surfaces,
    reference: Reference,
    freestream: Freestream,
    symmetric=False,
    surface_id=None,
    trailing_vortices=True,
    xhat=(1.0, 0.0, 0.0),
    ratios=None,
    fcore=None,
    near_field_analysis=True,
    derivatives=True,
):
    """Run a steady vortex-lattice analysis.

    Parameters
    ----------
    surfaces : sequence
        Either grids of shape ``(3, nc+1, ns+1)`` (as returned by
        :func:`~flightlab.vlm.wing_to_grid`) or :class:`~flightlab.vlm.Surface`
        objects.  A single grid or surface may be passed on its own.
    reference : Reference
        Reference area, chord, span, moment reference point and velocity.
    freestream : Freestream
        Flow condition.  ``alpha`` and ``beta`` are in **radians**; build one
        with :meth:`Freestream.from_degrees` if you are working in degrees.
    symmetric : bool or sequence of bool
        Per surface, use a mirror image across the X-Z plane instead of
        modelling both halves.  Halves the panel count for the same accuracy.
        Do not combine with a geometry that is already mirrored.

        **This zeroes the antisymmetric results.**  With ``symmetric=True`` the
        side force ``CY`` and the rolling and yawing moments ``Cl`` and ``Cn``
        come back as exactly zero, along with every derivative of them --
        ``Cl_beta``, ``Cn_beta``, ``Cl_p``, ``Cn_r`` and the rest.  That is
        correct (the mirror image cancels them), but it means **any lateral or
        roll/yaw analysis must use a mirrored geometry** and
        ``symmetric=False``.  Build it with ``wing_to_grid(..., mirror=True)``.
        Longitudinal results agree between the two approaches.
    surface_id : sequence of int, optional
        Surfaces that share an ID do not apply the finite-core model to each
        other.  Defaults to a distinct ID per surface; set them all equal to
        match AVL with its finite-core model switched off.
    trailing_vortices : bool or sequence of bool
        Shed trailing vortices to infinity.  Almost always ``True``.
    xhat : array_like, shape (3,)
        Direction the trailing vortices are shed in.
    ratios : sequence of ndarray, optional
        Panel placement ratios from :func:`~flightlab.vlm.wing_to_grid`.  Pass them
        whenever you pass grids, or the control points fall back to mid-panel
        and 3/4 chord.
    fcore : callable, optional
        ``fcore(chord, dspan) -> core_size`` in metres.  Defaults to ``1e-3``.
    near_field_analysis : bool
        Compute per-panel forces.  Required by :func:`~flightlab.vlm.body_forces`
        and :func:`~flightlab.vlm.lifting_line_coefficients`.
    derivatives : bool
        Compute freestream derivatives.  Required by
        :func:`~flightlab.vlm.stability_derivatives`.

    Returns
    -------
    System
        Holds the circulation solution and, if requested, panel properties and
        derivatives.

    Examples
    --------
    A single symmetric wing::

        grid, ratios = wing_to_grid(xle, yle, zle, chord, theta, phi, ns, nc)
        ref = Reference(S=30.0, c=2.0, b=15.0, r=[0.5, 0, 0], V=1.0)
        fs = Freestream.from_degrees(Vinf=1.0, alpha=1.0)
        system = steady_analysis([grid], ref, fs, symmetric=True, ratios=[ratios])
        CF, CM = body_forces(system, frame=Stability())
        CDi = far_field_drag(system)
    """
    grids, surfs, ratio_list = _normalize_geometry(surfaces, ratios, fcore)
    n = len(surfs)

    symmetric = _as_list(symmetric, n, "symmetric", False)
    trailing_vortices = _as_list(trailing_vortices, n, "trailing_vortices", True)
    if surface_id is None:
        surface_id = list(range(1, n + 1))
    surface_id = _as_list(surface_id, n, "surface_id", None)
    xhat = np.asarray(xhat, dtype=float).reshape(3)

    system = System(
        surfaces=surfs,
        grids=grids,
        ratios=ratio_list,
        reference=reference,
        freestream=freestream,
        symmetric=symmetric,
        surface_id=surface_id,
        trailing_vortices=trailing_vortices,
        xhat=xhat,
    )

    system.AIC = influence_coefficients(
        surfs, symmetric, surface_id, trailing_vortices, xhat
    )
    system.w, dw = _normal_velocity(surfs, reference, freestream, derivatives)

    lu = sla.lu_factor(system.AIC)
    system.gamma = sla.lu_solve(lu, system.w)
    if derivatives:
        system.dgamma = {
            name: sla.lu_solve(lu, dw[name]) for name in DERIVS
        }
    else:
        system.dgamma = None

    if near_field_analysis:
        near_field_forces(system)

    return system


def _normalize_geometry(surfaces, ratios, fcore):
    """Accept grids, Surfaces, or a single one of either; return all three forms."""
    if isinstance(surfaces, Surface):
        surfaces = [surfaces]
    elif isinstance(surfaces, np.ndarray) and surfaces.ndim == 3:
        surfaces = [surfaces]

    if ratios is None:
        ratios = [None] * len(surfaces)
    elif isinstance(ratios, np.ndarray) and ratios.ndim == 3:
        ratios = [ratios]

    grids, surfs, ratio_list = [], [], []
    for item, rat in zip(surfaces, ratios):
        if isinstance(item, Surface):
            grids.append(None)
            surfs.append(item)
            ratio_list.append(rat)
        else:
            grid, rat_out, surf = grid_to_surface_panels(
                item, ratios=rat, fcore=fcore
            )
            grids.append(grid)
            surfs.append(surf)
            ratio_list.append(rat_out)

    if any(g is None for g in grids):
        # lifting_line_coefficients needs grids; keep the list only if complete
        grids = [g for g in grids if g is not None]
        if len(grids) != len(surfs):
            grids = []

    return grids, surfs, ratio_list


def _normal_velocity(surfaces, ref, fs, derivatives):
    """Right-hand side of the circulation system: minus the normal throughflow.

    Returns
    -------
    w : ndarray, shape (N,)
    dw : dict or None
        Derivatives of ``w`` with respect to each freestream variable.
    """
    Vfs, (Vfs_a, Vfs_b) = freestream_velocity_derivatives(fs)

    w_parts = []
    dw_parts = {name: [] for name in DERIVS}
    for surf in surfaces:
        P = surf.flat("rcp")
        nhat = surf.flat("ncp")
        Vrot, dVrot = rotational_velocity_derivatives(P, fs, ref)
        V = Vfs + Vrot
        w_parts.append(-np.einsum("ij,ij->i", V, nhat))
        if derivatives:
            base = {
                "alpha": np.broadcast_to(Vfs_a, P.shape),
                "beta": np.broadcast_to(Vfs_b, P.shape),
                "p": dVrot[0],
                "q": dVrot[1],
                "r": dVrot[2],
            }
            for name in DERIVS:
                dw_parts[name].append(-np.einsum("ij,ij->i", base[name], nhat))

    w = np.concatenate(w_parts)
    if not derivatives:
        return w, None
    dw = {name: np.concatenate(dw_parts[name]) for name in DERIVS}
    return w, dw
