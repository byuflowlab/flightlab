"""Near-field forces: Kutta-Joukowski on each filament, then integration.

Ported from VortexLattice.jl (``src/nearfield.jl``).

The near-field force on a panel is ``rho * Gamma * (V x ds)`` applied to each of
the ring's bound filaments.  Summed over the aircraft this gives ``CF`` and
``CM``; collected chordwise it gives the per-span-station coefficients that a
strip-integration drag buildup needs.

Near-field drag from a vortex lattice is the *pressure* drag obtained from the
local velocity, and it is noisier than the Trefftz-plane value.  When you want
induced drag, use :func:`flightlab.vlm.far_field_drag`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .freestream import (
    Freestream,
    body_to_stability,
    body_to_wind,
    freestream_velocity_derivatives,
    rotational_velocity,
    rotational_velocity_derivatives,
)
from .induced import induced_velocity
from .reference import Body, Reference, Stability, Wind
from .system import RHO, PanelProperties, System

__all__ = [
    "near_field_forces",
    "body_forces",
    "body_forces_derivatives",
    "lifting_line_coefficients",
    "body_to_frame",
]

#: Order in which freestream derivatives are carried around.
DERIVS = ("alpha", "beta", "p", "q", "r")

#: ``Mx`` and ``Mz`` come out of the geometric axes with the opposite sign to
#: the roll and yaw moments of flight dynamics.
_CONVENTION = np.array([-1.0, 1.0, -1.0])


def _stack_gamma(gamma, dgamma):
    """Stack ``gamma`` and its five derivatives into a ``(k, N)`` array."""
    if dgamma is None:
        return gamma[None, :], False
    rows = [gamma] + [dgamma[name] for name in DERIVS]
    return np.stack(rows, axis=0), True


def _net_circulation(g2d):
    """Circulation jump across each panel's bound vortex.

    A vortex ring's bound filament carries the difference between its own
    circulation and that of the panel ahead of it, because the two rings share
    that edge with opposite sense.  The leading-edge row carries its full value.
    """
    out = np.empty_like(g2d)
    out[..., 0, :] = g2d[..., 0, :]
    out[..., 1:, :] = g2d[..., 1:, :] - g2d[..., :-1, :]
    return out


def near_field_forces(system: System) -> List[PanelProperties]:
    """Compute per-panel forces and store them on ``system``.

    Parameters
    ----------
    system : System
        Must already hold a circulation solution in ``system.gamma``.

    Returns
    -------
    list of PanelProperties
    """
    ref = system.reference
    fs = system.freestream
    surfaces = system.surfaces
    offs = system.offsets

    G, want_deriv = _stack_gamma(system.gamma, system.dgamma)
    k = G.shape[0]

    Vfs, (Vfs_a, Vfs_b) = freestream_velocity_derivatives(fs)
    q = 0.5 * RHO * ref.V**2
    qS = q * ref.S

    props: List[PanelProperties] = []
    dprops: dict = {name: [] for name in DERIVS}

    for isurf, surf in enumerate(surfaces):
        nc, ns = surf.shape
        n = nc * ns

        # --- velocity at the bound vortex centres --------------------------
        Ptop = surf.flat("rtc")
        Vrot, dVrot = rotational_velocity_derivatives(Ptop, fs, ref)

        # index of the filament(s) coincident with each evaluation point
        idx = np.arange(n)
        skip_top = idx
        skip_bottom = np.where(idx >= ns, idx - ns, -1)

        Vind = induced_velocity(
            Ptop,
            surfaces,
            G,
            system.symmetric,
            system.surface_id,
            system.trailing_vortices,
            system.xhat,
            receiving_surface=isurf,
            skip_top=skip_top,
            skip_bottom=skip_bottom,
        )  # (k, n, 3)

        Vtop = np.empty((k, n, 3))
        Vtop[0] = Vfs + Vrot + Vind[0]
        if want_deriv:
            base = {
                "alpha": np.broadcast_to(Vfs_a, (n, 3)),
                "beta": np.broadcast_to(Vfs_b, (n, 3)),
                "p": dVrot[0],
                "q": dVrot[1],
                "r": dVrot[2],
            }
            for m, name in enumerate(DERIVS, start=1):
                Vtop[m] = base[name] + Vind[m]

        # --- velocity at the side legs (no induced contribution) -----------
        # AVL makes the same approximation: the side legs are nearly parallel
        # to the induced velocity, so the cross product is insensitive to it.
        Vleft = _leg_velocity(surf.left_center.reshape(-1, 3), fs, ref, Vfs,
                              Vfs_a, Vfs_b, k, want_deriv)
        Vright = _leg_velocity(surf.right_center.reshape(-1, 3), fs, ref, Vfs,
                               Vfs_a, Vfs_b, k, want_deriv)

        # --- Kutta-Joukowski ----------------------------------------------
        g_surf = G[:, offs[isurf] : offs[isurf + 1]].reshape(k, nc, ns)
        g_net = _net_circulation(g_surf).reshape(k, n)

        ds_top = surf.top_vector.reshape(-1, 3)
        ds_left = surf.left_vector.reshape(-1, 3)
        ds_right = surf.right_vector.reshape(-1, 3)

        Fb = _kutta(g_net, Vtop, ds_top)
        Fl = _kutta(G[:, offs[isurf] : offs[isurf + 1]], Vleft, ds_left)
        Fr = _kutta(G[:, offs[isurf] : offs[isurf + 1]], Vright, ds_right)

        shape3 = (nc, ns, 3)
        props.append(
            PanelProperties(
                gamma=(G[0, offs[isurf] : offs[isurf + 1]] / ref.V).reshape(nc, ns),
                velocity=(Vtop[0] / ref.V).reshape(shape3),
                cfb=(Fb[0] / qS).reshape(shape3),
                cfl=(Fl[0] / qS).reshape(shape3),
                cfr=(Fr[0] / qS).reshape(shape3),
            )
        )
        if want_deriv:
            for m, name in enumerate(DERIVS, start=1):
                dprops[name].append(
                    PanelProperties(
                        gamma=(
                            system.dgamma[name][offs[isurf] : offs[isurf + 1]] / ref.V
                        ).reshape(nc, ns),
                        velocity=(Vtop[m] / ref.V).reshape(shape3),
                        cfb=(Fb[m] / qS).reshape(shape3),
                        cfl=(Fl[m] / qS).reshape(shape3),
                        cfr=(Fr[m] / qS).reshape(shape3),
                    )
                )

    system.properties = props
    system.dproperties = dprops if want_deriv else None
    return props


def _leg_velocity(P, fs, ref, Vfs, Vfs_a, Vfs_b, k, want_deriv):
    """Velocity and its freestream derivatives at the ring side legs."""
    n = P.shape[0]
    Vrot, dVrot = rotational_velocity_derivatives(P, fs, ref)
    V = np.empty((k, n, 3))
    V[0] = Vfs + Vrot
    if want_deriv:
        base = {
            "alpha": np.broadcast_to(Vfs_a, (n, 3)),
            "beta": np.broadcast_to(Vfs_b, (n, 3)),
            "p": dVrot[0],
            "q": dVrot[1],
            "r": dVrot[2],
        }
        for m, name in enumerate(DERIVS, start=1):
            V[m] = base[name]
    return V


def _kutta(g, V, ds):
    """``rho * Gamma * (V x ds)`` and its derivatives, by the product rule.

    Parameters
    ----------
    g : ndarray, shape (k, n)
        Circulation and its derivatives.
    V : ndarray, shape (k, n, 3)
        Velocity and its derivatives.
    ds : ndarray, shape (n, 3)
        Filament vector.

    Returns
    -------
    ndarray, shape (k, n, 3)
    """
    cross0 = np.cross(V[0], ds)  # (n, 3)
    F = np.empty_like(V)
    F[0] = RHO * g[0][:, None] * cross0
    for m in range(1, g.shape[0]):
        F[m] = RHO * (
            g[m][:, None] * cross0 + g[0][:, None] * np.cross(V[m], ds)
        )
    return F


# --- integrated forces ------------------------------------------------------


def _integrate(surfaces, props, ref, symmetric):
    """Sum panel forces and moments over every surface."""
    CF = np.zeros(3)
    CM = np.zeros(3)
    for isurf, surf in enumerate(surfaces):
        p = props[isurf]
        CFi = np.zeros(3)
        CMi = np.zeros(3)
        for rc, cf in (
            (surf.rtc, p.cfb),
            (surf.left_center, p.cfl),
            (surf.right_center, p.cfr),
        ):
            dr = rc - ref.r
            CFi += cf.reshape(-1, 3).sum(axis=0)
            CMi += np.cross(dr.reshape(-1, 3), cf.reshape(-1, 3)).sum(axis=0)

        if symmetric[isurf]:
            # the mirror image doubles the symmetric components and cancels
            # the antisymmetric ones exactly
            CFi = np.array([2.0 * CFi[0], 0.0, 2.0 * CFi[2]])
            CMi = np.array([0.0, 2.0 * CMi[1], 0.0])

        CF += CFi
        CM += CMi

    CM = CM / ref.lengths * _CONVENTION
    return CF, CM


def body_forces(system: System, frame=None):
    """Total force and moment coefficients.

    Parameters
    ----------
    system : System
        Must hold panel properties from a near-field analysis.
    frame : Body, Stability or Wind
        Output frame.  Defaults to :class:`~flightlab.vlm.Body`.  Use
        :class:`~flightlab.vlm.Stability` to get ``(CD, CY, CL)`` and
        ``(Cl, Cm, Cn)``, which is what a textbook means by those symbols.

    Returns
    -------
    CF : ndarray, shape (3,)
        Force coefficients in the requested frame.
    CM : ndarray, shape (3,)
        Moment coefficients ``(Cl, Cm, Cn)``, normalized by ``(b, c, b)``.
    """
    if system.properties is None:
        raise RuntimeError("near field analysis required before body_forces")
    frame = Body() if frame is None else frame

    CF, CM = _integrate(
        system.surfaces, system.properties, system.reference, system.symmetric
    )
    return body_to_frame(CF, CM, system.reference, system.freestream, frame)


def body_forces_derivatives(system: System):
    """Body force coefficients in the body frame plus freestream derivatives.

    Returns
    -------
    CF, CM : ndarray, shape (3,)
        Body-frame coefficients.
    dCF, dCM : dict
        Derivatives keyed by ``'alpha'``, ``'beta'`` (per radian) and ``'p'``,
        ``'q'``, ``'r'`` (per rad/s).

    Notes
    -----
    Upstream VortexLattice.jl 0.2.3 applies the roll/yaw sign convention
    ``diag(-1, 1, -1)`` to the moment *derivatives* but not to ``CM`` itself in
    this function, so the ``CM`` it returns disagrees in sign with the one from
    :func:`body_forces`.  Both are then used in the same product rule inside
    :func:`~flightlab.vlm.stability_derivatives`, which rotates the moment vector
    into the stability frame as ``R @ dCM + R_a @ CM``.

    Either convention is fine used consistently -- ``diag(-1, 1, -1)`` commutes
    with a rotation about y, so applying it before or after the alpha rotation
    gives the same answer.  Mixing them does not.  This port applies it to both.

    Central-differencing :func:`body_forces` settles which is right without
    appealing to a convention.  At alpha = 4 deg, beta = 10 deg on a wing with
    dihedral and washout, this port matches that finite difference to better
    than 1e-4 relative, while the mixed-convention version puts ``Cl_alpha`` 3%
    off and ``Cn_alpha`` off by a factor of five.  ``Cm_alpha`` is unaffected
    either way, because ``R_a``'s middle row is zero -- which is why a symmetric
    aircraft at zero sideslip never shows the difference.  See
    ``tests/test_vlm_physics.py`` for the pinned comparison.
    """
    if system.dproperties is None:
        raise RuntimeError(
            "derivatives required -- run steady_analysis with derivatives=True"
        )
    ref, surfaces, sym = system.reference, system.surfaces, system.symmetric

    CF, CM = _integrate(surfaces, system.properties, ref, sym)
    dCF, dCM = {}, {}
    for name in DERIVS:
        dCF[name], dCM[name] = _integrate(
            surfaces, system.dproperties[name], ref, sym
        )
    return CF, CM, dCF, dCM


# --- per-station coefficients ----------------------------------------------


def lifting_line_coefficients(system: System, r=None, c=None, frame=None, xc=0.25):
    """Force and moment coefficients per unit span at each spanwise station.

    This is what a strip-integration drag buildup and a span-loading plot both
    consume, and it is the one output the rest of the course leans on hardest.

    Parameters
    ----------
    system : System
    r : sequence of ndarray, optional
        Lifting-line coordinates per surface, shape ``(3, ns+1)``.  Computed
        from ``system.grids`` at ``xc`` if omitted.
    c : sequence of ndarray, optional
        Chord at each lifting-line station, shape ``(ns+1,)``.
    frame : Body, Stability or Wind
        Output frame; defaults to :class:`~flightlab.vlm.Body`.  Pass
        :class:`~flightlab.vlm.Stability` to read ``cf[2]`` as the section lift
        coefficient and ``cf[0]`` as the section drag coefficient.
    xc : float
        Chordwise location of the lifting line if ``r`` and ``c`` are computed
        here.  Defaults to the quarter chord.

    Returns
    -------
    cf : list of ndarray
        One ``(3, ns)`` array per surface: ``(cx, cy, cz)`` per unit span,
        normalized by the local chord.  In the stability frame the third row is
        the section lift coefficient ``c_l``.
    cm : list of ndarray
        One ``(3, ns)`` array per surface, normalized by local chord squared.
    """
    if system.properties is None:
        raise RuntimeError(
            "near field analysis required before lifting_line_coefficients"
        )
    frame = Body() if frame is None else frame

    if r is None or c is None:
        from .geometry import lifting_line_geometry

        if not system.grids:
            raise RuntimeError(
                "no grids stored on the system; pass r and c explicitly"
            )
        r, c = lifting_line_geometry(system.grids, xc)

    ref, fs = system.reference, system.freestream
    cf_out, cm_out = [], []

    for isurf, surf in enumerate(system.surfaces):
        nc, ns = surf.shape
        p = system.properties[isurf]
        ri = np.asarray(r[isurf], dtype=float)
        ci = np.asarray(c[isurf], dtype=float)

        rl = ri[:, :-1].T  # (ns, 3)
        rr = ri[:, 1:].T
        ds = np.linalg.norm(rr - rl, axis=1)  # (ns,)
        rs = 0.5 * (rl + rr)  # (ns, 3) station reference point
        cs = 0.5 * (ci[:-1] + ci[1:])  # (ns,)

        cfj = np.zeros((ns, 3))
        cmj = np.zeros((ns, 3))
        for rc, cval in (
            (surf.rtc, p.cfb),
            (surf.left_center, p.cfl),
            (surf.right_center, p.cfr),
        ):
            cfj += cval.sum(axis=0)
            cmj += np.cross(rc - rs[None, :, :], cval).sum(axis=0)

        # renormalize from the reference area to the local strip area
        cfj *= (ref.S / (ds * cs))[:, None]
        cmj *= (ref.S / (ds * cs**2))[:, None]

        cfj, cmj = body_to_frame(cfj, cmj, ref, fs, frame, vector_axis=-1)
        cf_out.append(cfj.T.copy())
        cm_out.append(cmj.T.copy())

    return cf_out, cm_out


# --- frame transforms -------------------------------------------------------


def body_to_frame(CF, CM, ref: Reference, fs: Freestream, frame, vector_axis=None):
    """Rotate force and moment coefficients from the body frame into ``frame``.

    Parameters
    ----------
    CF, CM : ndarray
        Shape ``(3,)``, or ``(..., 3)`` when ``vector_axis=-1``.
    frame : Body, Stability or Wind
    vector_axis : int, optional
        Pass ``-1`` for stacked vectors.

    Returns
    -------
    CF, CM : ndarray
        Same shape as the inputs.
    """
    CF = np.asarray(CF, dtype=float)
    CM = np.asarray(CM, dtype=float)

    if isinstance(frame, Body):
        return CF, CM

    if isinstance(frame, Stability):
        R = body_to_stability(fs)
        return _rot(R, CF, vector_axis), _rot(R, CM, vector_axis)

    if isinstance(frame, Wind):
        # moments must be rotated with their reference lengths restored
        lengths = ref.lengths
        R = body_to_wind(fs)
        CMd = CM * lengths
        return _rot(R, CF, vector_axis), _rot(R, CMd, vector_axis) / lengths

    raise TypeError(f"unknown frame: {frame!r}")


def _rot(R, V, vector_axis):
    if vector_axis is None:
        return R @ V
    return V @ R.T
