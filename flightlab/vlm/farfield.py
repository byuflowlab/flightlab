"""Trefftz-plane (far-field) induced drag.

Ported from VortexLattice.jl (``src/farfield.jl``).

The trailing vorticity is projected onto a plane far downstream and the kinetic
energy left in the wake is evaluated there.  This is the induced drag you want:
it is much less sensitive to panel resolution than the near-field pressure drag,
which is why every span-efficiency calculation in the course uses it.

Even so, it is *not* insensitive.  ``e_inv`` from a vortex lattice depends on how
the wing was sectioned near the tip. Calibrate on a planform whose answer you
know before trusting one whose
answer you do not.
"""

from __future__ import annotations

import numpy as np

from .freestream import body_to_wind
from .panel import TrefftzPanels, flipy
from .system import RHO, System

__all__ = ["far_field_drag", "trefftz_panels"]


def trefftz_panels(system: System):
    """Project each surface's trailing edge into the Trefftz plane.

    Returns
    -------
    list of TrefftzPanels
        One entry per surface.
    """
    R = body_to_wind(system.freestream)
    offs = system.offsets
    out = []
    for isurf, surf in enumerate(system.surfaces):
        nc, ns = surf.shape
        g = system.gamma[offs[isurf] : offs[isurf + 1]].reshape(nc, ns)
        pts = []
        for arr in (surf.rbl, surf.rbc, surf.rbr):
            p = arr[-1] @ R.T  # (ns, 3) rotated into the wind frame
            p = p.copy()
            p[:, 0] = 0.0  # the Trefftz plane is normal to the freestream
            pts.append(p)
        out.append(
            TrefftzPanels(rl=pts[0], rc=pts[1], rr=pts[2], gamma=g[-1].copy())
        )
    return out


def _vortex_induced_drag(rj, gj, ri, gi, ni):
    """Drag on receiving panels from a line vortex at ``rj`` with strength ``gj``.

    Parameters
    ----------
    rj : ndarray, shape (ns, 3)
        Sending filament locations in the Trefftz plane.
    gj : ndarray, shape (ns,)
        Sending circulation (already signed for which edge this is).
    ri : ndarray, shape (nr, 3)
        Receiving panel centres.
    gi : ndarray, shape (nr,)
        Receiving circulation.
    ni : ndarray, shape (nr, 3)
        Receiving panel normal, magnitude included.

    Returns
    -------
    ndarray, shape (nr, ns)
    """
    rij = ri[:, None, :] - rj[None, :, :]
    y, z = rij[..., 1], rij[..., 2]
    den = 2.0 * np.pi * (y**2 + z**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(den == 0.0, 0.0, 1.0 / np.where(den == 0.0, 1.0, den))
    g = gj[None, :]
    # tangential velocity of a 2-D vortex, x-component is identically zero
    vy = -g * z * inv
    vz = g * y * inv
    Vn = -(vy * ni[:, None, 1] + vz * ni[:, None, 2])
    return 0.5 * RHO * gi[:, None] * Vn


def far_field_drag(system: System) -> float:
    """Induced drag coefficient from a Trefftz-plane analysis.

    Parameters
    ----------
    system : System
        Must hold a circulation solution.

    Returns
    -------
    float
        ``CDi``, non-dimensionalized by the reference dynamic pressure and area.

    Examples
    --------
    Inviscid span efficiency follows directly::

        CDi = far_field_drag(system)
        e_inv = CL**2 / (np.pi * AR * CDi)
    """
    if system.gamma is None:
        raise RuntimeError("no circulation solution; run steady_analysis first")

    panels = trefftz_panels(system)
    ref = system.reference
    nsurf = len(panels)

    total = 0.0
    for i in range(nsurf):
        recv = panels[i]
        ni = recv.normal
        for j in range(nsurf):
            send = panels[j]
            sym = bool(system.symmetric[j])

            Di = _vortex_induced_drag(send.rl, -send.gamma, recv.rc, recv.gamma, ni)
            Di += _vortex_induced_drag(send.rr, send.gamma, recv.rc, recv.gamma, ni)
            if sym:
                Di += _vortex_induced_drag(
                    flipy(send.rr), -send.gamma, recv.rc, recv.gamma, ni
                )
                Di += _vortex_induced_drag(
                    flipy(send.rl), send.gamma, recv.rc, recv.gamma, ni
                )

            Dsum = float(Di.sum())
            if sym:
                Dsum *= 2.0
            total += Dsum

    q = 0.5 * RHO * ref.V**2
    return total / (q * ref.S)
