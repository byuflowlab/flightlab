"""Vortex filament induced velocities and influence-coefficient assembly.

Ported from VortexLattice.jl (``src/induced.jl``).

This is the part of the method nobody learns from writing, which is why it is
provided.  What matters for using it correctly:

* Each panel carries a **vortex ring**: a closed loop running
  ``rtl -> rtr -> rbr -> rbl -> rtl``.  On the trailing edge the aft leg is
  replaced by two filaments trailing to infinity along ``xhat``, which turns
  the ring into a horseshoe.
* ``symmetric=True`` adds the influence of a mirror image across the X-Z plane
  without adding panels.  Use it *or* a mirrored geometry, never both.
* The **finite core** model desingularizes a filament that passes very close to
  a control point.  It is switched on automatically between surfaces with
  different ``surface_id`` and off within a surface, because a panel's own
  shared edges are supposed to cancel exactly.

Upstream reuses shared edges between neighbouring panels as an optimization.
This port evaluates each ring independently, which is algebraically identical
(a shared edge appears twice with opposite sense) and lets numpy vectorize the
whole assembly.
"""

from __future__ import annotations

import numpy as np

from .panel import Surface, flipy

__all__ = [
    "bound_influence",
    "trailing_influence",
    "influence_coefficients",
    "induced_velocity",
]

_FOUR_PI = 4.0 * np.pi

#: Chunk size target in bytes for the (chunk, N, 3) temporaries.  Keeps memory
#: bounded for large panel counts without hurting speed at course-sized cases.
_CHUNK_BYTES = 32 << 20


def _chunk_rows(n_send: int) -> int:
    """Number of receiving points to process at once."""
    per_row = max(1, n_send * 3 * 8 * 8)  # ~8 live temporaries of (N,3)
    return max(1, min(4096, _CHUNK_BYTES // per_row))


def bound_influence(r1, r2, finite_core=False, core_size=0.0):
    """Velocity per unit circulation induced by a straight bound filament.

    Parameters
    ----------
    r1, r2 : ndarray, shape (..., 3)
        Position of the evaluation point relative to the start and end of the
        filament, m.
    finite_core : bool
        Enable the finite-core desingularization.
    core_size : float or ndarray
        Core radius, m.  Broadcasts against the leading dimensions of ``r1``.

    Returns
    -------
    ndarray, shape (..., 3)

    Notes
    -----
    A filament induces no velocity on a point lying on its own axis; that case
    is returned as exactly zero rather than as a division by zero.
    """
    nr1 = np.linalg.norm(r1, axis=-1)
    nr2 = np.linalg.norm(r2, axis=-1)
    rdot = np.einsum("...i,...i->...", r1, r2)
    cr = np.cross(r1, r2)

    if finite_core:
        eps2 = np.asarray(core_size) ** 2
        r1s, r2s = nr1**2, nr2**2
        den1 = r1s * r2s - rdot * rdot + eps2 * (r1s + r2s - 2.0 * nr1 * nr2)
        f2 = (r1s - rdot) / np.sqrt(r1s + eps2) + (r2s - rdot) / np.sqrt(r2s + eps2)
    else:
        den1 = nr1 * nr2 + rdot
        f2 = _safe_recip(nr1) + _safe_recip(nr2)

    scale = _safe_div(f2, den1)
    return cr * scale[..., None] / _FOUR_PI


def trailing_influence(r, xhat, finite_core=False, core_size=0.0):
    """Velocity per unit circulation induced by a semi-infinite trailing filament.

    Parameters
    ----------
    r : ndarray, shape (..., 3)
        Position of the evaluation point relative to the filament's origin, m.
    xhat : ndarray, shape (3,)
        Unit vector along which the filament trails downstream.
    finite_core : bool
        Enable the finite-core desingularization.
    core_size : float or ndarray
        Core radius, m.

    Returns
    -------
    ndarray, shape (..., 3)
    """
    nr = np.linalg.norm(r, axis=-1)
    rdot = np.einsum("...i,i->...", r, xhat)
    cr = np.cross(r, xhat)

    if finite_core:
        eps2 = np.asarray(core_size) ** 2
        tmp = _safe_div(eps2, nr + rdot)
        den = nr * (nr - rdot + tmp)
    else:
        den = nr * (nr - rdot)

    return -cr * _safe_div(1.0, den)[..., None] / _FOUR_PI


def _safe_div(num, den):
    """``num/den``, returning 0 where ``den`` vanishes exactly.

    A point lying exactly on a filament's axis gives ``0/0``.  Physically that
    filament induces nothing there, and in the one case where the point is on
    the filament *segment* the caller has already been told to skip it, so
    returning zero is both safe and correct.
    """
    den = np.asarray(den, dtype=float)
    bad = den == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, np.where(bad, 1.0, den))
    return np.where(bad, 0.0, out)


def _safe_recip(x):
    return _safe_div(1.0, x)


# --- ring influence ---------------------------------------------------------


def _ring_block(
    P,
    send: Surface,
    finite_core: bool,
    symmetric: bool,
    trailing_vortices: bool,
    xhat,
    skip_top=None,
    skip_bottom=None,
    on_symmetry_plane=None,
):
    """Influence of every panel in ``send`` at each point in ``P``.

    Parameters
    ----------
    P : ndarray, shape (m, 3)
        Evaluation points.
    send : Surface
        Sending surface with ``N = nc*ns`` panels.
    finite_core, symmetric, trailing_vortices : bool
        Model flags.
    xhat : ndarray, shape (3,)
        Trailing vortex direction.
    skip_top, skip_bottom : ndarray of int, shape (m,), optional
        For each evaluation point, the flattened index of a sending panel whose
        top (resp. bottom) filament passes through that point and must be
        omitted.  Use ``-1`` for "nothing to skip".
    on_symmetry_plane : ndarray of bool, shape (m,), optional
        True where the evaluation point lies on ``y = 0``.  A skipped
        filament's mirror image is retained unless the point is on the plane,
        where the image is the same filament.

    Returns
    -------
    ndarray, shape (m, N, 3)
        Velocity per unit circulation of each sending panel.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    xhat = np.asarray(xhat, dtype=float).reshape(3)
    nc, ns = send.shape
    N = nc * ns

    r11 = send.flat("rtl")
    r12 = send.flat("rtr")
    r21 = send.flat("rbl")
    r22 = send.flat("rbr")
    cs = send.flat("core_size") if finite_core else 0.0

    # panels on the trailing edge shed the trailing filaments
    te = np.zeros(N, dtype=bool)
    te[(nc - 1) * ns :] = True
    use_trail = te & trailing_vortices
    use_bottom = ~use_trail

    V = np.zeros((P.shape[0], N, 3))

    def edges(a1, a2, a3, a4, reflected):
        """Accumulate one image's worth of ring filaments into ``V``."""
        # top edge (a1 -> a2 forward, a2 -> a1 for the mirror image)
        vt = (
            bound_influence(a2, a1, finite_core, cs)
            if reflected
            else bound_influence(a1, a2, finite_core, cs)
        )
        if skip_top is not None:
            _zero_skipped(vt, skip_top, reflected, on_symmetry_plane)
        V[...] += vt
        del vt

        # right edge
        V[...] += (
            bound_influence(a3, a2, finite_core, cs)
            if reflected
            else bound_influence(a2, a3, finite_core, cs)
        )

        # left edge
        V[...] += (
            bound_influence(a1, a4, finite_core, cs)
            if reflected
            else bound_influence(a4, a1, finite_core, cs)
        )

        # bottom edge, except where trailing filaments replace it
        vb = (
            bound_influence(a4, a3, finite_core, cs)
            if reflected
            else bound_influence(a3, a4, finite_core, cs)
        )
        vb[:, use_trail, :] = 0.0
        if skip_bottom is not None:
            _zero_skipped(vb, skip_bottom, reflected, on_symmetry_plane)
        V[...] += vb
        del vb

        # trailing filaments; the mirror image trails with opposite sense
        if trailing_vortices:
            sgn = -1.0 if reflected else 1.0
            vr = trailing_influence(a3, xhat, finite_core, cs)
            vl = trailing_influence(a4, xhat, finite_core, cs)
            vtr = sgn * (vr - vl)
            vtr[:, ~use_trail, :] = 0.0
            V[...] += vtr

    a1 = P[:, None, :] - r11[None, :, :]
    a2 = P[:, None, :] - r12[None, :, :]
    a3 = P[:, None, :] - r22[None, :, :]
    a4 = P[:, None, :] - r21[None, :, :]
    edges(a1, a2, a3, a4, reflected=False)
    del a1, a2, a3, a4

    if symmetric:
        a1 = P[:, None, :] - flipy(r11)[None, :, :]
        a2 = P[:, None, :] - flipy(r12)[None, :, :]
        a3 = P[:, None, :] - flipy(r22)[None, :, :]
        a4 = P[:, None, :] - flipy(r21)[None, :, :]
        edges(a1, a2, a3, a4, reflected=True)

    return V


def _zero_skipped(v, skip, reflected, on_symmetry_plane):
    """Zero the entries of ``v`` named by ``skip`` (an index per row)."""
    rows = np.nonzero(skip >= 0)[0]
    if rows.size == 0:
        return
    if reflected:
        # the mirror image of a skipped filament is only coincident with the
        # evaluation point when that point lies on the symmetry plane
        if on_symmetry_plane is None:
            return
        rows = rows[on_symmetry_plane[rows]]
        if rows.size == 0:
            return
    v[rows, skip[rows], :] = 0.0


# --- assembly ---------------------------------------------------------------


def influence_coefficients(
    surfaces,
    symmetric,
    surface_id,
    trailing_vortices,
    xhat=(1.0, 0.0, 0.0),
):
    """Assemble the aerodynamic influence coefficient matrix.

    ``AIC[i, j]`` is the normal velocity induced at control point ``i`` by unit
    circulation on panel ``j``.  Panels are numbered surface by surface, and
    within a surface in C order over ``(nc, ns)``.

    Parameters
    ----------
    surfaces : sequence of Surface
    symmetric : sequence of bool
        Per surface: mirror its panels across the X-Z plane when computing
        induced velocities.
    surface_id : sequence of int
        Surfaces sharing an ID do not use the finite-core model on each other.
    trailing_vortices : sequence of bool
    xhat : array_like, shape (3,)

    Returns
    -------
    ndarray, shape (N, N)
    """
    xhat = np.asarray(xhat, dtype=float).reshape(3)
    sizes = [s.size for s in surfaces]
    offs = np.concatenate(([0], np.cumsum(sizes)))
    N = int(offs[-1])
    AIC = np.zeros((N, N))

    for i, recv in enumerate(surfaces):
        Pcp = recv.flat("rcp")
        nhat = recv.flat("ncp")
        for j, send in enumerate(surfaces):
            fc = surface_id[i] != surface_id[j]
            block = np.zeros((sizes[i], sizes[j]))
            step = _chunk_rows(sizes[j])
            for a in range(0, sizes[i], step):
                b = min(a + step, sizes[i])
                Vb = _ring_block(
                    Pcp[a:b],
                    send,
                    fc,
                    bool(symmetric[j]),
                    bool(trailing_vortices[j]),
                    xhat,
                )
                block[a:b] = np.einsum("mnk,mk->mn", Vb, nhat[a:b])
            AIC[offs[i] : offs[i + 1], offs[j] : offs[j + 1]] = block

    return AIC


def induced_velocity(
    P,
    surfaces,
    gamma,
    symmetric,
    surface_id,
    trailing_vortices,
    xhat=(1.0, 0.0, 0.0),
    receiving_surface=None,
    skip_top=None,
    skip_bottom=None,
):
    """Velocity induced at points ``P`` by the circulation on ``surfaces``.

    Parameters
    ----------
    P : ndarray, shape (m, 3)
        Evaluation points, m.
    surfaces : sequence of Surface
    gamma : ndarray, shape (N,) or (k, N)
        Circulation on every panel.  A 2-D array evaluates several circulation
        distributions at once (used for the stability derivatives, which share
        the same geometry).
    symmetric, surface_id, trailing_vortices : sequence
        As in :func:`influence_coefficients`.
    xhat : array_like, shape (3,)
    receiving_surface : int, optional
        Index of the surface the points belong to.  Only that surface applies
        the ``skip_*`` arguments.
    skip_top, skip_bottom : ndarray of int, shape (m,), optional
        Filaments coincident with the evaluation points; see :func:`_ring_block`.

    Returns
    -------
    ndarray
        Shape ``(m, 3)`` for 1-D ``gamma``, else ``(k, m, 3)``.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    xhat = np.asarray(xhat, dtype=float).reshape(3)
    gamma = np.asarray(gamma, dtype=float)
    squeeze = gamma.ndim == 1
    G = gamma[None, :] if squeeze else gamma

    sizes = [s.size for s in surfaces]
    offs = np.concatenate(([0], np.cumsum(sizes)))

    on_sym = np.abs(P[:, 1]) <= 1e-12 * max(
        1.0, float(np.abs(P[:, 1]).max(initial=0.0))
    )

    out = np.zeros((G.shape[0], P.shape[0], 3))

    for j, send in enumerate(surfaces):
        own = receiving_surface is not None and j == receiving_surface
        fc = (
            surface_id[receiving_surface] != surface_id[j]
            if receiving_surface is not None
            else True
        )
        Gj = G[:, offs[j] : offs[j + 1]]
        step = _chunk_rows(sizes[j])
        for a in range(0, P.shape[0], step):
            b = min(a + step, P.shape[0])
            Vb = _ring_block(
                P[a:b],
                send,
                fc,
                bool(symmetric[j]),
                bool(trailing_vortices[j]),
                xhat,
                skip_top=skip_top[a:b] if (own and skip_top is not None) else None,
                skip_bottom=(
                    skip_bottom[a:b] if (own and skip_bottom is not None) else None
                ),
                on_symmetry_plane=on_sym[a:b],
            )
            out[:, a:b, :] += np.einsum("mnk,gn->gmk", Vb, Gj)

    return out[0] if squeeze else out
