"""Geometry construction: spacing schemes, grids, and panel generation.

Ported from VortexLattice.jl (``src/geometry.jl``).

The workflow is always the same two steps:

1. ``grid, ratios = wing_to_grid(...)`` turns *design parameters* -- leading
   edge positions, chords, twist, dihedral -- into a grid of panel corners.
2. ``grid, ratios, surface = grid_to_surface_panels(grid, ratios=ratios)``
   turns that grid into vortex-ring panels.

``steady_analysis`` will do step 2 for you if you hand it grids, so most of the
time you only call ``wing_to_grid``.

Angle convention
----------------
``theta`` (twist) and ``phi`` (dihedral) are in **radians** here, matching the
Julia package.  Prefer :func:`flightlab.vlm.wing` in the top-level API, which takes
degrees.
"""

from __future__ import annotations

import numpy as np

from .panel import Surface

__all__ = [
    "AbstractSpacing",
    "Uniform",
    "Sine",
    "Cosine",
    "spanwise_spacing",
    "chordwise_spacing",
    "interpolate_grid",
    "wing_to_grid",
    "grid_to_surface_panels",
    "lifting_line_geometry",
    "translate",
    "rotate",
]


# --- spacing schemes --------------------------------------------------------


class AbstractSpacing:
    """Base class for panel spacing schemes."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self).__name__)


class Uniform(AbstractSpacing):
    """Equal-width panels."""

    __slots__ = ()


class Sine(AbstractSpacing):
    """Sine spacing -- panels bunched toward the *outboard* end.

    Applied to the right half of a wing that is then mirrored, this gives
    cosine spacing across the full span.
    """

    __slots__ = ()


class Cosine(AbstractSpacing):
    """Cosine spacing -- panels bunched toward *both* ends.

    Usually the most accurate spanwise scheme, because induced drag is set by
    the loading near the tip where the gradient is steepest.
    """

    __slots__ = ()


def _linearinterp(eta, rstart, rend):
    return (1.0 - eta) * rstart + eta * rend


def spanwise_spacing(n: int, spacing: AbstractSpacing):
    """Distribute ``n`` panel edges and ``n-1`` midpoints on ``[0, 1]``.

    Returns
    -------
    eta : ndarray, shape (n,)
        Panel edge locations.
    eta_mid : ndarray, shape (n-1,)
        Midpoint locations.  Note that for :class:`Sine` and :class:`Cosine`
        these are the *parameter* midpoints mapped through the spacing
        function, not the arithmetic midpoints of ``eta``.  Placing control
        points this way improves accuracy substantially.
    """
    if isinstance(spacing, Uniform):
        eta = np.linspace(0.0, 1.0, n)
        eta_mid = _linearinterp(0.5, eta[:-1], eta[1:])
    elif isinstance(spacing, Sine):
        theta = np.linspace(0.0, np.pi / 2, n)
        eta = np.sin(theta)
        theta_mid = _linearinterp(0.5, theta[:-1], theta[1:])
        eta_mid = np.sin(theta_mid)
    elif isinstance(spacing, Cosine):
        theta = np.linspace(0.0, np.pi, n)
        eta = (1.0 - np.cos(theta)) / 2.0
        theta_mid = _linearinterp(0.5, theta[:-1], theta[1:])
        eta_mid = (1.0 - np.cos(theta_mid)) / 2.0
    else:
        raise TypeError(f"unknown spanwise spacing scheme: {spacing!r}")
    return eta, eta_mid


def chordwise_spacing(n: int, spacing: AbstractSpacing):
    """Distribute ``n`` panel edges plus quarter- and three-quarter-chord points.

    Returns
    -------
    eta : ndarray, shape (n,)
        Panel edge locations.
    eta_qtr : ndarray, shape (n-1,)
        Quarter-chord (bound vortex) location within each panel.
    eta_thrqtr : ndarray, shape (n-1,)
        Three-quarter-chord (control point) location within each panel.
    """
    if isinstance(spacing, Uniform):
        eta = np.linspace(0.0, 1.0, n)
    elif isinstance(spacing, Sine):
        eta = np.sin(np.linspace(0.0, np.pi / 2, n))
    elif isinstance(spacing, Cosine):
        eta = (1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0
    else:
        raise TypeError(f"unknown chordwise spacing scheme: {spacing!r}")

    eta_qtr = _linearinterp(0.25, eta[:-1], eta[1:])
    eta_thrqtr = _linearinterp(0.75, eta[:-1], eta[1:])
    return eta, eta_qtr, eta_thrqtr


# --- grid interpolation -----------------------------------------------------


def interpolate_grid(xyz, eta, ydir=1):
    """Re-interpolate a grid along one direction using normalized arc length.

    Parameters
    ----------
    xyz : ndarray, shape (3, ni, nj)
        Grid of points.  ``ni`` runs chordwise, ``nj`` spanwise.
    eta : array_like, shape (m,)
        New normalized coordinates, ``0 <= eta <= 1``.
    ydir : {1, 2}
        Direction to interpolate along: ``1`` for the chordwise (``i``) axis,
        ``2`` for the spanwise (``j``) axis.

    Returns
    -------
    ndarray
        Shape ``(3, m, nj)`` if ``ydir == 1``, else ``(3, ni, m)``.

    Notes
    -----
    Interpolation is piecewise linear in normalized arc length along the
    chosen direction.

    Upstream VortexLattice.jl 0.2.3 computes ``x`` and ``z`` for every slice but
    reuses the ``y`` interpolation from the *first* slice when ``ydir == 2``.
    That discards the chordwise variation in ``y`` which dihedral produces once
    the section is also twisted -- ``y = cos(phi)*y_le - sin(phi)*z``, and ``z``
    varies along a twisted chord -- so every chordwise row is placed at the
    leading edge's ``y`` and the panels are slightly sheared.  This port
    interpolates all three components of every slice.

    The difference is exactly zero unless dihedral **and** twist are both
    nonzero.  Where it does appear it is small: on a 15 m wing with 6 degrees of
    dihedral and 3 degrees of washout the offset grows from zero at the leading
    edge to about 4.4 mm at the trailing edge, moving ``CL`` by 0.36% and ``Cm``
    by 2%.
    """
    xyz = np.asarray(xyz, dtype=float)
    eta = np.asarray(eta, dtype=float).ravel()

    if ydir == 1:
        # interpolate along axis 1, loop over axis 2
        n_other = xyz.shape[2]
        out = np.empty((3, eta.size, n_other))
        for k in range(n_other):
            line = xyz[:, :, k]  # (3, ni)
            out[:, :, k] = _interp_line(line, eta)
    elif ydir == 2:
        n_other = xyz.shape[1]
        out = np.empty((3, n_other, eta.size))
        for k in range(n_other):
            line = xyz[:, k, :]  # (3, nj)
            out[:, k, :] = _interp_line(line, eta)
    else:
        raise ValueError("ydir must be 1 or 2")

    return out


def _interp_line(line, eta):
    """Linearly interpolate a (3, n) polyline at normalized arc lengths ``eta``."""
    ds = np.zeros(line.shape[1])
    ds[1:] = np.linalg.norm(np.diff(line, axis=1), axis=0)
    t = np.cumsum(ds)
    if t[-1] <= 0.0:
        # degenerate (all points coincident); return the single point
        return np.repeat(line[:, :1], eta.size, axis=1)
    t = t / t[-1]
    return np.stack([np.interp(eta, t, line[i, :]) for i in range(3)], axis=0)


# --- wing construction ------------------------------------------------------


def _default_fcore(c, ds):
    return 1e-3


def wing_to_grid(
    xle,
    yle,
    zle,
    chord,
    theta,
    phi,
    ns,
    nc,
    fc=None,
    reference_line=None,
    mirror=False,
    spacing_s=None,
    spacing_c=None,
):
    """Build a panel-corner grid for a wing from its design parameters.

    Parameters
    ----------
    xle, yle, zle : array_like, shape (n,)
        Leading-edge coordinates of each defining section, m.
    chord : array_like, shape (n,)
        Chord length of each section, m.
    theta : array_like, shape (n,)
        Twist of each section, **radians**, positive nose up.
    phi : array_like, shape (n,)
        Dihedral of each section, **radians**, a right-handed rotation about x.
    ns, nc : int
        Number of spanwise and chordwise panels.
    fc : sequence of callables, optional
        Camber line ``z/c = f(x/c)`` for each section.  Defaults to flat plates.
        A vortex lattice sees camber only through the panel normal vectors, so
        this is the only place airfoil shape enters an inviscid solve.
    reference_line : array_like, shape (n, 2), optional
        ``(x/c, z/c)`` of the point that ``xle, yle, zle`` actually locate, so
        you can define a wing about its quarter chord instead of its leading
        edge.  Defaults to the leading edge.
    mirror : bool
        Mirror the geometry across the X-Z plane.  Use this *or* the
        ``symmetric`` flag in :func:`steady_analysis`, not both.
    spacing_s, spacing_c : AbstractSpacing
        Spanwise and chordwise spacing.  Default ``Cosine()`` spanwise and
        ``Uniform()`` chordwise.

    Returns
    -------
    grid : ndarray, shape (3, nc+1, ns_total+1)
        Panel corner coordinates, where ``ns_total`` is ``2*ns`` if mirrored.
    ratios : ndarray, shape (2, nc, ns_total)
        Where to place the bound-vortex centre and control point within each
        panel when the grid is converted to panels.  Pass this straight to
        :func:`grid_to_surface_panels`.
    """
    spacing_s = Cosine() if spacing_s is None else spacing_s
    spacing_c = Uniform() if spacing_c is None else spacing_c

    xle = np.asarray(xle, dtype=float).ravel()
    yle = np.asarray(yle, dtype=float).ravel()
    zle = np.asarray(zle, dtype=float).ravel()
    chord = np.asarray(chord, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()

    n = xle.size
    if not (
        n == yle.size == zle.size == chord.size == theta.size == phi.size
    ):
        raise ValueError(
            "xle, yle, zle, chord, theta and phi must all have the same length"
        )

    if fc is None:
        fc = [(lambda x: 0.0 * np.asarray(x, dtype=float))] * n
    if len(fc) != n:
        raise ValueError("fc must have one camber function per section")

    if reference_line is None:
        reference_line = np.zeros((n, 2))
    reference_line = np.asarray(reference_line, dtype=float).reshape(n, 2)

    etas, etabar = spanwise_spacing(ns + 1, spacing_s)
    etac, eta_qtr, eta_thrqtr = chordwise_spacing(nc + 1, spacing_c)

    # the bound-vortex grid also carries the leading and trailing edges
    eta_bound = np.concatenate(([0.0], eta_qtr, [1.0]))

    xyz_edge = np.empty((3, nc + 1, n))
    xyz_bound = np.empty((3, nc + 2, n))
    xyz_cp = np.empty((3, nc, n))

    for j in range(n):
        st, ct = np.sin(theta[j]), np.cos(theta[j])
        Rt = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        sp, cp = np.sin(phi[j]), np.cos(phi[j])
        Rp = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])

        ref_offset = (
            np.array([reference_line[j, 0], 0.0, reference_line[j, 1]]) * chord[j]
        )
        rle = np.array([xle[j], yle[j], zle[j]]) - Rt @ ref_offset

        for target, etas_c in (
            (xyz_edge, etac),
            (xyz_bound, eta_bound),
            (xyz_cp, eta_thrqtr),
        ):
            xc = np.asarray(etas_c, dtype=float)
            zc = np.asarray(fc[j](xc), dtype=float) * np.ones_like(xc)
            r = np.stack([xc, np.zeros_like(xc), zc], axis=0)  # (3, m)
            r = chord[j] * r
            r = Rt @ r
            r = r + rle[:, None]
            r = Rp @ r
            target[:, :, j] = r

    # interpolate to the requested spanwise spacing
    xyz_panels = interpolate_grid(xyz_edge, etas, ydir=2)
    ratios = _ratios_from_grids(
        xyz_edge, xyz_bound, xyz_cp, etas, etabar, nc, ns
    )

    if mirror:
        xyz_panels, ratios = _mirror_grid(xyz_panels, ratios, yle)

    return xyz_panels, ratios


def _ratios_from_grids(xyz_edge, xyz_bound, xyz_cp, etas, etabar, nc, ns):
    """Compute bound-vortex-centre and control-point placement ratios."""
    xyz_panels = interpolate_grid(xyz_edge, etas, ydir=2)
    xyz_corner = interpolate_grid(xyz_bound, etas, ydir=2)
    xyz_center = interpolate_grid(xyz_bound, etabar, ydir=2)
    xyz_cp_i = interpolate_grid(xyz_cp, etabar, ydir=2)

    ratios = np.zeros((2, nc, ns))
    for j in range(ns):
        for i in range(nc):
            rtl = xyz_corner[:, i + 1, j]
            rtc = xyz_center[:, i + 1, j]
            rtr = xyz_corner[:, i + 1, j + 1]
            rcp = xyz_cp_i[:, i, j]

            r1 = xyz_panels[:, i, j]
            r2 = xyz_panels[:, i, j + 1]
            r3 = xyz_panels[:, i + 1, j]
            r4 = xyz_panels[:, i + 1, j + 1]

            den = np.hypot(rtr[1] - rtl[1], rtr[2] - rtl[2])
            ratios[0, i, j] = (
                np.hypot(rtc[1] - rtl[1], rtc[2] - rtl[2]) / den if den > 0 else 0.5
            )

            rtop = _linearinterp(ratios[0, i, j], r1, r2)
            rbot = _linearinterp(ratios[0, i, j], r3, r4)
            den2 = np.hypot(rbot[0] - rtop[0], rbot[2] - rtop[2])
            ratios[1, i, j] = (
                np.hypot(rcp[0] - rtop[0], rcp[2] - rtop[2]) / den2
                if den2 > 0
                else 0.75
            )
    return ratios


def _mirror_grid(xyz_panels, ratios, yle):
    """Mirror a half-span grid and its ratios across the X-Z plane."""
    right_side = np.sum(yle) > 0

    mirrored = xyz_panels[:, :, ::-1].copy()
    mirrored[1, :, :] *= -1.0

    # the spanwise placement ratio measures from the left edge, which becomes
    # the right edge under reflection
    ratios_ref = ratios[:, :, ::-1].copy()
    ratios_ref[0, :, :] = 1.0 - ratios_ref[0, :, :]

    if right_side:
        grid = np.concatenate([mirrored, xyz_panels[:, :, 1:]], axis=2)
        ratios_out = np.concatenate([ratios_ref, ratios], axis=2)
    else:
        grid = np.concatenate([xyz_panels, mirrored[:, :, 1:]], axis=2)
        ratios_out = np.concatenate([ratios, ratios_ref], axis=2)

    return grid, ratios_out


def grid_to_surface_panels(
    xyz,
    ns=None,
    nc=None,
    ratios=None,
    mirror=False,
    fcore=None,
    spacing_s=None,
    spacing_c=None,
):
    """Convert a grid of panel corners into vortex-ring panels.

    Two modes:

    * ``grid_to_surface_panels(xyz, ratios=ratios)`` uses the grid as given.
    * ``grid_to_surface_panels(xyz, ns, nc)`` re-discretizes the grid into
      ``ns`` spanwise and ``nc`` chordwise panels first.

    The bound vortex goes at the 1/4 chord of each panel and the control point
    at the 3/4 chord, which is what makes a flat plate return ``2*pi`` per
    radian.

    Parameters
    ----------
    xyz : ndarray, shape (3, ni, nj)
        Panel corner grid; ``ni`` chordwise from leading to trailing edge,
        ``nj`` spanwise from left to right.
    ns, nc : int, optional
        Target panel counts.  Given together, the grid is re-interpolated.
    ratios : ndarray, shape (2, nc, ns), optional
        Bound-vortex-centre and control-point placement, from
        :func:`wing_to_grid`.  Defaults to mid-panel and 3/4 chord.
    mirror : bool
        Mirror across the X-Z plane.
    fcore : callable, optional
        ``fcore(chord, dspan) -> core_size``.  Defaults to ``1e-3`` m.

    Returns
    -------
    grid : ndarray
        The grid actually used (mirrored and/or re-interpolated).
    ratios : ndarray
        The ratios actually used.
    surface : Surface
        The generated panels.
    """
    xyz = np.asarray(xyz, dtype=float)
    fcore = _default_fcore if fcore is None else fcore

    if (ns is None) != (nc is None):
        raise ValueError("give both ns and nc, or neither")

    if ns is not None:
        spacing_s = Cosine() if spacing_s is None else spacing_s
        spacing_c = Uniform() if spacing_c is None else spacing_c
        etas, etabar = spanwise_spacing(ns + 1, spacing_s)
        etac, eta_qtr, eta_thrqtr = chordwise_spacing(nc + 1, spacing_c)
        eta_bound = np.concatenate(([0.0], eta_qtr, [1.0]))

        xyz_edge = interpolate_grid(xyz, etac, ydir=1)
        xyz_bound = interpolate_grid(xyz, eta_bound, ydir=1)
        xyz_cp_c = interpolate_grid(xyz, eta_thrqtr, ydir=1)

        ratios = _ratios_from_grids(
            xyz_edge, xyz_bound, xyz_cp_c, etas, etabar, nc, ns
        )
        xyz = interpolate_grid(xyz_edge, etas, ydir=2)

    nc_g = xyz.shape[1] - 1
    ns_g = xyz.shape[2] - 1
    if ratios is None:
        ratios = np.zeros((2, nc_g, ns_g))
        ratios[0] = 0.5
        ratios[1] = 0.75
    ratios = np.asarray(ratios, dtype=float)

    if mirror:
        xyz, ratios = _mirror_grid(xyz, ratios, xyz[1, :, :].ravel())

    surface = _panels_from_grid(xyz, ratios, fcore)
    return xyz, ratios, surface


def _panels_from_grid(grid, ratios, fcore):
    """Build a :class:`Surface` from a corner grid and placement ratios.

    This is the port of ``update_surface_panels!``.  The bound vortex sits 25%
    of the way from each panel's leading-edge corners to its trailing-edge
    corners, except on the trailing edge where the aft leg sits on the grid
    trailing edge so that the wake leaves the geometry cleanly.
    """
    grid = np.asarray(grid, dtype=float)
    nc = grid.shape[1] - 1
    ns = grid.shape[2] - 1

    # corner arrays, shape (nc, ns, 3)
    P = np.moveaxis(grid, 0, -1)  # (ni, nj, 3)
    r1 = P[:-1, :-1, :]  # top left
    r2 = P[:-1, 1:, :]  # top right
    r3 = P[1:, :-1, :]  # bottom left
    r4 = P[1:, 1:, :]  # bottom right

    rtl = _linearinterp(0.25, r1, r3)
    rtr = _linearinterp(0.25, r2, r4)

    # the aft leg of panel i is the forward leg of panel i+1, except at the
    # trailing edge where it lies on the grid trailing edge
    rbl = np.empty_like(rtl)
    rbr = np.empty_like(rtr)
    rbl[:-1] = rtl[1:]
    rbr[:-1] = rtr[1:]
    rbl[-1] = r3[-1]
    rbr[-1] = r4[-1]

    fy = ratios[0][..., None]  # (nc, ns, 1)
    fz = ratios[1][..., None]

    rtc = _linearinterp(fy, rtl, rtr)
    rbc = _linearinterp(fy, rbl, rbr)

    rtop = _linearinterp(fy, r1, r2)
    rbot = _linearinterp(fy, r3, r4)
    rcp = _linearinterp(fz, rtop, rbot)

    ncp = np.cross(rcp - rtr, rcp - rtl)
    ncp = ncp / np.linalg.norm(ncp, axis=-1, keepdims=True)

    chord = np.linalg.norm(0.5 * (r1 + r2) - 0.5 * (r3 + r4), axis=-1)

    # finite core size, from the section chord and the panel width
    cl = np.linalg.norm(P[-1, :-1, :] - P[0, :-1, :], axis=-1)
    cr = np.linalg.norm(P[-1, 1:, :] - P[0, 1:, :], axis=-1)
    c_sec = 0.5 * (cl + cr)  # (ns,)
    dspan = np.hypot(rtr[..., 1] - rtl[..., 1], rtr[..., 2] - rtl[..., 2])
    core_size = np.empty((nc, ns))
    for j in range(ns):
        for i in range(nc):
            core_size[i, j] = fcore(c_sec[j], dspan[i, j])

    return Surface(
        rtl=rtl.copy(),
        rtc=rtc,
        rtr=rtr.copy(),
        rbl=rbl,
        rbc=rbc,
        rbr=rbr,
        rcp=rcp,
        ncp=ncp,
        core_size=core_size,
        chord=chord,
    )


# --- lifting line -----------------------------------------------------------


def lifting_line_geometry(grids, xc=0.25):
    """Build a lifting-line representation of one or more grids.

    Parameters
    ----------
    grids : sequence of ndarray
        One grid of shape ``(3, nc+1, ns+1)`` per surface.
    xc : float
        Normalized chordwise location of the line, measured from the leading
        edge.  Defaults to the quarter chord.

    Returns
    -------
    r : list of ndarray
        One ``(3, ns+1)`` array per surface: the line's coordinates at each
        spanwise station boundary.
    c : list of ndarray
        One ``(ns+1,)`` array per surface: the chord length at each station.
    """
    if isinstance(grids, np.ndarray) and grids.ndim == 3:
        grids = [grids]

    r_out, c_out = [], []
    for grid in grids:
        grid = np.asarray(grid, dtype=float)
        le = grid[:, 0, :]  # (3, ns+1)
        te = grid[:, -1, :]
        r_out.append(_linearinterp(xc, le, te))
        c_out.append(np.linalg.norm(le - te, axis=0))
    return r_out, c_out


# --- transformations --------------------------------------------------------


def translate(grid, r):
    """Return a copy of ``grid`` translated by the vector ``r`` (m)."""
    grid = np.asarray(grid, dtype=float)
    r = np.asarray(r, dtype=float).reshape(3)
    return grid + r[:, None, None]


def rotate(grid, R, r=(0.0, 0.0, 0.0)):
    """Return a copy of ``grid`` rotated by matrix ``R`` about the point ``r``."""
    grid = np.asarray(grid, dtype=float)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    r = np.asarray(r, dtype=float).reshape(3)
    shifted = grid - r[:, None, None]
    out = np.einsum("ij,jkl->ikl", R, shifted)
    return out + r[:, None, None]
