"""Surface panels with attached vortex rings, and Trefftz-plane panels.

Ported from VortexLattice.jl (``src/panel.jl``).

Where the Julia package stores a matrix of ``SurfacePanel`` structs, this port
stores a :class:`Surface` holding parallel arrays of shape ``(nc, ns, 3)``.
The physics is identical; arrays let numpy do the influence-coefficient
assembly in one vectorized pass instead of a Python loop over panels.

Vortex ring layout for panel ``(i, j)``::

      rtl -------- rtc -------- rtr        <- top (bound) vortex, at 1/4 chord
       |                         |
       |          rcp            |         <- control point, at 3/4 chord
       |                         |
      rbl -------- rbc -------- rbr        <- bottom vortex

The ring circulation runs ``rtl -> rtr -> rbr -> rbl -> rtl``.  For a panel on
the trailing edge the bottom leg is replaced by two vortices trailing to
infinity in the ``xhat`` direction, which is what makes the ring a horseshoe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Surface", "flipy"]


def flipy(r):
    """Reflect vector(s) across the X-Z plane (negate the y-component)."""
    r = np.asarray(r, dtype=float)
    out = r.copy()
    out[..., 1] *= -1.0
    return out


@dataclass
class Surface:
    """A lifting surface discretized into a ``(nc, ns)`` grid of vortex rings.

    Attributes
    ----------
    rtl, rtc, rtr : ndarray, shape (nc, ns, 3)
        Left, centre and right points of the top (bound) vortex, m.
    rbl, rbc, rbr : ndarray, shape (nc, ns, 3)
        Left, centre and right points of the bottom vortex, m.
    rcp : ndarray, shape (nc, ns, 3)
        Control point where the flow-tangency condition is applied, m.
    ncp : ndarray, shape (nc, ns, 3)
        Unit normal vector at the control point.
    core_size : ndarray, shape (nc, ns)
        Finite-core smoothing radius, m.  Only used when the finite-core model
        is active (i.e. between surfaces with different ``surface_id``).
    chord : ndarray, shape (nc, ns)
        Panel chord length, m.

    Notes
    -----
    ``nc`` indexes chordwise from leading edge to trailing edge; ``ns`` indexes
    spanwise from left to right.
    """

    rtl: np.ndarray
    rtc: np.ndarray
    rtr: np.ndarray
    rbl: np.ndarray
    rbc: np.ndarray
    rbr: np.ndarray
    rcp: np.ndarray
    ncp: np.ndarray
    core_size: np.ndarray
    chord: np.ndarray

    # --- shape ------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """``(nc, ns)`` -- chordwise and spanwise panel counts."""
        return self.rtl.shape[:2]

    @property
    def nc(self) -> int:
        """Number of chordwise panels."""
        return self.rtl.shape[0]

    @property
    def ns(self) -> int:
        """Number of spanwise panels."""
        return self.rtl.shape[1]

    @property
    def size(self) -> int:
        """Total number of panels, ``nc * ns``."""
        return self.nc * self.ns

    # --- derived vortex geometry ------------------------------------------

    @property
    def left_center(self) -> np.ndarray:
        """Midpoint of each ring's left leg, shape ``(nc, ns, 3)``."""
        return 0.5 * (self.rtl + self.rbl)

    @property
    def right_center(self) -> np.ndarray:
        """Midpoint of each ring's right leg, shape ``(nc, ns, 3)``."""
        return 0.5 * (self.rtr + self.rbr)

    @property
    def top_vector(self) -> np.ndarray:
        """Top bound vortex vector ``rtr - rtl``, shape ``(nc, ns, 3)``."""
        return self.rtr - self.rtl

    @property
    def left_vector(self) -> np.ndarray:
        """Left leg vector ``rtl - rbl``, shape ``(nc, ns, 3)``."""
        return self.rtl - self.rbl

    @property
    def right_vector(self) -> np.ndarray:
        """Right leg vector ``rbr - rtr``, shape ``(nc, ns, 3)``."""
        return self.rbr - self.rtr

    # --- transformations ---------------------------------------------------

    def copy(self) -> "Surface":
        """Return a deep copy."""
        return Surface(*(getattr(self, f).copy() for f in _FIELDS))

    def translate(self, r) -> "Surface":
        """Return a copy translated by the vector ``r`` (m)."""
        r = np.asarray(r, dtype=float).reshape(3)
        out = self.copy()
        for name in _POINT_FIELDS:
            getattr(out, name)[...] += r
        return out

    def rotate(self, R, r=(0.0, 0.0, 0.0)) -> "Surface":
        """Return a copy rotated by matrix ``R`` about the point ``r``."""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        r = np.asarray(r, dtype=float).reshape(3)
        out = self.copy()
        for name in _POINT_FIELDS:
            pts = getattr(out, name)
            pts[...] = (pts - r) @ R.T + r
        out.ncp[...] = out.ncp @ R.T
        return out

    def set_normal(self, ncp) -> "Surface":
        """Return a copy with the control-point normal vectors replaced.

        ``ncp`` broadcasts against shape ``(nc, ns, 3)``.  Used by the AVL
        verification cases, where AVL's normal-vector construction differs
        slightly from this package's for twisted dihedral panels.
        """
        out = self.copy()
        out.ncp[...] = np.broadcast_to(
            np.asarray(ncp, dtype=float), out.ncp.shape
        )
        return out

    def reflect(self) -> "Surface":
        """Return a copy reflected across the X-Z plane.

        Left and right are swapped along with the sign of y, so the resulting
        panels keep a consistent circulation sense.
        """
        rev = lambda a: a[:, ::-1]  # noqa: E731
        return Surface(
            rtl=flipy(rev(self.rtr)),
            rtc=flipy(rev(self.rtc)),
            rtr=flipy(rev(self.rtl)),
            rbl=flipy(rev(self.rbr)),
            rbc=flipy(rev(self.rbc)),
            rbr=flipy(rev(self.rbl)),
            rcp=flipy(rev(self.rcp)),
            ncp=flipy(rev(self.ncp)),
            core_size=rev(self.core_size).copy(),
            chord=rev(self.chord).copy(),
        )

    # --- flattened views used by the solver --------------------------------

    def flat(self, name: str) -> np.ndarray:
        """Return field ``name`` reshaped to ``(nc*ns, ...)`` in C order."""
        a = getattr(self, name)
        if a.ndim == 3:
            return a.reshape(-1, 3)
        return a.reshape(-1)


_FIELDS = (
    "rtl",
    "rtc",
    "rtr",
    "rbl",
    "rbc",
    "rbr",
    "rcp",
    "ncp",
    "core_size",
    "chord",
)
_POINT_FIELDS = ("rtl", "rtc", "rtr", "rbl", "rbc", "rbr", "rcp")


# --- Trefftz plane ----------------------------------------------------------


@dataclass
class TrefftzPanels:
    """Trailing-edge filaments projected into the Trefftz (far-field) plane.

    Attributes
    ----------
    rl, rc, rr : ndarray, shape (ns, 3)
        Left, centre and right points of each panel, rotated into the wind
        frame with the x-component zeroed out.
    gamma : ndarray, shape (ns,)
        Circulation of the corresponding trailing-edge panel.
    """

    rl: np.ndarray
    rc: np.ndarray
    rr: np.ndarray
    gamma: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        """Panel normal *including magnitude*, shape ``(ns, 3)``.

        The magnitude carries the panel width, which is why the Trefftz sum
        does not need a separate length factor.
        """
        dy = self.rr[:, 1] - self.rl[:, 1]
        dz = self.rr[:, 2] - self.rl[:, 2]
        return np.stack([np.zeros_like(dy), -dz, dy], axis=-1)
