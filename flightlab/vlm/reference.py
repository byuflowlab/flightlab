"""Reference quantities and reference frames.

Ported from VortexLattice.jl (``src/reference.jl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Reference", "Body", "Stability", "Wind"]


@dataclass(frozen=True)
class Reference:
    """Reference quantities used to non-dimensionalize forces and moments.

    Parameters
    ----------
    S : float
        Reference area, m^2.
    c : float
        Reference chord, m.  Pitching moments and the ``q`` rate derivative are
        normalized by this length.
    b : float
        Reference span, m.  Rolling and yawing moments and the ``p`` and ``r``
        rate derivatives are normalized by this length.
    r : array_like, shape (3,)
        Reference location (x, y, z) in metres about which all moments are
        taken and all body rotations are applied.  This is the moment reference
        point, *not* necessarily the centre of mass.
    V : float
        Reference velocity magnitude, m/s.  Used to form the reference dynamic
        pressure ``q = rho V^2 / 2``.  Normally equal to ``Freestream.Vinf``.

    Notes
    -----
    All lengths in metres, all areas in m^2.  The solver itself is inviscid and
    scale free; ``V`` only enters through the non-dimensionalization.
    """

    S: float
    c: float
    b: float
    r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    V: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "S", float(self.S))
        object.__setattr__(self, "c", float(self.c))
        object.__setattr__(self, "b", float(self.b))
        object.__setattr__(self, "V", float(self.V))
        r = np.asarray(self.r, dtype=float).reshape(3)
        object.__setattr__(self, "r", r)

    @property
    def lengths(self) -> np.ndarray:
        """Moment reference lengths ``(b, c, b)`` for ``(Cl, Cm, Cn)``."""
        return np.array([self.b, self.c, self.b])


class _Frame:
    """Base class for the output reference frames."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self).__name__)


class Body(_Frame):
    """Reference frame aligned with the global X-Y-Z axes."""

    __slots__ = ()


class Stability(_Frame):
    """Body frame rotated about y to align with the freestream ``alpha``.

    In this frame the x-component of the force coefficient is drag and the
    z-component is lift, which is what almost every textbook means by ``CD``
    and ``CL``.
    """

    __slots__ = ()


class Wind(_Frame):
    """Frame aligned with the freestream ``alpha`` *and* ``beta``."""

    __slots__ = ()
