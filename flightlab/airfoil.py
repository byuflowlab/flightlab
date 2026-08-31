"""``flightlab.airfoil`` -- section polars, and the ``cd(cl, Re)`` lookup.

:mod:`flightlab.foil` is the raw wrapper around NeuralFoil: give it an angle of
attack and a Reynolds number, get back coefficients.  This module is the layer
above it -- the one that answers the questions a wing analysis actually asks.

A strip integration does not know its angle of attack.  It knows that station
17 is running at ``cl = 0.62`` and ``Re = 4.1e5`` and it wants the section drag
there.  Going from one to the other means inverting the lift curve, which is
double-valued past the stall, and interpolating in two dimensions across a
grid that has to be dense enough near ``cl_max`` and cheap enough to build.
That is what :class:`Table` does.

    >>> from flightlab import airfoil
    >>> p = airfoil.polar("naca2412", Re=3e6)
    >>> round(p.cl_alpha(-2, 8), 2), round(p.cl_max, 3)
    (6.28, 1.783)
    >>> t = airfoil.table("sd7037", Re=(5e4, 5e5))
    >>> round(float(t.cd(cl=0.6, Re=1.5e5)), 5)
    0.01119

Units
-----
Angles in **degrees**.  Reynolds numbers are chord Reynolds numbers.  ``cm`` is
about the quarter chord.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from . import foil

__all__ = ["Polar", "Table", "polar", "table", "cl_max", "cd_min", "clear_cache"]


# --- a polar at one Reynolds number -----------------------------------------


@dataclass(frozen=True)
class Polar:
    """A section's coefficients swept over angle of attack at one condition.

    Attributes
    ----------
    section : str
    alpha : ndarray
        Degrees.
    cl, cd, cm, confidence : ndarray
    Re, mach, n_crit, xtr_upper, xtr_lower : float
    """

    section: str
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    confidence: np.ndarray
    Re: float
    mach: float = 0.0
    n_crit: float = 9.0
    xtr_upper: float = 1.0
    xtr_lower: float = 1.0

    # -- headline numbers ---------------------------------------------------

    @property
    def stall_index(self) -> int:
        """Index of maximum ``cl`` -- the last point on the usable branch."""
        return int(np.argmax(self.cl))

    @property
    def cl_max(self) -> float:
        """Maximum section lift coefficient over the swept range.

        Depends on how far you swept.  If the maximum lands on the last point
        of the sweep, the sweep stopped before the stall and this is a lower
        bound, not a stall value; :attr:`stall_captured` says which.
        """
        return float(np.max(self.cl))

    @property
    def stall_captured(self) -> bool:
        """Whether ``cl_max`` is an interior maximum rather than the last point."""
        return 0 < self.stall_index < len(self.alpha) - 1

    @property
    def alpha_stall(self) -> float:
        """Angle of attack at ``cl_max``, degrees."""
        return float(self.alpha[self.stall_index])

    @property
    def cd_min(self) -> float:
        """Minimum section drag coefficient over the swept range."""
        return float(np.min(self.cd))

    @property
    def cl_at_cd_min(self) -> float:
        """Lift coefficient at minimum drag -- the bottom of the bucket."""
        return float(self.cl[int(np.argmin(self.cd))])

    @property
    def alpha_0(self) -> float:
        """Zero-lift angle of attack, degrees, by interpolation."""
        i = self.stall_index
        cl, a = self.cl[: i + 1], self.alpha[: i + 1]
        if cl[0] > 0 or cl[-1] < 0:
            return float("nan")
        return float(np.interp(0.0, cl, a))

    @property
    def ld_max(self) -> float:
        """Maximum section ``cl/cd``."""
        return float(np.max(self.cl / self.cd))

    def cl_alpha(self, a_lo: float = -2.0, a_hi: float = 8.0) -> float:
        """Lift-curve slope fitted over ``[a_lo, a_hi]`` degrees, **per radian**.

        The interval is an argument and not a default buried in the code
        because the answer depends on it.  A NACA 2412 at ``Re = 3e6`` gives
        6.28 /rad over -2 to 8 degrees.  The same section at ``Re = 1e5`` gives
        7.87 over -2 to 4, 5.77 over 1 to 4, and 6.35 over -2 to 8 -- because
        at low Reynolds number the curve has a steep segment near zero lift and
        is simply not straight.  Quoting a slope without quoting the interval
        is not a result.
        """
        m = (self.alpha >= a_lo) & (self.alpha <= a_hi)
        if m.sum() < 2:
            raise ValueError(
                f"only {m.sum()} swept points fall in [{a_lo}, {a_hi}] degrees"
            )
        slope_per_deg = np.polyfit(self.alpha[m], self.cl[m], 1)[0]
        return float(np.degrees(slope_per_deg))

    def cd_at(self, cl) -> np.ndarray:
        """Section drag at a given lift coefficient, on the pre-stall branch.

        Interpolates ``cd`` against ``cl`` up to the stall.  Values of ``cl``
        beyond ``cl_max`` return the stall value -- the section is not flying
        there, and returning a plausible-looking number for an impossible
        request is worse than returning the boundary.
        """
        i = self.stall_index
        return np.interp(np.asarray(cl, dtype=float), self.cl[: i + 1], self.cd[: i + 1])

    def alpha_at(self, cl) -> np.ndarray:
        """Angle of attack for a given ``cl``, pre-stall branch, degrees."""
        i = self.stall_index
        return np.interp(
            np.asarray(cl, dtype=float), self.cl[: i + 1], self.alpha[: i + 1]
        )

    def bucket(self, tol: float = 1.10) -> Tuple[float, float]:
        """The low-drag bucket: the ``cl`` range within ``tol`` of ``cd_min``.

        A laminar section's defining feature and the reason a sailplane picks
        one.  Returns ``(cl_low, cl_high)``.  For a section with no bucket the
        range is narrow rather than absent, so read the width, not its
        existence.
        """
        inside = self.cd <= tol * self.cd_min
        if not inside.any():  # pragma: no cover - cd_min is always inside
            return (float("nan"), float("nan"))
        return float(np.min(self.cl[inside])), float(np.max(self.cl[inside]))

    @property
    def min_confidence(self) -> float:
        """Lowest NeuralFoil confidence over the sweep.

        Worth looking at, and worth not trusting too far: the three low-Re
        lift-slope fits above disagree by more than 30% while confidence stays
        above 0.90 throughout.  A confidence metric is not validation of the
        quantity you extracted.
        """
        return float(np.min(self.confidence))

    def __len__(self) -> int:
        return len(self.alpha)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<Polar {self.section} Re={self.Re:.3g} "
            f"cl_max={self.cl_max:.3f} cd_min={self.cd_min:.5f} "
            f"{'stalled' if self.stall_captured else 'no stall in sweep'}>"
        )


def polar(
    section,
    Re: float = 1e6,
    alpha=None,
    mach: float = 0.0,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
) -> Polar:
    """Sweep a section over angle of attack at one Reynolds number.

    Parameters
    ----------
    section : str, Section, or ndarray
        Anything :func:`flightlab.foil.load` understands.
    Re : float
        Chord Reynolds number.
    alpha : array_like, optional
        Degrees.  Defaults to -10 to 20 in 0.25-degree steps, which captures
        the stall of every section in the fleet at every Reynolds number the
        course uses.  Narrow it when you only want the linear range.
    mach : float
        Prandtl-Glauert correction on ``cl`` and ``cm``.  **There is no drag
        rise in it**; see :mod:`flightlab.foil`.
    n_crit : float
        9 is a clean tunnel.  Drop to 5-7 for a rougher surface or a more
        turbulent stream -- a foam wing sanded by hand is not a clean tunnel.
    xtr_upper, xtr_lower : float
        Forced transition ``x/c``; 1.0 lets the model decide.
    """
    if alpha is None:
        alpha = np.arange(-10.0, 20.0 + 1e-9, 0.25)
    alpha = np.asarray(alpha, dtype=float)
    raw = foil.aero(section, alpha, Re, mach, n_crit, xtr_upper, xtr_lower)
    name = section if isinstance(section, str) else getattr(section, "name", "section")
    return Polar(
        section=str(name),
        alpha=alpha,
        cl=raw["cl"],
        cd=raw["cd"],
        cm=raw["cm"],
        confidence=raw["confidence"],
        Re=float(Re),
        mach=float(mach),
        n_crit=float(n_crit),
        xtr_upper=float(xtr_upper),
        xtr_lower=float(xtr_lower),
    )


# --- the two-dimensional table ----------------------------------------------


class Table:
    """``cd(cl, Re)`` and ``cl_max(Re)`` for one section, by interpolation.

    Built by sweeping angle of attack at each of several Reynolds numbers, then
    inverting.  A wing strip integration calls :meth:`cd` once per station with
    that station's own ``cl`` and its own ``Re``, which is the whole point:
    the tip of a tapered wing runs at a third the Reynolds number of the root,
    and pretending otherwise is worth several percent of the wing's drag.

    Interpolation is linear in ``cl`` and in ``log(Re)``, because section drag
    varies far more nearly linearly with the logarithm of Reynolds number than
    with Reynolds number itself.

    Parameters
    ----------
    section : str, Section, or ndarray
    Re : tuple or array_like
        Either ``(Re_min, Re_max)``, in which case ``n_Re`` values are spaced
        logarithmically between them, or an explicit list of Reynolds numbers.
    alpha : array_like, optional
        Angles of attack to sweep, degrees.
    n_Re : int
        Number of Reynolds numbers when a range is given.
    mach, n_crit, xtr_upper, xtr_lower : float

    Attributes
    ----------
    alpha, Re : ndarray
        The tabulated axes.
    cl_grid, cd_grid, cm_grid, confidence : ndarray
        Shape ``(n_alpha, n_Re)``.  The raw sweep, kept so a plot can show the
        polars the interpolation was built from.

    Notes
    -----
    Requests outside the tabulated Reynolds range are **clamped to the
    endpoints**, not extrapolated, and :attr:`Table.Re_range` says where the
    edges are.  Extrapolating a section polar in Reynolds number is how a
    strip integration quietly produces drag coefficients below the laminar
    flat plate.
    """

    def __init__(
        self,
        section,
        Re=(5e4, 5e6),
        alpha=None,
        n_Re: int = 12,
        mach: float = 0.0,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
    ):
        Re = np.asarray(Re, dtype=float)
        if Re.size == 2:
            Re = np.logspace(np.log10(Re[0]), np.log10(Re[1]), n_Re)
        Re = np.unique(Re)
        if alpha is None:
            alpha = np.arange(-10.0, 20.0 + 1e-9, 0.25)
        alpha = np.asarray(alpha, dtype=float)

        # one vectorized NeuralFoil call over the whole (alpha, Re) grid
        A, R = np.meshgrid(alpha, Re, indexing="ij")
        raw = foil.aero(section, A, R, mach, n_crit, xtr_upper, xtr_lower)

        name = section if isinstance(section, str) else getattr(section, "name", "sec")
        self.section = str(name)
        self.alpha = alpha
        self.Re = Re
        self.log_Re = np.log(Re)
        self.cl_grid = raw["cl"]
        self.cd_grid = raw["cd"]
        self.cm_grid = raw["cm"]
        self.confidence = raw["confidence"]
        self.mach, self.n_crit = float(mach), float(n_crit)
        self.xtr_upper, self.xtr_lower = float(xtr_upper), float(xtr_lower)

        # per-Reynolds-number stall index, so every lookup uses the branch that
        # is actually monotone in cl
        self._i_stall = np.argmax(self.cl_grid, axis=0)
        self._cl_max = self.cl_grid[self._i_stall, np.arange(len(Re))]
        self._cd_min = np.min(self.cd_grid, axis=0)

    # -- properties ---------------------------------------------------------

    @property
    def Re_range(self) -> Tuple[float, float]:
        """The tabulated Reynolds range; outside it, lookups clamp."""
        return float(self.Re[0]), float(self.Re[-1])

    def polar_at(self, Re: float) -> Polar:
        """The tabulated :class:`Polar` nearest to ``Re``."""
        j = int(np.argmin(np.abs(self.log_Re - np.log(Re))))
        return Polar(
            section=self.section,
            alpha=self.alpha,
            cl=self.cl_grid[:, j],
            cd=self.cd_grid[:, j],
            cm=self.cm_grid[:, j],
            confidence=self.confidence[:, j],
            Re=float(self.Re[j]),
            mach=self.mach,
            n_crit=self.n_crit,
            xtr_upper=self.xtr_upper,
            xtr_lower=self.xtr_lower,
        )

    # -- lookups ------------------------------------------------------------

    def _weights(self, Re):
        """Bracketing column indices and blend weight for ``log(Re)``."""
        lr = np.log(np.clip(np.asarray(Re, dtype=float), self.Re[0], self.Re[-1]))
        j = np.clip(np.searchsorted(self.log_Re, lr) - 1, 0, len(self.Re) - 2)
        span = self.log_Re[j + 1] - self.log_Re[j]
        w = np.where(span > 0, (lr - self.log_Re[j]) / span, 0.0)
        return j, w

    def _per_column(self, values, cl, Re):
        """Interpolate ``values[:, j](cl)`` at both bracketing columns, blend."""
        cl = np.asarray(cl, dtype=float)
        j, w = self._weights(Re)
        cl, j, w = np.broadcast_arrays(cl, j, w)
        out = np.empty(cl.shape, dtype=float)
        flat_cl, flat_j, flat_w = cl.ravel(), j.ravel(), w.ravel()
        result = np.empty(flat_cl.shape, dtype=float)
        for k in np.unique(flat_j):
            m = flat_j == k
            lo = self._branch(values, k)
            hi = self._branch(values, k + 1)
            result[m] = (1.0 - flat_w[m]) * np.interp(
                flat_cl[m], *lo
            ) + flat_w[m] * np.interp(flat_cl[m], *hi)
        out[...] = result.reshape(cl.shape)
        return out

    def _branch(self, values, k):
        """``(cl, value)`` on the pre-stall branch of column ``k``."""
        i = self._i_stall[k] + 1
        return self.cl_grid[:i, k], values[:i, k]

    def cd(self, cl, Re) -> np.ndarray:
        """Section drag coefficient at lift coefficient ``cl`` and ``Re``.

        Both arguments broadcast, so a whole span's worth of stations is one
        call.  ``cl`` beyond the local ``cl_max`` clamps to the stall value.
        """
        return self._per_column(self.cd_grid, cl, Re)

    def cm(self, cl, Re) -> np.ndarray:
        """Quarter-chord moment coefficient at ``cl`` and ``Re``."""
        return self._per_column(self.cm_grid, cl, Re)

    def alpha_at(self, cl, Re) -> np.ndarray:
        """Angle of attack for ``cl`` at ``Re``, degrees, pre-stall branch."""
        grid = np.repeat(self.alpha[:, None], len(self.Re), axis=1)
        return self._per_column(grid, cl, Re)

    def cl_max(self, Re) -> np.ndarray:
        """Maximum section lift coefficient at ``Re``.

        Interpolated in ``log(Re)``.  This is the number a wing ``CL_max``
        search compares each station's local ``cl`` against, and it rises with
        Reynolds number -- which is why the tip of a tapered wing, running at
        the lowest local Reynolds number on the wing, is doubly disadvantaged.
        """
        j, w = self._weights(Re)
        return (1.0 - w) * self._cl_max[j] + w * self._cl_max[j + 1]

    def cd_min(self, Re) -> np.ndarray:
        """Minimum section drag coefficient at ``Re``."""
        j, w = self._weights(Re)
        return (1.0 - w) * self._cd_min[j] + w * self._cd_min[j + 1]

    def out_of_range(self, Re) -> np.ndarray:
        """True where ``Re`` falls outside the tabulated range and was clamped."""
        Re = np.asarray(Re, dtype=float)
        return (Re < self.Re[0]) | (Re > self.Re[-1])

    def state_key(self):
        """Hashable identity, for :mod:`flightlab.cache`."""
        return (
            self.section,
            self.mach,
            self.n_crit,
            self.xtr_upper,
            self.xtr_lower,
            float(self.Re[0]),
            float(self.Re[-1]),
            len(self.Re),
            len(self.alpha),
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<airfoil.Table {self.section} "
            f"Re {self.Re[0]:.2g}-{self.Re[-1]:.2g} ({len(self.Re)}), "
            f"{len(self.alpha)} alphas>"
        )


# --- module-level conveniences ----------------------------------------------

_TABLES: Dict[Tuple, Table] = {}


def table(section, Re=(5e4, 5e6), **kwargs) -> Table:
    """A cached :class:`Table`.

    Building one costs a few hundred milliseconds of NeuralFoil, and the same
    section at the same condition gets asked for repeatedly across a sweep, so
    they are memoized on their arguments for the life of the session.
    """
    name = section if isinstance(section, str) else getattr(section, "name", None)
    if name is None:
        return Table(section, Re, **kwargs)
    Re_arr = np.asarray(Re, dtype=float)
    key = (
        str(name),
        (float(Re_arr.min()), float(Re_arr.max()), Re_arr.size),
        tuple(sorted((k, _hashable(v)) for k, v in kwargs.items())),
    )
    if key not in _TABLES:
        _TABLES[key] = Table(section, Re, **kwargs)
    return _TABLES[key]


def _hashable(v):
    if isinstance(v, np.ndarray):
        return ("arr", v.shape, v.tobytes())
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def clear_cache() -> None:
    """Drop the memoized section tables."""
    _TABLES.clear()


def cl_max(section, Re: float, **kwargs) -> float:
    """Maximum section lift coefficient at one Reynolds number."""
    return polar(section, Re, **kwargs).cl_max


def cd_min(section, Re: float, **kwargs) -> float:
    """Minimum section drag coefficient at one Reynolds number."""
    return polar(section, Re, **kwargs).cd_min
