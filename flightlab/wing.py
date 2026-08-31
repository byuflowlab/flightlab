"""``flightlab.wing`` -- vortex-lattice wing analysis: loading, efficiency, stall.

Wraps :mod:`flightlab.vlm` in the questions a wing designer asks.  The solver
returns panel circulations; this module returns span loading, span efficiency,
where the wing stalls first, and how much lift it makes before it does.

Everything here is **inviscid**.  The vortex lattice knows nothing about
Reynolds number, separation, or section drag, and it will happily report lift
at 30 degrees of angle of attack.  What makes stall computable is combining it
with :mod:`flightlab.airfoil`: the lattice supplies each station's local ``cl`` and
local Reynolds number, the section table supplies that station's own
``cl_max``, and the wing stalls when the first station reaches its own limit.

    >>> from flightlab import wing
    >>> from flightlab.fleet import ASW27
    >>> s = wing.analyze(ASW27, alpha=4.0, V=30.0, altitude=1000.0)
    >>> round(s.CL, 4), round(s.e_inv, 4)
    (0.8442, 0.9652)

Units
-----
SI; angles in **degrees** at this module's interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from . import airfoil as _airfoil
from . import atmos, geom
from .fleet import Aircraft, Planform
from .geom import Panel
from .vlm import (
    Freestream,
    Stability,
    body_forces,
    far_field_drag,
    lifting_line_coefficients,
    lifting_line_geometry,
    steady_analysis,
)

__all__ = [
    "Solution",
    "Sweep",
    "analyze",
    "sweep",
    "trim_to_CL",
    "CL_max",
    "span_efficiency",
    "elliptical_loading",
]

G0 = 9.80665


# --- results ----------------------------------------------------------------


@dataclass(frozen=True)
class Solution:
    """One vortex-lattice solve at one angle of attack.

    Attributes
    ----------
    alpha, beta : float
        Degrees.
    V, altitude : float
        m/s and m.
    air : flightlab.atmos.State
    CL, CD_i, CY : float
        Lift, Trefftz-plane induced drag, and side-force coefficients, on the
        wing reference area, in the stability frame.
    Cl, Cm, Cn : float
        Roll, pitch and yaw moment coefficients about the reference point.
    e_inv : float
        Inviscid span efficiency, ``CL^2 / (pi AR CD_i)``.
    y, chord, cl, cm_section, Re, ds : ndarray
        Per-station span loading.  ``y`` is the station centre (m from the
        centreline), ``chord`` the local chord, ``cl`` the local section lift
        coefficient, ``Re`` the local chord Reynolds number, and ``ds`` the
        station width along the lifting line.
    surfaces : tuple of str
        Which surfaces the per-station arrays cover, in order.
    surface_slices : dict
        Name to slice, for pulling one surface's stations out of the arrays.
    reference : dict
        ``area``, ``mac``, ``span``, ``aspect_ratio``, ``x_ref``.
    """

    alpha: float
    beta: float
    V: float
    altitude: float
    air: "atmos.State"
    CL: float
    CD_i: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    e_inv: float
    y: np.ndarray
    chord: np.ndarray
    cl: np.ndarray
    cm_section: np.ndarray
    Re: np.ndarray
    ds: np.ndarray
    surfaces: Tuple[str, ...]
    surface_slices: Dict[str, slice]
    reference: Dict[str, float]
    _system: object = field(default=None, repr=False, compare=False)

    # -- derived ------------------------------------------------------------

    @property
    def q(self) -> float:
        """Dynamic pressure, Pa."""
        return self.air.q(self.V)

    @property
    def lift(self) -> float:
        """Total lift, N."""
        return self.CL * self.q * self.reference["area"]

    @property
    def induced_drag(self) -> float:
        """Induced drag, N."""
        return self.CD_i * self.q * self.reference["area"]

    @property
    def ccl(self) -> np.ndarray:
        """Local ``c * cl`` -- the span loading proper, m."""
        return self.chord * self.cl

    @property
    def strip_lift(self) -> float:
        """Lift recovered by integrating the strips, N.

        Equals :attr:`lift` to solver precision.  It is worth checking once on
        any new geometry: if the strips do not integrate to the total, the
        panelling is not what you think it is.
        """
        return float(np.sum(self.ccl * self.ds)) * self.q

    def surface(self, name: str) -> "Solution":
        """A view of one surface's stations, for a multi-surface solve."""
        s = self.surface_slices[name]
        return Solution(
            **{
                **self.__dict__,
                "y": self.y[s],
                "chord": self.chord[s],
                "cl": self.cl[s],
                "cm_section": self.cm_section[s],
                "Re": self.Re[s],
                "ds": self.ds[s],
                "surfaces": (name,),
                "surface_slices": {name: slice(0, s.stop - s.start)},
            }
        )

    def elliptical(self) -> np.ndarray:
        """The elliptical loading carrying the same total lift, as ``c*cl``.

        The reference curve a span-loading plot is read against.  Scaled to the
        same total lift rather than the same root value, so the comparison is
        about *distribution* and the areas under the two curves match.
        """
        b = self.reference["span"]
        eta = np.clip(2.0 * np.abs(self.y) / b, 0.0, 1.0)
        shape = np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
        total = float(np.sum(self.ccl * self.ds))
        scale = total / max(float(np.sum(shape * self.ds)), 1e-30)
        return scale * shape

    def cl_max_local(self, table: "_airfoil.Table") -> np.ndarray:
        """Each station's own section ``cl_max``, at its own local Reynolds number."""
        return np.asarray(table.cl_max(self.Re))

    def stall_margin(self, table: "_airfoil.Table") -> np.ndarray:
        """``cl_max_local - cl`` at each station.  Zero is the stall."""
        return self.cl_max_local(table) - self.cl

    def critical_station(self, table: "_airfoil.Table") -> int:
        """Index of the station closest to its own section ``cl_max``.

        Where the wing stalls first.  For a strongly tapered wing this is out
        near the tip, which is the aileron and is why washout exists.
        """
        return int(np.argmin(self.stall_margin(table)))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<wing.Solution a={self.alpha:.2f} deg  CL={self.CL:.4f}  "
            f"CDi={self.CD_i:.6f}  e={self.e_inv:.4f}  Cm={self.Cm:.4f}>"
        )


@dataclass(frozen=True)
class Sweep:
    """An angle-of-attack sweep: the inviscid half of a drag polar.

    Attributes
    ----------
    alpha : ndarray
        Degrees.
    CL, CD_i, Cm, e_inv : ndarray
    solutions : tuple of Solution
    """

    alpha: np.ndarray
    CL: np.ndarray
    CD_i: np.ndarray
    Cm: np.ndarray
    e_inv: np.ndarray
    solutions: Tuple[Solution, ...]

    @property
    def CL_alpha(self) -> float:
        """Wing lift-curve slope, **per radian**, by least squares over the sweep.

        Inviscid and linear, so unlike a section lift slope this one does not
        depend on the fitting interval -- which makes it a clean number to
        compare against lifting-line theory, and a poor model of a real wing
        near the stall.
        """
        return float(np.degrees(np.polyfit(self.alpha, self.CL, 1)[0]))

    @property
    def alpha_0(self) -> float:
        """Zero-lift angle of attack, degrees."""
        return float(np.interp(0.0, self.CL, self.alpha))

    @property
    def Cm_alpha(self) -> float:
        """Pitching-moment slope about the reference point, per radian."""
        return float(np.degrees(np.polyfit(self.alpha, self.Cm, 1)[0]))

    def at(self, alpha: float) -> Solution:
        """The solution nearest ``alpha``."""
        return self.solutions[int(np.argmin(np.abs(self.alpha - alpha)))]

    def __len__(self) -> int:
        return len(self.alpha)


# --- the driver -------------------------------------------------------------


def _as_panels(obj, tail: bool, fin: bool) -> Tuple[List[Panel], List[str]]:
    if isinstance(obj, Aircraft):
        out, names = [], []
        for label, plan, include in (
            ("wing", obj.wing, True),
            ("htail", obj.htail, tail),
            ("vtail", obj.vtail, fin),
        ):
            if plan is not None and include:
                out.append(geom.resolve(plan))
                names.append(label)
        if not out:
            raise ValueError(f"{obj.label} has no lifting surfaces")
        return out, names
    if isinstance(obj, Planform):
        return [geom.resolve(obj)], ["wing"]
    if isinstance(obj, Panel):
        return [obj], ["wing"]
    raise TypeError(f"cannot analyze a {type(obj).__name__}")


def analyze(
    obj,
    alpha: float,
    V: float,
    altitude: float = 0.0,
    beta: float = 0.0,
    ns: int = 40,
    nc: int = 6,
    tail: bool = False,
    fin: bool = False,
    mirror: bool = False,
    camber: bool = True,
    tail_incidence_deg: Optional[float] = None,
    x_ref: Optional[float] = None,
    dT: float = 0.0,
) -> Solution:
    """Solve one flight condition.

    Parameters
    ----------
    obj : Aircraft, Planform, or Panel
        The geometry.  An :class:`~flightlab.fleet.Aircraft` brings its tail along
        if ``tail`` or ``fin`` is set.
    alpha, beta : float
        Angle of attack and sideslip, degrees.
    V : float
        True airspeed, m/s.
    altitude : float
        Geometric altitude, m -- sets density, and through it the local
        Reynolds numbers the section lookups need.
    ns, nc : int
        Spanwise and chordwise panels per semispan.  Defaults are converged for
        span efficiency to about half a percent on the fleet's planforms. A
        resolution study is still worth doing before trusting a new planform.
    tail, fin : bool
        Include the horizontal tail and vertical fin.
    mirror : bool
        Model both sides explicitly instead of using solver symmetry.  Forced
        on when ``beta`` is nonzero or ``fin`` is set, because **with symmetry
        every lateral coefficient comes back exactly zero** -- correct, and
        indistinguishable from a broken model.
    camber : bool
        Put the panels on the section camber line.
    tail_incidence_deg : float, optional
        Override the horizontal tail incidence; the variable a trim solve turns.
    x_ref : float, optional
        Moment reference ``x``, m.  Defaults to the wing MAC quarter chord.
    dT : float
        Atmospheric temperature offset, K.

    Returns
    -------
    Solution
    """
    panels, names = _as_panels(obj, tail, fin)
    if tail_incidence_deg is not None:
        panels = [
            Panel(**{**p.__dict__, "incidence_deg": tail_incidence_deg})
            if n == "htail"
            else p
            for p, n in zip(panels, names)
        ]

    lateral = bool(beta) or fin or mirror
    mirror = lateral
    symmetric = not lateral

    air = atmos.at(altitude, dT)
    ref_panel = panels[0]
    x_ref = ref_panel.x_c4_mac if x_ref is None else x_ref

    from .vlm import Reference, translate

    grids, ratios = [], []
    for p in panels:
        g, r = geom.surface_grid(p, ns=ns, nc=nc, mirror=mirror, camber=camber)
        if p.x_le or p.z:
            g = translate(g, (p.x_le, 0.0, p.z))
        grids.append(g)
        ratios.append(r)

    reference = Reference(
        ref_panel.area, ref_panel.mac, ref_panel.span, [x_ref, 0.0, 0.0], V
    )
    fs = Freestream.from_degrees(V, alpha=alpha, beta=beta)
    system = steady_analysis(
        grids, reference, fs, symmetric=symmetric, ratios=ratios
    )

    CF, CM = body_forces(system, frame=Stability())
    CD_i = far_field_drag(system)
    CL, CY = float(CF[2]), float(CF[1])
    AR = ref_panel.aspect_ratio
    e_inv = float(CL**2 / (np.pi * AR * CD_i)) if CD_i > 0 else float("nan")

    # per-station loading, at the same stations for every quantity
    r_ll, c_ll = lifting_line_geometry(system.grids)
    cf, cm = lifting_line_coefficients(system, r_ll, c_ll, frame=Stability())

    ys, chords, cls, cms, dss, slices = [], [], [], [], [], {}
    start = 0
    for i, name in enumerate(names):
        r_i = r_ll[i]
        y_edge = r_i[1, :]
        d = np.linalg.norm(np.diff(r_i, axis=1), axis=0)
        c_mid = 0.5 * (c_ll[i][:-1] + c_ll[i][1:])
        ys.append(0.5 * (y_edge[:-1] + y_edge[1:]))
        chords.append(c_mid)
        cls.append(cf[i][2, :])
        cms.append(cm[i][1, :])
        dss.append(d)
        slices[name] = slice(start, start + len(d))
        start += len(d)

    y = np.concatenate(ys)
    chord = np.concatenate(chords)
    Re = air.reynolds(V, chord)
    ds = np.concatenate(dss)
    # a symmetric solve models one side; the strips must still integrate to the
    # whole aircraft's lift, so count both
    if symmetric:
        ds = 2.0 * ds

    return Solution(
        alpha=float(alpha),
        beta=float(beta),
        V=float(V),
        altitude=float(altitude),
        air=air,
        CL=CL,
        CD_i=float(CD_i),
        CY=CY,
        Cl=float(CM[0]),
        Cm=float(CM[1]),
        Cn=float(CM[2]),
        e_inv=e_inv,
        y=y,
        chord=chord,
        cl=np.concatenate(cls),
        cm_section=np.concatenate(cms),
        Re=Re,
        ds=ds,
        surfaces=tuple(names),
        surface_slices=slices,
        reference={
            "area": ref_panel.area,
            "mac": ref_panel.mac,
            "span": ref_panel.span,
            "aspect_ratio": AR,
            "x_ref": x_ref,
        },
        _system=system,
    )


def sweep(obj, alpha, V: float, altitude: float = 0.0, **kwargs) -> Sweep:
    """Solve a range of angles of attack.

    Parameters
    ----------
    obj : Aircraft, Planform, or Panel
    alpha : array_like
        Degrees.
    V, altitude : float
    **kwargs
        Passed to :func:`analyze`.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    sols = tuple(analyze(obj, a, V, altitude, **kwargs) for a in alpha)
    return Sweep(
        alpha=alpha,
        CL=np.array([s.CL for s in sols]),
        CD_i=np.array([s.CD_i for s in sols]),
        Cm=np.array([s.Cm for s in sols]),
        e_inv=np.array([s.e_inv for s in sols]),
        solutions=sols,
    )


def trim_to_CL(
    obj,
    CL: float,
    V: float,
    altitude: float = 0.0,
    bracket: Tuple[float, float] = (-8.0, 16.0),
    tol: float = 1e-9,
    **kwargs,
) -> Solution:
    """Find the angle of attack that produces ``CL``, and return that solution.

    The inviscid lift curve is linear, so this converges in a handful of
    iterations regardless of the bracket.  Widen ``bracket`` for a heavily
    cambered section at negative lift.
    """

    def residual(a):
        return analyze(obj, a, V, altitude, **kwargs).CL - CL

    lo, hi = bracket
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"CL = {CL:.4f} is not reachable between {lo} and {hi} degrees "
            f"(CL spans {CL + f_lo:.4f} to {CL + f_hi:.4f}); widen the bracket"
        )
    a = brentq(residual, lo, hi, xtol=tol)
    return analyze(obj, a, V, altitude, **kwargs)


def trim_to_weight(
    obj,
    mass: float,
    V: float,
    altitude: float = 0.0,
    n: float = 1.0,
    **kwargs,
) -> Solution:
    """Trim to carry ``mass`` kilograms at load factor ``n``.

    The identity every analysis rests on: at steady flight, lift equals weight.
    """
    panels, _ = _as_panels(obj, kwargs.get("tail", False), kwargs.get("fin", False))
    air = atmos.at(altitude, kwargs.get("dT", 0.0))
    CL = n * mass * G0 / (air.q(V) * panels[0].area)
    return trim_to_CL(obj, CL, V, altitude, **kwargs)


def CL_max(
    obj,
    V: float,
    altitude: float = 0.0,
    table: Optional["_airfoil.Table"] = None,
    section: Optional[str] = None,
    alpha_bracket: Tuple[float, float] = (0.0, 25.0),
    **kwargs,
) -> Dict[str, float]:
    """Maximum wing lift coefficient, by first-station stall.

    Raises the angle of attack until the first span station reaches its **own**
    section ``cl_max`` at its **own** local Reynolds number, and reports the
    wing ``CL`` there.

    This is a section-stall criterion, not a viscous wing solution.  It has no
    knowledge of how stall spreads once it starts, and it will be optimistic
    for a wing whose tip stalls first, because a real wing loses lift as the
    separated region grows.  It is nonetheless the right criterion for design
    work: it says *where* the wing runs out first, which is the actionable
    part.

    Parameters
    ----------
    obj : Aircraft, Planform, or Panel
    V, altitude : float
    table : flightlab.airfoil.Table, optional
        Section table.  Built from the wing's own section over the local
        Reynolds range if omitted.
    section : str, optional
        Override the section used to build that table.
    alpha_bracket : tuple
        Search range, degrees.

    Returns
    -------
    dict
        ``CL_max``, ``alpha`` (degrees), ``y_critical`` (m), ``eta_critical``
        (fraction of semispan), ``cl_critical``, ``Re_critical``, and
        ``solution``.
    """
    panels, _ = _as_panels(obj, kwargs.get("tail", False), kwargs.get("fin", False))
    panel = panels[0]

    if table is None:
        probe = analyze(obj, 0.0, V, altitude, **kwargs)
        s = probe.surface_slices[probe.surfaces[0]]
        Re_lo, Re_hi = float(probe.Re[s].min()), float(probe.Re[s].max())
        table = _airfoil.table(
            section or panel.section, Re=(0.5 * Re_lo, 2.0 * Re_hi)
        )

    def margin(a):
        sol = analyze(obj, a, V, altitude, **kwargs)
        s = sol.surface_slices[sol.surfaces[0]]
        return float(np.min(table.cl_max(sol.Re[s]) - sol.cl[s]))

    lo, hi = alpha_bracket
    m_lo, m_hi = margin(lo), margin(hi)
    if m_lo < 0:
        raise ValueError(
            f"the wing is already stalled at alpha = {lo} degrees; "
            "lower the bottom of alpha_bracket"
        )
    if m_hi > 0:
        raise ValueError(
            f"no station reaches its cl_max by alpha = {hi} degrees "
            f"(closest margin {m_hi:.3f}); raise the top of alpha_bracket"
        )
    a_stall = brentq(margin, lo, hi, xtol=1e-6)
    sol = analyze(obj, a_stall, V, altitude, **kwargs)
    s = sol.surface_slices[sol.surfaces[0]]
    i = int(np.argmin(table.cl_max(sol.Re[s]) - sol.cl[s])) + s.start

    return {
        "CL_max": sol.CL,
        "alpha": a_stall,
        "y_critical": float(sol.y[i]),
        "eta_critical": float(abs(sol.y[i]) / panel.semispan),
        "cl_critical": float(sol.cl[i]),
        "Re_critical": float(sol.Re[i]),
        "solution": sol,
    }


def stall_speed(
    obj, mass: float, altitude: float = 0.0, n: float = 1.0, **kwargs
) -> float:
    """Stall speed, m/s, by iterating ``CL_max`` and the speed it implies.

    ``CL_max`` depends on Reynolds number, which depends on speed, which
    depends on ``CL_max``.  Two or three passes converge.
    """
    panels, _ = _as_panels(obj, kwargs.get("tail", False), kwargs.get("fin", False))
    panel = panels[0]
    air = atmos.at(altitude, kwargs.get("dT", 0.0))
    W = n * mass * G0
    V = np.sqrt(2.0 * W / (air.density * panel.area * 1.2))  # first guess
    for _ in range(6):
        clmax = CL_max(obj, V, altitude, **kwargs)["CL_max"]
        V_new = float(np.sqrt(2.0 * W / (air.density * panel.area * clmax)))
        if abs(V_new - V) < 1e-4 * V:
            return V_new
        V = V_new
    return V


def span_efficiency(sol: Solution) -> float:
    """``CL^2 / (pi AR CD_i)`` from a solution.  Same as :attr:`Solution.e_inv`."""
    return sol.e_inv


def elliptical_loading(y, span: float, total_lift_coefficient: float, area: float):
    """The elliptical ``c*cl`` distribution carrying a given ``CL``, m."""
    eta = np.clip(2.0 * np.abs(np.asarray(y, dtype=float)) / span, 0.0, 1.0)
    root = 4.0 * total_lift_coefficient * area / (np.pi * span)
    return root * np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
