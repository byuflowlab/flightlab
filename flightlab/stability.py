"""``flightlab.stability`` -- mass properties, trim, static margin, dynamic modes.

Four things that are usually taught as four topics and are really one, because
each depends on the one before it.  You cannot trim without a centre of
gravity, you cannot quote a static margin without a neutral point, and you
cannot get a phugoid period without both plus the inertias.

    >>> from flightlab import stability
    >>> from flightlab.fleet import RC1
    >>> mp = stability.mass_properties(RC1)
    >>> round(mp.mass, 4), round(mp.x_cg, 4)
    (0.7502, 0.0479)
    >>> t = stability.trim(RC1, V=11.0, altitude=1400.0)
    >>> round(t.alpha, 2), round(t.static_margin, 3)
    (2.16, 0.226)

What is modelled and what is not
--------------------------------
The neutral point and every derivative come from the vortex lattice, which
models **lifting surfaces only**.  A fuselage or a pod ahead of the wing is
destabilizing and the lattice does not see it, so a lifting-surfaces-only
neutral point is optimistic.  :func:`body_pitching_moment` supplies the
standard Munk slender-body correction, and :func:`neutral_point` will apply it
when asked -- but it is a correction, not a solution, and it is off by default
so that the number's provenance stays visible.

Downwash lag (the ``alpha_dot`` derivatives) is likewise not something a steady
vortex lattice can produce.  It is estimated from the tail's contribution and
the flow time from wing to tail; see :func:`alpha_dot_derivatives`.

Units
-----
SI.  Angles in **degrees** at the interfaces, radians inside.  Inertias in
kg m^2 about the centre of gravity, body axes, ``x`` aft, ``y`` right, ``z`` up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, fsolve

from . import atmos, geom, wing
from .fleet import Aircraft, Component
from .geom import Panel
from .vlm import (
    Freestream,
    Reference,
    Stability,
    body_forces,
    stability_derivatives,
    steady_analysis,
    translate,
)

__all__ = [
    "MassProperties",
    "Trim",
    "Derivatives",
    "Mode",
    "Modes",
    "mass_properties",
    "neutral_point",
    "static_margin",
    "trim",
    "derivatives",
    "tail_volume",
    "body_pitching_moment",
    "body_derivatives",
    "longitudinal_modes",
    "lateral_modes",
    "modes",
]

G0 = 9.80665


# --- mass properties --------------------------------------------------------


@dataclass(frozen=True)
class MassProperties:
    """Mass, centre of gravity, and inertia tensor from a component table.

    Attributes
    ----------
    mass : float
        kg.
    x_cg, y_cg, z_cg : float
        Centre of gravity in body axes, m.
    Ixx, Iyy, Izz, Ixz : float
        Inertias about the centre of gravity, kg m^2.  ``Ixy`` and ``Iyz``
        vanish for a laterally symmetric aircraft and are not carried.
    components : tuple of Component
    """

    mass: float
    x_cg: float
    y_cg: float
    z_cg: float
    Ixx: float
    Iyy: float
    Izz: float
    Ixz: float
    components: Tuple[Component, ...] = ()

    @property
    def cg(self) -> np.ndarray:
        """Centre of gravity as a 3-vector, m."""
        return np.array([self.x_cg, self.y_cg, self.z_cg])

    @property
    def inertia(self) -> np.ndarray:
        """The 3x3 inertia tensor about the centre of gravity, kg m^2."""
        return np.array(
            [
                [self.Ixx, 0.0, -self.Ixz],
                [0.0, self.Iyy, 0.0],
                [-self.Ixz, 0.0, self.Izz],
            ]
        )

    @property
    def weight(self) -> float:
        """Weight, N."""
        return self.mass * G0

    def x_cg_over_mac(self, panel: Panel) -> float:
        """Centre of gravity as a fraction of the mean aerodynamic chord."""
        return (self.x_cg - (panel.x_le + panel.x_mac)) / panel.mac

    def table(self) -> str:
        """A printable component table with each row's moment contribution."""
        w = max(14, max((len(c.name) for c in self.components), default=14))
        lines = [
            f"{'component':<{w}} {'mass (kg)':>10} {'x (m)':>9} {'z (m)':>9} "
            f"{'m*x':>10} {'share':>7}",
            "-" * (w + 48),
        ]
        for c in sorted(self.components, key=lambda c: -c.mass):
            lines.append(
                f"{c.name:<{w}} {c.mass:10.4f} {c.x:9.4f} {c.z:9.4f} "
                f"{c.mass * c.x:10.5f} {100 * c.mass / self.mass:6.1f}%"
            )
        lines.append("-" * (w + 48))
        lines.append(
            f"{'TOTAL':<{w}} {self.mass:10.4f} {self.x_cg:9.4f} {self.z_cg:9.4f}"
        )
        lines.append(
            f"\nIxx = {self.Ixx:.6g}   Iyy = {self.Iyy:.6g}   "
            f"Izz = {self.Izz:.6g}   Ixz = {self.Ixz:.6g}  kg m^2"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<MassProperties {self.mass:.4f} kg  "
            f"cg=({self.x_cg:.4f}, {self.y_cg:.4f}, {self.z_cg:.4f}) m>"
        )


def mass_properties(source) -> MassProperties:
    """Mass, centre of gravity and inertias from a component table.

    Parameters
    ----------
    source : Aircraft or sequence of Component
        The rows.  A component with ``distributed="span"`` is treated as spread
        uniformly over ``span`` metres rather than concentrated at a point --
        which does not move the centre of gravity, but does change ``Ixx`` and
        ``Izz`` substantially, because a wing's mass is most of the roll
        inertia and it is nowhere near the centreline.

    Returns
    -------
    MassProperties

    Notes
    -----
    Point masses contribute ``m d^2`` about the centre of gravity.  A mass
    spread uniformly over a span ``s``, centred on the aircraft's plane of
    symmetry, contributes an additional ``m s^2 / 12`` about the axes it is
    not aligned with -- the standard result for a uniform rod, and the reason a
    long-span sailplane's roll inertia is dominated by the wing even when the
    fuselage is heavier.
    """
    comps = tuple(source.components if isinstance(source, Aircraft) else source)
    if not comps:
        label = getattr(source, "label", "this aircraft")
        raise ValueError(
            f"{label} has no component mass table, so mass properties cannot "
            "be computed.  fleet.Aircraft.mass holds published totals instead."
        )

    m = np.array([c.mass for c in comps])
    x = np.array([c.x for c in comps])
    y = np.array([c.y for c in comps])
    z = np.array([c.z for c in comps])
    total = float(m.sum())

    x_cg = float((m * x).sum() / total)
    y_cg = float((m * y).sum() / total)
    z_cg = float((m * z).sum() / total)

    dx, dy, dz = x - x_cg, y - y_cg, z - z_cg
    Ixx = float((m * (dy**2 + dz**2)).sum())
    Iyy = float((m * (dx**2 + dz**2)).sum())
    Izz = float((m * (dx**2 + dy**2)).sum())
    Ixz = float((m * dx * dz).sum())

    # Geometry-attached project components may already carry their inertia
    # about their own centroid. Ordinary fleet point masses leave these zero.
    Ixx += float(sum(getattr(c, "Ixx_cg", 0.0) for c in comps))
    Iyy += float(sum(getattr(c, "Iyy_cg", 0.0) for c in comps))
    Izz += float(sum(getattr(c, "Izz_cg", 0.0) for c in comps))
    Ixz += float(sum(getattr(c, "Ixz_cg", 0.0) for c in comps))

    # spread the distributed rows along their own span
    for c in comps:
        if c.distributed == "span" and c.span:
            spread = c.mass * c.span**2 / 12.0
            Ixx += spread
            Izz += spread

    return MassProperties(total, x_cg, y_cg, z_cg, Ixx, Iyy, Izz, Ixz, comps)


# --- tail volumes and the body correction -----------------------------------


def tail_volume(aircraft: Aircraft, x_cg: Optional[float] = None) -> Dict[str, float]:
    """Horizontal and vertical tail volume coefficients.

    ``V_h = S_h l_h / (S c)`` and ``V_v = S_v l_v / (S b)``, with the moment
    arms measured from the centre of gravity to each tail's quarter-chord MAC.

    Typical values: 0.3-0.6 horizontal and 0.02-0.05 vertical for a
    conventional aircraft.  They are the first thing to look at when a tail
    seems the wrong size, and the last thing to trust when it seems right --
    a tail volume says nothing about where the centre of gravity is.
    """
    w = geom.resolve(aircraft.wing)
    if x_cg is None:
        x_cg = mass_properties(aircraft).x_cg
    out = {"x_cg": x_cg}
    if aircraft.htail is not None:
        h = geom.resolve(aircraft.htail)
        l_h = h.x_c4_mac - x_cg
        out["l_h"] = l_h
        out["V_h"] = h.area * l_h / (w.area * w.mac)
    if aircraft.vtail is not None:
        v = geom.resolve(aircraft.vtail)
        l_v = v.x_c4_mac - x_cg
        out["l_v"] = l_v
        out["V_v"] = v.area * l_v / (w.area * w.span)
    return out


def body_pitching_moment(aircraft: Aircraft, panel: Optional[Panel] = None) -> float:
    """``dCm/dalpha`` contributed by the bodies, per radian.  Positive is destabilizing.

    The Munk slender-body result, in the practical form used for preliminary
    design: a body in a flow at incidence carries a nose-up moment
    proportional to its volume, and a fuselage ahead of the wing is therefore
    destabilizing.

    ``Cm_alpha_body = (k2 - k1) * integral(w^2 dx) / (S c)``, approximated here
    as ``2 * V_body / (S c)`` with the apparent-mass factor folded in -- good
    to perhaps 20%, which is the accuracy of every fuselage stability estimate
    at this stage.

    The RC-1's pod is the case that matters in this course: the lifting-surface
    neutral point puts the static margin at about 23%, and the pod eats several
    points of that.  Which way it moves is more important than the exact
    number, and the sign is not in doubt.
    """
    w = panel or geom.resolve(aircraft.wing)
    total = 0.0
    for body in aircraft.bodies:
        try:
            f = body.fineness
        except ValueError:
            continue
        d = body.length / f
        volume = 0.7 * (np.pi / 4.0) * d**2 * body.length * body.count
        total += 2.0 * volume / (w.area * w.mac)
    return float(total)


def body_derivatives(aircraft: Aircraft, x_cg: Optional[float] = None) -> Dict[str, float]:
    """Empirical non-dimensional stability-derivative increments from bodies.

    The static pitching increment is the Munk slender-body estimate used by
    :func:`body_pitching_moment`. Lateral force and damping increments use a
    strip crossflow model over each body's side projection. The yawing static
    derivative also includes the destabilizing Munk volume term. These are
    preliminary-design corrections, not a coupled body-panel solution.
    """
    w = geom.resolve(aircraft.wing)
    if x_cg is None:
        try:
            x_cg = mass_properties(aircraft).x_cg
        except ValueError:
            x_cg = w.x_c4_mac
    values = {
        "Cm_alpha": body_pitching_moment(aircraft, w),
        "CY_beta": 0.0, "Cl_beta": 0.0, "Cn_beta": 0.0,
        "CY_p": 0.0, "CY_r": 0.0, "Cl_p": 0.0,
        "Cn_p": 0.0, "Cl_r": 0.0, "Cn_r": 0.0,
    }
    for body in aircraft.bodies:
        diameter = body.diameter if body.diameter is not None else body.width
        height = body.height if body.height is not None else diameter
        width = body.width if body.width is not None else diameter
        if diameter is None or height is None or width is None:
            continue
        n = 80
        u = (np.arange(n) + 0.5) / n
        cone = float(np.clip(getattr(body, "cone_fraction", 0.4), 0.0, 1.0))
        end = cone / 2.0
        scale = np.ones_like(u)
        if end > 0:
            scale = np.minimum(scale, u / end)
            scale = np.minimum(scale, (1.0 - u) / end)
        scale = np.clip(scale, 0.0, 1.0)
        x0 = body.x_nose or 0.0
        x = x0 + u * body.length
        lever = (x - x_cg) / w.span
        zbar = float(getattr(body, "z", 0.0)) / w.span
        # Crossflow side-force slope on each projected side-area strip.
        dcy = -2.0 * height * scale * (body.length / n) * body.count / w.area
        values["CY_beta"] += float(np.sum(dcy))
        values["Cl_beta"] += float(np.sum(-zbar * dcy))
        values["Cn_beta"] += float(np.sum(lever * dcy))
        values["CY_p"] += float(np.sum(dcy * (-2.0 * zbar)))
        values["CY_r"] += float(np.sum(dcy * (2.0 * lever)))
        values["Cl_p"] += float(np.sum(dcy * 2.0 * zbar**2))
        values["Cn_p"] += float(np.sum(lever * dcy * (-2.0 * zbar)))
        values["Cl_r"] += float(np.sum(-zbar * dcy * (2.0 * lever)))
        values["Cn_r"] += float(np.sum(dcy * 2.0 * lever**2))
        # Destabilizing yaw counterpart of the Munk pitch-volume term.
        volume = (
            np.pi * width * height / 4.0 * body.length
            * (1.0 - 2.0 * cone / 3.0) * body.count
        )
        values["Cn_beta"] -= 2.0 * volume / (w.area * w.span)
    return {key: float(value) for key, value in values.items()}


# --- neutral point and static margin ----------------------------------------


def neutral_point(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    alpha: float = 2.0,
    ns: int = 30,
    nc: int = 6,
    include_body: bool = False,
    **kwargs,
) -> Dict[str, float]:
    """Longitudinal neutral point, m aft of the wing root leading edge.

    Found from the lattice's own derivatives, ``x_np = x_ref - Cm_alpha /
    CL_alpha * c``, which is independent of where the moment reference was put
    -- a fact worth checking once, because if it is not independent the
    reference length or the moment normalization is wrong.

    Parameters
    ----------
    aircraft : Aircraft
    V, altitude : float
    alpha : float
        Angle of attack the derivatives are taken at, degrees.
    ns, nc : int
    include_body : bool
        Add the destabilizing body contribution from
        :func:`body_pitching_moment`.  Off by default so the lifting-surface
        number and the corrected one stay distinguishable.

    Returns
    -------
    dict
        ``x_np`` (m), ``x_np_over_mac``, ``CL_alpha``, ``Cm_alpha`` (per
        radian, about the wing MAC quarter chord), and ``body_increment``.
    """
    w = geom.resolve(aircraft.wing)
    x_ref = w.x_c4_mac

    grids, ratios = [], []
    for plan in (aircraft.wing, aircraft.htail):
        if plan is None:
            continue
        p = geom.resolve(plan)
        g, r = geom.surface_grid(p, ns=ns, nc=nc, **kwargs)
        if p.x_le or p.z:
            g = translate(g, (p.x_le, 0.0, p.z))
        grids.append(g)
        ratios.append(r)

    reference = Reference(w.area, w.mac, w.span, [x_ref, 0.0, 0.0], V)
    fs = Freestream.from_degrees(V, alpha=alpha)
    system = steady_analysis(
        grids, reference, fs, symmetric=True, ratios=ratios, derivatives=True
    )
    dCF, dCM = stability_derivatives(system)
    CL_a = float(dCF["alpha"][2])
    Cm_a = float(dCM["alpha"][1])

    body = body_pitching_moment(aircraft, w) if include_body else 0.0
    x_np = x_ref - (Cm_a + body) / CL_a * w.mac

    return {
        "x_np": float(x_np),
        "x_np_over_mac": float((x_np - (w.x_le + w.x_mac)) / w.mac),
        "CL_alpha": CL_a,
        "Cm_alpha": Cm_a,
        "body_increment": float(body),
        "x_ref": x_ref,
    }


def static_margin(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    x_cg: Optional[float] = None,
    **kwargs,
) -> float:
    """``(x_np - x_cg) / mac``.  Positive is statically stable."""
    w = geom.resolve(aircraft.wing)
    if x_cg is None:
        x_cg = mass_properties(aircraft).x_cg
    np_ = neutral_point(aircraft, V, altitude, **kwargs)
    return float((np_["x_np"] - x_cg) / w.mac)


# --- trim -------------------------------------------------------------------


@dataclass(frozen=True)
class Trim:
    """A trimmed longitudinal flight condition.

    Attributes
    ----------
    alpha : float
        Body angle of attack, degrees.
    tail_incidence : float
        Horizontal tail incidence that trims, degrees.
    V, altitude, mass, x_cg : float
    CL, CD_i, Cm : float
    x_np, static_margin : float
    lift_residual, moment_residual : float
        How closely the solve closed.  Both should be at solver precision;
        a nonzero one means the trim did not converge and nothing downstream
        of it means anything.
    solution : flightlab.wing.Solution
    """

    alpha: float
    tail_incidence: float
    V: float
    altitude: float
    mass: float
    x_cg: float
    CL: float
    CD_i: float
    Cm: float
    lift_residual: float
    moment_residual: float
    solution: object = field(default=None, repr=False)
    _np_args: tuple = field(default=(), repr=False, compare=False)

    @property
    def converged(self) -> bool:
        """Whether both residuals closed to a sensible tolerance."""
        return abs(self.lift_residual) < 1e-6 and abs(self.moment_residual) < 1e-8

    @property
    def neutral_point_data(self) -> Dict[str, float]:
        """The neutral point solve, computed on first access and remembered.

        Trimming needs only the lift and moment balance; the neutral point
        needs a second solve with derivatives switched on, which costs more
        than the trim itself.  Nothing pays for it unless it is asked for.
        """
        cached = self.__dict__.get("_np_cache")
        if cached is None:
            aircraft, V, altitude, ns, nc, include_body = self._np_args
            cached = neutral_point(
                aircraft, V, altitude, ns=ns, nc=nc, include_body=include_body
            )
            object.__setattr__(self, "_np_cache", cached)
        return cached

    @property
    def x_np(self) -> float:
        """Neutral point, m aft of the wing root leading edge."""
        return float(self.neutral_point_data["x_np"])

    @property
    def static_margin(self) -> float:
        """``(x_np - x_cg) / mac``.  Positive is statically stable."""
        aircraft = self._np_args[0]
        mac = geom.resolve(aircraft.wing).mac
        return float((self.x_np - self.x_cg) / mac)

    def __repr__(self) -> str:  # pragma: no cover - display only
        sm = self.__dict__.get("_np_cache")
        tail = f"  SM={self.static_margin:.3f}" if sm is not None else ""
        return (
            f"<Trim alpha={self.alpha:.3f} deg  i_t={self.tail_incidence:.3f} deg  "
            f"CL={self.CL:.4f}{tail}>"
        )


def trim(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    mass: Optional[float] = None,
    x_cg: Optional[float] = None,
    n: float = 1.0,
    ns: int = 30,
    nc: int = 6,
    guess: Tuple[float, float] = (2.0, -2.0),
    include_body: bool = False,
) -> Trim:
    """Trim the aircraft: lift equals weight and pitching moment is zero.

    Solves simultaneously for angle of attack and horizontal tail incidence.
    Two equations, two unknowns, and both have to close -- an aircraft that
    makes its weight in lift but not zero moment is not in steady flight, it
    is pitching.

    Parameters
    ----------
    aircraft : Aircraft
    V, altitude : float
    mass : float, optional
        Defaults to the component table total.
    x_cg : float, optional
        Defaults to the component table centre of gravity.
    n : float
        Load factor.
    ns, nc : int
    guess : tuple
        Starting ``(alpha, tail_incidence)`` in degrees.

    Returns
    -------
    Trim
    """
    if aircraft.htail is None:
        raise ValueError(
            f"{aircraft.label} has no horizontal tail, so there is nothing to "
            "trim with; use flightlab.wing.trim_to_weight for a wing alone"
        )
    mp = None
    if mass is None or x_cg is None:
        mp = mass_properties(aircraft)
    mass = mp.mass if mass is None else mass
    x_cg = mp.x_cg if x_cg is None else x_cg

    w = geom.resolve(aircraft.wing)
    air = atmos.at(altitude)
    q = air.q(V)
    CL_req = n * mass * G0 / (q * w.area)
    body_slope = body_pitching_moment(aircraft, w) if include_body else 0.0

    def solve(alpha, i_t):
        return wing.analyze(
            aircraft, alpha, V, altitude, ns=ns, nc=nc, tail=True,
            tail_incidence_deg=i_t, x_ref=x_cg,
        )

    # The lattice is exactly linear in angle of attack, and very nearly linear
    # in tail incidence -- "nearly" because incidence rotates the panels, which
    # is a trigonometric change in geometry rather than a change of freestream
    # direction.  So: build the 2x2 Jacobian once with three solves, then take
    # Newton steps reusing it.  Two or three steps close it to machine
    # precision, for five or six solves in total against the twenty-odd a
    # general-purpose root finder needs.
    alpha, i_t = float(guess[0]), float(guess[1])
    da = di = 1.0
    s00 = solve(alpha, i_t)
    s10 = solve(alpha + da, i_t)
    s01 = solve(alpha, i_t + di)

    def total_cm(solution, alpha_deg):
        return solution.Cm + body_slope * np.radians(alpha_deg)

    J = np.array(
        [
            [(s10.CL - s00.CL) / da, (s01.CL - s00.CL) / di],
            [
                (total_cm(s10, alpha + da) - total_cm(s00, alpha)) / da,
                (total_cm(s01, alpha) - total_cm(s00, alpha)) / di,
            ],
        ]
    )
    if abs(np.linalg.det(J)) < 1e-14:
        raise ValueError(
            "the trim Jacobian is singular: angle of attack and tail "
            "incidence are not independently controlling lift and moment.  "
            "Usually this means the horizontal tail has no moment arm."
        )
    J_inv = np.linalg.inv(J)

    sol = s00
    for _ in range(12):
        residual = np.array([sol.CL - CL_req, total_cm(sol, alpha)])
        if abs(residual[0]) < 1e-10 and abs(residual[1]) < 1e-11:
            break
        step = J_inv @ (-residual)
        alpha += float(step[0])
        i_t += float(step[1])
        sol = solve(alpha, i_t)
    else:
        raise ValueError(
            f"trim did not converge for {aircraft.label} at {V} m/s: "
            f"residuals {sol.CL - CL_req:.3e} in lift and {total_cm(sol, alpha):.3e} in "
            "moment after 12 Newton steps"
        )
    return Trim(
        alpha=alpha,
        tail_incidence=i_t,
        V=float(V),
        altitude=float(altitude),
        mass=float(mass),
        x_cg=float(x_cg),
        CL=sol.CL,
        CD_i=sol.CD_i,
        Cm=total_cm(sol, alpha),
        lift_residual=sol.CL - CL_req,
        moment_residual=total_cm(sol, alpha),
        solution=sol,
        _np_args=(aircraft, V, altitude, ns, nc, include_body),
    )


# --- derivatives ------------------------------------------------------------


@dataclass(frozen=True)
class Derivatives:
    """Non-dimensional stability derivatives in the stability frame.

    Longitudinal derivatives are per radian of ``alpha`` and per
    non-dimensional pitch rate ``q c / (2V)``.  Lateral derivatives are per
    radian of ``beta`` and per ``p b / (2V)`` and ``r b / (2V)``.

    Attributes
    ----------
    CL, CD, Cm : float
        At the condition itself.
    CL_alpha, CD_alpha, Cm_alpha : float
    CL_q, Cm_q : float
    CY_beta, Cl_beta, Cn_beta : float
    Cl_p, Cn_p, Cl_r, Cn_r, CY_p, CY_r : float
    CL_alphadot, Cm_alphadot : float
        Downwash-lag derivatives, estimated rather than solved; see
        :func:`alpha_dot_derivatives`.
    lateral_valid : bool
        False when the solve was symmetric, in which case every lateral
        derivative is exactly zero by construction and means nothing.
    """

    CL: float
    CD: float
    Cm: float
    CL_alpha: float
    CD_alpha: float
    Cm_alpha: float
    CL_q: float
    Cm_q: float
    CY_beta: float = 0.0
    Cl_beta: float = 0.0
    Cn_beta: float = 0.0
    Cl_p: float = 0.0
    Cn_p: float = 0.0
    Cl_r: float = 0.0
    Cn_r: float = 0.0
    CY_p: float = 0.0
    CY_r: float = 0.0
    CL_alphadot: float = 0.0
    Cm_alphadot: float = 0.0
    lateral_valid: bool = False

    def table(self) -> str:
        """A printable derivative table."""
        rows = [
            ("CL", self.CL), ("CD", self.CD), ("Cm", self.Cm),
            ("CL_alpha", self.CL_alpha), ("CD_alpha", self.CD_alpha),
            ("Cm_alpha", self.Cm_alpha),
            ("CL_q", self.CL_q), ("Cm_q", self.Cm_q),
            ("CL_alphadot", self.CL_alphadot), ("Cm_alphadot", self.Cm_alphadot),
        ]
        lat = [
            ("CY_beta", self.CY_beta), ("Cl_beta", self.Cl_beta),
            ("Cn_beta", self.Cn_beta), ("Cl_p", self.Cl_p),
            ("Cn_p", self.Cn_p), ("Cl_r", self.Cl_r), ("Cn_r", self.Cn_r),
        ]
        lines = ["longitudinal", "------------"]
        lines += [f"  {k:<14} {v:11.6f}" for k, v in rows]
        lines += ["", "lateral-directional", "-------------------"]
        if not self.lateral_valid:
            lines.append("  (solved symmetric -- every value below is exactly")
            lines.append("   zero by construction and means nothing)")
        lines += [f"  {k:<14} {v:11.6f}" for k, v in lat]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<Derivatives CL_alpha={self.CL_alpha:.3f}  "
            f"Cm_alpha={self.Cm_alpha:.3f}  "
            f"{'lateral ok' if self.lateral_valid else 'longitudinal only'}>"
        )


def alpha_dot_derivatives(
    aircraft: Aircraft, CL_alpha_tail: float = 4.5, downwash: float = 0.4
) -> Tuple[float, float]:
    """Estimated ``CL_alphadot`` and ``Cm_alphadot``, per radian.

    A steady vortex lattice cannot produce these: they exist because the
    downwash arriving at the tail reflects the wing's angle of attack one flow
    time ago, and a steady solver has no memory.  The standard estimate is

        ``CL_alphadot = 2 * a_t * eta * V_h * de/dalpha``
        ``Cm_alphadot = -CL_alphadot * l_h / c``

    They matter for the short period's damping and hardly at all for anything
    else, so an estimate is the right level of effort -- but the number is an
    estimate and the mode damping inherits that.
    """
    if aircraft.htail is None:
        return 0.0, 0.0
    tv = tail_volume(aircraft)
    w = geom.resolve(aircraft.wing)
    V_h = tv.get("V_h", 0.0)
    CL_ad = 2.0 * CL_alpha_tail * V_h * downwash
    Cm_ad = -CL_ad * tv.get("l_h", 0.0) / w.mac
    return float(CL_ad), float(Cm_ad)


def derivatives(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    alpha: float = 2.0,
    beta: float = 0.0,
    x_cg: Optional[float] = None,
    ns: int = 30,
    nc: int = 6,
    lateral: bool = True,
    tail_incidence_deg: Optional[float] = None,
) -> Derivatives:
    """All the stability derivatives at one condition.

    Parameters
    ----------
    aircraft : Aircraft
    V, altitude, alpha, beta : float
    x_cg : float, optional
        Moment reference.  Defaults to the component table centre of gravity,
        which is where derivatives have to be taken for a dynamics model.
    ns, nc : int
    lateral : bool
        Build a mirrored, non-symmetric model so the lateral derivatives are
        real.  Costs about four times the solve time.  **Turn it off only if
        you do not need lateral results**, never to save time and then read the
        zeros.
    tail_incidence_deg : float, optional

    Returns
    -------
    Derivatives
    """
    w = geom.resolve(aircraft.wing)
    if x_cg is None:
        try:
            x_cg = mass_properties(aircraft).x_cg
        except ValueError:
            x_cg = w.x_c4_mac

    grids, ratios = [], []
    for plan, include in (
        (aircraft.wing, True),
        (aircraft.htail, True),
        (aircraft.vtail, lateral),
    ):
        if plan is None or not include:
            continue
        p = geom.resolve(plan)
        if plan is aircraft.htail and tail_incidence_deg is not None:
            p = Panel(**{**p.__dict__, "incidence_deg": tail_incidence_deg})
        g, r = geom.surface_grid(p, ns=ns, nc=nc, mirror=lateral)
        if p.x_le or p.z:
            g = translate(g, (p.x_le, 0.0, p.z))
        grids.append(g)
        ratios.append(r)

    reference = Reference(w.area, w.mac, w.span, [x_cg, 0.0, 0.0], V)
    fs = Freestream.from_degrees(V, alpha=alpha, beta=beta)
    system = steady_analysis(
        grids, reference, fs, symmetric=not lateral, ratios=ratios,
        derivatives=True,
    )
    CF, CM = body_forces(system, frame=Stability())
    dCF, dCM = stability_derivatives(system)
    CL_ad, Cm_ad = alpha_dot_derivatives(aircraft)

    return Derivatives(
        CL=float(CF[2]), CD=float(CF[0]), Cm=float(CM[1]),
        CL_alpha=float(dCF["alpha"][2]),
        CD_alpha=float(dCF["alpha"][0]),
        Cm_alpha=float(dCM["alpha"][1]),
        CL_q=float(dCF["q"][2]),
        Cm_q=float(dCM["q"][1]),
        CY_beta=float(dCF["beta"][1]),
        Cl_beta=float(dCM["beta"][0]),
        Cn_beta=float(dCM["beta"][2]),
        Cl_p=float(dCM["p"][0]),
        Cn_p=float(dCM["p"][2]),
        Cl_r=float(dCM["r"][0]),
        Cn_r=float(dCM["r"][2]),
        CY_p=float(dCF["p"][1]),
        CY_r=float(dCF["r"][1]),
        CL_alphadot=CL_ad,
        Cm_alphadot=Cm_ad,
        lateral_valid=bool(lateral),
    )


# --- dynamic modes ----------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """One dynamic mode: an eigenvalue and what it means.

    Attributes
    ----------
    name : str
    eigenvalue : complex
        1/s.
    """

    name: str
    eigenvalue: complex

    @property
    def real(self) -> float:
        return float(self.eigenvalue.real)

    @property
    def imag(self) -> float:
        return float(abs(self.eigenvalue.imag))

    @property
    def oscillatory(self) -> bool:
        """Whether the mode oscillates rather than simply converging."""
        return self.imag > 1e-9

    @property
    def frequency(self) -> float:
        """Undamped natural frequency, rad/s."""
        return float(abs(self.eigenvalue))

    @property
    def damping(self) -> float:
        """Damping ratio.  Negative means the mode grows."""
        w = self.frequency
        return float(-self.real / w) if w > 1e-12 else float("nan")

    @property
    def period(self) -> float:
        """Period, s.  Infinite for a non-oscillatory mode."""
        return float(2.0 * np.pi / self.imag) if self.oscillatory else float("inf")

    @property
    def time_to_half(self) -> float:
        """Time to halve (or, if unstable, to double) the amplitude, s."""
        return float(np.log(2.0) / abs(self.real)) if abs(self.real) > 1e-12 else float("inf")

    @property
    def stable(self) -> bool:
        return self.real < 0.0

    def __repr__(self) -> str:  # pragma: no cover - display only
        if self.oscillatory:
            return (
                f"<{self.name}: {self.eigenvalue:.4f} /s  "
                f"T={self.period:.2f} s  zeta={self.damping:.4f}  "
                f"{'stable' if self.stable else 'UNSTABLE'}>"
            )
        return (
            f"<{self.name}: {self.real:.4f} /s  "
            f"t_half={self.time_to_half:.2f} s  "
            f"{'stable' if self.stable else 'UNSTABLE'}>"
        )


@dataclass(frozen=True)
class Modes:
    """A set of dynamic modes with the state matrix they came from."""

    modes: Tuple[Mode, ...]
    A: np.ndarray
    states: Tuple[str, ...]

    def __getitem__(self, key):
        if isinstance(key, str):
            for m in self.modes:
                if m.name == key:
                    return m
            raise KeyError(
                f"no mode named {key!r}; this set has "
                f"{[m.name for m in self.modes]}"
            )
        return self.modes[key]

    def __iter__(self):
        return iter(self.modes)

    def __len__(self):
        return len(self.modes)

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.array([m.eigenvalue for m in self.modes])

    @property
    def stable(self) -> bool:
        """Whether every mode converges."""
        return all(m.stable for m in self.modes)

    def table(self) -> str:
        """A printable mode summary."""
        lines = [
            f"{'mode':<14} {'eigenvalue':>26} {'period (s)':>11} "
            f"{'zeta':>8} {'t_half (s)':>11}",
            "-" * 74,
        ]
        for m in self.modes:
            ev = f"{m.eigenvalue.real:+.5f} {m.eigenvalue.imag:+.5f}j"
            per = f"{m.period:.3f}" if m.oscillatory else "--"
            zeta = f"{m.damping:.4f}" if m.oscillatory else "--"
            th = f"{m.time_to_half:.3f}" if np.isfinite(m.time_to_half) else "--"
            flag = "" if m.stable else "  UNSTABLE"
            lines.append(f"{m.name:<14} {ev:>26} {per:>11} {zeta:>8} {th:>11}{flag}")
        return "\n".join(lines)

    def simulate(self, x0, t):
        """Free response from initial state ``x0`` over times ``t``.

        Returns an array of shape ``(len(t), n_states)``.  Linear, so it is
        valid only for perturbations small enough that the derivatives still
        apply -- which for a phugoid is a good deal smaller than the amplitude
        the mode actually reaches.
        """
        t = np.asarray(t, dtype=float)
        vals, vecs = np.linalg.eig(self.A)
        c = np.linalg.solve(vecs, np.asarray(x0, dtype=float))
        return np.real((vecs @ (c[:, None] * np.exp(np.outer(vals, t)))).T)


def longitudinal_modes(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    mass: Optional[float] = None,
    Iyy: Optional[float] = None,
    x_cg: Optional[float] = None,
    derivs: Optional[Derivatives] = None,
    thrust_dT_dV: float = 0.0,
    thrust_dM_dV: float = 0.0,
    **kwargs,
) -> Modes:
    """Phugoid and short-period modes.

    Builds the dimensional longitudinal state matrix in the states
    ``(u, w, q, theta)`` -- forward and vertical velocity perturbations in m/s,
    pitch rate in rad/s, pitch attitude in rad -- and takes its eigenvalues.

    Parameters
    ----------
    aircraft : Aircraft
    V, altitude : float
    mass, Iyy, x_cg : float, optional
        Default to the component table.
    derivs : Derivatives, optional
        Reuse a derivative set rather than re-solving.
    **kwargs
        Passed to :func:`derivatives`.

    Returns
    -------
    Modes
        Two modes named ``"short period"`` and ``"phugoid"``, identified by
        frequency: the short period is the faster pair.  When the roots are
        real rather than complex -- which happens on light aircraft with large
        static margins -- they are named ``"longitudinal 1"`` and
        ``"longitudinal 2"``, because calling a real root a phugoid is a
        category error.

    Notes
    -----
    Compressibility and thrust terms are omitted: ``CD_u``, ``CL_u``, ``Cm_u``
    and the thrust derivatives are all taken as zero.  For an aircraft below
    Mach 0.3 with a fixed throttle that is a good approximation for the short
    period and a poor one for the phugoid, whose damping depends on exactly
    those terms.  The phugoid period is robust; the phugoid damping is not.
    """
    mp = None
    if mass is None or Iyy is None or x_cg is None:
        mp = mass_properties(aircraft)
    mass = mp.mass if mass is None else mass
    Iyy = mp.Iyy if Iyy is None else Iyy
    x_cg = mp.x_cg if x_cg is None else x_cg

    w_panel = geom.resolve(aircraft.wing)
    air = atmos.at(altitude)
    q_bar = air.q(V)
    S, c = w_panel.area, w_panel.mac

    # steady level flight: the reference lift carries the weight.  This is not
    # a modelling choice, it is the definition of the condition the modes are
    # perturbations about, and getting it wrong scales the phugoid frequency
    # by the square root of the error.
    CW = mass * G0 / (q_bar * S)

    if derivs is None:
        derivs = _trimmed_derivatives(
            aircraft, V, altitude, mass=mass, x_cg=x_cg, **kwargs
        )
    d = derivs
    _check_trimmed(d, CW, V)

    CL0 = CW
    CD0 = d.CD

    qS = q_bar * S
    # dimensional derivatives, per the standard small-perturbation form
    Xu = -qS * (2.0 * CD0) / (mass * V) + thrust_dT_dV / mass
    Xw = qS * (CL0 - d.CD_alpha) / (mass * V)
    Zu = -qS * (2.0 * CL0) / (mass * V)
    Zw = -qS * (d.CL_alpha + CD0) / (mass * V)
    Zq = -qS * c * d.CL_q / (2.0 * mass * V)
    Zwdot = -qS * c * d.CL_alphadot / (2.0 * mass * V**2)
    Mu = thrust_dM_dV / Iyy
    Mw = qS * c * d.Cm_alpha / (Iyy * V)
    Mq = qS * c**2 * d.Cm_q / (2.0 * Iyy * V)
    Mwdot = qS * c**2 * d.Cm_alphadot / (2.0 * Iyy * V**2)

    A = np.array(
        [
            [Xu, Xw, 0.0, -G0],
            [Zu / (1.0 - Zwdot), Zw / (1.0 - Zwdot),
             (Zq + V) / (1.0 - Zwdot), 0.0],
            [
                Mu + Mwdot * Zu / (1.0 - Zwdot),
                Mw + Mwdot * Zw / (1.0 - Zwdot),
                Mq + Mwdot * (Zq + V) / (1.0 - Zwdot),
                0.0,
            ],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    vals = np.linalg.eigvals(A)
    return Modes(_name_longitudinal(vals), A, ("u", "w", "q", "theta"))


def _name_longitudinal(vals) -> Tuple[Mode, ...]:
    vals = np.asarray(vals)
    osc = vals[np.abs(vals.imag) > 1e-9]
    if len(osc) >= 4:
        pairs = sorted(
            {complex(round(v.real, 12), round(abs(v.imag), 12)) for v in osc},
            key=lambda z: -abs(z.imag),
        )
        names = ["short period", "phugoid"]
        out = []
        for name, p in zip(names, pairs):
            for v in vals:
                if abs(v.real - p.real) < 1e-9 and abs(abs(v.imag) - p.imag) < 1e-9:
                    out.append(Mode(name, complex(v)))
                    break
        return tuple(out)
    if len(osc) == 2:
        real = vals[np.abs(vals.imag) <= 1e-9]
        pair = osc[np.argmax(osc.imag)]
        name = "short period" if abs(pair.imag) > 0.5 else "phugoid"
        modes = [Mode(name, complex(pair))]
        modes += [
            Mode(f"longitudinal {i + 1}", complex(v)) for i, v in enumerate(real)
        ]
        return tuple(modes)
    return tuple(
        Mode(f"longitudinal {i + 1}", complex(v))
        for i, v in enumerate(sorted(vals, key=lambda z: z.real))
    )


def lateral_modes(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    mass: Optional[float] = None,
    Ixx: Optional[float] = None,
    Izz: Optional[float] = None,
    Ixz: Optional[float] = None,
    x_cg: Optional[float] = None,
    derivs: Optional[Derivatives] = None,
    **kwargs,
) -> Modes:
    """Dutch roll, roll subsidence and spiral modes.

    States are ``(v, p, r, phi)`` -- side velocity in m/s, roll and yaw rate in
    rad/s, bank angle in rad.

    The three modes are identified by character rather than by position:
    the oscillatory pair is the dutch roll, the fastest real root is roll
    subsidence, and what remains is the spiral -- which is very often mildly
    unstable on a real aircraft and is not a defect.  A slowly diverging
    spiral is a normal design outcome; the number that matters is its time to
    double, not its sign.

    Requires lateral derivatives, so :func:`derivatives` is called with
    ``lateral=True`` and a mirrored model.
    """
    mp = None
    if any(v is None for v in (mass, Ixx, Izz, Ixz, x_cg)):
        mp = mass_properties(aircraft)
    mass = mp.mass if mass is None else mass
    Ixx = mp.Ixx if Ixx is None else Ixx
    Izz = mp.Izz if Izz is None else Izz
    Ixz = mp.Ixz if Ixz is None else Ixz
    x_cg = mp.x_cg if x_cg is None else x_cg

    if derivs is None:
        derivs = _trimmed_derivatives(
            aircraft, V, altitude, mass=mass, x_cg=x_cg, lateral=True, **kwargs
        )
    d = derivs
    if not d.lateral_valid:
        raise ValueError(
            "these derivatives came from a symmetric solve, so every lateral "
            "value in them is exactly zero; re-run derivatives(lateral=True)"
        )

    w = geom.resolve(aircraft.wing)
    air = atmos.at(altitude)
    q_bar = air.q(V)
    S, b = w.area, w.span
    qS = q_bar * S

    Yv = qS * d.CY_beta / (mass * V)
    Yp = qS * b * d.CY_p / (2.0 * mass * V)
    Yr = qS * b * d.CY_r / (2.0 * mass * V)
    Lv = qS * b * d.Cl_beta / (Ixx * V)
    Lp = qS * b**2 * d.Cl_p / (2.0 * Ixx * V)
    Lr = qS * b**2 * d.Cl_r / (2.0 * Ixx * V)
    Nv = qS * b * d.Cn_beta / (Izz * V)
    Np = qS * b**2 * d.Cn_p / (2.0 * Izz * V)
    Nr = qS * b**2 * d.Cn_r / (2.0 * Izz * V)

    # inertial cross-coupling through Ixz
    g1 = Ixz / Ixx
    g2 = Ixz / Izz
    den = 1.0 - g1 * g2
    Lv, Lp, Lr = (Lv + g1 * Nv) / den, (Lp + g1 * Np) / den, (Lr + g1 * Nr) / den
    Nv, Np, Nr = (Nv + g2 * Lv) / den, (Np + g2 * Lp) / den, (Nr + g2 * Lr) / den

    A = np.array(
        [
            [Yv, Yp, Yr - V, G0],
            [Lv, Lp, Lr, 0.0],
            [Nv, Np, Nr, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    vals = np.linalg.eigvals(A)
    return Modes(_name_lateral(vals), A, ("v", "p", "r", "phi"))


def _name_lateral(vals) -> Tuple[Mode, ...]:
    vals = np.asarray(vals)
    osc = [v for v in vals if abs(v.imag) > 1e-9]
    real = sorted([v for v in vals if abs(v.imag) <= 1e-9], key=lambda z: z.real)
    out = []
    if osc:
        out.append(Mode("dutch roll", complex(max(osc, key=lambda z: z.imag))))
    if real:
        out.append(Mode("roll subsidence", complex(real[0])))
    for v in real[1:]:
        out.append(Mode("spiral", complex(v)))
    if not out:  # pragma: no cover
        out = [Mode(f"lateral {i}", complex(v)) for i, v in enumerate(vals)]
    return tuple(out)


def _trimmed_derivatives(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    mass: Optional[float] = None,
    x_cg: Optional[float] = None,
    ns: int = 30,
    nc: int = 6,
    **kwargs,
) -> Derivatives:
    """Derivatives taken at the trimmed attitude, which is where they belong.

    A dynamics model linearizes about a steady flight condition.  Taking the
    derivatives at some convenient angle of attack instead gives a reference
    lift coefficient that does not carry the weight, and the phugoid frequency
    -- which depends on it directly -- comes out wrong.
    """
    if aircraft.htail is not None:
        t = trim(aircraft, V, altitude, mass=mass, x_cg=x_cg, ns=ns, nc=nc)
        return derivatives(
            aircraft, V, altitude, alpha=t.alpha, x_cg=t.x_cg,
            tail_incidence_deg=t.tail_incidence, ns=ns, nc=nc, **kwargs
        )
    sol = wing.trim_to_weight(aircraft.wing, mass, V, altitude, ns=ns, nc=nc)
    return derivatives(
        aircraft, V, altitude, alpha=sol.alpha, x_cg=x_cg, ns=ns, nc=nc, **kwargs
    )


def _check_trimmed(d: Derivatives, CW: float, V: float) -> None:
    """Warn loudly when the derivatives were not taken at the trim condition."""
    if abs(d.CL - CW) > 0.05 * max(abs(CW), 1e-9):
        import warnings

        warnings.warn(
            f"the supplied derivatives were taken at CL = {d.CL:.4f}, but "
            f"level flight at {V:.3g} m/s needs CL = {CW:.4f}.  The mode "
            "eigenvalues assume the reference condition carries the weight; "
            "pass derivatives taken at trim, or let this function trim for "
            "you by omitting derivs.",
            RuntimeWarning,
            stacklevel=3,
        )


def modes(aircraft: Aircraft, V: float, altitude: float = 0.0, **kwargs):
    """Both mode sets: ``(longitudinal, lateral)``.

    Trims the aircraft, takes one mirrored derivative set at that condition,
    and reuses it for both -- which is most of the cost.
    """
    d = _trimmed_derivatives(aircraft, V, altitude, lateral=True, **kwargs)
    return (
        longitudinal_modes(aircraft, V, altitude, derivs=d),
        lateral_modes(aircraft, V, altitude, derivs=d),
    )
