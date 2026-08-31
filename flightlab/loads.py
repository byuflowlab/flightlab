"""``flightlab.loads`` -- flight envelope loads, span loads, and spar sizing.

The structural half of the course, which is short by design.  Three questions:
how hard is the wing pulled, how is that load distributed along the span, and
is the spar big enough.

    >>> from flightlab import loads
    >>> from flightlab.fleet import ASW27
    >>> vn = loads.vn_diagram(ASW27, mass=525.0, CL_max=1.4)
    >>> round(vn["V_A"], 1), vn["n_pos"]
    (44.9, 5.3)

A note on scale
---------------
Load factors do not care how big the aircraft is, but stresses do, and not in
the way intuition suggests.  Hold the shape and the load factor fixed and scale
the aircraft up: weight goes as the cube of the length, section modulus as the
cube, but the bending moment as the *fourth* power, because the moment arm
grows too.  So stress grows linearly with size.  This is why a 750 g foam
model has no structural problem worth solving and a 15 m sailplane's entire
design is its spar -- and it is the point of comparing the two.

Units
-----
SI: newtons, metres, pascals, kg.  Angles in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from . import atmos, geom, wing
from .fleet import Aircraft
from .geom import Panel

__all__ = [
    "SpanLoad",
    "vn_diagram",
    "gust_load_factor",
    "span_load",
    "elliptical_root_bending_moment",
    "spar_stress",
    "spar_sizing",
    "tip_deflection",
]

G0 = 9.80665


# --- the V-n diagram --------------------------------------------------------


def vn_diagram(
    aircraft: Aircraft,
    mass: Optional[float] = None,
    CL_max: Optional[float] = None,
    CL_min: Optional[float] = None,
    altitude: float = 0.0,
    n_pos: Optional[float] = None,
    n_neg: Optional[float] = None,
    V_max: Optional[float] = None,
    n_points: int = 200,
    reference_area: Optional[float] = None,
) -> Dict[str, object]:
    """The manoeuvre envelope: load factor against speed.

    Below the corner speed the wing stalls before it reaches the limit load
    factor, so the aerodynamics protect the structure.  Above it they do not,
    and the limit is whatever the certification basis says.  The corner speed
    ``V_A`` is where those two boundaries meet, and it is the fastest the
    aircraft can be manoeuvred at full control deflection without either
    stalling or breaking.

    Parameters
    ----------
    aircraft : Aircraft
    mass : float, optional
        Defaults to the gross mass on record.
    CL_max, CL_min : float, optional
        Positive and negative maximum lift coefficients.  ``CL_min`` defaults
        to ``-0.6 CL_max``, the usual asymmetry of a cambered section.
    altitude : float
    n_pos, n_neg : float, optional
        Limit load factors.  Default to the aircraft's published limits.
    V_max : float, optional
        Dive speed.  Defaults to ``1.4 V_cruise``.
    n_points : int
    reference_area : float, optional
        Wing/reference area used for wing loading. Defaults to the area of
        ``aircraft.wing``. Pass the aircraft-level reference area for a
        multi-surface project such as a biplane.

    Returns
    -------
    dict
        ``V_stall``, ``V_A`` (corner speed), ``V_max``, ``n_pos``, ``n_neg``,
        and the boundary curves ``V_upper``/``n_upper`` and
        ``V_lower``/``n_lower`` ready to plot.

    Notes
    -----
    Limit load is what the structure must carry without permanent deformation;
    **ultimate** load is 1.5 times that, and is what it must carry without
    breaking.  Sizing a spar to limit load is a design error, and the factor of
    1.5 is not a safety factor in the usual sense -- it is a fixed regulatory
    multiplier that has been 1.5 since the 1930s.
    """
    p = geom.resolve(aircraft.wing)
    mass = mass if mass is not None else (
        aircraft.mass.get("gross") or aircraft.mass.get("mtow")
    )
    if mass is None:
        raise ValueError(f"{aircraft.label} has no gross mass on record")
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass!r} kg")
    CL_max = CL_max if CL_max is not None else aircraft.placeholders.get("CLmax")
    if CL_max is None:
        raise ValueError(
            f"{aircraft.label} has no CL_max on record; compute one with "
            "flightlab.wing.CL_max or pass it"
        )
    if CL_max <= 0:
        raise ValueError(f"CL_max must be positive; got {CL_max!r}")
    CL_min = -0.6 * CL_max if CL_min is None else CL_min
    n_pos = n_pos if n_pos is not None else aircraft.limits.get("n_pos", 3.8)
    n_neg = n_neg if n_neg is not None else aircraft.limits.get("n_neg", -1.5)
    if CL_min >= 0:
        raise ValueError(f"CL_min must be negative; got {CL_min!r}")
    if n_pos <= 0 or n_neg >= 0:
        raise ValueError(
            f"limit factors must have n_pos > 0 and n_neg < 0; got {n_pos}, {n_neg}"
        )

    air = atmos.at(altitude)
    W = mass * G0
    S_ref = p.area if reference_area is None else float(reference_area)
    if S_ref <= 0:
        raise ValueError(f"reference_area must be positive; got {S_ref!r} m^2")
    ws = W / S_ref

    V_stall = float(np.sqrt(2.0 * ws / (air.density * CL_max)))
    V_A = float(V_stall * np.sqrt(abs(n_pos)))
    V_S_neg = float(np.sqrt(2.0 * ws / (air.density * abs(CL_min))))
    V_G = float(V_S_neg * np.sqrt(abs(n_neg)))
    if V_max is None:
        V_c = aircraft.operating.get("cruise_speed", 1.5 * V_stall)
        V_max = 1.4 * V_c
    if V_max <= 0:
        raise ValueError(f"V_max must be positive; got {V_max!r} m/s")

    # upper boundary: stall parabola to V_A, then the limit to V_max
    V_par = np.linspace(0.0, V_A, n_points // 2)
    n_par = 0.5 * air.density * V_par**2 * CL_max / ws
    V_upper = np.concatenate([V_par, [V_max]])
    n_upper = np.concatenate([n_par, [n_pos]])

    V_par_n = np.linspace(0.0, V_G, n_points // 2)
    n_par_n = -0.5 * air.density * V_par_n**2 * abs(CL_min) / ws
    V_lower = np.concatenate([V_par_n, [V_max]])
    n_lower = np.concatenate([n_par_n, [n_neg]])

    return {
        "V_stall": V_stall,
        "V_A": V_A,
        "V_G": V_G,
        "V_max": float(V_max),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "n_ultimate_pos": float(1.5 * n_pos),
        "n_ultimate_neg": float(1.5 * n_neg),
        "wing_loading": float(ws),
        "V_upper": V_upper,
        "n_upper": n_upper,
        "V_lower": V_lower,
        "n_lower": n_lower,
        "mass": float(mass),
        "altitude": float(altitude),
    }


def gust_load_factor(
    aircraft: Aircraft,
    V,
    gust: float = 15.24,
    mass: Optional[float] = None,
    CL_alpha: float = 5.0,
    altitude: float = 0.0,
    reference_area: Optional[float] = None,
    reference_chord: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Load factor from a sharp-edged gust, with the standard alleviation factor.

    ``n = 1 + (rho * V * CL_alpha * K_g * U) / (2 * W/S)``

    ``K_g`` is the gust alleviation factor, which accounts for the aircraft
    beginning to respond before the gust is fully developed.  A lightly loaded
    aircraft is thrown around more by the same gust, which is why a model
    aircraft in wind is a much rougher ride than an airliner in the same air --
    the wing loading is in the denominator.

    ``gust`` defaults to 15.24 m/s (50 ft/s), the standard rough-air value.
    ``V`` may be one speed or an array of speeds, so the result can be plotted
    directly over a manoeuvre envelope.
    """
    p = geom.resolve(aircraft.wing)
    mass = mass if mass is not None else (
        aircraft.mass.get("gross") or aircraft.mass.get("mtow")
    )
    if mass is None:
        raise ValueError(f"{aircraft.label} has no gross mass on record; pass mass")
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass!r} kg")
    if CL_alpha <= 0:
        raise ValueError(f"CL_alpha must be positive and in 1/rad; got {CL_alpha!r}")
    V_arr = np.asarray(V, dtype=float)
    if np.any(V_arr < 0):
        raise ValueError("V must contain nonnegative true airspeeds")
    S_ref = p.area if reference_area is None else float(reference_area)
    c_ref = p.mac if reference_chord is None else float(reference_chord)
    if S_ref <= 0 or c_ref <= 0:
        raise ValueError("reference_area and reference_chord must be positive")
    air = atmos.at(altitude)
    ws = mass * G0 / S_ref
    mu = 2.0 * ws / (air.density * c_ref * CL_alpha * G0)
    K_g = 0.88 * mu / (5.3 + mu)
    result = 1.0 + air.density * V_arr * CL_alpha * K_g * gust / (2.0 * ws)
    return float(result) if result.ndim == 0 else result


# --- span loads -------------------------------------------------------------


@dataclass(frozen=True)
class SpanLoad:
    """Distributed lift and the shear and bending moment it produces.

    Attributes
    ----------
    y : ndarray
        Span stations from root to tip, m.
    lift : ndarray
        Running lift per unit span, N/m.
    shear : ndarray
        Shear force, N, integrated inboard from the tip.
    moment : ndarray
        Bending moment, N m, integrated inboard from the tip.
    n : float
        Load factor this was computed at.
    total_lift : float
        Net load integrated over both sides of the selected surface, N.
    root_moment : float
        Bending moment at the selected surface root, N m.
    """

    y: np.ndarray
    lift: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    n: float
    total_lift: float
    root_moment: float

    @property
    def root_shear(self) -> float:
        """Shear at the root, N -- half the total lift for a symmetric wing."""
        return float(self.shear[0])

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<SpanLoad n={self.n:.2f}  root shear={self.root_shear:.1f} N  "
            f"root moment={self.root_moment:.1f} N m>"
        )


def span_load(
    aircraft: Aircraft,
    mass: float,
    n: float = 1.0,
    V: Optional[float] = None,
    altitude: float = 0.0,
    ns: int = 60,
    relief: Optional[np.ndarray] = None,
    solution: Optional["wing.Solution"] = None,
    surface: Optional[str] = None,
) -> SpanLoad:
    """Running lift, shear and bending moment along the semispan.

    The lift distribution comes from the vortex lattice at the trimmed
    condition, so it is the real distribution rather than an assumed elliptical
    one -- which matters, because a tapered wing carries proportionally more
    load inboard and a lower root bending moment than the elliptical estimate.

    Parameters
    ----------
    aircraft : Aircraft
    mass : float
    n : float
        Load factor.
    V, altitude : float
        The condition.  ``V`` defaults to the cruise speed on record.
    ns : int
        Span stations.
    relief : ndarray, optional
        Running weight per unit span of the wing structure and anything
        carried in it, N/m, at the same stations.  Subtracted from the lift
        before integrating.  This is inertial relief, and on a sailplane with
        water ballast or an airliner with fuel in the wing it removes a large
        fraction of the root bending moment -- which is why those aircraft
        carry mass out there on purpose.
    solution : flightlab.wing.Solution, optional
        Reuse an existing solve.
    surface : str, optional
        Name of the lifting surface to integrate when ``solution`` contains
        more than one. Defaults to the first surface in the solution.

    Returns
    -------
    SpanLoad
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass!r} kg")
    V = V if V is not None else aircraft.operating.get("cruise_speed")
    if V is None:
        raise ValueError(f"{aircraft.label} has no cruise speed on record; pass V")
    if V <= 0:
        raise ValueError(f"V must be positive; got {V!r} m/s")

    sol = solution or wing.trim_to_weight(
        aircraft.wing, mass, V, altitude, n=n, ns=ns
    )
    surface_name = sol.surfaces[0] if surface is None else surface
    if surface_name not in sol.surface_slices:
        available = ", ".join(sol.surfaces)
        raise ValueError(
            f"unknown solution surface {surface_name!r}; choose one of: {available}"
        )
    s = sol.surface_slices[surface_name]
    y, ccl, ds = sol.y[s], sol.ccl[s], sol.ds[s]

    # one side only, root to tip
    right = y >= 0
    y_r = y[right]
    order = np.argsort(y_r)
    y_r = y_r[order]
    running = (ccl[right][order]) * sol.q  # N per metre of span

    if relief is not None:
        relief_arr = np.asarray(relief, dtype=float)
        if relief_arr.shape != running.shape:
            raise ValueError(
                "relief must have one value at each semispan station "
                f"(expected shape {running.shape}, got {relief_arr.shape})"
            )
        if np.any(~np.isfinite(relief_arr)):
            raise ValueError("relief must contain only finite values in N/m")
        running = running - relief_arr

    # integrate inboard from the tip
    shear = np.array(
        [np.trapezoid(running[i:], y_r[i:]) for i in range(len(y_r))]
    )
    moment = np.array(
        [np.trapezoid(running[i:] * (y_r[i:] - y_r[i]), y_r[i:]) for i in range(len(y_r))]
    )

    return SpanLoad(
        y=y_r,
        lift=running,
        shear=shear,
        moment=moment,
        n=float(n),
        # This is the selected symmetric surface's net load, not the complete
        # aircraft lift when a multi-surface solution was supplied.
        total_lift=float(2.0 * shear[0]),
        root_moment=float(moment[0]),
    )


def elliptical_root_bending_moment(lift: float, span: float) -> float:
    """Root bending moment of an elliptical distribution, N m.

    ``M = L b / (3 pi)`` for the whole aircraft's lift ``L`` and span ``b``.
    The closed form worth knowing, because it is the sanity check on any
    computed span load: a tapered wing should come out a little below it, and
    a wing with washout further below still.
    """
    return float(lift * span / (3.0 * np.pi))


# --- spar sizing ------------------------------------------------------------


def spar_stress(
    moment: float,
    height: float,
    cap_area: float,
) -> float:
    """Bending stress in a two-cap spar, Pa.

    ``sigma = M / (A h)``: the caps carry the bending as a force couple, the
    tension cap pulled and the compression cap pushed, separated by the spar
    height.  The shear web carries the shear and contributes essentially
    nothing to bending stiffness, which is why it can be so much thinner.

    This is the idealization that makes spar sizing a one-line calculation, and
    it is accurate to a few percent for a real cap-and-web spar.
    """
    if height <= 0 or cap_area <= 0:
        raise ValueError("height and cap_area must both be positive")
    return float(moment / (cap_area * height))


def spar_sizing(
    moment: float,
    height: float,
    sigma_allow: float,
    safety_factor: float = 1.5,
) -> Dict[str, float]:
    """Cap area needed to carry a bending moment, m^2.

    Parameters
    ----------
    moment : float
        **Limit** bending moment, N m.
    height : float
        Spar height, m -- typically 0.8 to 0.9 of the section thickness at that
        station.
    sigma_allow : float
        Allowable stress, Pa.
    safety_factor : float
        1.5, the regulatory limit-to-ultimate factor.  Applied here, so
        ``moment`` should be the limit value.

    Returns
    -------
    dict
        ``cap_area``, ``ultimate_moment``, ``margin`` -- and note that doubling
        the spar height halves the required cap area, which is why depth is the
        cheapest structural parameter there is and why a thin wing is expensive.
    """
    if moment < 0:
        raise ValueError("moment must be a nonnegative magnitude")
    if height <= 0 or sigma_allow <= 0 or safety_factor <= 0:
        raise ValueError("height, sigma_allow, and safety_factor must be positive")
    M_ult = moment * safety_factor
    area = M_ult / (sigma_allow * height)
    return {
        "cap_area": float(area),
        "ultimate_moment": float(M_ult),
        "limit_moment": float(moment),
        "height": float(height),
        "sigma_allow": float(sigma_allow),
        "safety_factor": float(safety_factor),
    }


def tip_deflection(
    span_load_result: SpanLoad,
    EI,
    semispan: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Bending deflection along the span by double integration, m.

    ``d2w/dy2 = M / EI``, integrated twice from the root with zero deflection
    and zero slope there -- a cantilever built in at the centreline.

    Parameters
    ----------
    span_load_result : SpanLoad
    EI : float or array_like
        Bending stiffness, N m^2.  A constant value models a spar of uniform
        section; an array at the same stations models a tapered one, which is
        what real wings have and which roughly doubles the tip deflection for
        the same root stress.

    Returns
    -------
    dict
        ``y``, ``slope`` (rad), ``deflection`` (m), and ``tip_deflection``.
        Divide the last by the semispan for the number that actually gets
        quoted, since a deflection means nothing without the span it happened
        over.
    """
    y = span_load_result.y
    M = span_load_result.moment
    try:
        EI = np.broadcast_to(np.asarray(EI, dtype=float), y.shape)
    except ValueError as exc:
        raise ValueError(
            f"EI must be scalar or have one value per station (shape {y.shape})"
        ) from exc
    if np.any(~np.isfinite(EI)) or np.any(EI <= 0):
        raise ValueError("EI must contain only positive, finite stiffness values")

    curvature = M / EI
    slope = np.concatenate([[0.0], np.cumsum(np.diff(y) * 0.5 * (curvature[1:] + curvature[:-1]))])
    defl = np.concatenate([[0.0], np.cumsum(np.diff(y) * 0.5 * (slope[1:] + slope[:-1]))])
    b2 = float(y[-1]) if semispan is None else float(semispan)
    if b2 <= 0:
        raise ValueError(f"semispan must be positive; got {b2!r} m")
    return {
        "y": y,
        "curvature": curvature,
        "slope": slope,
        "deflection": defl,
        "tip_deflection": float(defl[-1]),
        "tip_over_semispan": float(defl[-1] / b2),
    }
