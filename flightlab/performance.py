"""``flightlab.performance`` -- what the aircraft can actually do.

Everything in this module is the drag polar crossed with something else.  Cross
it with weight and you get the speed for best glide; with the thrust available
and you get climb rate and ceiling; with a fuel or energy supply and you get
range and endurance.  The polar is the object; this is what you ask it.

    >>> from flightlab import performance as perf
    >>> from flightlab import drag
    >>> from flightlab.fleet import RC1
    >>> pol = drag.polar(RC1, V=12.0, altitude=1400.0, markup=0.15)
    >>> s = perf.speeds(pol, mass=0.75)
    >>> round(s["V_LD_max"], 2), round(s["LD_max"], 2)
    (11.32, 9.35)

Two families, and they are not interchangeable
----------------------------------------------
An **electric** aircraft holds its weight constant and drains a finite energy
store, so its endurance is energy divided by power and its range is that times
speed.  A **fuel-burning** aircraft gets lighter as it flies, which is what
makes the Breguet equation logarithmic.  Applying the electric formula to a
transport underestimates its range badly, and applying Breguet to a battery
aircraft divides by the logarithm of one.

Units
-----
SI.  Speeds are true airspeed unless a name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import atmos, drag as _drag, geom
from .drag import Polar

__all__ = [
    "speeds",
    "drag_curve",
    "power_required",
    "climb",
    "ceiling",
    "glide",
    "endurance_electric",
    "range_electric",
    "range_breguet",
    "endurance_breguet",
    "takeoff_ground_roll",
    "landing_ground_roll",
    "turn",
    "envelope",
]

G0 = 9.80665


# --- the basic curves -------------------------------------------------------


def drag_curve(
    polar: Polar, mass: float, V, altitude: Optional[float] = None, n: float = 1.0
) -> Dict[str, np.ndarray]:
    """Drag and power required against speed.

    Parameters
    ----------
    polar : flightlab.drag.Polar
    mass : float
        kg.
    V : array_like
        True airspeeds, m/s.
    altitude : float, optional
        Defaults to the polar's own altitude.  Overriding it re-evaluates the
        polar's ``CL`` at a different density but keeps its ``CD(CL)`` shape,
        which is a good approximation while the Reynolds and Mach numbers do
        not move much and a poor one when they do.
    n : float
        Load factor.

    Returns
    -------
    dict
        ``V``, ``CL``, ``CD``, ``drag`` (N), ``power`` (W), ``LD``.
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass!r} kg")
    V = np.atleast_1d(np.asarray(V, dtype=float))
    if np.any(V <= 0):
        raise ValueError("V must contain only positive true airspeeds")
    h = polar.altitude if altitude is None else altitude
    air = atmos.at(h)
    q = air.q(V)
    S = polar.S_ref
    CL = n * mass * G0 / (q * S)
    CD = polar.CD_at(CL)
    D = CD * q * S
    return {
        "V": V,
        "CL": CL,
        "CD": CD,
        "drag": D,
        "power": D * V,
        "LD": CL / CD,
        "altitude": h,
    }


def power_required(polar: Polar, mass: float, V, **kwargs) -> np.ndarray:
    """Power required for steady level flight, W."""
    return drag_curve(polar, mass, V, **kwargs)["power"]


def speeds(
    polar: Polar,
    mass: float,
    altitude: Optional[float] = None,
    CL_max: Optional[float] = None,
) -> Dict[str, float]:
    """The characteristic speeds, all from one polar.

    Returns
    -------
    dict
        ``V_LD_max`` and ``LD_max`` -- best glide and best range for a
        propeller aircraft; ``V_min_power`` and ``P_min`` -- best endurance for
        a propeller or electric aircraft, and minimum sink for a glider;
        ``V_stall`` when a ``CL_max`` is supplied; and the lift coefficients at
        each.

    Notes
    -----
    ``V_min_power`` sits at ``3^-0.25 ~ 0.76`` of ``V_LD_max`` for an ideal
    parabolic polar, and close to that for a real one.  Flying for endurance is
    therefore slower than flying for range, which is not obvious and is the
    reason a loiter speed is published separately.
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass!r} kg")
    if not np.any(polar.CL > 0):
        raise ValueError("polar must contain at least one positive lift coefficient")
    if CL_max is not None and CL_max <= 0:
        raise ValueError(f"CL_max must be positive; got {CL_max!r}")
    h = polar.altitude if altitude is None else altitude
    air = atmos.at(h)
    S = polar.S_ref
    W = mass * G0

    i = int(np.argmax(polar.LD))
    CL_ld = float(polar.CL[i])
    V_ld = float(np.sqrt(2.0 * W / (air.density * S * CL_ld)))

    # minimum power is minimum CD/CL^1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = np.where(polar.CL > 1e-6, polar.CD / polar.CL**1.5, np.inf)
    j = int(np.argmin(metric))
    CL_pow = float(polar.CL[j])
    V_pow = float(np.sqrt(2.0 * W / (air.density * S * CL_pow)))

    out = {
        "V_LD_max": V_ld,
        "LD_max": float(polar.LD[i]),
        "CL_LD_max": CL_ld,
        "V_min_power": V_pow,
        "CL_min_power": CL_pow,
        "P_min": float(polar.CD[j] * air.q(V_pow) * S * V_pow),
        "altitude": h,
    }
    if CL_max is not None:
        out["V_stall"] = float(np.sqrt(2.0 * W / (air.density * S * CL_max)))
        out["CL_max"] = float(CL_max)
        out["V_LD_over_V_stall"] = out["V_LD_max"] / out["V_stall"]
    return out


# --- climb ------------------------------------------------------------------


def climb(
    polar: Polar,
    mass: float,
    thrust: Callable[[float], float],
    V,
    altitude: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Rate of climb against speed.

    ``RoC = V (T - D) / W``, the excess-power form.  Small-angle: it assumes
    the climb angle is shallow enough that lift still equals weight, which
    holds for everything in the fleet except a fighter and is stated rather
    than hidden.

    Parameters
    ----------
    polar : flightlab.drag.Polar
    mass : float
    thrust : callable
        ``thrust(V)`` in newtons.  For an electric aircraft this comes from
        :func:`flightlab.propulsion.operating_point`; for a jet, from
        :func:`flightlab.propulsion.turbofan_thrust`.
    V : array_like
    altitude : float, optional

    Returns
    -------
    dict
        ``V``, ``thrust``, ``drag``, ``RoC`` (m/s), ``climb_angle`` (degrees),
        plus ``V_best_climb``, ``RoC_max``, ``V_best_angle`` and
        ``angle_max``.
    """
    d = drag_curve(polar, mass, V, altitude)
    V = d["V"]
    T = np.array([float(thrust(float(v))) for v in V])
    W = mass * G0
    roc = V * (T - d["drag"]) / W
    angle = np.degrees(np.arcsin(np.clip((T - d["drag"]) / W, -1.0, 1.0)))
    i, j = int(np.argmax(roc)), int(np.argmax(angle))
    return {
        "V": V,
        "thrust": T,
        "drag": d["drag"],
        "RoC": roc,
        "climb_angle": angle,
        "V_best_climb": float(V[i]),
        "RoC_max": float(roc[i]),
        "V_best_angle": float(V[j]),
        "angle_max": float(angle[j]),
    }


def ceiling(
    polar_at: Callable[[float], Polar],
    mass: float,
    thrust_at: Callable[[float, float], float],
    V,
    h_bracket: Tuple[float, float] = (0.0, 15000.0),
    roc_target: float = 0.5,
) -> float:
    """Altitude at which the best rate of climb falls to ``roc_target``, m.

    ``roc_target = 0`` is the absolute ceiling, which an aircraft approaches
    asymptotically and never reaches.  The service ceiling is conventionally
    where the best climb rate falls to 0.5 m/s (100 ft/min), and that is the
    default here because it is a number an aircraft can actually demonstrate.

    Parameters
    ----------
    polar_at : callable
        ``polar_at(altitude)`` returning a :class:`flightlab.drag.Polar`.
    mass : float
    thrust_at : callable
        ``thrust_at(V, altitude)`` in newtons.
    V : array_like
        Speeds to search over at each altitude.
    h_bracket : tuple
    roc_target : float
    """

    def best_roc(h):
        pol = polar_at(h)
        c = climb(pol, mass, lambda v: thrust_at(v, h), V, altitude=h)
        return c["RoC_max"] - roc_target

    lo, hi = h_bracket
    f_lo, f_hi = best_roc(lo), best_roc(hi)
    if f_lo < 0:
        raise ValueError(
            f"the aircraft cannot climb at {roc_target} m/s even at {lo} m; "
            "it has no ceiling in the usual sense"
        )
    if f_hi > 0:
        return float(hi)
    return float(brentq(best_roc, lo, hi, xtol=1.0))


# --- glide ------------------------------------------------------------------


def glide(polar: Polar, mass: float, altitude: Optional[float] = None) -> Dict[str, float]:
    """Best glide and minimum sink.

    Returns
    -------
    dict
        ``glide_ratio`` (equal to ``LD_max``), ``V_best_glide``,
        ``sink_at_best_glide``, ``min_sink``, ``V_min_sink``, and
        ``glide_angle`` in degrees.

    Notes
    -----
    The two speeds are different and the difference matters.  Best glide covers
    the most ground; minimum sink stays up the longest.  A sailplane pilot
    circling in a thermal flies minimum sink and one crossing to the next
    thermal flies best glide, and the same distinction decides whether a
    student's glide test is measuring what they think it is.
    """
    h = polar.altitude if altitude is None else altitude
    air = atmos.at(h)
    S = polar.S_ref
    W = mass * G0

    i = int(np.argmax(polar.LD))
    CL = polar.CL[i]
    V_bg = float(np.sqrt(2.0 * W / (air.density * S * CL)))
    ld = float(polar.LD[i])

    with np.errstate(divide="ignore", invalid="ignore"):
        metric = np.where(polar.CL > 1e-6, polar.CD / polar.CL**1.5, np.inf)
    j = int(np.argmin(metric))
    V_ms = float(np.sqrt(2.0 * W / (air.density * S * polar.CL[j])))
    sink_ms = float(V_ms * polar.CD[j] / polar.CL[j])

    return {
        "glide_ratio": ld,
        "V_best_glide": V_bg,
        "sink_at_best_glide": float(V_bg / ld),
        "glide_angle": float(np.degrees(np.arctan(1.0 / ld))),
        "min_sink": sink_ms,
        "V_min_sink": V_ms,
        "CL_best_glide": float(CL),
        "CL_min_sink": float(polar.CL[j]),
    }


# --- range and endurance ----------------------------------------------------


def endurance_electric(
    energy: float,
    polar: Polar,
    mass: float,
    V: Optional[float] = None,
    efficiency: float = 0.5,
    altitude: Optional[float] = None,
) -> Dict[str, float]:
    """Endurance on a fixed energy store, s.

    ``t = E * eta / P_required``.  Weight is constant, so unlike a fuel-burning
    aircraft the best speed does not drift as the flight goes on: it is the
    minimum-power speed from beginning to end.

    Parameters
    ----------
    energy : float
        Usable energy, J.  See :func:`flightlab.propulsion.pack_energy`.
    polar, mass : Polar, float
    V : float, optional
        Speed to fly.  Defaults to the minimum-power speed.
    efficiency : float
        Whole-chain efficiency from stored energy to useful power.  0.4 to 0.55
        is realistic for a small electric aircraft, and it is the single
        largest uncertainty in an endurance prediction -- larger than the
        polar.
    altitude : float, optional
    """
    if energy <= 0 or not 0.0 < efficiency <= 1.0:
        raise ValueError("energy must be positive and efficiency must be in (0, 1]")
    s = speeds(polar, mass, altitude)
    V = s["V_min_power"] if V is None else V
    P = float(power_required(polar, mass, V, altitude=altitude)[0])
    t = energy * efficiency / P
    return {
        "endurance": float(t),
        "V": float(V),
        "power_required": P,
        "power_electrical": P / efficiency,
        "efficiency": float(efficiency),
    }


def range_electric(
    energy: float,
    polar: Polar,
    mass: float,
    V: Optional[float] = None,
    efficiency: float = 0.5,
    altitude: Optional[float] = None,
) -> Dict[str, float]:
    """Range on a fixed energy store, m.

    ``R = E * eta * (L/D) / W``, which is worth staring at: it contains no
    speed at all.  A battery aircraft's range depends on lift-to-drag ratio and
    on nothing else about how fast it flies -- so the best-range speed is the
    best-``L/D`` speed, and flying faster costs time but not distance.  That is
    a genuinely different result from the fuel-burning case, and it comes from
    the weight staying constant.
    """
    if energy <= 0 or not 0.0 < efficiency <= 1.0:
        raise ValueError("energy must be positive and efficiency must be in (0, 1]")
    s = speeds(polar, mass, altitude)
    V = s["V_LD_max"] if V is None else V
    d = drag_curve(polar, mass, V, altitude)
    t = energy * efficiency / float(d["power"][0])
    return {
        "range": float(t * V),
        "endurance": float(t),
        "V": float(V),
        "LD": float(d["LD"][0]),
        "efficiency": float(efficiency),
    }


def range_breguet(
    mass_start: float,
    mass_end: float,
    LD: float,
    V: float,
    tsfc: float,
) -> float:
    """Jet range by the Breguet equation, m.

    ``R = (V / (g * TSFC)) * (L/D) * ln(m_start / m_end)``

    Parameters
    ----------
    mass_start, mass_end : float
        kg, at the start and end of cruise.
    LD : float
    V : float
        Cruise true airspeed, m/s.
    tsfc : float
        Thrust-specific fuel consumption, **kg/(N s)**.  Vendor figures are
        often in lb/(lbf hr); the conversion is ``1 lb/(lbf hr) = 2.832e-5
        kg/(N s)``, and getting it wrong is a factor of 35,000.

    Notes
    -----
    Assumes constant ``V``, ``L/D`` and ``TSFC`` through the cruise, which
    means it describes a cruise-climb: as fuel burns off, the aircraft drifts
    upward to hold the same lift coefficient.  Real aircraft fly step climbs
    because air traffic control assigns discrete levels, and lose a little to
    that.
    """
    if mass_start <= mass_end or mass_end <= 0:
        raise ValueError("require mass_start > mass_end > 0")
    if LD <= 0 or V <= 0 or tsfc <= 0:
        raise ValueError("LD, V, and tsfc must all be positive")
    return float((V / (G0 * tsfc)) * LD * np.log(mass_start / mass_end))


def endurance_breguet(mass_start: float, mass_end: float, LD: float, tsfc: float) -> float:
    """Jet endurance by the Breguet equation, s.

    ``E = (1 / (g * TSFC)) * (L/D) * ln(m_start / m_end)`` -- the range
    equation without the speed, so best endurance is best ``L/D`` and best
    range is best ``V L/D``.  For a jet those are different speeds; for a
    propeller aircraft the roles swap, which is a good check that you have the
    right family of equations for the aircraft in front of you.
    """
    if mass_start <= mass_end or mass_end <= 0:
        raise ValueError("require mass_start > mass_end > 0")
    if LD <= 0 or tsfc <= 0:
        raise ValueError("LD and tsfc must both be positive")
    return float((1.0 / (G0 * tsfc)) * LD * np.log(mass_start / mass_end))


# --- field performance ------------------------------------------------------


def takeoff_ground_roll(
    mass: float,
    S: float,
    CL_max: float,
    thrust: Callable[[float], float],
    polar: Polar,
    altitude: float = 0.0,
    mu: float = 0.04,
    CL_ground: Optional[float] = None,
    dT: float = 0.0,
    n_steps: int = 200,
) -> Dict[str, float]:
    """Ground roll to lift-off, m, by integrating the acceleration.

    Lift-off is taken at ``1.1 V_stall``.  The integration is
    ``s = integral V dV / a`` with ``a = (T - D - mu(W - L))/m``, stepped in
    speed rather than time because the thrust is a function of speed.

    Parameters
    ----------
    mass, S, CL_max : float
    thrust : callable
        ``thrust(V)``, N.
    polar : flightlab.drag.Polar
    altitude : float
    mu : float
        Rolling friction coefficient.  0.02-0.05 on pavement, 0.08-0.10 on
        grass -- and a foam model aircraft launched by hand skips this
        calculation entirely, which is itself the answer for RC-1.
    CL_ground : float, optional
        Lift coefficient in the ground attitude.  Defaults to ``0.1 * CL_max``.
    """
    air = atmos.at(altitude, dT)
    W = mass * G0
    V_stall = float(np.sqrt(2.0 * W / (air.density * S * CL_max)))
    V_lof = 1.1 * V_stall
    CL_g = 0.1 * CL_max if CL_ground is None else CL_ground
    CD_g = float(polar.CD_at(CL_g))

    V = np.linspace(1e-3, V_lof, n_steps)
    q = air.q(V)
    T = np.array([float(thrust(float(v))) for v in V])
    L = CL_g * q * S
    D = CD_g * q * S
    a = (T - D - mu * np.maximum(W - L, 0.0)) / mass
    if np.any(a <= 0):
        raise ValueError(
            "the aircraft cannot accelerate to lift-off: thrust never exceeds "
            "drag plus rolling friction.  Either the propeller is badly "
            "matched or the runway is at too high an altitude."
        )
    s = float(np.trapezoid(V / a, V))
    return {
        "ground_roll": s,
        "V_stall": V_stall,
        "V_liftoff": V_lof,
        "acceleration_initial": float(a[0]),
        "time": float(np.trapezoid(1.0 / a, V)),
    }


def landing_ground_roll(
    mass: float,
    S: float,
    CL_max: float,
    polar: Polar,
    altitude: float = 0.0,
    mu: float = 0.4,
    CL_ground: Optional[float] = None,
    reverse_thrust: float = 0.0,
) -> Dict[str, float]:
    """Ground roll from touchdown to stop, m.

    Touchdown at ``1.15 V_stall``, braking coefficient ``mu`` (0.3-0.5 on dry
    pavement), no thrust unless ``reverse_thrust`` is given.
    """
    air = atmos.at(altitude)
    W = mass * G0
    V_stall = float(np.sqrt(2.0 * W / (air.density * S * CL_max)))
    V_td = 1.15 * V_stall
    CL_g = 0.1 * CL_max if CL_ground is None else CL_ground
    CD_g = float(polar.CD_at(CL_g))

    V = np.linspace(1e-3, V_td, 200)
    q = air.q(V)
    L = CL_g * q * S
    D = CD_g * q * S
    decel = (D + mu * np.maximum(W - L, 0.0) + reverse_thrust) / mass
    return {
        "ground_roll": float(np.trapezoid(V / decel, V)),
        "V_stall": V_stall,
        "V_touchdown": V_td,
        "time": float(np.trapezoid(1.0 / decel, V)),
    }


# --- manoeuvre --------------------------------------------------------------


def turn(
    polar: Polar,
    mass: float,
    V: float,
    n: Optional[float] = None,
    bank_deg: Optional[float] = None,
    altitude: Optional[float] = None,
) -> Dict[str, float]:
    """A steady level turn at load factor ``n`` or bank angle ``bank_deg``.

    ``n = 1/cos(phi)``, radius ``V^2/(g sqrt(n^2-1))``, rate
    ``g sqrt(n^2-1)/V``.  The drag comes from the polar at the turning lift
    coefficient, which is ``n`` times the level one -- so induced drag goes as
    ``n^2`` and a 60-degree bank costs four times the induced drag of level
    flight.
    """
    if n is None and bank_deg is None:
        raise ValueError("give either a load factor n or a bank angle")
    if n is None:
        n = 1.0 / np.cos(np.radians(bank_deg))
    if bank_deg is None:
        bank_deg = float(np.degrees(np.arccos(1.0 / n)))
    if n < 1.0:
        raise ValueError("a level turn needs a load factor of at least 1")

    d = drag_curve(polar, mass, V, altitude, n=n)
    root = np.sqrt(max(n**2 - 1.0, 0.0))
    return {
        "n": float(n),
        "bank_deg": float(bank_deg),
        "V": float(V),
        "radius": float(V**2 / (G0 * root)) if root > 0 else float("inf"),
        "rate_deg_s": float(np.degrees(G0 * root / V)) if root > 0 else 0.0,
        "CL": float(d["CL"][0]),
        "drag": float(d["drag"][0]),
        "power": float(d["power"][0]),
    }


# --- the envelope -----------------------------------------------------------


def envelope(
    polar_at: Callable[[float], Polar],
    mass: float,
    thrust_at: Callable[[float, float], float],
    CL_max: float,
    altitudes,
    V,
) -> Dict[str, np.ndarray]:
    """The speed-altitude flight envelope.

    At each altitude, finds where thrust available crosses drag required, and
    the stall speed.  The envelope closes at the top when the two crossings
    meet -- the aircraft's absolute ceiling, where the only speed it can
    sustain is the one speed it can sustain.

    Returns
    -------
    dict
        ``altitude``, ``V_min``, ``V_max``, ``V_stall``, and ``RoC_max``,
        each an array over ``altitudes``.  ``nan`` marks an altitude the
        aircraft cannot sustain level flight at.
    """
    altitudes = np.atleast_1d(np.asarray(altitudes, dtype=float))
    V = np.atleast_1d(np.asarray(V, dtype=float))
    v_min, v_max, v_stall, roc = [], [], [], []

    for h in altitudes:
        pol = polar_at(float(h))
        air = atmos.at(float(h))
        d = drag_curve(pol, mass, V, altitude=float(h))
        T = np.array([float(thrust_at(float(v), float(h))) for v in V])
        excess = T - d["drag"]
        vs = float(np.sqrt(2.0 * mass * G0 / (air.density * pol.S_ref * CL_max)))
        v_stall.append(vs)

        ok = excess > 0
        if not ok.any():
            v_min.append(np.nan)
            v_max.append(np.nan)
            roc.append(np.nan)
            continue
        idx = np.where(ok)[0]
        v_lo, v_hi = float(V[idx[0]]), float(V[idx[-1]])
        v_min.append(max(v_lo, vs))
        v_max.append(v_hi)
        roc.append(float(np.max(V * excess / (mass * G0))))

    return {
        "altitude": altitudes,
        "V_min": np.array(v_min),
        "V_max": np.array(v_max),
        "V_stall": np.array(v_stall),
        "RoC_max": np.array(roc),
    }
