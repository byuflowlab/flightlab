"""``flightlab.atmos`` -- the 1976 U.S. Standard Atmosphere, and what flows from it.

The atmosphere is the first thing every other module asks for.  Density sets
dynamic pressure, viscosity sets Reynolds number, and the speed of sound sets
Mach number, so a single call here feeds the section polars, the vortex
lattice, the propeller match and the range integral alike.

    >>> from flightlab import atmos
    >>> air = atmos.at(11000.0)
    >>> round(air.density, 4), round(air.temperature, 2)
    (0.3639, 216.65)
    >>> round(atmos.mach(250.0, 11000.0), 3)
    0.847

Coverage is 0 to 86 km geopotential altitude, all seven layers.  Above the
troposphere nothing in this course flies, but the 787 cruises within 200 m of
the tropopause and the Saturn V leaves the table entirely, so the layers are
here rather than a troposphere-only fit that fails silently at 12 km.

Units
-----
SI throughout: metres, kelvin, pascals, kg/m^3, m/s, Pa*s.  Altitudes are
**geometric** unless a function says ``geopotential``; the difference reaches
0.3% at 20 km and the conversion is applied internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = [
    "State",
    "at",
    "density",
    "temperature",
    "pressure",
    "viscosity",
    "speed_of_sound",
    "mach",
    "reynolds",
    "eas_to_tas",
    "tas_to_eas",
    "density_altitude",
    "SEA_LEVEL",
    "R_AIR",
    "GAMMA",
    "G0",
    "EARTH_RADIUS",
]

# --- constants, 1976 U.S. Standard Atmosphere -------------------------------

R_AIR = 287.0528  # specific gas constant for dry air, J/(kg K)
GAMMA = 1.4  # ratio of specific heats
G0 = 9.80665  # standard gravity, m/s^2
EARTH_RADIUS = 6356766.0  # effective Earth radius for geopotential, m

_S_MU = 1.458e-6  # Sutherland's constant, kg/(m s sqrt(K))
_S_T = 110.4  # Sutherland's temperature, K

# base geopotential altitude (m), lapse rate (K/m), base temperature (K)
_LAYERS = (
    (0.0, -0.0065, 288.15),
    (11000.0, 0.0, 216.65),
    (20000.0, 0.001, 216.65),
    (32000.0, 0.0028, 228.65),
    (47000.0, 0.0, 270.65),
    (51000.0, -0.0028, 270.65),
    (71000.0, -0.002, 214.65),
    (84852.0, 0.0, 186.946),
)

_P0 = 101325.0  # sea-level pressure, Pa


def _base_pressures():
    """Pressure at the base of each layer, integrated upward from sea level."""
    p = [_P0]
    for i in range(len(_LAYERS) - 1):
        h0, lapse, T0 = _LAYERS[i]
        h1 = _LAYERS[i + 1][0]
        if lapse == 0.0:
            p.append(p[i] * np.exp(-G0 * (h1 - h0) / (R_AIR * T0)))
        else:
            T1 = T0 + lapse * (h1 - h0)
            p.append(p[i] * (T1 / T0) ** (-G0 / (R_AIR * lapse)))
    return tuple(p)


_PBASE = _base_pressures()

_HBASE = np.array([layer[0] for layer in _LAYERS])
_LAPSE = np.array([layer[1] for layer in _LAYERS])
_TBASE = np.array([layer[2] for layer in _LAYERS])
_PB = np.array(_PBASE)

Number = Union[float, np.ndarray]


@dataclass(frozen=True)
class State:
    """The atmosphere at one altitude.

    Attributes
    ----------
    altitude : float or ndarray
        Geometric altitude, m.
    temperature : float or ndarray
        K.
    pressure : float or ndarray
        Pa.
    density : float or ndarray
        kg/m^3.
    viscosity : float or ndarray
        Dynamic viscosity, Pa*s.
    speed_of_sound : float or ndarray
        m/s.
    """

    altitude: Number
    temperature: Number
    pressure: Number
    density: Number
    viscosity: Number
    speed_of_sound: Number

    @property
    def kinematic_viscosity(self) -> Number:
        """``mu / rho``, m^2/s."""
        return self.viscosity / self.density

    @property
    def sigma(self) -> Number:
        """Density ratio ``rho / rho_sl``, the quantity that converts EAS."""
        return self.density / SEA_LEVEL.density

    def q(self, V: Number) -> Number:
        """Dynamic pressure ``0.5 rho V^2`` at true airspeed ``V``, Pa."""
        return 0.5 * self.density * np.asarray(V) ** 2

    def mach(self, V: Number) -> Number:
        """Mach number at true airspeed ``V``."""
        return np.asarray(V) / self.speed_of_sound

    def reynolds(self, V: Number, length: Number) -> Number:
        """Reynolds number at true airspeed ``V`` on a reference ``length``."""
        return self.density * np.asarray(V) * np.asarray(length) / self.viscosity

    def __repr__(self) -> str:  # pragma: no cover - display only
        try:
            h = float(self.altitude)
        except (TypeError, ValueError):
            return "<atmos.State (array)>"
        return (
            f"<atmos.State h={h:.0f} m  T={float(self.temperature):.2f} K  "
            f"p={float(self.pressure):.0f} Pa  rho={float(self.density):.4f} kg/m^3>"
        )


def geopotential(h: Number) -> Number:
    """Convert geometric altitude to geopotential altitude, m."""
    h = np.asarray(h, dtype=float)
    return EARTH_RADIUS * h / (EARTH_RADIUS + h)


def geometric(hgp: Number) -> Number:
    """Convert geopotential altitude to geometric altitude, m."""
    hgp = np.asarray(hgp, dtype=float)
    return EARTH_RADIUS * hgp / (EARTH_RADIUS - hgp)


def at(altitude: Number, dT: Number = 0.0) -> State:
    """The atmosphere at a geometric ``altitude``, m.

    Parameters
    ----------
    altitude : float or array_like
        Geometric altitude, m.  Valid from 0 to about 86 km.
    dT : float or array_like, optional
        Temperature offset from standard, K.  A hot day is a positive offset;
        it lowers the density at fixed pressure, which is why hot-and-high
        takeoff performance is a separate calculation and not a footnote.

    Returns
    -------
    State

    Notes
    -----
    The offset ``dT`` is applied to the temperature *after* the pressure is
    computed from the standard profile, which is the usual convention: the
    pressure altitude is what the aircraft's instruments read, and the
    temperature is what the thermometer reads.
    """
    h = np.asarray(altitude, dtype=float)
    scalar = h.ndim == 0
    hgp = geopotential(h)

    if np.any(hgp < -5000.0) or np.any(hgp > 84852.0):
        raise ValueError(
            "altitude outside the 1976 standard atmosphere (-5 km to 86 km); "
            f"got {np.min(h):.0f} to {np.max(h):.0f} m geometric"
        )

    i = np.clip(np.searchsorted(_HBASE, hgp, side="right") - 1, 0, len(_HBASE) - 1)
    h0, lapse, T0, p0 = _HBASE[i], _LAPSE[i], _TBASE[i], _PB[i]

    T_std = T0 + lapse * (hgp - h0)
    isothermal = lapse == 0.0
    # np.where evaluates both branches, so guard the lapse division
    safe_lapse = np.where(isothermal, 1.0, lapse)
    p = np.where(
        isothermal,
        p0 * np.exp(-G0 * (hgp - h0) / (R_AIR * T0)),
        p0 * (T_std / T0) ** (-G0 / (R_AIR * safe_lapse)),
    )

    T = T_std + np.asarray(dT, dtype=float)
    rho = p / (R_AIR * T)
    mu = _S_MU * T**1.5 / (T + _S_T)
    a = np.sqrt(GAMMA * R_AIR * T)

    if scalar and np.ndim(dT) == 0:
        return State(float(h), float(T), float(p), float(rho), float(mu), float(a))
    return State(h, T, p, rho, mu, a)


def temperature(altitude: Number, dT: Number = 0.0) -> Number:
    """Temperature at ``altitude``, K."""
    return at(altitude, dT).temperature


def pressure(altitude: Number) -> Number:
    """Pressure at ``altitude``, Pa."""
    return at(altitude).pressure


def density(altitude: Number, dT: Number = 0.0) -> Number:
    """Density at ``altitude``, kg/m^3."""
    return at(altitude, dT).density


def viscosity(altitude: Number, dT: Number = 0.0) -> Number:
    """Dynamic viscosity at ``altitude``, Pa*s."""
    return at(altitude, dT).viscosity


def speed_of_sound(altitude: Number, dT: Number = 0.0) -> Number:
    """Speed of sound at ``altitude``, m/s."""
    return at(altitude, dT).speed_of_sound


def mach(V: Number, altitude: Number, dT: Number = 0.0) -> Number:
    """Mach number for true airspeed ``V`` at ``altitude``."""
    return np.asarray(V) / speed_of_sound(altitude, dT)


def reynolds(V: Number, length: Number, altitude: Number, dT: Number = 0.0) -> Number:
    """Reynolds number for ``V`` on ``length`` at ``altitude``."""
    return at(altitude, dT).reynolds(V, length)


def eas_to_tas(eas: Number, altitude: Number, dT: Number = 0.0) -> Number:
    """Equivalent airspeed to true airspeed, m/s.

    ``V_true = V_eq / sqrt(sigma)``.  The stall speed of an aircraft is fixed
    in EAS and grows in TAS with altitude, which is the whole reason the
    flight envelope narrows at the top.
    """
    return np.asarray(eas) / np.sqrt(at(altitude, dT).sigma)


def tas_to_eas(tas: Number, altitude: Number, dT: Number = 0.0) -> Number:
    """True airspeed to equivalent airspeed, m/s."""
    return np.asarray(tas) * np.sqrt(at(altitude, dT).sigma)


def density_altitude(rho: Number) -> Number:
    """The standard altitude at which the density is ``rho``, m geometric.

    Inverts the standard profile numerically over the troposphere and lower
    stratosphere.  Useful for turning a measured field density into the
    altitude the aircraft's performance charts think it is at.
    """
    rho = np.asarray(rho, dtype=float)
    grid = np.linspace(-1000.0, 30000.0, 6201)
    table = at(grid).density
    # table is monotonically decreasing, so reverse for np.interp
    return np.interp(rho, table[::-1], grid[::-1])


SEA_LEVEL = at(0.0)
"""The standard sea-level state, precomputed because everything ratios to it."""
