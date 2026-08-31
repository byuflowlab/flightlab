"""``flightlab.propulsion`` -- battery, motor, ESC, propeller, and the match between them.

The electric power chain, end to end, plus the two other propulsion models the
fleet needs: momentum theory for a rotor and a thrust-lapse/TSFC model for a
turbofan.

The chain is a chain, and that is the whole difficulty.  A battery's voltage
sags under current.  The motor's current depends on the torque it is asked for.
The torque it is asked for is the propeller's, which depends on the propeller's
speed, which depends on the motor's voltage.  Nothing in that loop can be
solved without the rest of it, so :func:`operating_point` closes it numerically
and everything else here is either an input to it or a consequence of it.

    >>> from flightlab import propulsion as prop
    >>> op = prop.operating_point("M1000", "apce_10x7", "B3S1300", V=12.0)
    >>> round(op.thrust, 3), round(op.rpm), round(op.efficiency_total, 4)
    (8.703, 8178.0, 0.3689)

Units
-----
SI: newtons, watts, volts, amps, radians per second.  Rotational speed is
carried as ``omega`` in rad/s internally and reported as ``rpm`` as well,
because propeller data is tabulated in RPM and every ``2*pi`` in this module is
a chance to be wrong by a factor of about six.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from . import atmos, catalog, props

__all__ = [
    "PropellerModel",
    "MotorPoint",
    "OperatingPoint",
    "propeller_model",
    "motor",
    "battery_voltage",
    "operating_point",
    "thrust_available",
    "sweep_speed",
    "static_thrust",
    "ideal_propulsive_efficiency",
    "rotor_hover",
    "turbofan_thrust",
    "turbofan_tsfc",
]

RPM_TO_RAD = 2.0 * np.pi / 60.0
RAD_TO_RPM = 60.0 / (2.0 * np.pi)


# --- the propeller ----------------------------------------------------------


class PropellerModel:
    """Interpolated ``CT(J, n)`` and ``CP(J, n)`` from measured UIUC data.

    :mod:`flightlab.props` parses and exposes the measurements; this turns them
    into functions.  Interpolation is bilinear in advance ratio and rotational
    speed, the second because propeller coefficients at these sizes still drift
    with Reynolds number and the measured sweeps show it.

    Parameters
    ----------
    name : str
        A propeller in :func:`flightlab.props.available`, e.g. ``"apce_10x7"``.
    include_static : bool
        Splice the separately measured static run in at ``J = 0``.  Without it
        the model has no data below ``J ~ 0.09`` and static thrust -- the
        number that decides whether the aircraft leaves the ground -- is pure
        extrapolation.

    Attributes
    ----------
    diameter, radius, disk_area, pitch : float
        m, m, m^2, m.
    J_range : tuple
        The measured advance-ratio range **excluding** the static point.
        Outside it, :meth:`CT` extrapolates and says so through
        :meth:`out_of_range`.

    Notes
    -----
    The measured sweeps stop below ``J = 1``, and level flight at top speed
    often sits past the last point.  Both ends of the flight envelope are
    therefore outside the data, which is not a reason to avoid the model -- it
    is a reason to report what fraction of a result rests on extrapolation.
    """

    def __init__(self, name: str, include_static: bool = True):
        self.name = name
        self._initialize(props.load(name), include_static)

    @classmethod
    def from_points(cls, name: str, diameter: float, pitch: float, points) -> "PropellerModel":
        """Build a model from project-owned measured coefficient points.

        Parameters use SI geometry, RPM, and the standard nondimensional
        ``J``, ``CT``, and ``CP`` columns. At least two nonzero advance-ratio
        points are required. Multiple RPM sweeps improve Reynolds interpolation;
        a single sweep is accepted and duplicated internally without claiming
        additional measured coverage.
        """
        rows = list(points)
        if diameter <= 0 or pitch <= 0:
            raise ValueError("propeller diameter and pitch must be positive")
        if len(rows) < 2:
            raise ValueError("measured propeller data needs at least two coefficient points")
        grouped = {}
        static = []
        for row in rows:
            rpm = float(row["rpm"])
            J = float(row["J"])
            CT = float(row["CT"])
            CP = float(row["CP"])
            if rpm <= 0 or J < 0 or CT < 0 or CP <= 0:
                raise ValueError("require rpm > 0, J >= 0, CT >= 0, and CP > 0")
            if J == 0.0:
                static.append((rpm, CT, CP))
            else:
                grouped.setdefault(rpm, []).append((J, CT, CP))
        if not grouped:
            raise ValueError("measured propeller data needs nonzero-J points")
        runs = []
        for rpm, values in sorted(grouped.items()):
            values = sorted(values)
            if len(values) < 2:
                raise ValueError(f"RPM sweep {rpm:g} needs at least two nonzero-J points")
            array = np.asarray(values, dtype=float)
            runs.append(props.Run(
                rpm=rpm, J=array[:, 0], CT=array[:, 1], CP=array[:, 2],
                eta=np.divide(
                    array[:, 0] * array[:, 1], array[:, 2],
                    out=np.zeros(len(array)), where=array[:, 2] > 0,
                ), file="project measurement",
            ))
        if len(runs) == 1:
            run = runs[0]
            runs.append(props.Run(
                rpm=run.rpm * (1.0 + 1e-6), J=run.J.copy(), CT=run.CT.copy(),
                CP=run.CP.copy(), eta=run.eta.copy(), file=run.file,
            ))
        static_run = None
        if static:
            array = np.asarray(sorted(static), dtype=float)
            static_run = props.StaticRun(
                rpm=array[:, 0], CT=array[:, 1], CP=array[:, 2],
                file="project measurement",
            )
        prop = props.Propeller(
            name=name, family="project", diameter_in=diameter / props.INCH,
            pitch_in=pitch / props.INCH, runs=tuple(runs), static=static_run,
            source="project-owned measurements",
        )
        model = cls.__new__(cls)
        model.name = name
        model._initialize(prop, include_static=True)
        return model

    def _initialize(self, prop, include_static: bool) -> None:
        self._prop = prop
        self.diameter = self._prop.diameter
        self.radius = self._prop.radius
        self.disk_area = self._prop.disk_area
        self.pitch = self._prop.pitch
        self.J_range = self._prop.J_range

        # The static run is measured separately, as a sweep over RPM at J = 0.
        # Interpolating it against RPM gives the J = 0 end of each advancing
        # sweep; treating its points as their own rotational speeds instead
        # would build single-point curves, and a single-point interpolation is
        # a constant -- which silently makes thrust independent of airspeed.
        static_n = static_CT = static_CP = None
        if include_static:
            st = getattr(self._prop, "static", None)
            if st is not None:
                order = np.argsort(np.asarray(st.n, dtype=float))
                static_n = np.asarray(st.n, dtype=float)[order]
                static_CT = np.asarray(st.CT, dtype=float)[order]
                static_CP = np.asarray(st.CP, dtype=float)[order]

        raw_curves = {}
        speeds = []
        for rpm in self._prop.rpm_values:
            run = self._prop.run(rpm)
            n = float(run.n)
            J = np.asarray(run.J, dtype=float)
            CT = np.asarray(run.CT, dtype=float)
            CP = np.asarray(run.CP, dtype=float)
            if static_n is not None:
                J = np.concatenate(([0.0], J))
                CT = np.concatenate(([np.interp(n, static_n, static_CT)], CT))
                CP = np.concatenate(([np.interp(n, static_n, static_CP)], CP))
            order = np.argsort(J)
            raw_curves[n] = (J[order], CT[order], CP[order])
            speeds.append(n)

        self._n = np.array(sorted(speeds))
        self.has_static = static_n is not None
        self.n_range = (float(self._n[0]), float(self._n[-1]))

        # Every sweep was run to whatever advance ratio the tunnel could reach
        # at that speed, so the fast sweeps stop early -- the 10x7's 6,531 rpm
        # run ends at J = 0.44 while its 5,001 rpm run reaches 0.84.  Left
        # alone, a lookup at high rpm and high J would clamp at the end of the
        # short curve and report a thrust that is far too high.
        #
        # The coefficients are functions of advance ratio; rotational speed
        # enters only as a blade Reynolds number, and the measured curves very
        # nearly collapse on top of one another.  So each short curve is
        # extended over the full measured J range by borrowing the shape of the
        # curve with the widest coverage, offset to match where the two
        # overlap.  That is an interpolation in the variable that matters and a
        # constant offset in the one that does not.
        widest = max(raw_curves, key=lambda k: raw_curves[k][0][-1])
        self._J_min = min(c[0][0] for c in raw_curves.values())
        self._J_max = max(c[0][-1] for c in raw_curves.values())

        self._table = {}
        self._extended_above = {}
        donor_J, donor_CT, donor_CP = raw_curves[widest]
        for n, (J, CT, CP) in raw_curves.items():
            if J[-1] >= self._J_max - 1e-12:
                self._table[n] = (J, CT, CP)
                self._extended_above[n] = J[-1]
                continue
            tail = donor_J > J[-1]
            if not np.any(tail):  # pragma: no cover - donor is the widest
                self._table[n] = (J, CT, CP)
                self._extended_above[n] = J[-1]
                continue
            dCT = float(CT[-1] - np.interp(J[-1], donor_J, donor_CT))
            dCP = float(CP[-1] - np.interp(J[-1], donor_J, donor_CP))
            self._table[n] = (
                np.concatenate([J, donor_J[tail]]),
                np.concatenate([CT, donor_CT[tail] + dCT]),
                np.concatenate([CP, donor_CP[tail] + dCP]),
            )
            self._extended_above[n] = float(J[-1])

    # -- coefficient lookups ------------------------------------------------

    def _blend(self, which: int, J, n):
        J = np.asarray(J, dtype=float)
        n = np.asarray(n, dtype=float)
        J, n = np.broadcast_arrays(J, n)
        n_c = np.clip(n, self._n[0], self._n[-1])
        i = np.clip(np.searchsorted(self._n, n_c) - 1, 0, len(self._n) - 2)
        span = self._n[i + 1] - self._n[i]
        w = np.where(span > 0, (n_c - self._n[i]) / span, 0.0)

        out = np.empty(J.shape, dtype=float)
        flat_J, flat_i, flat_w = J.ravel(), i.ravel(), w.ravel()
        res = np.empty(flat_J.shape, dtype=float)
        for k in np.unique(flat_i):
            m = flat_i == k
            lo_J, *lo_v = self._table[self._n[k]]
            hi_J, *hi_v = self._table[self._n[k + 1]]
            res[m] = (1.0 - flat_w[m]) * np.interp(
                flat_J[m], lo_J, lo_v[which]
            ) + flat_w[m] * np.interp(flat_J[m], hi_J, hi_v[which])
        out[...] = res.reshape(J.shape)
        return out

    def CT(self, J, n) -> np.ndarray:
        """Thrust coefficient ``T / (rho n^2 D^4)``, ``n`` in rev/s."""
        return self._blend(0, J, n)

    def CP(self, J, n) -> np.ndarray:
        """Power coefficient ``P / (rho n^3 D^5)``, ``n`` in rev/s."""
        return self._blend(1, J, n)

    def CQ(self, J, n) -> np.ndarray:
        """Torque coefficient ``Q / (rho n^2 D^5) = CP / (2 pi)``."""
        return self.CP(J, n) / (2.0 * np.pi)

    def efficiency(self, J, n) -> np.ndarray:
        """Propeller efficiency ``J CT / CP``."""
        CP = self.CP(J, n)
        return np.where(CP > 1e-12, np.asarray(J) * self.CT(J, n) / CP, 0.0)

    def out_of_range(self, J, n=None) -> np.ndarray:
        """True where the **advance ratio** falls outside the measured data.

        Advance ratio is the variable the data is tabulated against, and it is
        the one that decides whether a lookup is interpolation or invention.
        Rotational speed is a separate and much weaker question -- see
        :meth:`reynolds_ratio`.
        """
        J = np.asarray(J, dtype=float)
        lo, hi = self.J_range
        return (J > hi) | (J < 0.0)

    def reynolds_ratio(self, n) -> np.ndarray:
        """How far outside the measured rotational speeds ``n`` sits, as a ratio.

        Returns 1.0 inside the measured range, ``n / n_max`` above it and
        ``n_min / n`` below, so the value is always at least 1 and reads as
        "this many times outside".

        **Rotational speed is not a coverage limit in the way advance ratio
        is.**  The coefficients ``CT`` and ``CP`` are functions of advance
        ratio; ``n`` enters only through the blade-section Reynolds number, and
        its effect is a second-order drift rather than a change of shape.  The
        measured sweeps at different speeds lie nearly on top of one another
        for exactly this reason.  A modest excursion -- call it a factor of two
        -- is a normal engineering extrapolation.  A large one is worth saying
        out loud, because blade sections at these sizes really are Reynolds
        sensitive.
        """
        n = np.asarray(n, dtype=float)
        lo, hi = self._n[0], self._n[-1]
        return np.where(n > hi, n / hi, np.where(n < lo, lo / n, 1.0))

    def coverage(self, J, n) -> Dict[str, object]:
        """What, if anything, about ``(J, n)`` sits outside the measured data."""
        n_arr = np.asarray(n, dtype=float)
        n_c = np.clip(n_arr, self._n[0], self._n[-1])
        i = int(np.clip(np.searchsorted(self._n, n_c) - 1, 0, len(self._n) - 2))
        borrowed_above = min(
            self._extended_above[self._n[i]], self._extended_above[self._n[i + 1]]
        )
        return {
            "J_ok": bool(self._J_min - 1e-9 <= float(J) <= self._J_max + 1e-9
                         or float(J) == 0.0),
            "J_max": float(self._J_max),
            "borrowed_above_J": float(borrowed_above),
            "reynolds_ratio": float(self.reynolds_ratio(n_arr)),
            "n_range": self.n_range,
        }

    # -- dimensional -------------------------------------------------------

    def thrust(self, V: float, omega: float, rho: float = 1.225) -> float:
        """Thrust, N, at speed ``V`` and shaft speed ``omega`` in rad/s."""
        n = omega * RAD_TO_RPM / 60.0
        if n <= 0:
            return 0.0
        J = V / (n * self.diameter)
        return float(self.CT(J, n) * rho * n**2 * self.diameter**4)

    def torque(self, V: float, omega: float, rho: float = 1.225) -> float:
        """Shaft torque, N m."""
        n = omega * RAD_TO_RPM / 60.0
        if n <= 0:
            return 0.0
        J = V / (n * self.diameter)
        return float(self.CQ(J, n) * rho * n**2 * self.diameter**5)

    def power(self, V: float, omega: float, rho: float = 1.225) -> float:
        """Shaft power, W."""
        return self.torque(V, omega, rho) * omega

    def advance_ratio(self, V: float, omega: float) -> float:
        """``J = V / (n D)``."""
        n = omega * RAD_TO_RPM / 60.0
        return float(V / (n * self.diameter)) if n > 0 else float("inf")

    def state_key(self):
        return (self.name, self.diameter)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<PropellerModel {self.name} D={self.diameter:.4f} m  "
            f"J {self.J_range[0]:.3f}-{self.J_range[1]:.3f} measured>"
        )


_PROP_MODELS: Dict[Tuple[str, bool], PropellerModel] = {}


def propeller_model(name, include_static: bool = True) -> PropellerModel:
    """A cached :class:`PropellerModel`."""
    if isinstance(name, PropellerModel):
        return name
    key = (str(name), bool(include_static))
    if key not in _PROP_MODELS:
        _PROP_MODELS[key] = PropellerModel(str(name), include_static)
    return _PROP_MODELS[key]


# --- the motor --------------------------------------------------------------


@dataclass(frozen=True)
class MotorPoint:
    """A brushed-DC-equivalent motor operating point.

    Attributes
    ----------
    voltage, current : float
        V and A at the motor terminals.
    omega : float
        rad/s.
    torque : float
        N m.
    power_in, power_out : float
        Electrical in and mechanical out, W.
    efficiency : float
    """

    voltage: float
    current: float
    omega: float
    torque: float
    power_in: float
    power_out: float
    efficiency: float

    @property
    def rpm(self) -> float:
        return self.omega * RAD_TO_RPM

    @property
    def losses(self) -> Dict[str, float]:
        """Where the wasted watts go: copper, iron/friction, and the total."""
        return {
            "resistive": self.current**2 * (
                (self.power_in - self.power_out) / max(self.current**2, 1e-12)
            ),
            "total": self.power_in - self.power_out,
        }


def motor(spec) -> catalog.Motor:
    """Resolve a motor key or object to a :class:`flightlab.catalog.Motor`."""
    if isinstance(spec, catalog.Motor):
        return spec
    try:
        return catalog.MOTORS[str(spec)]
    except KeyError:
        raise KeyError(
            f"no motor {spec!r} in the catalog; available are "
            f"{sorted(catalog.MOTORS)}"
        ) from None


def Kv_rad(spec) -> float:
    """Motor speed constant in **rad/s per volt**, from the catalog's RPM/V.

    The single most common unit error in the whole course.  A 1000 Kv motor is
    1000 RPM/V, which is 104.7 rad/s/V, and using the RPM figure directly in a
    torque calculation is wrong by a factor of 9.55.  The torque constant
    equals the reciprocal: ``Kt = 1/Kv`` in N m/A when ``Kv`` is in rad/s/V.
    """
    return motor(spec).Kv_rpm * RPM_TO_RAD


def motor_point(spec, voltage: float, omega: float) -> MotorPoint:
    """The motor's operating point at a given terminal voltage and shaft speed.

    The standard three-parameter model: back-EMF ``omega/Kv``, winding
    resistance ``R``, and a no-load current ``I0`` standing in for iron and
    bearing losses.

        ``I = (V - omega/Kv) / R``
        ``Q = (I - I0) / Kv``

    Both ``R`` and ``I0`` are exactly the two parameters hobby vendors do not
    publish, which is why the course measures them on the thrust stand.
    """
    m = motor(spec)
    kv = m.Kv_rpm * RPM_TO_RAD
    current = (voltage - omega / kv) / m.resistance
    torque = (current - m.current_no_load) / kv
    p_in = voltage * current
    p_out = torque * omega
    return MotorPoint(
        voltage=float(voltage),
        current=float(current),
        omega=float(omega),
        torque=float(torque),
        power_in=float(p_in),
        power_out=float(p_out),
        efficiency=float(p_out / p_in) if p_in > 0 else 0.0,
    )


def motor_peak_efficiency(spec, voltage: Optional[float] = None) -> Dict[str, float]:
    """Peak efficiency of a motor at a given terminal voltage.

    The closed form is ``eta_max = (1 - sqrt(I0 R / V))^2``, which says
    something worth noticing: peak efficiency depends on the motor only through
    the product ``I0 * R``, and not on ``Kv`` at all.  A high-``Kv`` motor is
    not inherently less efficient -- it is a different gear ratio, and the
    catalog's four motors rank differently by ``Kv`` and by peak efficiency
    precisely because of this.
    """
    m = motor(spec)
    V = m.no_load_voltage if voltage is None else voltage
    x = m.current_no_load * m.resistance / V
    eta = (1.0 - np.sqrt(x)) ** 2
    kv = m.Kv_rpm * RPM_TO_RAD
    omega = kv * (V - np.sqrt(V * m.current_no_load * m.resistance))
    return {
        "efficiency": float(eta),
        "I0R": float(m.current_no_load * m.resistance),
        "I0R_over_V": float(x),
        "omega": float(omega),
        "rpm": float(omega * RAD_TO_RPM),
        "voltage": float(V),
    }


# --- the battery ------------------------------------------------------------


def battery(spec) -> catalog.Battery:
    """Resolve a battery key or object to a :class:`flightlab.catalog.Battery`."""
    if isinstance(spec, catalog.Battery):
        return spec
    try:
        return catalog.BATTERIES[str(spec)]
    except KeyError:
        raise KeyError(
            f"no battery {spec!r} in the catalog; available are "
            f"{sorted(catalog.BATTERIES)}"
        ) from None


def battery_voltage(spec, current: float = 0.0, soc: float = 1.0) -> float:
    """Terminal voltage, V, at a given current draw and state of charge.

    Open-circuit voltage from the cell curve, minus ``I R`` through the pack's
    internal resistance.  The sag is not a detail: a 3S 1300 mAh pack at 20 A
    loses about a volt, which is nearly 9% of the pack voltage and therefore
    nearly 9% of the propeller's speed.
    """
    if not 0.0 <= soc <= 1.0:
        raise ValueError(f"soc must be between 0 and 1; got {soc!r}")
    b = battery(spec)
    ocv_cell = _cell_ocv(soc, b)
    r_pack = b.cell_resistance * b.cells_series / max(b.cells_parallel, 1)
    return float(ocv_cell * b.cells_series - current * r_pack)


def _cell_ocv(soc, b: catalog.Battery) -> float:
    """Open-circuit cell voltage against state of charge.

    A LiPo's discharge curve is flat across the middle and falls off a cliff at
    each end.  Modelled as a shaped interpolation between the empty, nominal
    and full cell voltages rather than a straight line, because a straight line
    predicts a graceful decline that does not happen.
    """
    soc = np.asarray(soc, dtype=float)
    pts = np.array([0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0])
    vals = np.array(
        [
            b.cell_voltage_empty,
            b.cell_voltage_empty + 0.35 * (b.cell_voltage_nominal - b.cell_voltage_empty),
            b.cell_voltage_nominal - 0.06,
            b.cell_voltage_nominal + 0.02,
            b.cell_voltage_nominal + 0.28,
            b.cell_voltage_full - 0.12,
            b.cell_voltage_full,
        ]
    )
    return float(np.interp(soc, pts, vals))


def pack_energy(spec, usable: float = 0.8) -> float:
    """Usable pack energy, J.

    ``usable`` is the fraction of nameplate capacity you are willing to take
    out.  0.8 is the normal working limit for a LiPo that is expected to
    survive the semester; taking a pack to zero once is how it becomes a
    paperweight.
    """
    if not 0.0 < usable <= 1.0:
        raise ValueError(f"usable must be greater than 0 and no more than 1; got {usable!r}")
    b = battery(spec)
    return float(
        b.capacity_ah * 3600.0 * b.cell_voltage_nominal * b.cells_series * usable
    )


# --- the match --------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """One converged point of the whole electric chain.

    Attributes
    ----------
    V : float
        Airspeed, m/s.
    omega, rpm : float
    thrust : float
        N.
    torque : float
        N m.
    J : float
        Advance ratio.
    voltage, current : float
        At the motor terminals, V and A.
    throttle : float
        Fraction of pack voltage the ESC passes.
    power_electrical, power_shaft, power_useful : float
        W.  ``power_useful`` is ``thrust * V``, which is zero at zero airspeed
        no matter how much noise the propeller makes.
    efficiency_motor, efficiency_prop, efficiency_esc, efficiency_total : float
    extrapolated : bool
        True when the **advance ratio** sits outside the measured data, which
        is the coverage question that matters.
    reynolds_ratio : float
        How far the rotational speed sits outside the measured sweeps, as a
        multiple (1.0 = inside).  A blade Reynolds-number excursion, not an
        extrapolation in the tabulated variable: modest values are ordinary,
        and it is reported so that a large one is visible.
    extrapolated_reason : str
        What is outside, and by how much.  Empty when nothing is.
    """

    V: float
    omega: float
    thrust: float
    torque: float
    J: float
    voltage: float
    current: float
    throttle: float
    power_electrical: float
    power_shaft: float
    power_useful: float
    efficiency_motor: float
    efficiency_prop: float
    efficiency_esc: float
    efficiency_total: float
    extrapolated: bool = False
    reynolds_ratio: float = 1.0
    extrapolated_reason: str = ""
    soc: float = 1.0

    @property
    def rpm(self) -> float:
        return self.omega * RAD_TO_RPM

    @property
    def well_covered(self) -> bool:
        """Inside the measured advance ratios and within 2x on Reynolds number."""
        return (not self.extrapolated) and self.reynolds_ratio < 2.0

    def table(self) -> str:
        """A printable chain breakdown, stage by stage."""
        lines = [
            f"airspeed        {self.V:10.3f} m/s",
            f"shaft speed     {self.rpm:10.0f} rpm   (J = {self.J:.4f}"
            + (")" if not self.extrapolated else ")\n                "
               f"   ** {self.extrapolated_reason} **"),
            f"thrust          {self.thrust:10.3f} N",
            f"torque          {self.torque:10.5f} N m",
            "",
            f"pack -> ESC     {self.voltage / max(self.throttle, 1e-9):10.3f} V"
            f"  at {self.current:.2f} A",
            f"ESC -> motor    {self.voltage:10.3f} V   "
            f"(throttle {100 * self.throttle:.0f}%)",
            "",
            f"electrical in   {self.power_electrical:10.2f} W",
            f"shaft power     {self.power_shaft:10.2f} W   "
            f"(motor {100 * self.efficiency_motor:.1f}%)",
            f"useful power    {self.power_useful:10.2f} W   "
            f"(prop {100 * self.efficiency_prop:.1f}%)",
            "",
            f"chain efficiency {100 * self.efficiency_total:9.1f}%   "
            f"= {100 * self.efficiency_esc:.0f}% x "
            f"{100 * self.efficiency_motor:.1f}% x "
            f"{100 * self.efficiency_prop:.1f}%",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<OperatingPoint V={self.V:.2f} m/s  T={self.thrust:.3f} N  "
            f"{self.rpm:.0f} rpm  {self.current:.2f} A  "
            f"eta={self.efficiency_total:.3f}>"
        )


def operating_point(
    motor_spec,
    prop_spec,
    battery_spec,
    V: float,
    throttle: float = 1.0,
    altitude: float = 0.0,
    soc: float = 1.0,
    esc=None,
    dT: float = 0.0,
    supply_voltage: Optional[float] = None,
    omega_bracket: Tuple[float, float] = (5.0, 25000.0),
) -> OperatingPoint:
    """Close the motor-propeller torque match at one airspeed.

    The propeller's torque demand rises with shaft speed; the motor's torque
    supply falls with it.  There is exactly one speed where they are equal, and
    that is where the system runs.  Everything else -- thrust, current, chain
    efficiency -- follows from it.

    Parameters
    ----------
    motor_spec, battery_spec : str or object
        A :class:`flightlab.catalog.Motor` and :class:`flightlab.catalog.Battery`,
        including objects updated with measured parameters. A catalog key is a
        shorthand that selects the corresponding starter dataset; it is not a
        constant built into this solver.
    prop_spec : str or object
        A name in :func:`flightlab.props.available` or a :class:`PropellerModel`.
    V : float
        True airspeed, m/s.
    throttle : float
        Fraction of pack voltage the ESC passes, 0 to 1.  A real ESC modulates
        by duty cycle, so this is a good model of one.
    altitude : float
        Geometric altitude, m -- sets the density the propeller works against.
    soc : float
        Battery state of charge, 0 to 1.
    esc : str or ESC, optional
        Defaults to the first catalog ESC that can carry the current.
    dT : float
    supply_voltage : float, optional
        Fixed voltage at the shared battery bus, before the selected throttle
        duty cycle. When omitted, this function computes sag for one motor.
        Project analyses with several propulsors solve their common bus voltage
        from total battery current and pass it here.
    omega_bracket : tuple
        Shaft speed search range, rad/s.

    Returns
    -------
    OperatingPoint

    Raises
    ------
    ValueError
        If no torque balance exists in the bracket -- which for a sensible
        combination means the propeller is far too large for the motor, and is
        a real answer rather than a numerical failure.
    """
    if V < 0:
        raise ValueError(f"V must be nonnegative; got {V!r} m/s")
    if not 0.0 < throttle <= 1.0:
        raise ValueError(f"throttle must be greater than 0 and no more than 1; got {throttle!r}")
    if not 0.0 <= soc <= 1.0:
        raise ValueError(f"soc must be between 0 and 1; got {soc!r}")
    m = motor(motor_spec)
    b = battery(battery_spec)
    p = propeller_model(prop_spec)
    rho = atmos.at(altitude, dT).density

    esc_obj = _resolve_esc(esc)
    eta_esc = esc_obj.efficiency

    def residual(omega):
        if supply_voltage is None:
            # A single-chain call closes battery sag on its own motor current.
            v_open = battery_voltage(b, 0.0, soc) * throttle
            i = (v_open - omega / (m.Kv_rpm * RPM_TO_RAD)) / m.resistance
            v = battery_voltage(b, max(i, 0.0), soc) * throttle
        else:
            v = float(supply_voltage) * throttle
        mp = motor_point(m, v, omega)
        return mp.torque * eta_esc - p.torque(V, omega, rho)

    lo, hi = omega_bracket
    try:
        f_lo, f_hi = residual(lo), residual(hi)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"could not evaluate the torque balance: {exc}") from exc
    if f_lo * f_hi > 0:
        raise ValueError(
            f"no motor-propeller torque balance between {lo} and {hi} rad/s "
            f"for {m.key} + {getattr(p, 'name', prop_spec)} at V = {V} m/s "
            f"(residual {f_lo:.4g} to {f_hi:.4g} N m).  Usually this means the "
            "propeller is far too large or too small for the motor."
        )
    omega = brentq(residual, lo, hi, xtol=1e-8, rtol=1e-12)

    if supply_voltage is None:
        v_open = battery_voltage(b, 0.0, soc) * throttle
        i_est = (v_open - omega / (m.Kv_rpm * RPM_TO_RAD)) / m.resistance
        v = battery_voltage(b, max(i_est, 0.0), soc) * throttle
    else:
        v = float(supply_voltage) * throttle
    mp = motor_point(m, v, omega)

    T = p.thrust(V, omega, rho)
    Q = p.torque(V, omega, rho)
    J = p.advance_ratio(V, omega)
    p_shaft = Q * omega
    p_elec = mp.power_in / eta_esc
    p_useful = T * V

    return OperatingPoint(
        V=float(V),
        omega=float(omega),
        thrust=float(T),
        torque=float(Q),
        J=float(J),
        voltage=float(v),
        current=float(mp.current),
        throttle=float(throttle),
        power_electrical=float(p_elec),
        power_shaft=float(p_shaft),
        power_useful=float(p_useful),
        efficiency_motor=float(mp.efficiency),
        efficiency_prop=float(p_useful / p_shaft) if p_shaft > 0 else 0.0,
        efficiency_esc=float(eta_esc),
        efficiency_total=float(p_useful / p_elec) if p_elec > 0 else 0.0,
        extrapolated=bool(np.any(p.out_of_range(J))),
        reynolds_ratio=float(p.reynolds_ratio(omega * RAD_TO_RPM / 60.0)),
        extrapolated_reason=_coverage_note(p, J, omega),
        soc=float(soc),
    )


def _coverage_note(p: PropellerModel, J: float, omega: float) -> str:
    """A human-readable note on how the operating point sits in the data."""
    n = omega * RAD_TO_RPM / 60.0
    c = p.coverage(J, n)
    notes = []
    if not c["J_ok"]:
        notes.append(
            f"J = {J:.3f} is past {c['J_max']:.3f}, the largest advance ratio "
            "measured at any speed -- an extrapolation in the tabulated variable"
        )
    elif J > c["borrowed_above_J"] + 1e-9:
        notes.append(
            f"J = {J:.3f} is past {c['borrowed_above_J']:.3f}, where this "
            "speed's own sweep ended, so the curve shape is borrowed from a "
            "slower sweep that reaches further"
        )
    r = c["reynolds_ratio"]
    if r > 1.02:
        lo, hi = c["n_range"]
        notes.append(
            f"{n * 60:.0f} rpm is {r:.2f}x outside the measured "
            f"{lo * 60:.0f}-{hi * 60:.0f} rpm sweeps (a blade Reynolds-number "
            + ("excursion, ordinary at this size)" if r < 2.0
               else "excursion large enough to be worth checking)")
        )
    return "; ".join(notes)


def _resolve_esc(esc):
    if esc is None:
        return list(catalog.ESCS.values())[0]
    if isinstance(esc, catalog.ESC):
        return esc
    try:
        return catalog.ESCS[str(esc)]
    except KeyError:
        raise KeyError(
            f"no ESC {esc!r} in the catalog; available are {sorted(catalog.ESCS)}"
        ) from None


def sweep_speed(
    motor_spec, prop_spec, battery_spec, V, **kwargs
) -> List[OperatingPoint]:
    """Close the match at each of a range of airspeeds.

    The result is a thrust-available curve, which crossed with a drag curve
    gives every performance number in the course.
    """
    return [
        operating_point(motor_spec, prop_spec, battery_spec, float(v), **kwargs)
        for v in np.atleast_1d(np.asarray(V, dtype=float))
    ]


def thrust_available(motor_spec, prop_spec, battery_spec, V, **kwargs) -> np.ndarray:
    """Thrust at each speed in ``V``, N."""
    return np.array(
        [op.thrust for op in sweep_speed(motor_spec, prop_spec, battery_spec, V, **kwargs)]
    )


def static_thrust(motor_spec, prop_spec, battery_spec, **kwargs) -> OperatingPoint:
    """The static (zero airspeed) operating point.

    Only meaningful because :class:`PropellerModel` splices in the separately
    measured static run.  Without that data this is an extrapolation of a curve
    fitted a long way from ``J = 0``, and it is typically wrong by tens of
    percent in the optimistic direction.
    """
    return operating_point(motor_spec, prop_spec, battery_spec, V=0.0, **kwargs)


# --- momentum theory --------------------------------------------------------


def ideal_propulsive_efficiency(V: float, thrust: float, disk_area: float,
                                rho: float = 1.225) -> float:
    """Froude efficiency of an ideal actuator disk.

    ``eta = 2 / (1 + sqrt(1 + 2T/(rho A V^2)))``.  The ceiling any propeller,
    rotor or fan works below, and the reason a large slow-turning disk beats a
    small fast one at the same thrust.  It says nothing about blade design and
    cannot be exceeded by improving one.
    """
    if V <= 0:
        return 0.0
    if thrust < 0 or disk_area <= 0 or rho <= 0:
        raise ValueError("thrust must be nonnegative; disk_area and rho must be positive")
    return float(2.0 / (1.0 + np.sqrt(1.0 + 2.0 * thrust / (rho * disk_area * V**2))))


def rotor_hover(
    thrust: float, disk_area: float, rho: float = 1.225,
    figure_of_merit: float = 0.75,
) -> Dict[str, float]:
    """Hover power for a rotor, from momentum theory plus a figure of merit.

    ``P_ideal = T^1.5 / sqrt(2 rho A)``, and the real power is that divided by
    the figure of merit -- 0.7 to 0.8 for a good rotor.

    The scaling is the point: hover power goes as thrust to the three-halves
    and as the inverse square root of disk area.  An eVTOL that wants to hover
    on a battery needs disk area more than it needs anything else, which is why
    they all have so many large rotors and why the number of them is a
    structural and acoustic compromise rather than an aerodynamic one.
    """
    if thrust <= 0 or disk_area <= 0 or rho <= 0:
        raise ValueError("thrust, disk_area, and rho must all be positive")
    if not 0.0 < figure_of_merit <= 1.0:
        raise ValueError("figure_of_merit must be greater than 0 and no more than 1")
    p_ideal = thrust**1.5 / np.sqrt(2.0 * rho * disk_area)
    return {
        "power_ideal": float(p_ideal),
        "power": float(p_ideal / figure_of_merit),
        "disk_loading": float(thrust / disk_area),
        "induced_velocity": float(np.sqrt(thrust / (2.0 * rho * disk_area))),
        "figure_of_merit": float(figure_of_merit),
    }


# --- turbofan ---------------------------------------------------------------


def turbofan_thrust(
    thrust_sl: float,
    altitude,
    mach=0.0,
    n: float = 0.7,
    mach_slope: float = 0.25,
    minimum_mach_factor: float = 0.65,
):
    """Available thrust, N, from the course high-bypass-turbofan model.

    ``T = T_sl * sigma^n * max(f_min, 1 - k_M M)``.  The default constants are
    the transparent reduced-order model stored in :mod:`flightlab.ref`.  It is a
    teaching correlation, not an engine deck.  Use it from Mach 0--0.90 and
    geometric altitude 0--13 km; the low-Mach end is only a smooth extension
    to the static sea-level rating.

    ``altitude`` and ``mach`` may be scalars or broadcastable arrays.
    """
    if thrust_sl <= 0 or n <= 0:
        raise ValueError("thrust_sl and density exponent n must be positive")
    mach_arr = np.asarray(mach, dtype=float)
    altitude_arr = np.asarray(altitude, dtype=float)
    if np.any((mach_arr < 0) | (mach_arr > 0.90)):
        raise ValueError("the course turbofan model requires 0 <= mach <= 0.90")
    if np.any((altitude_arr < 0) | (altitude_arr > 13_000.0)):
        raise ValueError("the course turbofan model requires altitude from 0 to 13000 m")
    if mach_slope < 0 or not 0.0 < minimum_mach_factor <= 1.0:
        raise ValueError("mach_slope must be nonnegative and minimum_mach_factor in (0, 1]")
    sigma = atmos.at(altitude).density / atmos.SEA_LEVEL.density
    mach_factor = np.maximum(minimum_mach_factor, 1.0 - mach_slope * mach_arr)
    result = thrust_sl * sigma**n * mach_factor
    return float(result) if np.ndim(result) == 0 else result


def turbofan_tsfc(tsfc_sl: float, altitude: float, mach: float = 0.0) -> float:
    """Thrust-specific fuel consumption at altitude, kg/(N s).

    Rises with the square root of the temperature ratio, which is why a
    turbofan is more efficient in the cold air at altitude and why cruise
    altitude is chosen as high as the wing will allow.
    """
    theta = atmos.at(altitude).temperature / atmos.SEA_LEVEL.temperature
    return float(tsfc_sl * np.sqrt(theta) * (1.0 + 0.35 * mach))
