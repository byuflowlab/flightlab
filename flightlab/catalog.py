"""``flightlab.catalog`` -- the bounded propulsion catalog.

Three motors, three propellers, two batteries, one ESC.  Eighteen combinations
is a genuine finite search with a right answer, small enough to stock and to
measure on the thrust stand, and the student orders the exact part they
analyzed.

Where the numbers come from
---------------------------
Every entry is a real, orderable part.

* **Propellers** carry measured UIUC wind-tunnel data (static and
  advance-ratio sweeps) and APC's published mass.
* **Motors** are SunnySky X-series outrunners, chosen because SunnySky, unlike
  most vendors in this price tier, publishes winding resistance, no-load
  current, and static thrust tables for each winding.  ``Kv``,
  ``resistance``, ``current_no_load``, ``current_max`` and ``mass`` are
  transcribed from those datasheets.  Compared at the same pack current, the
  model reproduces SunnySky's published static tables for the X2208 and
  X2212 windings to within about ten per cent in thrust and rpm.
* **Batteries** are generic hobby packs.  Capacity and cell count are the
  nameplate; mass and C rating are typical vendor figures for the class;
  ``cell_resistance`` is an estimate, because no pack vendor publishes it.

If you ever measure a motor or pack yourself, :meth:`Motor.with_measurements`
and :meth:`Battery.with_measurements` return a copy with your numbers in
place of the datasheet's.

Units
-----
SI: ``Kv`` is stored in **rad/s/V** as :attr:`Motor.Kv` and in the
manufacturer's RPM/V as :attr:`Motor.Kv_rpm`.  That conversion is the factor of
``2*pi/60 = 0.1047``: get it backwards and torques come
out wrong by 9.55, which is small enough to look like a modelling error and
large enough to ruin everything downstream.  Resistance in ohms, current in
amperes, voltage in volts, capacity in **coulombs** (with amp-hours available
as a property), energy in joules, mass in kilograms.

Examples
--------
::

    from flightlab import catalog

    catalog.MOTORS["M1260"].Kv        # rad/s/V, for the torque constant
    catalog.MOTORS["M1260"].Kv_rpm    # RPM/V, as the vendor quotes it
    catalog.BATTERIES["B3S1000"].energy_nominal / 3600   # W*h

    motor = catalog.MOTORS["M1260"].with_measurements(
        resistance=0.130, current_no_load=0.45, no_load_voltage=11.1
    )
    battery = catalog.BATTERIES["B3S1000"].with_measurements(
        cell_resistance=0.014
    )

    for m, p, b in catalog.combinations():
        ...                            # 3 * 3 * 2 = 18
"""


from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "Motor",
    "Battery",
    "PropellerEntry",
    "ESC",
    "MOTORS",
    "BATTERIES",
    "PROPELLERS",
    "ESCS",
    "combinations",
    "RC1_BASELINE",
]

RPM_PER_V_TO_RAD_PER_S_PER_V = 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class Motor:
    """A brushless outrunner, as the standard three-parameter model.

    Attributes
    ----------
    key, name : str
    Kv_rpm : float
        Speed constant as the vendor quotes it, **RPM/V** (no-load).
    resistance : float
        Winding resistance, ohms.  Datasheet value until measured.
    current_no_load : float
        No-load current ``I0``, amperes, at ``no_load_voltage``.  Datasheet
        value until measured.
    no_load_voltage : float
        Voltage at which ``I0`` was measured, V.
    current_max : float
        Manufacturer's continuous current limit, A.
    mass : float
        Mass in kg, motor only, without prop adapter.
    cells_min, cells_max : int
        Recommended LiPo cell count.
    notes : str
    """

    key: str
    name: str
    Kv_rpm: float
    resistance: float
    current_no_load: float
    current_max: float
    mass: float
    no_load_voltage: float = 11.1
    cells_min: int = 2
    cells_max: int = 3
    notes: str = ""

    def __post_init__(self) -> None:
        positive = {
            "Kv_rpm": self.Kv_rpm,
            "resistance": self.resistance,
            "current_no_load": self.current_no_load,
            "current_max": self.current_max,
            "mass": self.mass,
            "no_load_voltage": self.no_load_voltage,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"motor parameters must be positive: {', '.join(invalid)}")
        if self.cells_min < 1 or self.cells_max < self.cells_min:
            raise ValueError("require 1 <= cells_min <= cells_max")

    def with_measurements(
        self,
        *,
        resistance: float,
        current_no_load: float,
        no_load_voltage: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> "Motor":
        """Return this motor with measured electrical parameters.

        The catalog object is immutable and remains unchanged. ``resistance``
        is winding resistance in ohms; ``current_no_load`` is amperes measured
        at ``no_load_voltage``. The returned component can be passed directly
        to :func:`operating_point <flightlab.propulsion.operating_point>`.
        """
        voltage = self.no_load_voltage if no_load_voltage is None else no_load_voltage
        measurement_notes = notes or (
            f"Measured electrical parameters replace the starter estimates for {self.name}."
        )
        return replace(
            self,
            resistance=resistance,
            current_no_load=current_no_load,
            no_load_voltage=voltage,
            notes=measurement_notes,
        )

    @property
    def Kv(self) -> float:
        """Speed constant in **rad/s/V** -- the form the torque balance needs."""
        return self.Kv_rpm * RPM_PER_V_TO_RAD_PER_S_PER_V

    @property
    def Kt(self) -> float:
        """Torque constant in **N*m/A**, the reciprocal of :attr:`Kv`.

        For an ideal motor the torque constant and the speed constant are the
        same number in SI units.  Quoting one in RPM/V and the other in N*m/A
        is what hides the ``2*pi/60``.
        """
        return 1.0 / self.Kv

    def peak_efficiency(self, voltage) -> float:
        """Closed-form maximum efficiency of the three-parameter model.

        .. math::

            \\eta_{max} = \\left(1 - \\sqrt{\\frac{I_0 R}{V}}\\right)^2

        Parameters
        ----------
        voltage : float or array_like
            Applied terminal voltage, V.

        Returns
        -------
        float or ndarray

        Notes
        -----
        The important result is that peak efficiency depends on ``I0``, ``R`` and ``V``,
        and **not on Kv at all**: the middle-Kv motor has the best peak and
        the highest-Kv motor the worst.  The peaks sit within three points of
        each other; what really separates the motors is the current at which
        the peak sits (:meth:`current_at_peak_efficiency`), which is 6 A for
        the KV1260, 12 A for the KV1100 and 16 A for the KV1400.
        """
        v = np.asarray(voltage, dtype=float)
        return (1.0 - np.sqrt(self.current_no_load * self.resistance / v)) ** 2

    def current_at_peak_efficiency(self, voltage):
        """Current at which :meth:`peak_efficiency` occurs, A."""
        v = np.asarray(voltage, dtype=float)
        return np.sqrt(self.current_no_load * v / self.resistance)


@dataclass(frozen=True)
class Battery:
    """A lithium-polymer pack.

    Attributes
    ----------
    key, name : str
    cells_series, cells_parallel : int
        ``s`` and ``p``.
    capacity_ah : float
        Rated capacity, amp-hours.
    mass : float
        Pack mass, kg.
    c_rating : float
        Manufacturer's continuous discharge rating, in multiples of capacity.
    cell_resistance : float
        Internal resistance of one cell, ohms.
    cell_voltage_nominal, cell_voltage_full, cell_voltage_empty : float
        Volts per cell.
    notes : str
    """

    key: str
    name: str
    cells_series: int
    cells_parallel: int
    capacity_ah: float
    mass: float
    c_rating: float
    cell_resistance: float
    cell_voltage_nominal: float = 3.7
    cell_voltage_full: float = 4.2
    cell_voltage_empty: float = 3.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.cells_series < 1 or self.cells_parallel < 1:
            raise ValueError("battery series and parallel cell counts must be positive")
        positive = {
            "capacity_ah": self.capacity_ah,
            "mass": self.mass,
            "c_rating": self.c_rating,
            "cell_resistance": self.cell_resistance,
            "cell_voltage_nominal": self.cell_voltage_nominal,
            "cell_voltage_full": self.cell_voltage_full,
            "cell_voltage_empty": self.cell_voltage_empty,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"battery parameters must be positive: {', '.join(invalid)}")
        if not self.cell_voltage_empty < self.cell_voltage_nominal < self.cell_voltage_full:
            raise ValueError(
                "require cell_voltage_empty < cell_voltage_nominal < cell_voltage_full"
            )

    def with_measurements(
        self,
        *,
        cell_resistance: float,
        notes: Optional[str] = None,
    ) -> "Battery":
        """Return this pack with measured per-cell internal resistance.

        ``cell_resistance`` is in ohms for one cell. Pack resistance is derived
        from the series/parallel arrangement. The original catalog entry is not
        modified, and the returned component can be passed directly to
        :func:`flightlab.propulsion.operating_point`.
        """
        measurement_notes = notes or (
            f"Measured cell resistance replaces the starter estimate for {self.name}."
        )
        return replace(
            self,
            cell_resistance=cell_resistance,
            notes=measurement_notes,
        )

    @property
    def capacity(self) -> float:
        """Rated capacity in **coulombs**."""
        return self.capacity_ah * 3600.0

    @property
    def voltage_nominal(self) -> float:
        """Nominal pack voltage, V."""
        return self.cells_series * self.cell_voltage_nominal

    @property
    def voltage_full(self) -> float:
        """Fully charged pack voltage, V."""
        return self.cells_series * self.cell_voltage_full

    @property
    def voltage_empty(self) -> float:
        """Pack voltage at the discharge floor, V."""
        return self.cells_series * self.cell_voltage_empty

    @property
    def resistance(self) -> float:
        """Pack internal resistance, ohms: ``s/p`` times the cell value."""
        return self.cell_resistance * self.cells_series / self.cells_parallel

    @property
    def energy_nominal(self) -> float:
        """Nominal stored energy in **joules**.

        Using amp-hours where energy is required turns tens of minutes of
        endurance into hours; use this joule-valued property directly.
        """
        return self.voltage_nominal * self.capacity

    @property
    def specific_energy(self) -> float:
        """Nominal specific energy, J/kg."""
        return self.energy_nominal / self.mass

    @property
    def current_max(self) -> float:
        """Continuous current limit, A."""
        return self.c_rating * self.capacity_ah


@dataclass(frozen=True)
class PropellerEntry:
    """A catalog propeller, pointing at its measured UIUC data.

    Attributes
    ----------
    key : str
        Catalog key, e.g. ``"P10x4.7"``.
    data : str
        Key for :func:`flightlab.props.load`, e.g. ``"apcsf_10x4.7"``.
    name : str
    mass : float
        Mass in kg, including the hub but not the adapter, as APC publishes it.
    rpm_max : float, optional
        The manufacturer's maximum rotational speed, rev/min.  APC rates its
        thin electric line to ``190,000 / D`` and its slow flyer line to
        ``65,000 / D`` with ``D`` in inches; the thin, flexible slow flyer
        blades flutter and shed past that.  The analysis warns when an
        operating point exceeds it.
    notes : str
    """

    key: str
    data: str
    name: str
    mass: float
    rpm_max: Optional[float] = None
    notes: str = ""

    def load(self):
        """Load the measured data via :func:`flightlab.props.load`."""
        from . import props

        return props.load(self.data)


@dataclass(frozen=True)
class ESC:
    """An electronic speed controller.

    Attributes
    ----------
    key, name : str
    current_max : float
        Continuous current rating, A.
    mass : float
        kg.
    efficiency : float
        Assumed constant efficiency. Real ESC losses are not constant, so name
        that simplification when reconciling the model with measurements.
    """

    key: str
    name: str
    current_max: float
    mass: float
    efficiency: float = 0.95


# --- the catalog ------------------------------------------------------------

_MOTOR_NOTE = (
    "Kv, resistance, no-load current, current_max and mass are SunnySky's "
    "published datasheet figures for this winding (no-load current quoted at "
    "10 V); confirm them against the listing at order time."
)

MOTORS: Dict[str, Motor] = {
    m.key: m
    for m in (
        Motor(
            key="M1100",
            name="SunnySky X2216 II KV1100",
            Kv_rpm=1100.0,
            resistance=0.073,
            current_no_load=0.90,
            current_max=24.0,
            mass=0.072,
            no_load_voltage=10.0,
            cells_min=2,
            cells_max=4,
            notes=_MOTOR_NOTE + " Lowest Kv in the catalog on the larger "
            "22 x 16 mm stator: it turns the big slow-flyer propellers at a "
            "speed the data covers and carries 24 A doing it. 26 g heavier "
            "than the X2208, which on a 400 g airplane is not nothing. "
            "SunnySky recommends a 30 A ESC and the APC 9x4.7, 9x4.5, 9x6, "
            "8x6 and 10x4.7.",
        ),
        Motor(
            key="M1260",
            name="SunnySky X2208 KV1260",
            Kv_rpm=1260.0,
            resistance=0.122,
            current_no_load=0.40,
            current_max=15.0,
            mass=0.046,
            no_load_voltage=10.0,
            cells_min=2,
            cells_max=4,
            notes=_MOTOR_NOTE + " The small 22 x 8 mm stator: lightest and "
            "cheapest motor here, the best peak efficiency of the three, and a "
            "peak that lands at only 6 A -- the right motor for a light, slow "
            "airplane, and RC-1's. SunnySky's static table gives 780 gf at "
            "13.9 A on an APC 9x4.7 SF and 570 gf at 11.7 A on a 10x4.7 SF, "
            "both at 11.1 V, and recommends an 18 A ESC.",
        ),
        Motor(
            key="M1400",
            name="SunnySky X2216 II KV1400",
            Kv_rpm=1400.0,
            resistance=0.055,
            current_no_load=1.30,
            current_max=33.0,
            mass=0.072,
            no_load_voltage=10.0,
            cells_min=2,
            cells_max=4,
            notes=_MOTOR_NOTE + " Same can as the KV1100 with fewer turns of "
            "thicker wire: lower resistance, a no-load current of 1.3 A, and "
            "the WORST peak efficiency of the three despite the highest Kv, "
            "because peak efficiency ranks with I0*R. It needs about 16 A to "
            "reach that peak; on a small propeller at 3 A it is the least "
            "efficient motor here by a wide margin. SunnySky recommends a 40 A "
            "ESC and the APC 9x4.7, 9x4.5, 9x6, 8x6 and 7x6.",
        ),
    )
}


BATTERIES: Dict[str, Battery] = {
    b.key: b
    for b in (
        Battery(
            key="B2S1500",
            name="Turnigy 2S 7.4 V 1500 mAh 40C LiPo",
            cells_series=2,
            cells_parallel=1,
            capacity_ah=1.500,
            mass=0.085,
            c_rating=40.0,
            cell_resistance=0.010,
            notes=(
                "Turnigy 1500 mAh 2S 40C with XT60; capacity, C rating and "
                "the 85 g mass are HobbyKing's listing, cell_resistance is an "
                "estimate (no pack vendor publishes it). Nearly the same mass "
                "and stored energy as B3S1000, delivered at two thirds the "
                "voltage: the same motor and propeller turn about a third "
                "slower on it."
            ),
        ),
        Battery(
            key="B3S1000",
            name="Turnigy 3S 11.1 V 1000 mAh 40C LiPo",
            cells_series=3,
            cells_parallel=1,
            capacity_ah=1.000,
            mass=0.090,
            c_rating=40.0,
            cell_resistance=0.013,
            notes=(
                "Turnigy 1000 mAh 3S 40C with XT60; capacity and C rating are "
                "HobbyKing's listing, the mass is typical for the pack (the "
                "listings disagree between 88 and 105 g: weigh one), and "
                "cell_resistance is an estimate. RC-1's baseline pack."
            ),
        ),
    )
}


PROPELLERS: Dict[str, PropellerEntry] = {
    p.key: p
    for p in (
        PropellerEntry(
            key="P8x6",
            data="apce_8x6",
            name="APC 8x6E",
            mass=0.0139,
            rpm_max=190_000.0 / 8.0,
            notes=(
                "APC thin electric; UIUC measured performance, APC published "
                "mass (0.49 oz). The only fast propeller here: p/D = 0.75 puts "
                "its efficiency peak near J = 0.6, and it is the one the KV1400 "
                "motor is sized for."
            ),
        ),
        PropellerEntry(
            key="P9x4.7",
            data="apcsf_9x4.7",
            name="APC 9x4.7SF",
            mass=0.0091,
            rpm_max=65_000.0 / 9.0,
            notes=(
                "APC slow flyer; UIUC measured performance, APC published "
                "mass (0.32 oz). Thin, flexible blades meant for low rpm: APC "
                "rates the SF line to 65,000/D rpm, 7,200 rpm for this one, "
                "and the analysis warns when a motor spins it past that."
            ),
        ),
        PropellerEntry(
            key="P10x4.7",
            data="apcsf_10x4.7",
            name="APC 10x4.7SF",
            mass=0.0119,
            rpm_max=65_000.0 / 10.0,
            notes=(
                "APC slow flyer; UIUC measured performance, APC published "
                "mass (0.42 oz). Largest disk in the catalog, which is the "
                "efficient way to make static thrust, at the cost of the "
                "highest torque demand. Same pitch as P9x4.7, so the pair is "
                "a controlled diameter comparison. SF rpm limit 6,500."
            ),
        ),
    )
}

ESCS: Dict[str, ESC] = {
    e.key: e
    for e in (
        ESC(key="ESC40", name="SunnySky X-series 40 A airplane ESC, 4 A BEC",
            current_max=40.0, mass=0.040, efficiency=0.96),
    )
}

#: RC-1's baseline combination, as named on the fleet page.
#:
#: **It is deliberately a poor match, and analyzing it is how students find
#: that out.** The cheapest motor, the biggest propeller, and the 3S pack
#: "because 3S has more power" is how a beginner chooses. At full throttle the
#: torque balance sits near 9,000 rpm, past APC's 6,500 rpm rating for the
#: slow-flyer blade, and pulls about 30 A through a motor rated for 15 A.
#: Both are visible from the tools, and fixing it -- a smaller propeller, the
#: 2S pack, or a throttle limit -- is the design section of the propulsion
#: week.
RC1_BASELINE = ("M1260", "P10x4.7", "B3S1000")


def combinations() -> Iterator[Tuple[Motor, PropellerEntry, Battery]]:
    """Iterate over all 18 motor-propeller-battery combinations.

    Yields
    ------
    (Motor, PropellerEntry, Battery)

    Examples
    --------
    >>> len(list(combinations()))
    18
    """
    for m, p, b in itertools.product(
        MOTORS.values(), PROPELLERS.values(), BATTERIES.values()
    ):
        yield m, p, b
