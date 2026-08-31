"""``flightlab.catalog`` -- the bounded propulsion catalog.

Four motors, five propellers, three batteries.  Sixty combinations is a genuine
finite search with a right answer, small enough to stock and to measure on the
thrust stand, and the student orders the exact part they analyzed.

.. warning::

   **The motor and battery numbers here are provisional.**  Every propeller is
   real, with a real measured UIUC data file behind it.  The motors and packs
   are specified the way the fleet page specifies RC-1's -- generically ("the
   generic 1000 Kv motor") -- with physically plausible parameters chosen so
   the whole toolchain runs end to end.  They are **not** transcribed from a
   vendor datasheet, and ``entry.provisional`` is ``True`` for all of them.

   Replace two groups of values when course measurements are available:

   1. ``Kv``, ``mass`` and ``current_max`` from the datasheet of the part
      actually stocked;
   2. ``resistance`` and ``current_no_load`` from the **TA thrust-stand
      measurement**.  Vendors in this price tier do not publish winding
      resistance or no-load current, and estimates are the least trustworthy
      numbers in any catalog, which is why they should be measured.

   :func:`check_provisional` returns what still needs replacing.

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

    catalog.MOTORS["M1000"].Kv        # rad/s/V, for the torque constant
    catalog.MOTORS["M1000"].Kv_rpm    # RPM/V, as the vendor quotes it
    catalog.BATTERIES["B3S1300"].energy_nominal / 3600   # W*h

    motor = catalog.MOTORS["M1000"].with_measurements(
        resistance=0.095, current_no_load=0.82, no_load_voltage=11.1
    )
    battery = catalog.BATTERIES["B3S1300"].with_measurements(
        cell_resistance=0.010
    )

    for m, p, b in catalog.combinations():
        ...                            # 4 * 5 * 3 = 60

    catalog.check_provisional()        # what still needs real data
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
    "check_provisional",
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
        Winding resistance, ohms.  Measured, not published.
    current_no_load : float
        No-load current ``I0``, amperes, at ``no_load_voltage``.  Measured.
    no_load_voltage : float
        Voltage at which ``I0`` was measured, V.
    current_max : float
        Manufacturer's continuous current limit, A.
    mass : float
        Mass in kg, motor only, without prop adapter.
    cells_min, cells_max : int
        Recommended LiPo cell count.
    provisional : bool
        True while the electrical parameters are placeholders.
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
    provisional: bool = True
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
        at ``no_load_voltage``. The returned component is marked as no longer
        provisional and can be passed directly to :func:`operating_point
        <flightlab.propulsion.operating_point>`.
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
            provisional=False,
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
        and **not on Kv at all**.  Ranking these four motors by ``Kv`` and by
        peak efficiency gives two different orders.
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
    provisional : bool
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
    provisional: bool = True
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
            provisional=False,
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
        Catalog key, e.g. ``"P10x7"``.
    data : str
        Key for :func:`flightlab.props.load`, e.g. ``"apce_10x7"``.
    name : str
    mass : float
        Mass in kg, including the hub but not the adapter.
    provisional : bool
        The performance data is measured; only ``mass`` is a placeholder.
    notes : str
    """

    key: str
    data: str
    name: str
    mass: float
    provisional: bool = False
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
    provisional : bool
    """

    key: str
    name: str
    current_max: float
    mass: float
    efficiency: float = 0.95
    provisional: bool = True


# --- the catalog ------------------------------------------------------------

_MOTOR_NOTE = (
    "Kv, mass and current_max are the vendor's published figures for a real, "
    "orderable part and should be confirmed against the listing at order time. "
    "resistance and current_no_load are ESTIMATES -- vendors in this tier do "
    "not publish them -- and must be replaced by the TA thrust-stand "
    "measurement when measured course data are available."
)

MOTORS: Dict[str, Motor] = {
    m.key: m
    for m in (
        Motor(
            key="M820",
            name="EMAX XA2212-820, 820 Kv",
            Kv_rpm=820.0,
            resistance=0.200,
            current_no_load=0.60,
            current_max=16.0,
            mass=0.057,
            no_load_voltage=11.1,
            cells_min=2,
            cells_max=3,
            notes=_MOTOR_NOTE + " Lowest Kv here: turns a large propeller "
            "slowly, which is the efficient way to make thrust at low speed.",
        ),
        Motor(
            key="M1000",
            name="generic A2212-13T, 1000 Kv",
            Kv_rpm=1000.0,
            resistance=0.100,
            current_no_load=0.85,
            current_max=13.0,
            mass=0.050,
            no_load_voltage=11.1,
            cells_min=2,
            cells_max=3,
            notes=_MOTOR_NOTE + " The cheapest and most widely stocked motor "
            "of this class, and RC-1's baseline. Note the low current limit.",
        ),
        Motor(
            key="M1250",
            name="SunnySky X2216-11, 1250 Kv",
            Kv_rpm=1250.0,
            resistance=0.060,
            current_no_load=1.10,
            current_max=26.0,
            mass=0.068,
            no_load_voltage=11.1,
            cells_min=2,
            cells_max=3,
            notes=_MOTOR_NOTE + " Highest peak efficiency in the catalog, and "
            "18 g heavier than the lightest. Whether that is worth it depends "
            "on the mission, which is the point.",
        ),
        Motor(
            key="M1400",
            name="generic A2212-10T, 1400 Kv",
            Kv_rpm=1400.0,
            resistance=0.085,
            current_no_load=1.45,
            current_max=15.0,
            mass=0.050,
            no_load_voltage=11.1,
            cells_min=2,
            cells_max=3,
            notes=_MOTOR_NOTE + " Highest Kv and the WORST peak efficiency, "
            "because peak efficiency ranks with I0*R/V and this motor's "
            "no-load current is high. Students expect the opposite.",
        ),
    )
}


BATTERIES: Dict[str, Battery] = {
    b.key: b
    for b in (
        Battery(
            key="B2S1300",
            name="2S 7.4 V 1300 mAh 45C LiPo",
            cells_series=2,
            cells_parallel=1,
            capacity_ah=1.300,
            mass=0.075,
            c_rating=45.0,
            cell_resistance=0.010,
            notes=(
                "Widely stocked hobby pack. Mass is the vendor figure; "
                "cell_resistance is an ESTIMATE to be measured."
            ),
        ),
        Battery(
            key="B3S2200",
            name="3S 11.1 V 2200 mAh 50C LiPo",
            cells_series=3,
            cells_parallel=1,
            capacity_ah=2.200,
            mass=0.185,
            c_rating=50.0,
            cell_resistance=0.008,
            notes=(
                "70 g heavier than B3S1300 for 69% more energy. Whether that "
                "is a good trade is the endurance-versus-mass question, and it "
                "depends on the mission. cell_resistance is an ESTIMATE."
            ),
        ),
        Battery(
            key="B3S1300",
            name="3S 11.1 V 1300 mAh 45C LiPo",
            cells_series=3,
            cells_parallel=1,
            capacity_ah=1.300,
            mass=0.115,
            c_rating=45.0,
            cell_resistance=0.012,
            notes=(
                "RC-1's baseline pack; its 115 g matches the battery row of "
                "the RC-1 component mass table. cell_resistance is an ESTIMATE."
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
            mass=0.010,
            notes="performance measured; mass is a placeholder",
        ),
        PropellerEntry(
            key="P9x6",
            data="apce_9x6",
            name="APC 9x6E",
            mass=0.012,
            notes="performance measured; mass is a placeholder",
        ),
        PropellerEntry(
            key="P10x5",
            data="apce_10x5",
            name="APC 10x5E",
            mass=0.014,
            notes=(
                "performance measured; mass is a placeholder. Same diameter as "
                "P10x7 and less pitch, making it a controlled pitch comparison."
            ),
        ),
        PropellerEntry(
            key="P10x7",
            data="apce_10x7",
            name="APC 10x7E",
            mass=0.014,
            notes=(
                "performance measured; mass is a placeholder. RC-1's baseline "
                "propeller, and the one on the thrust stand."
            ),
        ),
        PropellerEntry(
            key="P11x7",
            data="apce_11x7",
            name="APC 11x7E",
            mass=0.017,
            notes="performance measured; mass is a placeholder",
        ),
    )
}

ESCS: Dict[str, ESC] = {
    e.key: e
    for e in (
        ESC(key="ESC30", name="30 A brushless ESC with 5 V BEC",
            current_max=30.0, mass=0.025, efficiency=0.95),
        ESC(key="ESC40", name="40 A brushless ESC with 5 V BEC",
            current_max=40.0, mass=0.038, efficiency=0.96),
    )
}

#: RC-1's baseline combination, as named on the fleet page.
#:
#: **It is deliberately a poor match, and analyzing it is how students find
#: that out.** A 1000 Kv motor on a 3S pack spins the 10x7 to about 7,700 rpm;
#: the fastest sweep UIUC measured for that propeller is 6,531 rpm, so every
#: thrust number for this combination rests on a clamped lookup. It also draws
#: well past the airframe's power allowance and past the motor's own current
#: limit. Every one of those is visible from the tools, and fixing it is the
#: design section of the propulsion week.
RC1_BASELINE = ("M1000", "P10x7", "B3S1300")


def combinations() -> Iterator[Tuple[Motor, PropellerEntry, Battery]]:
    """Iterate over all 60 motor-propeller-battery combinations.

    Yields
    ------
    (Motor, PropellerEntry, Battery)

    Examples
    --------
    >>> len(list(combinations()))
    60
    """
    for m, p, b in itertools.product(
        MOTORS.values(), PROPELLERS.values(), BATTERIES.values()
    ):
        yield m, p, b


def check_provisional() -> Dict[str, List[str]]:
    """Report which catalog entries still hold placeholder data.

    Returns
    -------
    dict
        Maps ``"motors"``, ``"batteries"``, ``"propellers"``, ``"escs"`` to the
        list of keys whose ``provisional`` flag is still set.  An empty list
        means that group has been replaced with real data.

    Examples
    --------
    >>> sorted(check_provisional()["motors"])
    ['M1000', 'M1200', 'M1400', 'M850']
    """
    return {
        "motors": [k for k, v in MOTORS.items() if v.provisional],
        "batteries": [k for k, v in BATTERIES.items() if v.provisional],
        "propellers": [k for k, v in PROPELLERS.items() if v.provisional],
        "escs": [k for k, v in ESCS.items() if v.provisional],
    }
