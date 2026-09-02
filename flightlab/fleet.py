"""``flightlab.fleet`` -- the focused course fleet as importable data.

Transcription errors look exactly like physics errors, so nobody should type
specification tables repeatedly.  Import the numbers instead::

    from flightlab.fleet import (RC1, B787, ASW27, ASG29, C172, JobyS4,
                                 SaturnV)

**Tables only.**  This module holds geometry, component masses, and published
performance.  Building grids from it is :mod:`flightlab.geom`; computing mass
properties from it is :mod:`flightlab.stability`.

Provenance
----------
Every scalar is either sourced or flagged as an engineering estimate.  Each
aircraft carries a ``sources`` tuple and an ``estimated`` frozenset naming the
fields that are estimates::

    B787.estimated                  # -> frozenset of dotted field names
    B787.is_estimated("wing.taper") # -> True

Do not cite an estimated value in a report without saying where it came from.

Units
-----
SI throughout: metres, m^2, kilograms, seconds, m/s, newtons, watts, pascals.
**Angles are in degrees**, and every field that holds one says so in its name.
Positions are in the aircraft's own body axes: ``x`` aft, ``y`` right, ``z`` up,
with the origin at the wing leading edge unless the aircraft's notes say
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

__all__ = [
    "Planform",
    "Body",
    "Component",
    "Aircraft",
    "Stage",
    "Rocket",
    "RC1",
    "B787",
    "ASW27",
    "ASG29",
    "C172",
    "JobyS4",
    "SaturnV",
    "AIRCRAFT",
    "ALL",
]

KT = 0.514444  # knots to m/s
KMH = 1.0 / 3.6  # km/h to m/s
FT = 0.3048  # feet to m
HP = 745.6999  # horsepower to W


# --- building blocks --------------------------------------------------------


@dataclass(frozen=True)
class Planform:
    """A lifting surface's planform.

    Attributes
    ----------
    span : float
        Tip-to-tip span, m.  For a vertical tail this is the height, and
        ``vertical=True``.
    area : float
        Reference (planform) area, m^2.
    section : str
        Airfoil designation, resolvable by :func:`flightlab.foil.load` where
        ``section_file`` is set.
    section_file : str, optional
        Bundled coordinate file to use, when the real section's coordinates are
        not public.  If this differs from ``section``, you are analyzing a
        **stand-in** and your report has to say so.
    root_chord, tip_chord : float, optional
        Chord at the centreline and tip, m.
    taper : float, optional
        ``tip_chord / root_chord``.
    mean_chord : float, optional
        Mean aerodynamic (or published mean) chord, m.
    sweep_le_deg, sweep_c4_deg : float, optional
        Leading-edge and quarter-chord sweep, degrees.
    dihedral_deg, twist_deg, incidence_deg : float
        Degrees.  ``twist_deg`` is the tip's twist relative to the root,
        negative for washout.
    thickness : float, optional
        ``t/c``.
    x_le, x_c4, z : float, optional
        Position of the surface's root leading edge, root quarter chord, and
        vertical offset, m, in the aircraft's body axes.
    vertical : bool
        True for a fin, whose ``span`` is a height and which has only one side.
    notes : str
    """

    span: float
    area: float
    section: str = ""
    section_file: str = ""
    root_chord: Optional[float] = None
    tip_chord: Optional[float] = None
    taper: Optional[float] = None
    mean_chord: Optional[float] = None
    sweep_le_deg: Optional[float] = None
    sweep_c4_deg: Optional[float] = None
    dihedral_deg: float = 0.0
    twist_deg: float = 0.0
    incidence_deg: float = 0.0
    thickness: Optional[float] = None
    x_le: Optional[float] = None
    x_c4: Optional[float] = None
    z: float = 0.0
    vertical: bool = False
    notes: str = ""

    @property
    def aspect_ratio(self) -> float:
        """``b^2 / S``.  For a fin this is the single-surface value."""
        return self.span**2 / self.area

    @property
    def is_stand_in(self) -> bool:
        """True if ``section_file`` is a substitute for the real section."""
        if not self.section_file:
            return False
        norm = lambda s: s.lower().replace(" ", "").replace("-", "")  # noqa: E731
        return norm(self.section_file) not in norm(self.section)


@dataclass(frozen=True)
class Body:
    """A non-lifting body: fuselage, pod, tail boom, or nacelle.

    Attributes
    ----------
    name : str
    length : float
        Length, m.
    width, height, diameter : float, optional
        Cross-section dimensions, m.  Use ``diameter`` for round bodies.
    x_nose : float, optional
        Axial position of the nose, m.
    y, z : float
        Cross-section centreline position, m. Most bodies lie on the aircraft
        centreline; offsets matter for geometry-derived inertia and moments.
    count : int
        How many of these the aircraft has.
    cone_fraction : float
        Fraction of the length made up by rounded nose and tail cones together.
        Wetted area is ``pi d L (1 - 0.25 * cone_fraction)``.
    drag_model : str
        ``"streamlined"`` -- a body of revolution, whose drag is skin friction
        on its wetted area times a form factor.  ``"crossflow"`` -- a strut,
        gear leg or other bluff item lying across the flow, whose drag is a
        coefficient on its *frontal* area and is an order of magnitude larger
        for the same size.  Getting this wrong on a strutted fixed-gear
        aircraft halves its drag.
    notes : str
    """

    name: str
    length: float
    width: Optional[float] = None
    height: Optional[float] = None
    diameter: Optional[float] = None
    x_nose: Optional[float] = None
    y: float = 0.0
    z: float = 0.0
    count: int = 1
    drag_model: str = "streamlined"
    cone_fraction: float = 0.4
    notes: str = ""

    @property
    def frontal_area(self) -> float:
        """Projected frontal area, m^2, times ``count``.

        What a bluff body's drag coefficient is referenced to.  A round strut
        or an unfaired gear leg is a cylinder in crossflow, and its drag has
        almost nothing to do with its wetted area.
        """
        if self.diameter is not None:
            return self.count * self.diameter * self.length
        if self.width is not None and self.height is not None:
            return self.count * self.height * self.length
        if self.width is not None:
            return self.count * self.width * self.length
        raise ValueError(f"{self.name}: no cross-section dimension given")

    @property
    def effective_diameter(self) -> float:
        """``sqrt(4 S_max / pi)``, m -- the diameter of the equivalent circle.

        A round body returns its own diameter.  A non-circular one is reduced
        to the circle of equal maximum cross-sectional area, which is how the
        course text defines the fineness ratio for one.
        """
        if self.diameter is not None:
            return self.diameter
        if self.width is not None and self.height is not None:
            s_max = 0.25 * 3.141592653589793 * self.width * self.height
            return (4.0 * s_max / 3.141592653589793) ** 0.5
        if self.width is not None:
            return self.width
        raise ValueError(f"{self.name}: no cross-section dimension given")

    @property
    def fineness(self) -> float:
        """Length over effective diameter."""
        return self.length / self.effective_diameter


@dataclass(frozen=True)
class Component:
    """One row of a component mass table.

    Attributes
    ----------
    name : str
    mass : float
        Mass in **kilograms** (the fleet page tabulates grams; converted here).
    x, y, z : float
        Position of the component's own centre of mass, m.
    distributed : str
        ``""`` for a point mass on the centreline, or ``"span"`` for a mass
        spread uniformly along the span of the named surface.  This matters for
        the inertias, not for the centre of mass.
    span : float, optional
        Span over which a distributed mass is spread, m.
    """

    name: str
    mass: float
    x: float
    y: float = 0.0
    z: float = 0.0
    distributed: str = ""
    span: Optional[float] = None
    Ixx_cg: float = 0.0
    Iyy_cg: float = 0.0
    Izz_cg: float = 0.0
    Ixz_cg: float = 0.0


@dataclass(frozen=True)
class Aircraft:
    """A fleet aircraft.

    Attributes
    ----------
    name, label : str
        Full name and short key.
    aircraft_class : str
    wing : Planform
    htail, vtail : Planform, optional
    bodies : tuple of Body
    mass : dict
        Masses in kg.  Common keys: ``empty``, ``gross``, ``mtow``,
        ``payload``, ``fuel``, ``cruise_start``, ``cruise_end``.
    components : tuple of Component
        The component mass table, where one is specified.
    operating : dict
        ``cruise_speed`` (m/s), ``cruise_altitude`` (m), ``cruise_mach``,
        ``field_altitude`` (m), and similar.
    limits : dict
        ``n_pos``, ``n_neg`` load factors and any published speed limits (m/s).
    placeholders : dict
        Early-design values to be replaced by the student's own analysis:
        ``CDp``, ``e``, ``CLmax``.
    published : dict
        Published performance to check against: ``LD_max``, ``glide_ratio``,
        ``min_sink`` (m/s), ``range`` (m), and similar.
    structure : dict
        Material allowables, e.g. ``sigma_allow`` (Pa), ``rho`` (kg/m^3).
    propulsion : dict
    sources : tuple of str
    estimated : frozenset of str
        Dotted field names whose values are engineering estimates.
    notes : str
    """

    name: str
    label: str
    aircraft_class: str
    wing: Optional[Planform] = None
    htail: Optional[Planform] = None
    vtail: Optional[Planform] = None
    bodies: Tuple[Body, ...] = ()
    mass: Dict[str, float] = field(default_factory=dict)
    components: Tuple[Component, ...] = ()
    operating: Dict[str, float] = field(default_factory=dict)
    limits: Dict[str, float] = field(default_factory=dict)
    placeholders: Dict[str, float] = field(default_factory=dict)
    published: Dict[str, float] = field(default_factory=dict)
    structure: Dict[str, float] = field(default_factory=dict)
    propulsion: Dict[str, object] = field(default_factory=dict)
    sources: Tuple[str, ...] = ()
    estimated: frozenset = frozenset()
    notes: str = ""

    def is_estimated(self, path: str) -> bool:
        """Whether the dotted field ``path`` is an engineering estimate."""
        return path in self.estimated

    @property
    def component_mass_total(self) -> float:
        """Sum of the component table, kg.  Should match ``mass['gross']``."""
        return sum(c.mass for c in self.components)

    @property
    def wing_loading(self) -> float:
        """``W/S`` at gross mass, N/m^2."""
        if self.wing is None:
            raise ValueError(f"{self.label} has no wing planform on record")
        g = 9.80665
        m = self.mass.get("gross") or self.mass.get("mtow")
        return m * g / self.wing.area

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Aircraft {self.label}: {self.name}>"


@dataclass(frozen=True)
class Stage:
    """One stage of a launch vehicle.

    Attributes
    ----------
    name : str
    propellant, dry : float
        Masses, kg.
    thrust : float
        Newtons, at the condition named in ``thrust_condition``.
    thrust_condition : str
        ``"SL"`` or ``"vac"``.
    isp_sl, isp_vac : float, optional
        Specific impulse, seconds.
    burn_time : float
        Seconds.  The S-IVB burns twice; both are given in ``burn_times``.
    burn_times : tuple of float
    height, diameter : float, optional
        Metres.
    """

    name: str
    propellant: float
    dry: float
    thrust: float
    thrust_condition: str
    isp_sl: Optional[float] = None
    isp_vac: Optional[float] = None
    burn_time: Optional[float] = None
    burn_times: Tuple[float, ...] = ()
    height: Optional[float] = None
    diameter: Optional[float] = None

    @property
    def gross(self) -> float:
        """Propellant plus dry mass, kg."""
        return self.propellant + self.dry

    @property
    def mass_ratio(self) -> float:
        """``gross / dry``."""
        return self.gross / self.dry


@dataclass(frozen=True)
class Rocket:
    """A launch vehicle: a stack of stages plus overall dimensions."""

    name: str
    label: str
    stages: Tuple[Stage, ...]
    height: float
    diameter: float
    payload_leo: float
    sources: Tuple[str, ...] = ()
    estimated: frozenset = frozenset()
    notes: str = ""

    def is_estimated(self, path: str) -> bool:
        """Whether the dotted field ``path`` is an engineering estimate."""
        return path in self.estimated

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"<Rocket {self.label}: {self.name}, {len(self.stages)} stages>"


# --- RC-1 -------------------------------------------------------------------

RC1 = Aircraft(
    name="RC-1",
    label="RC1",
    aircraft_class="750 g electric UAV, pod and boom",
    notes=(
        "The class of airplane the students are building, and deliberately "
        "mediocre: a plausible first attempt, with real problems they will "
        "find. Its propulsion was chosen the way a beginner chooses one -- the "
        "motor was in stock and the propeller fit. Entirely estimated, because "
        "no published 750 g model airplane is specified to this precision. "
        "Sits inside the course rules of <= 1 kg and installed maximum "
        "electrical input power <= 150 W/kg of converged mass. "
        "x is measured aft from the wing leading edge; z is up. "
        "Note that it cruises only 13% above its stall speed."
    ),
    wing=Planform(
        span=1.200,
        area=0.192,
        root_chord=0.160,
        tip_chord=0.160,
        taper=1.0,
        mean_chord=0.160,
        sweep_le_deg=0.0,
        sweep_c4_deg=0.0,
        dihedral_deg=0.0,
        twist_deg=0.0,
        incidence_deg=0.0,
        section="NACA 2412",
        section_file="naca2412",
        thickness=0.12,
        x_le=0.0,
        x_c4=0.040,
        notes="rectangular: no taper, twist, dihedral or sweep",
    ),
    htail=Planform(
        span=0.340,
        area=0.034,
        root_chord=0.100,
        tip_chord=0.100,
        taper=1.0,
        mean_chord=0.100,
        section="NACA 0009",
        section_file="naca0009",
        thickness=0.09,
        incidence_deg=0.0,
        x_c4=0.590,
        x_le=0.565,
        z=0.010,
        notes="rectangular",
    ),
    vtail=Planform(
        span=0.150,
        area=0.018,
        root_chord=0.120,
        tip_chord=0.120,
        taper=1.0,
        mean_chord=0.120,
        section="NACA 0009",
        section_file="naca0009",
        thickness=0.09,
        x_c4=0.590,
        x_le=0.560,
        vertical=True,
        notes="upward only; area and span are for the single surface",
    ),
    bodies=(
        Body(
            name="fuselage pod",
            length=0.400,
            width=0.110,
            height=0.100,
            x_nose=-0.130,
            notes=(
                "rounded box; sized to hold a battery that slides fore and aft "
                "for balance, an ESC, a receiver, three servos, and your hands"
            ),
        ),
        Body(
            name="tail boom",
            length=0.295,
            diameter=0.010,
            x_nose=0.270,
            notes="carbon tube, pod aft end to tail leading edge",
        ),
    ),
    mass={"gross": 0.750, "payload": 0.200, "empty": 0.550},
    components=(
        Component("Wing", 0.130, 0.064, 0.0, 0.0, "span", 1.200),
        Component("Fuselage", 0.081, 0.030, 0.0, -0.005),
        Component("Horizontal tail", 0.022, 0.605, 0.0, 0.010, "span", 0.340),
        Component("Vertical tail", 0.011, 0.605, 0.0, 0.085, "span", 0.150),
        Component("Tail boom", 0.008, 0.420, 0.0, 0.010, "span", 0.295),
        Component("Motor", 0.048, -0.135, 0.0, 0.0),
        Component("Propeller and adapter", 0.014, -0.155, 0.0, 0.0),
        Component("ESC", 0.012, -0.060, 0.0, -0.010),
        Component("Battery", 0.115, 0.009, 0.0, -0.015),
        Component("Servos (3)", 0.027, 0.030, 0.0, -0.010),
        Component("Receiver", 0.008, 0.060, 0.0, -0.010),
        Component("Control linkages", 0.018, 0.300, 0.0, 0.0),
        Component("Landing skid", 0.020, 0.010, 0.0, -0.045),
        Component("Adhesives, wiring, misc.", 0.036, 0.020, 0.0, 0.0),
        Component("Mission payload", 0.200, 0.013, 0.0, -0.010),
    ),
    operating={
        "field_altitude": 1400.0,
        "cruise_speed": 9.1,
        "cruise_altitude": 1400.0,
        "stall_speed": 8.1,
        "chord_reynolds_cruise": 9.0e4,
    },
    limits={
        "mass_max": 1.0,
        "installed_electrical_power_per_mass_guideline": 150.0,
        "CLmax": 1.10,
        "CLmin": -0.60,
        "n_pos": 3.8,
        "n_neg": -1.0,
        "VC_eas": 20.0,
        "VD_eas": 25.0,
    },
    placeholders={"CDp": 0.045, "e": 0.70, "CLmax": 1.10},
    published={"LD_max": 9.6, "CL_cruise": 0.86},
    structure={
        "cap_E": 120e9,
        "cap_sigma_allow": 600e6,
        "cap_density": 1600.0,
        "cap_area_min_per_semispan": 4e-6,
        "foam_areal_mass": 0.57,
        "tip_deflection_fraction_max": 0.05,
        "ultimate_factor": 1.5,
    },
    propulsion={
        "motor": "generic 1000 Kv",
        "propeller": "10 x 7",
        "battery": "3S 1300 mAh",
        "x_cg_published": 0.048,
        "x_cg_fraction_of_chord": 0.30,
    },
    sources=("course baseline; fully specified and deliberately mediocre",),
    estimated=frozenset(
        {
            "wing.span", "wing.area", "wing.root_chord", "wing.section",
            "htail.span", "htail.area", "htail.x_c4",
            "vtail.span", "vtail.area", "vtail.x_c4",
            "bodies", "mass.gross", "mass.payload", "components",
            "operating.cruise_speed", "operating.stall_speed",
            "limits.CLmax", "limits.CLmin", "limits.n_pos", "limits.n_neg",
            "limits.VC_eas", "limits.VD_eas",
            "structure.cap_E", "structure.cap_sigma_allow", "structure.cap_density",
            "structure.cap_area_min_per_semispan", "structure.foam_areal_mass",
            "structure.tip_deflection_fraction_max", "structure.ultimate_factor",
            "placeholders.CDp", "placeholders.e", "placeholders.CLmax",
            "published.LD_max", "published.CL_cruise", "propulsion",
        }
    ),
)


# --- Boeing 787-8 -----------------------------------------------------------

B787 = Aircraft(
    name="Boeing 787-8",
    label="B787",
    aircraft_class="widebody transport",
    notes=(
        "The default aircraft of aerospace engineering, and the one most of "
        "the book's correlations were built from. For a component drag buildup, "
        "analyze each component in isolation -- do not subtract the wing area "
        "buried in the fuselage. At this Reynolds number a fully turbulent "
        "boundary layer is the standard, conservative assumption."
    ),
    wing=Planform(
        span=60.12,
        area=377.0,
        mean_chord=6.27,
        taper=0.15,
        sweep_c4_deg=32.2,
        thickness=0.11,
        section="supercritical",
        notes="span 197 ft 3 in; area 4,058 ft^2",
    ),
    htail=Planform(
        span=19.81,
        area=77.3,
        sweep_c4_deg=34.0,
        thickness=0.09,
        notes="832.5 ft^2; AR 5.08",
    ),
    vtail=Planform(
        span=8.7,
        area=39.7,
        sweep_c4_deg=38.0,
        thickness=0.09,
        vertical=True,
        notes="427.5 ft^2; 8.7 m is the half span used by the course buildup",
    ),
    bodies=(
        Body(
            name="fuselage",
            length=56.72,
            width=5.77,
            height=5.77,
            x_nose=0.0,
            notes="external width",
        ),
        Body(
            name="nacelle",
            length=5.6,
            diameter=3.6,
            count=2,
            notes="fan cowl diameter",
        ),
    ),
    mass={
        "mtow": 227930.0,
        "empty": 119950.0,
        "payload_max": 41050.0,
        "cruise_start": 200000.0,
        "cruise_end": 150000.0,
        "gross": 200000.0,
    },
    operating={
        "cruise_mach": 0.85,
        "cruise_altitude": 11900.0,
    },
    limits={"n_pos": 2.5, "n_neg": -1.0},
    placeholders={"CDp": 0.018, "e": 0.85},
    published={"CLmax_clean": 1.4, "CLmax_flaps": 3.0},
    structure={},
    propulsion={
        "engine": "GEnx-1B",
        "count": 2,
        "fan_diameter": 2.82,
        "bypass_ratio": 9.1,
        "takeoff_thrust_each": 285e3,
        "cruise_thrust_each_course_estimate": 55e3,
    },
    sources=(
        "Boeing, 787 Airplane Characteristics for Airport Planning",
        "EASA type certificate data sheet",
        "Lissys Piano sample analysis",
    ),
    estimated=frozenset(
        {
            "wing.taper", "wing.thickness", "wing.section",
            "htail.sweep_c4_deg", "htail.thickness",
            "vtail.span", "vtail.sweep_c4_deg", "vtail.thickness",
            "bodies", "mass.cruise_start", "mass.cruise_end", "mass.gross",
            "operating.cruise_altitude",
            "propulsion.cruise_thrust_each_course_estimate",
            "placeholders.CDp", "placeholders.e", "published.CLmax_flaps",
        }
    ),
)


# --- ASW-27B ----------------------------------------------------------------

ASW27 = Aircraft(
    name="Schleicher ASW-27B",
    label="ASW27",
    aircraft_class="15 m competition sailplane",
    notes=(
        "Where span is not a free lunch. The real root section is Delft's "
        "DU 89-134/14, whose coordinates are not in the UIUC database, so the "
        "course substitutes the Wortmann FX 62-K-131: a genuine sailplane "
        "laminar section of essentially the same thickness and a relevant drag "
        "bucket. Say in your write-up that you used a "
        "representative section. 'A 13.1% laminar section representative of "
        "this class' is a true sentence; \"the ASW-27's airfoil\" is not. "
        "The water ballast is not decoration: being able to change wing "
        "loading by a large fraction between flights is a design feature."
    ),
    wing=Planform(
        span=15.0,
        area=9.0,
        taper=0.40,
        section="DU 89-134/14 root, DU 94-086 M4 tip",
        section_file="fx62k131",
        thickness=0.134,
        notes="course stand-in section FX 62-K-131, t/c 0.131",
    ),
    bodies=(Body(name="fuselage", length=6.55, notes="overall length"),),
    mass={"empty": 245.0, "gross": 500.0, "water_ballast_max": 175.0},
    operating={"best_glide_speed": 29.0},
    limits={
        "n_pos": 5.3,
        "n_neg": -2.65,
        "Vne": 285.0 * KMH,
        "V_rough_air": 215.0 * KMH,
        "V_min_control": 70.0 * KMH,
        "n_reference_speed": 215.0 * KMH,
    },
    placeholders={},
    published={"glide_ratio": 48.0, "min_sink": 0.58},
    structure={"sigma_allow": 600e6, "rho": 1600.0, "material": "carbon"},
    propulsion={},
    sources=("Alexander Schleicher ASW 27 flight manual",),
    estimated=frozenset(
        {
            "wing.taper",
            "operating.best_glide_speed",
            "structure.sigma_allow",
            "structure.rho",
        }
    ),
)


# --- ASG 29 -----------------------------------------------------------------

ASG29 = Aircraft(
    name="Schleicher ASG 29 (18 m)",
    label="ASG29",
    aircraft_class="18 m sailplane",
    notes=(
        "A different aircraft from the same manufacturer, one competition "
        "class up. It illustrates what three more metres of span buys and "
        "costs, and these two are the closest thing to "
        "a controlled experiment anyone actually sells. The ASG 29 itself is "
        "offered with interchangeable 15 m and 18 m tips, which is the "
        "commercial evidence that the tradeoff is genuinely close. These "
        "numbers are the 18 m configuration."
    ),
    wing=Planform(
        span=18.0,
        area=10.5,
        section="laminar sailplane section",
        section_file="fx62k131",
        notes=(
            "same course stand-in section as the ASW-27B. Aspect ratio is not "
            "stored: b^2/S is a definition, not an independent measurement, so "
            "`aspect_ratio` computes 30.86 from the published 18.0 m span and "
            "10.5 m^2 area. (An earlier draft of the fleet page listed 30.4, "
            "which is inconsistent with both; Schleicher publishes 30.9.)"
        ),
    ),
    bodies=(Body(name="fuselage", length=6.59, notes="overall length"),),
    mass={"empty": 280.0, "gross": 600.0, "water_ballast_max": 202.0},
    operating={"best_glide_speed": 30.0, "comparison_altitude": 0.0},
    limits={"V_max": 270.0 * KMH},
    placeholders={},
    published={"glide_ratio": 50.0, "min_sink": 0.47},
    structure={"sigma_allow": 600e6, "rho": 1600.0, "material": "carbon"},
    propulsion={},
    sources=("Alexander Schleicher published data",),
    estimated=frozenset({
        "wing.section", "operating.best_glide_speed", "operating.comparison_altitude",
        "structure.sigma_allow", "structure.rho"
    }),
)


# --- Cessna 172S Skyhawk ----------------------------------------------------

C172 = Aircraft(
    name="Cessna 172S Skyhawk SP",
    label="C172",
    aircraft_class="light general-aviation single",
    notes=(
        "The aircraft students have the most physical intuition about, and it "
        "provides a useful general-aviation scale between RC-1 and a transport: "
        "Re ~ 3e6, Mach 0.19, 1157 kg. Its wing is a NACA 2412, which is "
        "bundled, and its performance is published to the decimal in a POH "
        "that has to be right because pilots fly on it.\n\n"
        "Two things to know before analyzing it. First, the wing modelled here "
        "is a SIMPLIFIED single trapezoid; the real wing is constant-chord "
        "inboard and tapered only outboard. The span "
        "and area are the published ones and the taper is fitted to the "
        "outboard panel, so the chords below are area-consistent rather than "
        "measured. Second, Cessna publishes span 36 ft 1 in, area 174 sq ft "
        "AND aspect ratio 7.32, and those three numbers are not consistent: "
        "36.083^2 / 174 = 7.48. Reconciling those values is a useful data-sheet "
        "consistency check on a document that thousands of people fly behind."
    ),
    wing=Planform(
        span=11.00,
        area=16.17,
        taper=0.694,
        dihedral_deg=1.73,
        twist_deg=-3.0,
        incidence_deg=1.5,
        section="NACA 2412",
        section_file="naca2412",
        thickness=0.12,
        x_le=0.0,
        notes=(
            "simplified single taper; taper fitted to the outboard panel of "
            "the real semi-tapered wing. 3 degrees of washout, which is why "
            "the 172 is famous for stalling straight ahead with the ailerons "
            "still working"
        ),
    ),
    htail=Planform(
        span=3.45,
        area=2.00,
        taper=0.60,
        section="NACA 0009",
        section_file="naca0009",
        thickness=0.09,
        x_le=4.60,
        z=0.30,
        notes="fixed stabilizer plus elevator; x_le aft of the wing root LE",
    ),
    vtail=Planform(
        span=1.35,
        area=1.10,
        taper=0.55,
        sweep_le_deg=35.0,
        section="NACA 0009",
        section_file="naca0009",
        thickness=0.09,
        x_le=4.55,
        vertical=True,
        notes="swept fin plus rudder",
    ),
    bodies=(
        Body(
            name="fuselage",
            length=8.28,
            width=1.00,
            height=1.30,
            x_nose=-1.55,
            notes="overall length; cabin width 1.00 m",
        ),
        Body(
            name="main gear leg",
            length=0.90,
            diameter=0.06,
            count=2,
            drag_model="crossflow",
            notes="tubular spring steel, fixed, unfaired and fully exposed",
        ),
        Body(
            name="wheel fairing",
            length=0.55,
            diameter=0.22,
            count=3,
            notes=(
                "speed fairings over the wheels; the nose gear leg is inside "
                "the cowling.  Fineness 2.5, so the buildup uses Hoerner's "
                "low-Re body correlation -- see drag.body_cd_frontal"
            ),
        ),
        Body(
            name="wing strut",
            length=2.60,
            width=0.09,
            height=0.03,
            count=2,
            drag_model="crossflow",
            notes="streamline-section lift strut, each side",
        ),
    ),
    mass={
        "empty": 767.0,
        "gross": 1157.0,
        "mtow": 1157.0,
        "fuel": 153.0,
        "payload": 237.0,
    },
    operating={
        "cruise_speed": 63.8,
        "cruise_altitude": 8000.0 * FT,
        "cruise_speed_published": 124.0 * KT,
        "max_speed": 126.0 * KT,
        "stall_speed": 48.0 * KT,
        "stall_speed_flaps": 40.0 * KT,
        "service_ceiling": 14000.0 * FT,
        "V_a": 105.0 * KT,
        "V_ne": 163.0 * KT,
    },
    limits={"n_pos": 3.8, "n_neg": -1.52},
    placeholders={"CDp": 0.034, "e": 0.75, "CLmax": 1.55},
    published={
        "aspect_ratio_published": 7.32,
        "range": 640.0 * 1852.0,
        "rate_of_climb": 730.0 * FT / 60.0,
        "endurance": 5.0 * 3600.0,
        "fuel_usable_litres": 212.0,
        "takeoff_ground_roll": 960.0 * FT,
        "landing_ground_roll": 575.0 * FT,
    },
    structure={},
    propulsion={
        "engine": "Lycoming IO-360-L2A",
        "count": 1,
        "power_each": 180.0 * HP,
        "rpm_max": 2700.0,
        "propeller_diameter": 76.0 * 0.0254,
        "propeller_pitch": 60.0 * 0.0254,
        "propeller_blades": 2,
        "propeller_type": "fixed pitch",
        "propeller": "McCauley 1A170E/JHA7660",
    },
    sources=(
        "Cessna 172S Skyhawk SP Pilot's Operating Handbook and FAA-approved "
        "Airplane Flight Manual (Cessna Aircraft Company)",
        "FAA Type Certificate Data Sheet 3A12",
    ),
    estimated=frozenset(
        {
            "wing.taper",
            "htail.taper",
            "htail.x_le",
            "htail.z",
            "vtail.taper",
            "vtail.sweep_le_deg",
            "vtail.x_le",
            "bodies.width",
            "bodies.height",
            "bodies.main gear leg",
            "bodies.wheel fairing",
            "bodies.wing strut",
            "mass.payload",
            "placeholders.CDp",
            "placeholders.e",
            "placeholders.CLmax",
        }
    ),
)


# --- Joby S4 ----------------------------------------------------------------

JobyS4 = Aircraft(
    name="Joby Aviation S4",
    label="JobyS4",
    aircraft_class="eVTOL air taxi",
    notes=(
        "The vehicle where the battery is the design rather than a component "
        "of it. Note what is published and what is not: airframe-level "
        "performance and TOTAL PACK ENERGY are public; the course segment "
        "mission and cell properties are estimates. Nothing about the cells "
        "is. That is why the course analysis sweeps specific energy rather than assuming it -- "
        "the published pack energy gives you a number to check your sizing "
        "against, and the sweep tells you how much the unknown actually matters."
    ),
    wing=None,  # Joby publishes no span or area, and the course hover analysis needs neither:
    # hover power comes from the rotor disk area and cruise power from the
    # given cruise L/D. Inventing a planform here would put an unsourced
    # number where a report might cite it.
    bodies=(),
    mass={"mtow": 2404.0, "gross": 2404.0, "nonbattery_course_estimate": 1524.0},
    operating={
        "cruise_speed": 89.0,
        "occupants": 5.0,
        "passengers": 4.0,
    },
    limits={},
    placeholders={},
    published={
        "range": 161e3,
        "battery_energy_min": 150e3 * 3600.0,
        "battery_energy_max": 180e3 * 3600.0,
        "LD_cruise": 14.0,
    },
    structure={},
    propulsion={
        "rotors": 6,
        "rotor_type": "tilting",
        "rotor_diameter": 2.9,
        "figure_of_merit": 0.75,
        "drivetrain_efficiency": 0.90,
        "cell_specific_energy": 250.0 * 3600.0,
        "pack_packaging_efficiency": 0.75,
        "mission": (
            ("hover", 120.0),
            ("cruise", 1500.0),
            ("hover", 120.0),
            ("cruise_reserve", 600.0),
        ),
    },
    sources=(
        "Joby Aviation published figures",
        "eVTOL.news airframe data",
        "Stoll (NASA), Analysis and Full Scale Testing of the Joby S4 "
        "Propulsion System",
    ),
    estimated=frozenset(
        {
            "propulsion.rotor_diameter",
            "propulsion.figure_of_merit",
            "propulsion.drivetrain_efficiency",
            "propulsion.cell_specific_energy",
            "propulsion.pack_packaging_efficiency",
            "propulsion.mission",
            "mass.nonbattery_course_estimate",
            "published.LD_cruise",
        }
    ),
)


# --- Saturn V ---------------------------------------------------------------

SaturnV = Rocket(
    name="Saturn V",
    label="SaturnV",
    stages=(
        Stage(
            name="S-IC",
            propellant=2_077_000.0,
            dry=130_000.0,
            thrust=34.5e6,
            thrust_condition="SL",
            isp_sl=263.0,
            isp_vac=304.0,
            burn_time=168.0,
            burn_times=(168.0,),
            height=42.1,
            diameter=10.06,
        ),
        Stage(
            name="S-II",
            propellant=443_000.0,
            dry=36_000.0,
            thrust=5.14e6,
            thrust_condition="vac",
            isp_vac=421.0,
            burn_time=384.0,
            burn_times=(384.0,),
            diameter=10.06,
        ),
        Stage(
            name="S-IVB",
            propellant=107_000.0,
            dry=13_500.0,
            thrust=1.00e6,
            thrust_condition="vac",
            isp_vac=421.0,
            burn_time=500.0,
            burn_times=(165.0, 335.0),
            diameter=6.6,
        ),
    ),
    height=110.6,
    diameter=10.06,
    payload_leo=1.4e5,
    sources=(
        "NASA, Saturn V News Reference",
        "Orloff, Apollo by the Numbers",
    ),
    notes=(
        "R 1 asks you to reproduce the 42.1 m S-IC height and its propellant "
        "and dry masses from a sizing loop, so treat this table as the answer "
        "key rather than as an input. Published figures for this vehicle vary "
        "by a few percent between references -- cite the one you used."
    ),
)


# --- registry ---------------------------------------------------------------

#: All winged fleet aircraft, keyed by label.
AIRCRAFT = {
    a.label: a for a in (RC1, B787, ASW27, ASG29, C172, JobyS4)
}

#: Everything, including the launch vehicle.
ALL = dict(AIRCRAFT, SaturnV=SaturnV)
