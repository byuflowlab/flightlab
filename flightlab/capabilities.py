"""Discover what FlightLab can do before remembering function names.

This module is deliberately data-first.  The command-line browser and browser
workbench can render the same topic records, so
their descriptions do not drift away from the Python API.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Topic:
    """One student-facing family of analyses."""

    key: str
    title: str
    question: str
    analyses: Tuple[str, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    limits: Tuple[str, ...]
    homework: str
    example: str


TOPICS: Dict[str, Topic] = {
    "atmosphere": Topic(
        key="atmosphere",
        title="Atmosphere and flight condition",
        question="What air properties and nondimensional conditions does the vehicle see?",
        analyses=(
            "atmos.at — complete atmospheric state",
            "atmos.mach and atmos.reynolds — flight-regime measures",
            "atmos.eas_to_tas and atmos.tas_to_eas — airspeed conversion",
            "State.q, State.mach, and State.reynolds — reuse one atmospheric state",
        ),
        inputs=("geometric altitude [m]", "true or equivalent airspeed [m/s]", "reference length [m]", "temperature offset [K], optional"),
        outputs=("temperature, pressure, density, viscosity", "speed of sound, dynamic pressure", "Mach and Reynolds numbers"),
        limits=("1976 U.S. Standard Atmosphere", "Speeds are true airspeed unless explicitly labeled EAS."),
        homework="HW 1; reused throughout the course",
        example="""from flightlab import atmos

air = atmos.at(2438.4)
q = air.q(63.8)
M = air.mach(63.8)
Re = air.reynolds(63.8, 1.49)
print(air.density, q, M, Re)""",
    ),
    "airfoils": Topic(
        key="airfoils",
        title="Airfoil section aerodynamics",
        question="How does a two-dimensional section behave across angle of attack and Reynolds number?",
        analyses=(
            "airfoil.polar — CL, CD, CM, transition, and confidence versus angle",
            "airfoil.table — reusable CD(CL, Re) and CL-max lookup",
            "airfoil.cl_max and airfoil.cd_min — scalar section metrics",
            "plot.airfoil and plot.airfoil_polars — geometry and polar comparisons",
        ),
        inputs=("section name or coordinate file", "Reynolds number or range", "angle-of-attack range [deg]", "transition locations, optional"),
        outputs=("CL, CD, CM", "transition and boundary-layer quantities", "model confidence and interpolated lookup tables"),
        limits=("NeuralFoil is a reduced-order low-speed section model.", "Confidence measures model coverage, not experimental uncertainty.", "It does not model transonic wave drag."),
        homework="HW 2; supplies section data to HW 3, 5, and 8",
        example="""import numpy as np
from flightlab import airfoil

p = airfoil.polar("naca2412", Re=5.0e5,
                  alpha=np.linspace(-4.0, 14.0, 73))
print(p.cl, p.cd, p.cm, p.confidence)""",
    ),
    "wings": Topic(
        key="wings",
        title="Finite-wing loading and stall",
        question="How do planform, twist, and flight condition set loading, induced drag, and stall onset?",
        analyses=(
            "wing.analyze and wing.sweep — VLM solutions at specified angles",
            "wing.trim_to_CL and wing.trim_to_weight — solve the operating angle",
            "wing.CL_max and wing.stall_speed — couple local loading to section limits",
            "plot.span_loading and plot.stall_margin — distinguish induced-drag and stall distributions",
        ),
        inputs=("aircraft or planform", "speed [m/s] and altitude [m]", "angle [deg], target CL, or mass [kg]", "panel counts and optional section table"),
        outputs=("CL, induced CD, lift and induced drag", "span efficiency and spanwise loading", "local section cl and first-stall station"),
        limits=("The VLM is steady and inviscid.", "Section drag and stall enter through separate models.", "Use the high-level wing interface in degrees; flightlab.vlm uses radians."),
        homework="HW 3; reused in HW 4, 5, and 8",
        example="""from flightlab import wing
from flightlab.fleet import ASW27

sol = wing.trim_to_weight(
    ASW27, mass=500.0, V=29.0, altitude=0.0,
    ns=60, nc=6,
)
print(sol.alpha, sol.CL, sol.CD_i, sol.e_inv)""",
    ),
    "drag": Topic(
        key="drag",
        title="Component drag and complete-aircraft polars",
        question="Where does aircraft drag come from, and how does it change with lift and Mach number?",
        analyses=(
            "drag.buildup — component parasite-drag areas and markups",
            "drag.polar — parasite, viscous lift-dependent, and induced drag",
            "drag.strip_viscous_drag — spanwise section-drag integration",
            "drag.drag_divergence_mach and drag.wave_drag — conceptual transonic correlations",
        ),
        inputs=("aircraft, speed [m/s], and altitude [m]", "CL range or operating condition", "interference, protuberance, cooling, and transition assumptions", "section table for strip analysis"),
        outputs=("component drag-area table and CD0", "complete CD(CL) breakdown and maximum L/D", "drag-divergence Mach and wave-drag rise"),
        limits=("Component correlations use simplified geometry.", "Cooling and installation penalties require explicit allowances.", "The wave-drag model is an early-design correlation, not a transonic CFD solver."),
        homework="HW 1, 2, and 5; reused in HW 6 and 7",
        example="""from flightlab import drag
from flightlab.fleet import C172

b = drag.buildup(
    C172, 50.0, altitude=0.0,
    interference=0.05, protuberance=0.05,
)
print(b.table())""",
    ),
    "stability": Topic(
        key="stability",
        title="Stability, trim, and dynamic modes",
        question="Can the aircraft trim, how stable is it, and what motions follow a disturbance?",
        analyses=(
            "stability.mass_properties — mass, CG, and inertias",
            "stability.neutral_point and stability.static_margin — longitudinal stability",
            "stability.trim — trimmed angle, tail incidence, and loads",
            "stability.derivatives and stability.modes — static derivatives and dynamic modes",
        ),
        inputs=("aircraft and component masses", "speed [m/s], altitude [m], and load factor", "CG location and control/tail settings", "optional supplied derivatives"),
        outputs=("CG and inertia tensor", "neutral point and static margin", "trimmed state and tail load", "short-period, phugoid, roll, Dutch-roll, and spiral modes"),
        limits=("Dynamic modes are linearized near the trimmed condition.", "Results inherit simplifications in fleet geometry and component inertias."),
        homework="HW 4",
        example="""from flightlab import stability
from flightlab.fleet import RC1

tr = stability.trim(
    RC1, V=11.5, altitude=1400.0,
    mass=0.750, x_cg=0.30 * RC1.wing.mean_chord,
)
print(tr.alpha, tr.tail_incidence, tr.static_margin)""",
    ),
    "propulsion": Topic(
        key="propulsion",
        title="Propulsion-system matching",
        question="Where does a battery–motor–propeller system operate, and is that point feasible?",
        analyses=(
            "propulsion.operating_point — coupled electrical and propeller solve",
            "propulsion.sweep_speed and propulsion.static_thrust — operating-point comparisons",
            "propulsion.motor_point and propulsion.battery_voltage — component behavior",
            "propulsion.rotor_hover and propulsion.turbofan_thrust — rotor and jet course models",
        ),
        inputs=("motor, propeller, battery, and ESC", "speed [m/s], altitude [m], throttle, and state of charge", "measured or provisional electrical properties"),
        outputs=("thrust, torque, rpm, advance ratio, and tip speed", "current, terminal voltage, and power at each stage", "stage efficiencies, constraint margins, and data-coverage flags"),
        limits=("Catalog electrical values remain provisional until replaced by course measurements.", "A converged propeller solution may still be outside the measured advance-ratio range."),
        homework="HW 6; reused in HW 7",
        example="""from flightlab import catalog, propulsion

motor = catalog.MOTORS["M1000"]
prop = catalog.PROPELLERS["P10x7"]
battery = catalog.BATTERIES["B3S1300"]
op = propulsion.operating_point(
    motor, prop.data, battery,
    V=9.1, altitude=1400.0, soc=0.90, esc="ESC30",
)
print(op.thrust, op.current, op.extrapolated)""",
    ),
    "performance": Topic(
        key="performance",
        title="Aircraft performance",
        question="What speeds, climb, glide, range, endurance, and ceiling follow from the aircraft models?",
        analyses=(
            "performance.speeds and performance.glide — characteristic speeds",
            "performance.climb and performance.ceiling — available versus required performance",
            "performance.range_electric and performance.endurance_electric — battery aircraft",
            "performance.range_breguet and performance.endurance_breguet — fuel-burning aircraft",
            "performance.takeoff_ground_roll, landing_ground_roll, turn, and envelope — additional performance",
        ),
        inputs=("complete drag polar", "aircraft mass [kg] and altitude [m]", "thrust- or power-available model", "energy, efficiency, or fuel-consumption model"),
        outputs=("stall, minimum-drag, minimum-power, and best-climb speeds", "glide ratio, sink rate, climb rate, and ceiling", "range, endurance, turn, and field performance"),
        limits=("Performance inherits every assumption in the drag and propulsion models.", "Use the near-best region rather than reporting a falsely precise optimum."),
        homework="HW 7",
        example="""from flightlab import drag, performance
from flightlab.fleet import C172

p = drag.polar(C172, V=50.0, altitude=2400.0)
print(performance.speeds(p, mass=1111.0))
print(performance.glide(p, mass=1111.0))""",
    ),
    "loads": Topic(
        key="loads",
        title="Flight loads and simple structures",
        question="Which maneuver or gust case governs, and what loads reach the wing structure?",
        analyses=(
            "loads.vn_diagram and loads.gust_load_factor — maneuver and gust envelopes",
            "loads.span_load — running lift, shear, and bending moment",
            "loads.spar_stress and loads.spar_sizing — simple cap sizing",
            "loads.tip_deflection — Euler–Bernoulli deflection estimate",
        ),
        inputs=("aircraft, mass [kg], and altitude [m]", "CL limits, load-factor limits, and gust speeds", "trimmed wing solution or flight condition", "material allowables and section geometry"),
        outputs=("stall boundaries, corner speed, and governing load case", "spanwise lift, shear, and bending moment", "spar area, stress margin, and tip deflection"),
        limits=("Limit and ultimate loads are not interchangeable.", "The beam and spar models do not replace detailed structural analysis."),
        homework="HW 8",
        example="""from flightlab import loads
from flightlab.fleet import ASW27

vn = loads.vn_diagram(ASW27, mass=500.0, CL_max=1.4)
sl = loads.span_load(
    ASW27, mass=500.0, n=5.3,
    V=vn["V_A"], altitude=0.0,
)
print(vn["V_stall"], vn["V_A"], sl.root_moment)""",
    ),
}


RESOURCES = (
    "fleet — supplied aircraft geometry, mass data, and operating conditions",
    "catalog — bounded motor, propeller, battery, and ESC choices",
    "geom — planform resolution, wetted areas, and solver grids",
    "plot — course plots that accept an optional Matplotlib axes object",
    "case and live — cached cases and slider-driven design exploration",
    "foil, props, and vlm — lower-level interfaces behind the course analyses",
    "ref — analytic checks, published ranges, and labeled course correlations",
)


_ALIASES = {
    "atmos": "atmosphere",
    "airfoil": "airfoils",
    "wing": "wings",
    "stability-and-trim": "stability",
    "prop": "propulsion",
    "structures": "loads",
}


def topic_names() -> Tuple[str, ...]:
    """Return the canonical topic keys in course order."""
    return tuple(TOPICS)


def _resolve(name: str) -> Topic:
    key = name.strip().lower().replace(" ", "-")
    key = _ALIASES.get(key, key)
    try:
        return TOPICS[key]
    except KeyError as exc:
        choices = ", ".join(TOPICS)
        raise KeyError(f"unknown FlightLab topic {name!r}; choose from {choices}") from exc


def describe(name: str) -> Topic:
    """Return the structured record for a topic such as ``"wings"``."""
    return _resolve(name)


def example(name: str) -> str:
    """Return a small runnable example for one topic."""
    return _resolve(name).example


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def format_tools(name: Optional[str] = None) -> str:
    """Format the toolbox overview or one detailed topic for plain text."""
    if name is None:
        width = max(len(key) for key in TOPICS)
        lines = [
            "FlightLab analysis toolbox",
            "==========================",
            "Choose a topic with flightlab.show_tools('wings') or",
            "python -m flightlab wings.",
            "",
        ]
        for key, item in TOPICS.items():
            lines.append(f"{key:<{width}}  {item.question}")
        lines += ["", "Shared resources:", _bullets(RESOURCES)]
        return "\n".join(lines)

    item = _resolve(name)
    return "\n".join(
        (
            item.title,
            "=" * len(item.title),
            item.question,
            f"Course use: {item.homework}",
            "",
            "Analyses:",
            _bullets(item.analyses),
            "",
            "Inputs:",
            _bullets(item.inputs),
            "",
            "Named outputs:",
            _bullets(item.outputs),
            "",
            "Model limits:",
            _bullets(item.limits),
            "",
            "Starter code:",
            indent(item.example, "  "),
        )
    )


def show_tools(name: Optional[str] = None) -> None:
    """Print the toolbox overview or detailed help for one topic.

    Examples
    --------
    >>> import flightlab
    >>> flightlab.show_tools()          # all topic families
    >>> flightlab.show_tools("wings")  # analyses, inputs, outputs, limits
    """
    print(format_tools(name))


__all__ = [
    "Topic", "TOPICS", "RESOURCES", "topic_names", "describe", "example",
    "format_tools", "show_tools",
]
