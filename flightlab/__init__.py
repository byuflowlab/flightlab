"""``flightlab`` -- analysis tools for ME 415, Flight Vehicle Design.

Complete.  Section aerodynamics, vortex lattice, viscous coupling, drag
buildup, mass properties, trim, dynamic modes, the electric propulsion chain,
performance and loads -- everything the course analyzes, plus the fleet and
catalog data nobody learns from transcribing, and the plotting.

The course is about analysis and design, not about building the tools, so the
tools are here.  What is *not* here is the rocket work: those two assignments
are done in class, and building and verifying that code is the assignment.

Submodules
----------
``flightlab.atmos``
    1976 standard atmosphere, and Mach and Reynolds numbers from it.
``flightlab.geom``
    Planform arithmetic, wetted areas, and vortex-lattice grid building.
``flightlab.airfoil``
    Section polars and the ``cd(cl, Re)`` lookup a strip integration needs.
``flightlab.wing``
    Vortex-lattice wing analysis: span loading, span efficiency, stall.
``flightlab.drag``
    Component buildup, strip integration, and the complete drag polar.
``flightlab.stability``
    Mass properties, trim, neutral point, static margin, dynamic modes.
``flightlab.propulsion``
    Battery, motor, ESC, propeller, and the match between them.
``flightlab.performance``
    Speeds, climb, glide, range, endurance, field length, the envelope.
``flightlab.loads``
    V-n diagram, span loads, spar stress and stiffness.
``flightlab.case``
    The ``Case`` object: parameters, cached analysis modes.
``flightlab.live``
    Slider-driven figures for design exploration.
``flightlab.project``
    Saved student aircraft: station geometry, bodies, masses, and flight cases.
``flightlab.project_analysis``
    Integrated VLM, trim, stability, mass, and drag analysis of a project.
``flightlab.workbench``
    Optional local browser application for aircraft design.
``flightlab.cache``
    The dependency-tracked caching that makes design iteration fast.
``flightlab.vlm``
    Steady vortex lattice method, ported from VortexLattice.jl.  Inviscid.
``flightlab.foil``
    NeuralFoil section aerodynamics: ``cl``, ``cd``, ``cm`` and a confidence
    metric, from a NACA designation or coordinates.
``flightlab.fleet``
    The focused course fleet as importable geometry and component mass tables.
``flightlab.props``
    UIUC propeller data files and a reader.
``flightlab.catalog``
    The bounded propulsion catalog: motors, propellers, batteries.
``flightlab.plot``
    Plotting for geometry, span loading, polars, contours, V-n and eigenvalues.
``flightlab.ref``
    Reference values for verification, so an assertion is one line.

Units
-----
SI everywhere: metres, kilograms, seconds, newtons, watts, kelvin, pascals.
Angles are in **degrees** at every interface a human touches and radians
internally, except in ``flightlab.vlm``, which keeps the radian convention of the
Julia package it was ported from.  Every docstring says which.
"""

from importlib import import_module

from .capabilities import TOPICS, describe, example, show_tools


__all__ = [
    # analysis
    "atmos", "geom", "airfoil", "wing", "drag", "stability",
    "propulsion", "performance", "loads",
    # workflow
    "case", "live", "cache", "capabilities", "project",
    "project_analysis", "workbench",
    # data and solvers
    "vlm", "foil", "fleet", "props", "catalog", "plot", "ref",
    # discovery
    "TOPICS", "describe", "example", "show_tools",
]

__version__ = "0.9.0"


def __getattr__(name: str):
    """Load public submodules only when they are first used.

    This makes both common import styles work without importing the numerical
    stack up front::

        import flightlab
        air = flightlab.atmos.at(1400.0)

        from flightlab import atmos

    Misspelled module names still raise a normal, helpful ``AttributeError``.
    """
    if name in {
        "atmos", "geom", "airfoil", "wing", "drag", "stability",
        "propulsion", "performance", "loads", "case", "live", "cache",
        "capabilities", "project", "project_analysis", "workbench",
        "vlm", "foil", "fleet", "props", "catalog", "plot", "ref",
    }:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
