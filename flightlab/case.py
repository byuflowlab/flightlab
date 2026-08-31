"""``flightlab.case`` -- one aircraft, one condition, and every analysis of it.

A :class:`Case` is the object design work is done on.  It holds an aircraft's
parameters where they can be changed, and it knows how to run every analysis in
the package against them -- caching each one, so that changing the span
recomputes the aerodynamics and changing the battery mass does not.

    >>> from flightlab.case import Case
    >>> from flightlab.fleet import RC1
    >>> case = Case(RC1, V=12.0, altitude=1400.0)
    >>> case.wing_aero().CL              # computes
    0.4457...
    >>> case.wing.span = 1.4             # invalidates the aerodynamics
    >>> case.wing_aero().CL              # recomputes
    0.4319...
    >>> case.mass["battery"] = 0.140     # the aerodynamics do not care
    >>> case.wing_aero().CL              # served from cache, instantly
    0.4319...

Parameter groups
----------------
========== ====================================================================
``wing``   span, area, taper, sweep, twist, dihedral, incidence, section, t/c
``htail``  the same, for the horizontal tail
``vtail``  the same, for the fin
``mass``   the component table, as a mapping of name to kilograms
``cond``   V, altitude, dT, load factor, throttle
``prop``   motor, propeller, battery, ESC keys
``solver`` panel counts, section table resolution, drag buildup assumptions
========== ====================================================================

Every analysis mode declares which of these it reads, and that declaration is
the cache key.  :meth:`Case.explain` prints what is cached and what a given
parameter would invalidate, which is worth looking at once to see the
dependency structure the course is built on.

Units are SI, angles in degrees -- the same everywhere else in the package.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import airfoil as _airfoil
from . import atmos, cache, drag, geom, loads, performance, propulsion, stability, wing
from .cache import Watched, WatchedDict, mode
from .fleet import Aircraft, Component, Planform

__all__ = ["Case", "WingParams", "Condition", "SolverParams", "PropulsionParams"]


# --- parameter groups -------------------------------------------------------


class WingParams(Watched):
    """A lifting surface's parameters, as things you can assign to."""

    _fields = (
        "span", "area", "taper", "root_chord", "tip_chord",
        "sweep_le_deg", "sweep_c4_deg", "dihedral_deg", "twist_deg",
        "incidence_deg", "thickness", "section", "x_le", "z",
    )

    def to_planform(self, base: Planform) -> Planform:
        """Apply the non-``None`` overrides to a fleet planform."""
        changes = {k: v for k, v in self.asdict().items() if v is not None}
        if not changes:
            return base
        # a chord pair and a taper cannot both be honoured; the explicit
        # override wins and the stale partner is dropped
        if "taper" in changes and "tip_chord" not in changes:
            changes["tip_chord"] = None
        if "root_chord" in changes or "tip_chord" in changes:
            changes.setdefault("taper", None)
        if "section" in changes:
            changes["section_file"] = changes.pop("section")
        return replace(base, **changes)


class Condition(Watched):
    """The flight condition."""

    _fields = ("V", "altitude", "dT", "n", "throttle", "soc", "mass")


class PropulsionParams(Watched):
    """Which catalog parts are fitted."""

    _fields = ("motor", "propeller", "battery", "esc", "efficiency_chain")


class SolverParams(Watched):
    """Numerical and modelling settings that are not the aircraft."""

    _fields = (
        "ns", "nc", "camber", "interference", "protuberance", "f_other",
        "cooling", "xtr", "xtr_wing", "K", "e_assumed", "n_Re", "CL_grid",
        "method",
    )


# --- the case ---------------------------------------------------------------


class Case:
    """An aircraft plus a condition, with every analysis cached.

    Parameters
    ----------
    aircraft : Aircraft
    V : float, optional
        True airspeed, m/s.  Defaults to the cruise speed on record.
    altitude : float, optional
        Geometric altitude, m.  Defaults to the cruise altitude on record.
    mass : float, optional
        Defaults to the component table total, or the gross mass.
    **overrides
        Any parameter of any group, as ``wing_span=1.4`` or ``solver_ns=60``.

    Attributes
    ----------
    wing, htail, vtail : WingParams
    mass : WatchedDict
        Component masses in kg.  Editing a row moves the centre of gravity and
        the inertias, and invalidates everything that depends on them.
    cond : Condition
    prop : PropulsionParams
    solver : SolverParams
    """

    def __init__(self, aircraft: Aircraft, V=None, altitude=None, mass=None, **overrides):
        self.base = aircraft
        self._cache: Dict[str, Tuple] = {}
        cache._register(self)

        self.wing = WingParams()
        self.htail = WingParams()
        self.vtail = WingParams()

        comps = {c.name: c.mass for c in aircraft.components}
        self.mass = WatchedDict(comps)
        self._component_geometry = {
            c.name: (c.x, c.y, c.z, c.distributed, c.span) for c in aircraft.components
        }

        self.cond = Condition(
            V=V if V is not None else aircraft.operating.get("cruise_speed"),
            altitude=(
                altitude if altitude is not None
                else aircraft.operating.get("cruise_altitude", 0.0)
            ),
            dT=0.0,
            n=1.0,
            throttle=1.0,
            soc=1.0,
            mass=mass,
        )
        pbase = aircraft.propulsion or {}
        self.prop = PropulsionParams(
            motor=pbase.get("motor"),
            propeller=pbase.get("propeller"),
            battery=pbase.get("battery"),
            esc=pbase.get("esc"),
            efficiency_chain=0.45,
        )
        self.solver = SolverParams(
            ns=40, nc=6, camber=True, interference=0.05, protuberance=0.05,
            f_other=0.0, cooling=0.0, xtr=0.0, xtr_wing=None, K=0.38,
            e_assumed=0.98, n_Re=10, CL_grid=None, method="buildup",
        )

        for key, value in overrides.items():
            group, _, field = key.partition("_")
            target = getattr(self, group, None)
            if target is None or not field:
                raise AttributeError(
                    f"cannot set {key!r}: expected <group>_<parameter> with "
                    "group one of wing, htail, vtail, cond, prop, solver"
                )
            setattr(target, field, value)

    # -- derived geometry ---------------------------------------------------

    def aircraft(self) -> Aircraft:
        """The fleet aircraft with this case's overrides applied."""
        changes = {}
        for name in ("wing", "htail", "vtail"):
            base = getattr(self.base, name)
            if base is None:
                continue
            changed = getattr(self, name).to_planform(base)
            if changed is not base:
                changes[name] = changed
        if self.mass.version != self._mass_baseline_version():
            changes["components"] = self.components()
        return replace(self.base, **changes) if changes else self.base

    def _mass_baseline_version(self) -> int:
        return getattr(self, "_mass_v0", -1)

    def components(self) -> Tuple[Component, ...]:
        """The component table with this case's mass edits applied."""
        out = []
        for name, m in self.mass.items():
            x, y, z, dist, span = self._component_geometry.get(
                name, (0.0, 0.0, 0.0, "", None)
            )
            out.append(Component(name, float(m), x, y, z, dist, span))
        return tuple(out)

    def total_mass(self) -> float:
        """Flight mass, kg -- the override if set, else the component sum."""
        if self.cond.mass is not None:
            return float(self.cond.mass)
        if self.mass:
            return float(sum(self.mass.values()))
        m = self.base.mass.get("gross") or self.base.mass.get("mtow")
        if m is None:
            raise ValueError(f"{self.base.label} has no mass on record")
        return float(m)

    def panel(self) -> geom.Panel:
        """The resolved wing planform."""
        return geom.resolve(self.aircraft().wing)

    def _identity(self):
        """Part of the disk-cache key: which aircraft this case is of."""
        return (self.base.label,)

    # -- analysis modes -----------------------------------------------------

    @mode("cond")
    def atmosphere(self) -> "atmos.State":
        """The atmospheric state at this condition."""
        return atmos.at(self.cond.altitude, self.cond.dT or 0.0)

    @mode("wing", "cond", "solver")
    def section(self) -> "_airfoil.Table":
        """A section table covering this wing's local Reynolds range."""
        p = self.panel()
        air = self.atmosphere()
        Re_tip = air.reynolds(self.cond.V, p.tip_chord)
        Re_root = air.reynolds(self.cond.V, p.root_chord)
        return _airfoil.table(
            p.section, Re=(0.4 * Re_tip, 2.5 * Re_root), n_Re=self.solver.n_Re
        )

    @mode("wing", "htail", "vtail", "cond", "solver")
    def wing_aero(self) -> "wing.Solution":
        """The wing trimmed to carry the weight at this condition."""
        return wing.trim_to_weight(
            self.aircraft().wing,
            self.total_mass(),
            self.cond.V,
            self.cond.altitude,
            n=self.cond.n or 1.0,
            ns=self.solver.ns,
            nc=self.solver.nc,
            camber=self.solver.camber,
            dT=self.cond.dT or 0.0,
        )

    @mode("wing", "cond", "solver")
    def CL_max(self) -> Dict[str, float]:
        """Wing maximum lift coefficient and where it stalls first."""
        return wing.CL_max(
            self.aircraft().wing,
            self.cond.V,
            self.cond.altitude,
            table=self.section(),
            ns=self.solver.ns,
            nc=self.solver.nc,
        )

    @mode("wing", "htail", "vtail", "cond", "solver")
    def buildup(self) -> "drag.Buildup":
        """The component parasitic drag buildup."""
        return drag.buildup(
            self.aircraft(), self.cond.V, self.cond.altitude,
            interference=self.solver.interference,
            protuberance=self.solver.protuberance,
            f_other=self.solver.f_other, cooling=self.solver.cooling,
            xtr=self.solver.xtr, xtr_wing=self.solver.xtr_wing,
            dT=self.cond.dT or 0.0,
        )

    @mode("wing", "htail", "vtail", "cond", "solver")
    def polar(self) -> "drag.Polar":
        """The complete drag polar."""
        return drag.polar(
            self.aircraft(), self.cond.V, self.cond.altitude,
            CL=self.solver.CL_grid, method=self.solver.method,
            e=self.solver.e_assumed, K=self.solver.K,
            interference=self.solver.interference,
            protuberance=self.solver.protuberance,
            f_other=self.solver.f_other, cooling=self.solver.cooling,
            xtr=self.solver.xtr, xtr_wing=self.solver.xtr_wing,
            ns=self.solver.ns, nc=self.solver.nc,
            table=self.section() if self.solver.method == "strip" else None,
            dT=self.cond.dT or 0.0,
        )

    @mode("wing", "htail", "vtail", "mass", "cond", "solver")
    def mass_properties(self) -> "stability.MassProperties":
        """Mass, centre of gravity and inertias from the component table."""
        return stability.mass_properties(self.components() or self.base)

    @mode("wing", "htail", "vtail", "mass", "cond", "solver")
    def trim(self) -> "stability.Trim":
        """The trimmed longitudinal state."""
        mp = self.mass_properties()
        return stability.trim(
            self.aircraft(), self.cond.V, self.cond.altitude,
            mass=self.total_mass(), x_cg=mp.x_cg,
            n=self.cond.n or 1.0, ns=self.solver.ns, nc=self.solver.nc,
        )

    @mode("wing", "htail", "vtail", "mass", "cond", "solver")
    def derivatives(self) -> "stability.Derivatives":
        """Stability derivatives at the trimmed condition, mirrored model."""
        t = self.trim()
        return stability.derivatives(
            self.aircraft(), self.cond.V, self.cond.altitude,
            alpha=t.alpha, x_cg=t.x_cg, tail_incidence_deg=t.tail_incidence,
            ns=self.solver.ns, nc=self.solver.nc, lateral=True,
        )

    @mode("wing", "htail", "vtail", "mass", "cond", "solver")
    def modes(self):
        """``(longitudinal, lateral)`` dynamic modes."""
        d = self.derivatives()
        mp = self.mass_properties()
        return (
            stability.longitudinal_modes(
                self.aircraft(), self.cond.V, self.cond.altitude,
                mass=self.total_mass(), Iyy=mp.Iyy, x_cg=mp.x_cg, derivs=d,
            ),
            stability.lateral_modes(
                self.aircraft(), self.cond.V, self.cond.altitude,
                mass=self.total_mass(), Ixx=mp.Ixx, Izz=mp.Izz, Ixz=mp.Ixz,
                x_cg=mp.x_cg, derivs=d,
            ),
        )

    @mode("prop", "cond")
    def operating_point(self) -> "propulsion.OperatingPoint":
        """The motor-propeller match at this condition."""
        self._require_propulsion()
        return propulsion.operating_point(
            self.prop.motor, self.prop.propeller, self.prop.battery,
            V=self.cond.V, throttle=self.cond.throttle or 1.0,
            altitude=self.cond.altitude, soc=self.cond.soc or 1.0,
            esc=self.prop.esc, dT=self.cond.dT or 0.0,
        )

    @mode("wing", "htail", "vtail", "mass", "cond", "prop", "solver")
    def performance(self) -> Dict[str, object]:
        """Characteristic speeds, glide, and electric range and endurance."""
        pol = self.polar()
        m = self.total_mass()
        out = {
            "speeds": performance.speeds(pol, m, CL_max=self.CL_max()["CL_max"]),
            "glide": performance.glide(pol, m),
        }
        if self.prop.battery:
            energy = propulsion.pack_energy(self.prop.battery)
            eta = self.prop.efficiency_chain
            out["endurance"] = performance.endurance_electric(
                energy, pol, m, efficiency=eta
            )
            out["range"] = performance.range_electric(energy, pol, m, efficiency=eta)
        return out

    @mode("wing", "mass", "cond", "solver")
    def span_load(self) -> "loads.SpanLoad":
        """Running lift, shear and bending moment at this load factor."""
        return loads.span_load(
            self.aircraft(), self.total_mass(), n=self.cond.n or 1.0,
            V=self.cond.V, altitude=self.cond.altitude, ns=self.solver.ns,
        )

    @mode("wing", "mass", "cond", "solver")
    def vn(self) -> Dict[str, object]:
        """The manoeuvre envelope."""
        return loads.vn_diagram(
            self.aircraft(), mass=self.total_mass(),
            CL_max=self.CL_max()["CL_max"], altitude=self.cond.altitude,
        )

    def _require_propulsion(self) -> None:
        missing = [
            k for k in ("motor", "propeller", "battery")
            if getattr(self.prop, k) is None
        ]
        if missing:
            raise ValueError(
                f"{self.base.label} has no {', '.join(missing)} set.  Assign "
                f"them, e.g. case.prop.motor = 'M1000', or pick from "
                "flightlab.catalog."
            )

    # -- introspection ------------------------------------------------------

    def groups(self) -> Dict[str, object]:
        """The parameter groups, by name."""
        return {
            "wing": self.wing, "htail": self.htail, "vtail": self.vtail,
            "mass": self.mass, "cond": self.cond, "prop": self.prop,
            "solver": self.solver,
        }

    def cached(self) -> List[str]:
        """Which analysis modes currently have a valid cached result."""
        out = []
        for name in sorted(self._cache):
            fn = getattr(type(self), name)
            groups = getattr(fn, "_groups", ())
            if self._cache[name][0] == self._version_key(groups, (), {}):
                out.append(name)
        return out

    def invalidated_by(self, group: str) -> List[str]:
        """Which modes a change to ``group`` would force to recompute."""
        if group not in self.groups():
            raise KeyError(f"no parameter group {group!r}; have {list(self.groups())}")
        return sorted(
            name
            for name in dir(type(self))
            if group in getattr(getattr(type(self), name, None), "_groups", ())
        )

    def explain(self) -> str:
        """A printable map of what depends on what, and what is cached now."""
        lines = ["analysis mode        reads", "-" * 62]
        for name in sorted(dir(type(self))):
            fn = getattr(type(self), name, None)
            groups = getattr(fn, "_groups", None)
            if not groups:
                continue
            lines.append(f"{name:<20} {', '.join(groups)}")
        lines += ["", f"cached now: {', '.join(self.cached()) or 'nothing'}"]
        lines.append(f"{cache.stats()}")
        return "\n".join(lines)

    def _version_key(self, groups, args, kwargs):
        return (
            tuple(getattr(self, g).version for g in groups),
            args,
            tuple(sorted(kwargs.items())),
        )

    def copy(self) -> "Case":
        """An independent copy, with its own cache.

        Useful for a sweep that should not disturb the case you are looking at:
        ``variant = case.copy(); variant.wing.span = 1.4``.
        """
        new = Case(self.base)
        for name in ("wing", "htail", "vtail"):
            getattr(new, name).replace(**getattr(self, name).asdict())
        new.mass.update(dict(self.mass))
        new.cond.replace(**self.cond.asdict())
        new.prop.replace(**self.prop.asdict())
        new.solver.replace(**self.solver.asdict())
        return new

    def set(self, **overrides) -> "Case":
        """Assign several parameters at once; returns ``self`` for chaining."""
        for key, value in overrides.items():
            group, _, field = key.partition("_")
            target = self.groups().get(group)
            if target is None or not field:
                raise AttributeError(f"cannot set {key!r}")
            if isinstance(target, WatchedDict):
                target[field] = value
            else:
                setattr(target, field, value)
        return self

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<Case {self.base.label}  V={self.cond.V} m/s  "
            f"h={self.cond.altitude} m  {len(self.cached())} cached>"
        )
