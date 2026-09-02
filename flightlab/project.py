"""Serializable aircraft projects for open-ended FlightLab design work.

The fleet module is a reference catalog.  This module is the student's design
model: lifting surfaces are piecewise-linear lofts through any number of
stations, bodies and mass items are editable rows, and several named flight
conditions may belong to one aircraft.

The JSON representation contains only ordinary numbers, strings, lists, and
dictionaries.  It is intentionally independent of Panel and the solver so the
same project can be used by the workbench, scripts, and future import/export
adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import catalog, foil
from .fleet import Aircraft, Body, Component, Planform

__all__ = [
    "AirfoilDefinition",
    "SurfaceStation",
    "LiftingSurface",
    "ReferenceGeometry",
    "BodyDefinition",
    "MassItem",
    "FlightCase",
    "PropulsorSetup",
    "PropellerPoint",
    "PropellerDefinition",
    "PropulsionSetup",
    "StructuralSetup",
    "ProjectIssue",
    "AircraftProject",
    "blank_project",
    "example_project",
]

FORMAT_VERSION = 3


@dataclass
class AirfoilDefinition:
    """A custom section stored inside a project."""

    name: str
    coordinates: List[List[float]]
    source: str = "user supplied"

    @classmethod
    def from_section(cls, section: "foil.Section") -> "AirfoilDefinition":
        return cls(section.name, np.asarray(section.coordinates).tolist(), section.source)

    def section(self) -> "foil.Section":
        return foil.Section(self.name, np.asarray(self.coordinates, dtype=float), self.source)


@dataclass
class SurfaceStation:
    """One defining cross-section of a piecewise-linear lifting surface."""

    x_le: float
    y: float
    z: float
    chord: float
    twist_deg: float = 0.0
    airfoil: str = "naca0012"


@dataclass
class LiftingSurface:
    """A wing, horizontal tail, fin, or additional lifting surface.

    Stations are absolute body-axis coordinates and must run from root to tip.
    A symmetric surface stores only one side; its area and span include the
    reflected side.
    """

    name: str
    orientation: str
    purpose: str
    trim_control: str
    symmetric: bool
    stations: List[SurfaceStation] = field(default_factory=list)
    control_hinge_fraction: float = 0.75
    control_min_deg: float = -25.0
    control_max_deg: float = 25.0

    def path_lengths(self) -> np.ndarray:
        if len(self.stations) < 2:
            return np.array([], dtype=float)
        y = np.array([s.y for s in self.stations])
        z = np.array([s.z for s in self.stations])
        return np.hypot(np.diff(y), np.diff(z))

    @property
    def area(self) -> float:
        if len(self.stations) < 2:
            return 0.0
        chords = np.array([s.chord for s in self.stations])
        one_side = float(np.sum(0.5 * (chords[:-1] + chords[1:]) * self.path_lengths()))
        return one_side * (2.0 if self.symmetric else 1.0)

    @property
    def span(self) -> float:
        if len(self.stations) < 2:
            return 0.0
        length = float(np.sum(self.path_lengths()))
        return length * (2.0 if self.symmetric else 1.0)

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord from exact integration of each linear loft."""
        if len(self.stations) < 2 or self.area <= 0.0:
            return 0.0
        c = np.array([s.chord for s in self.stations])
        ds = self.path_lengths()
        int_c2 = np.sum(ds * (c[:-1] ** 2 + c[:-1] * c[1:] + c[1:] ** 2) / 3.0)
        one_side_area = self.area / (2.0 if self.symmetric else 1.0)
        return float(int_c2 / one_side_area)

    @property
    def aerodynamic_center_x(self) -> float:
        """Area-weighted quarter-chord location, suitable as a VLM reference."""
        if len(self.stations) < 2 or self.area <= 0.0:
            return 0.0
        total = moment = 0.0
        for left, right, ds in zip(self.stations[:-1], self.stations[1:], self.path_lengths()):
            # Three-point Gauss integration of c * x_c/4 along a linear section.
            for xi, weight in ((0.1127016654, 5 / 18), (0.5, 8 / 18), (0.8872983346, 5 / 18)):
                c = left.chord + xi * (right.chord - left.chord)
                x = left.x_le + xi * (right.x_le - left.x_le) + 0.25 * c
                total += weight * c * ds
                moment += weight * c * x * ds
        return float(moment / total)


@dataclass
class ReferenceGeometry:
    """Aircraft coefficient reference quantities, independent of surface purpose.

    ``mode="surface"`` derives all three quantities from ``surface``.
    ``mode="selected_surfaces"`` sums the named surfaces' planform area, uses
    their largest span, and area-weights their MACs. ``mode="manual"`` uses
    the explicitly entered values. This separation permits biplanes, tandem
    wings, canards, and user-selected reference conventions without changing
    how any surface participates in the aerodynamic model.
    """

    mode: str = "surface"
    surface: str = "Main wing"
    surfaces: List[str] = field(default_factory=list)
    area: Optional[float] = None
    span: Optional[float] = None
    chord: Optional[float] = None


@dataclass
class BodyDefinition:
    """A fuselage, nacelle, boom, pod, strut, or landing-gear item."""

    name: str
    length: float
    width: Optional[float] = None
    height: Optional[float] = None
    diameter: Optional[float] = None
    x_nose: Optional[float] = None
    y: float = 0.0
    z: float = 0.0
    count: int = 1
    drag_model: str = "streamlined_body"
    cone_fraction: float = 0.4

    def to_body(self) -> Body:
        model = "streamlined" if self.drag_model == "streamlined_body" else self.drag_model
        return Body(**{**asdict(self), "drag_model": model})


@dataclass
class MassItem:
    """A point mass or a mass distributed over project geometry.

    ``distributed`` may be empty/``"point"``, ``"span"``,
    ``"surface_area"``, ``"surface_volume"``, or ``"body_volume"``.
    Geometry-attached rows name their surface/body in ``attached_to``. Give
    either total ``mass`` or a volumetric ``density``; an area-distributed
    shell also requires ``skin_thickness`` when density supplies the total.
    """

    name: str
    mass: Optional[float]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    distributed: str = ""
    span: Optional[float] = None
    attached_to: str = ""
    density: Optional[float] = None
    skin_thickness: Optional[float] = None

    def to_component(self) -> Component:
        if self.mass is None:
            raise ValueError(f"{self.name}: a point/span mass requires total mass")
        return Component(
            self.name, self.mass, self.x, self.y, self.z,
            "span" if self.distributed == "span" else "", self.span,
        )


@dataclass
class FlightCase:
    """One named operating condition and its early-design drag assumptions.

    ``alpha_deg`` is the initial angle-of-attack guess for integrated trim; the
    trim solver adjusts it to satisfy the required lift and zero pitching
    moment. ``xtr_upper=xtr_lower=1`` imposes no artificial boundary-layer trip
    and is presented as natural transition in the workbench.
    """

    name: str
    speed: float
    altitude: float = 0.0
    load_factor: float = 1.0
    alpha_deg: float = 2.0
    interference: float = 0.05
    protuberance: float = 0.05
    f_other: float = 0.0
    cooling: float = 0.0
    n_crit: float = 9.0
    xtr_upper: float = 1.0
    xtr_lower: float = 1.0


@dataclass
class PropulsorSetup:
    """One motor–ESC–propeller unit and its thrust application point."""

    name: str = "Propulsor 1"
    motor: str = "M1000"
    propeller: str = "P10x7"
    esc: str = "ESC30"
    throttle: float = 1.0
    x: float = -0.08
    y: float = 0.0
    z: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class PropulsionSetup:
    """A shared battery bus feeding one or more positioned propulsors."""

    battery: str = "B3S1300"
    state_of_charge: float = 0.9
    battery_x: float = 0.02
    battery_y: float = 0.0
    battery_z: float = 0.0
    include_component_masses: bool = True
    propulsors: List[PropulsorSetup] = field(default_factory=lambda: [PropulsorSetup()])


@dataclass
class StructuralSetup:
    """Saved physical definition of the preliminary two-cap spar model.

    Flight condition, load factor, and numerical panel counts belong to an
    analysis request.  The selected structural surface, material properties,
    and spar geometry describe the aircraft and therefore travel with it.
    """

    surface: str = ""
    spar_height: float = 0.03
    allowable_stress: float = 300e6
    ultimate_factor: float = 1.5
    elastic_modulus: float = 70e9
    cap_width: float = 0.02


@dataclass
class PropellerPoint:
    """One measured propeller coefficient point.

    ``rpm`` is rev/min; ``J``, ``CT``, and ``CP`` use the standard UIUC
    nondimensional definitions. Multiple RPM sweeps may be stored in a project.
    """

    rpm: float
    J: float
    CT: float
    CP: float


@dataclass
class PropellerDefinition:
    """An editable propeller dataset stored in the aircraft project.

    A starter entry may refer to bundled UIUC data through ``data``. When
    ``points`` is nonempty, those project-owned measurements are used instead;
    diameter and pitch are then required in metres.
    """

    key: str
    name: str
    mass: float
    data: str = ""
    diameter: Optional[float] = None
    pitch: Optional[float] = None
    points: List[PropellerPoint] = field(default_factory=list)
    notes: str = ""

    def model(self):
        from .propulsion import PropellerModel, propeller_model

        if self.points:
            if self.diameter is None or self.pitch is None:
                raise ValueError(f"{self.key}: measured propeller data needs diameter and pitch")
            return PropellerModel.from_points(
                self.key, self.diameter, self.pitch,
                [asdict(point) for point in self.points],
            )
        if not self.data:
            raise ValueError(f"{self.key}: choose bundled data or import measured coefficient points")
        return propeller_model(self.data)


def _starter_motors() -> Dict[str, catalog.Motor]:
    return dict(catalog.MOTORS)


def _starter_batteries() -> Dict[str, catalog.Battery]:
    return dict(catalog.BATTERIES)


def _starter_escs() -> Dict[str, catalog.ESC]:
    return dict(catalog.ESCS)


def _starter_propellers() -> Dict[str, PropellerDefinition]:
    return {
        key: PropellerDefinition(
            key=entry.key, name=entry.name, mass=entry.mass, data=entry.data,
            notes=entry.notes,
        )
        for key, entry in catalog.PROPELLERS.items()
    }


@dataclass(frozen=True)
class ProjectIssue:
    level: str
    message: str


@dataclass
class AircraftProject:
    """One editable aircraft design and all of its analysis cases."""

    name: str
    surfaces: List[LiftingSurface] = field(default_factory=list)
    reference: ReferenceGeometry = field(default_factory=ReferenceGeometry)
    bodies: List[BodyDefinition] = field(default_factory=list)
    masses: List[MassItem] = field(default_factory=list)
    cases: List[FlightCase] = field(default_factory=list)
    airfoils: Dict[str, AirfoilDefinition] = field(default_factory=dict)
    propulsion: Optional[PropulsionSetup] = field(default_factory=PropulsionSetup)
    structure: StructuralSetup = field(default_factory=StructuralSetup)
    motors: Dict[str, catalog.Motor] = field(default_factory=_starter_motors)
    batteries: Dict[str, catalog.Battery] = field(default_factory=_starter_batteries)
    escs: Dict[str, catalog.ESC] = field(default_factory=_starter_escs)
    propellers: Dict[str, PropellerDefinition] = field(default_factory=_starter_propellers)
    notes: str = ""
    format_version: int = FORMAT_VERSION

    def surface(self, purpose: str) -> Optional[LiftingSurface]:
        """Return the first surface with a descriptive purpose."""
        return next((surface for surface in self.surfaces if surface.purpose == purpose), None)

    def surface_named(self, name: str) -> Optional[LiftingSurface]:
        return next((surface for surface in self.surfaces if surface.name == name), None)

    @property
    def horizontal_surfaces(self) -> Tuple[LiftingSurface, ...]:
        return tuple(surface for surface in self.surfaces if surface.orientation == "horizontal")

    @property
    def vertical_surfaces(self) -> Tuple[LiftingSurface, ...]:
        return tuple(surface for surface in self.surfaces if surface.orientation == "vertical")

    @property
    def trim_surfaces(self) -> Tuple[LiftingSurface, ...]:
        return tuple(
            surface for surface in self.horizontal_surfaces
            if surface.trim_control != "fixed"
        )

    @property
    def reference_surface(self) -> LiftingSurface:
        surface = self.surface_named(self.reference.surface)
        if surface is None:
            raise ValueError(f"reference surface {self.reference.surface!r} does not exist")
        return surface

    def reference_quantities(self) -> Tuple[float, float, float]:
        """Return ``(S_ref, b_ref, c_ref)`` for coefficient normalization."""
        if self.reference.mode == "surface":
            surface = self.reference_surface
            return surface.area, surface.span, surface.mac
        if self.reference.mode == "selected_surfaces":
            surfaces = tuple(
                surface for name in self.reference.surfaces
                if (surface := self.surface_named(name)) is not None
            )
            if not surfaces:
                raise ValueError("selected-surfaces reference needs at least one surface")
            area = sum(surface.area for surface in surfaces)
            span = max(surface.span for surface in surfaces)
            chord = sum(surface.area * surface.mac for surface in surfaces) / area
            return float(area), float(span), float(chord)
        if self.reference.mode == "manual":
            if self.reference.area is None or self.reference.span is None or self.reference.chord is None:
                raise ValueError("manual reference needs area, span, and chord")
            return float(self.reference.area), float(self.reference.span), float(self.reference.chord)
        raise ValueError(f"unknown reference mode {self.reference.mode!r}")

    @property
    def primary_horizontal_surface(self) -> LiftingSurface:
        """Geometry used by legacy single-wing handbook adapters."""
        if self.reference.mode == "surface":
            candidate = self.reference_surface
            if candidate.orientation == "horizontal":
                return candidate
        candidates = [surface for surface in self.horizontal_surfaces if surface.purpose == "wing"]
        candidates = candidates or list(self.horizontal_surfaces)
        if not candidates:
            raise ValueError("the project has no horizontal lifting surface")
        return max(candidates, key=lambda surface: surface.area)

    def motor(self, propulsor: PropulsorSetup) -> catalog.Motor:
        return self.motors[propulsor.motor]

    def battery(self) -> catalog.Battery:
        return self.batteries[self.propulsion.battery]

    def esc(self, propulsor: PropulsorSetup) -> catalog.ESC:
        return self.escs[propulsor.esc]

    def propeller(self, propulsor: PropulsorSetup) -> PropellerDefinition:
        return self.propellers[propulsor.propeller]

    def case(self, name: Optional[str] = None) -> FlightCase:
        if not self.cases:
            raise ValueError("the project has no flight cases")
        if name is None:
            return self.cases[0]
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown flight case {name!r}")

    def section(self, name: str) -> "foil.Section":
        if name in self.airfoils:
            return self.airfoils[name].section()
        return foil.load(name)

    def components(self) -> Tuple[Component, ...]:
        """Resolve point and geometry-attached mass rows into mass properties."""
        components = []
        for item in self.masses:
            model = item.distributed or "point"
            if model in {"point", "span"}:
                components.append(item.to_component())
            elif model in {"surface_area", "surface_volume"}:
                surface = next((s for s in self.surfaces if s.name == item.attached_to), None)
                if surface is None:
                    raise ValueError(
                        f"{item.name}: attached lifting surface {item.attached_to!r} does not exist"
                    )
                components.append(self._surface_mass_component(item, surface))
            elif model == "body_volume":
                body = next((b for b in self.bodies if b.name == item.attached_to), None)
                if body is None:
                    raise ValueError(f"{item.name}: attached body {item.attached_to!r} does not exist")
                components.append(self._body_mass_component(item, body))
            else:
                raise ValueError(f"{item.name}: unknown mass distribution {model!r}")
        components.extend(self.propulsion_components())
        return tuple(components)

    def propulsion_components(self) -> Tuple[Component, ...]:
        """Component-library masses positioned by the propulsion-system geometry."""
        setup = self.propulsion
        if setup is None or not setup.include_component_masses:
            return ()
        components = [Component(
            f"Propulsion battery ({setup.battery})", self.battery().mass,
            setup.battery_x, setup.battery_y, setup.battery_z,
        )]
        for propulsor in setup.propulsors:
            mass = (
                self.motor(propulsor).mass
                + self.esc(propulsor).mass
                + self.propeller(propulsor).mass
            )
            components.append(Component(
                f"{propulsor.name} hardware", mass,
                propulsor.x, propulsor.y, propulsor.z,
            ))
        return tuple(components)

    @staticmethod
    def _section_properties(section: "foil.Section") -> Tuple[float, float, float, float]:
        """Enclosed area, volume centroid, and perimeter in chord units."""
        xy = np.asarray(section.coordinates, dtype=float)
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        cross = xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]
        signed_area = 0.5 * np.sum(cross)
        area = abs(float(signed_area))
        if area < 1e-10:
            return 0.0, 0.5, 0.0, float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        cx = float(np.sum((xy[:-1, 0] + xy[1:, 0]) * cross) / (6.0 * signed_area))
        cz = float(np.sum((xy[:-1, 1] + xy[1:, 1]) * cross) / (6.0 * signed_area))
        perimeter = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        return area, cx, cz, perimeter

    @staticmethod
    def _component_from_samples(name, total_mass, points, weights, chord=None, radial=None):
        points = np.asarray(points, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if total_mass <= 0 or weights.sum() <= 0:
            raise ValueError(f"{name}: derived mass and geometry measure must be positive")
        dm = total_mass * weights / weights.sum()
        cg = np.sum(points * dm[:, None], axis=0) / total_mass
        delta = points - cg
        Ixx = float(np.sum(dm * (delta[:, 1] ** 2 + delta[:, 2] ** 2)))
        Iyy = float(np.sum(dm * (delta[:, 0] ** 2 + delta[:, 2] ** 2)))
        Izz = float(np.sum(dm * (delta[:, 0] ** 2 + delta[:, 1] ** 2)))
        Ixz = float(np.sum(dm * delta[:, 0] * delta[:, 2]))
        if chord is not None:
            # Each small spanwise element retains its chordwise extent.
            local = dm * np.asarray(chord, dtype=float) ** 2 / 12.0
            Iyy += float(np.sum(local))
            Izz += float(np.sum(local))
        if radial is not None:
            # Elliptical cross-section intrinsic inertia for each body slice.
            a, b = np.asarray(radial, dtype=float).T
            Ixx += float(np.sum(dm * (a * a + b * b) / 4.0))
            Iyy += float(np.sum(dm * b * b / 4.0))
            Izz += float(np.sum(dm * a * a / 4.0))
        return Component(
            name, float(total_mass), float(cg[0]), float(cg[1]), float(cg[2]),
            Ixx_cg=Ixx, Iyy_cg=Iyy, Izz_cg=Izz, Ixz_cg=Ixz,
        )

    def _surface_mass_component(self, item: MassItem, surface: LiftingSurface) -> Component:
        points, weights, chords = [], [], []
        props = [self._section_properties(self.section(st.airfoil)) for st in surface.stations]
        for j, (left, right, ds) in enumerate(
            zip(surface.stations[:-1], surface.stations[1:], surface.path_lengths())
        ):
            for xi, gw in ((0.1127016654, 5 / 18), (0.5, 8 / 18), (0.8872983346, 5 / 18)):
                c = left.chord + xi * (right.chord - left.chord)
                xle = left.x_le + xi * (right.x_le - left.x_le)
                y = left.y + xi * (right.y - left.y)
                z = left.z + xi * (right.z - left.z)
                p0, p1 = props[j], props[j + 1]
                area_coeff = p0[0] + xi * (p1[0] - p0[0])
                cx = p0[1] + xi * (p1[1] - p0[1])
                cz = p0[2] + xi * (p1[2] - p0[2])
                perimeter = p0[3] + xi * (p1[3] - p0[3])
                if item.distributed == "surface_volume":
                    measure = area_coeff * c * c * ds * gw
                else:
                    measure = perimeter * c * ds * gw
                for sign in ([1.0, -1.0] if surface.symmetric else [1.0]):
                    points.append((xle + cx * c, sign * y, z + cz * c))
                    weights.append(measure)
                    chords.append(c)
        measure = float(np.sum(weights))
        if item.mass is not None:
            mass = item.mass
        elif item.density is not None:
            factor = item.skin_thickness if item.distributed == "surface_area" else 1.0
            if factor is None:
                raise ValueError(f"{item.name}: a density-based surface shell needs skin thickness")
            mass = item.density * factor * measure
        else:
            raise ValueError(f"{item.name}: give total mass or material density")
        return self._component_from_samples(item.name, mass, points, weights, chord=chords)

    def _body_mass_component(self, item: MassItem, body: BodyDefinition) -> Component:
        if body.diameter is not None:
            width = height = body.diameter
        else:
            width = body.width
            height = body.height if body.height is not None else body.width
        if width is None or height is None:
            raise ValueError(f"{item.name}: attached body needs a cross-section dimension")
        n = 80
        u = (np.arange(n) + 0.5) / n
        cone = float(np.clip(body.cone_fraction, 0.0, 1.0))
        end = cone / 2.0
        scale = np.ones_like(u)
        if end > 0:
            scale = np.minimum(scale, u / end)
            scale = np.minimum(scale, (1.0 - u) / end)
        scale = np.clip(scale, 0.0, 1.0)
        dx = body.length / n
        area = np.pi * width * height * scale**2 / 4.0
        weights = area * dx * body.count
        x0 = body.x_nose or 0.0
        points = np.column_stack((x0 + u * body.length, np.full(n, body.y), np.full(n, body.z)))
        volume = float(np.sum(weights))
        if item.mass is not None:
            mass = item.mass
        elif item.density is not None:
            mass = item.density * volume
        else:
            raise ValueError(f"{item.name}: give total mass or material density")
        radial = np.column_stack((0.5 * width * scale, 0.5 * height * scale))
        return self._component_from_samples(item.name, mass, points, weights, radial=radial)

    def validate(self) -> List[ProjectIssue]:
        issues: List[ProjectIssue] = []
        if self.format_version != FORMAT_VERSION:
            issues.append(ProjectIssue("error", f"unsupported project format {self.format_version}"))
        if self.reference.mode not in {"surface", "selected_surfaces", "manual"}:
            issues.append(ProjectIssue("error", f"unknown reference mode {self.reference.mode!r}"))
        elif self.reference.mode == "surface" and self.surface_named(self.reference.surface) is None:
            issues.append(ProjectIssue("error", f"choose an existing coefficient reference surface"))
        elif (
            self.reference.mode == "surface"
            and self.reference_surface.orientation != "horizontal"
        ):
            issues.append(ProjectIssue("error", "the coefficient reference surface must be horizontal"))
        elif self.reference.mode == "selected_surfaces":
            if not self.reference.surfaces:
                issues.append(ProjectIssue("error", "select at least one coefficient reference surface"))
            missing = [name for name in self.reference.surfaces if self.surface_named(name) is None]
            if missing:
                issues.append(ProjectIssue("error", f"unknown coefficient reference surfaces: {', '.join(missing)}"))
            vertical = [
                name for name in self.reference.surfaces
                if self.surface_named(name) is not None
                and self.surface_named(name).orientation != "horizontal"
            ]
            if vertical:
                issues.append(ProjectIssue("error", f"coefficient reference surfaces must be horizontal: {', '.join(vertical)}"))
        elif self.reference.mode == "manual":
            values = (self.reference.area, self.reference.span, self.reference.chord)
            if any(value is None or value <= 0 for value in values):
                issues.append(ProjectIssue("error", "manual reference area, span, and chord must be positive"))
        if not self.cases:
            issues.append(ProjectIssue("error", "define at least one flight case"))
        has_automatic_propulsion_mass = (
            self.propulsion is not None and self.propulsion.include_component_masses
        )
        if not self.masses and not has_automatic_propulsion_mass:
            issues.append(ProjectIssue("warning", "no mass components: trim and CG are unavailable"))
        if not self.trim_surfaces:
            issues.append(ProjectIssue(
                "warning", "no horizontal pitch-trim control: integrated trim is unavailable"
            ))
        elif len(self.trim_surfaces) > 1:
            issues.append(ProjectIssue(
                "warning", "multiple pitch-trim surfaces share one commanded deflection"
            ))

        names = set()
        for surface in self.surfaces:
            if surface.name in names:
                issues.append(ProjectIssue("error", f"duplicate surface name {surface.name!r}"))
            names.add(surface.name)
            if surface.orientation not in {"horizontal", "vertical"}:
                issues.append(ProjectIssue("error", f"{surface.name}: unknown orientation {surface.orientation!r}"))
            if surface.purpose not in {"wing", "tail", "canard", "fin", "other"}:
                issues.append(ProjectIssue("error", f"{surface.name}: unknown purpose {surface.purpose!r}"))
            if surface.trim_control not in {"fixed", "whole_surface", "elevator"}:
                issues.append(ProjectIssue("error", f"{surface.name}: unknown trim control {surface.trim_control!r}"))
            if surface.orientation == "vertical" and surface.trim_control != "fixed":
                issues.append(ProjectIssue("error", f"{surface.name}: a vertical surface cannot be a pitch-trim surface"))
            if not 0.05 <= surface.control_hinge_fraction <= 0.95:
                issues.append(ProjectIssue("error", f"{surface.name}: control hinge x/c must be between 0.05 and 0.95"))
            if surface.control_min_deg >= surface.control_max_deg:
                issues.append(ProjectIssue("error", f"{surface.name}: control minimum must be below its maximum"))
            if len(surface.stations) < 2:
                issues.append(ProjectIssue("error", f"{surface.name}: at least two stations are required"))
                continue
            if any(station.chord <= 0 for station in surface.stations):
                issues.append(ProjectIssue("error", f"{surface.name}: every chord must be positive"))
            if np.any(surface.path_lengths() <= 0):
                issues.append(ProjectIssue("error", f"{surface.name}: consecutive stations must be distinct"))
            for station in surface.stations:
                try:
                    self.section(station.airfoil)
                except Exception as exc:
                    issues.append(ProjectIssue("error", f"{surface.name}: airfoil {station.airfoil!r}: {exc}"))

        if self.structure.surface:
            structural_surface = self.surface_named(self.structure.surface)
            if structural_surface is None or structural_surface.orientation != "horizontal":
                issues.append(ProjectIssue(
                    "error", "the saved structural surface must be an existing horizontal surface"
                ))
        for label, value in (
            ("spar-cap centroid spacing", self.structure.spar_height),
            ("cap allowable stress", self.structure.allowable_stress),
            ("limit-to-ultimate factor", self.structure.ultimate_factor),
            ("cap elastic modulus", self.structure.elastic_modulus),
            ("available cap width", self.structure.cap_width),
        ):
            if value <= 0:
                issues.append(ProjectIssue("error", f"structural {label} must be positive"))

        for body in self.bodies:
            if body.length <= 0 or body.count < 1:
                issues.append(ProjectIssue("error", f"{body.name}: length and count must be positive"))
            if body.diameter is None and body.width is None:
                issues.append(ProjectIssue("warning", f"{body.name}: no cross-section; drag row will be skipped"))
            if body.drag_model not in {
                "streamlined_body", "bluff_round_member", "faired_member", "streamlined_strut",
            }:
                issues.append(ProjectIssue("error", f"{body.name}: unknown drag model {body.drag_model!r}"))
        for item in self.masses:
            allowed_mass_models = {"", "point", "span", "surface_area", "surface_volume", "body_volume"}
            if item.distributed not in allowed_mass_models:
                issues.append(ProjectIssue("error", f"{item.name}: unknown mass distribution {item.distributed!r}"))
            if item.mass is not None and item.mass <= 0:
                issues.append(ProjectIssue("error", f"{item.name}: mass must be positive"))
            if item.mass is None and (item.density is None or item.density <= 0):
                issues.append(ProjectIssue("error", f"{item.name}: give a positive mass or density"))
            if item.distributed in {"", "point", "span"} and item.mass is None:
                issues.append(ProjectIssue("error", f"{item.name}: point/span distributions require total mass"))
            if item.density is not None and item.density <= 0:
                issues.append(ProjectIssue("error", f"{item.name}: density must be positive"))
            if item.skin_thickness is not None and item.skin_thickness <= 0:
                issues.append(ProjectIssue("error", f"{item.name}: skin thickness must be positive"))
            if item.distributed in {"surface_area", "surface_volume"}:
                if item.attached_to not in {surface.name for surface in self.surfaces}:
                    issues.append(ProjectIssue("error", f"{item.name}: choose an existing lifting surface"))
            if item.distributed == "body_volume":
                if item.attached_to not in {body.name for body in self.bodies}:
                    issues.append(ProjectIssue("error", f"{item.name}: choose an existing body"))
            if item.distributed == "surface_area" and item.mass is None and not item.skin_thickness:
                issues.append(ProjectIssue("error", f"{item.name}: density-based surface shell needs skin thickness"))
        for case in self.cases:
            if case.speed <= 0:
                issues.append(ProjectIssue("error", f"{case.name}: speed must be positive"))
            if case.altitude < -500 or case.altitude > 50_000:
                issues.append(ProjectIssue("warning", f"{case.name}: check the altitude {case.altitude:g} m"))
            if case.n_crit <= 0:
                issues.append(ProjectIssue("error", f"{case.name}: n_crit must be positive"))
            if not 0.0 < case.xtr_upper <= 1.0 or not 0.0 < case.xtr_lower <= 1.0:
                issues.append(ProjectIssue("error", f"{case.name}: transition x/c must be in (0, 1]"))
        if self.propulsion is not None:
            if self.propulsion.battery not in self.batteries:
                issues.append(ProjectIssue("error", f"unknown propulsion battery {self.propulsion.battery!r}"))
            if not self.propulsion.propulsors:
                issues.append(ProjectIssue("error", "define at least one propulsor"))
            propulsor_names = set()
            for propulsor in self.propulsion.propulsors:
                if propulsor.name in propulsor_names:
                    issues.append(ProjectIssue("error", f"duplicate propulsor name {propulsor.name!r}"))
                propulsor_names.add(propulsor.name)
                selections = (
                    ("motor", propulsor.motor, self.motors),
                    ("propeller", propulsor.propeller, self.propellers),
                    ("ESC", propulsor.esc, self.escs),
                )
                for label, key, choices in selections:
                    if key not in choices:
                        issues.append(ProjectIssue(
                            "error", f"{propulsor.name}: unknown {label} {key!r}"
                        ))
                if not 0.0 < propulsor.throttle <= 1.0:
                    issues.append(ProjectIssue(
                        "error", f"{propulsor.name}: throttle must be greater than 0 and no more than 1"
                    ))
            if not 0.0 <= self.propulsion.state_of_charge <= 1.0:
                issues.append(ProjectIssue("error", "battery state of charge must be between 0 and 1"))
        for key, motor in self.motors.items():
            if key != motor.key:
                issues.append(ProjectIssue("error", f"motor library key {key!r} does not match component key {motor.key!r}"))
        for key, battery in self.batteries.items():
            if key != battery.key:
                issues.append(ProjectIssue("error", f"battery library key {key!r} does not match component key {battery.key!r}"))
        for key, esc in self.escs.items():
            if key != esc.key:
                issues.append(ProjectIssue("error", f"ESC library key {key!r} does not match component key {esc.key!r}"))
        for key, propeller in self.propellers.items():
            if key != propeller.key:
                issues.append(ProjectIssue("error", f"propeller library key {key!r} does not match component key {propeller.key!r}"))
            if propeller.mass <= 0:
                issues.append(ProjectIssue("error", f"{key}: propeller mass must be positive"))
            if propeller.points and (not propeller.diameter or not propeller.pitch):
                issues.append(ProjectIssue("error", f"{key}: measured propeller data needs diameter and pitch"))
        return issues

    def require_valid(self) -> None:
        errors = [issue.message for issue in self.validate() if issue.level == "error"]
        if errors:
            raise ValueError("invalid aircraft project:\n- " + "\n- ".join(errors))

    def total_mass(self) -> float:
        components = self.components()
        if not components:
            raise ValueError("the project has no mass components")
        return float(sum(component.mass for component in components))

    def _mean_thickness(self, surface: LiftingSurface) -> float:
        values = []
        for station in surface.stations:
            try:
                values.append(self.section(station.airfoil).thickness)
            except Exception:
                pass
        return float(np.mean(values)) if values else 0.12

    def _equivalent_planform(self, surface: LiftingSurface) -> Planform:
        """Preserve S, b, MAC, placement, and endpoint incidence for handbook models."""
        S, b, mac = surface.area, surface.span, surface.mac
        if S <= 0 or b <= 0:
            raise ValueError(f"{surface.name}: cannot create an equivalent planform")
        mean_chord = S / b
        target = float(np.clip(mac / mean_chord, 1.0, 4.0 / 3.0 - 1e-8))

        def ratio(taper):
            return (4.0 / 3.0) * (1.0 + taper + taper**2) / (1.0 + taper) ** 2

        lo, hi = 1e-4, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ratio(mid) > target:
                lo = mid
            else:
                hi = mid
        taper = 0.5 * (lo + hi)
        root = 2.0 * mean_chord / (1.0 + taper)
        tip = taper * root
        first, last = surface.stations[0], surface.stations[-1]
        semispan = max(float(np.sum(surface.path_lengths())), 1e-12)
        sweep_le = math.degrees(math.atan2(last.x_le - first.x_le, semispan))
        dihedral = 90.0 if surface.orientation == "vertical" else math.degrees(
            math.atan2(last.z - first.z, max(abs(last.y - first.y), 1e-12))
        )
        return Planform(
            span=b,
            area=S,
            section=first.airfoil,
            section_file=first.airfoil,
            root_chord=root,
            tip_chord=tip,
            taper=taper,
            mean_chord=mac,
            sweep_le_deg=sweep_le,
            dihedral_deg=dihedral,
            twist_deg=last.twist_deg - first.twist_deg,
            incidence_deg=first.twist_deg,
            thickness=self._mean_thickness(surface),
            x_le=first.x_le,
            z=first.z,
            vertical=surface.orientation == "vertical",
            notes=(
                "Equivalent single trapezoid preserving area, span, and MAC for "
                "handbook drag calculations; VLM uses the complete station geometry."
            ),
        )

    def equivalent_aircraft(self) -> Aircraft:
        """Return an Aircraft adapter for handbook drag and legacy helpers."""
        wing = self.primary_horizontal_surface
        htail = next((surface for surface in self.trim_surfaces if surface is not wing), None)
        vtail = max(self.vertical_surfaces, key=lambda surface: surface.area, default=None)
        components = self.components()
        total = sum(component.mass for component in components)
        return Aircraft(
            name=self.name,
            label="PROJECT",
            aircraft_class="student design",
            wing=self._equivalent_planform(wing),
            htail=self._equivalent_planform(htail) if htail else None,
            vtail=self._equivalent_planform(vtail) if vtail else None,
            bodies=tuple(body.to_body() for body in self.bodies),
            mass={"gross": total} if total else {},
            components=components,
            operating={"cruise_speed": self.cases[0].speed if self.cases else 0.0},
            notes=self.notes,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "AircraftProject":
        return cls(
            name=data["name"],
            surfaces=[
                LiftingSurface(
                    name=item["name"], orientation=item["orientation"],
                    purpose=item["purpose"], trim_control=item["trim_control"],
                    symmetric=item["symmetric"],
                    stations=[SurfaceStation(**station) for station in item.get("stations", [])],
                    control_hinge_fraction=item.get("control_hinge_fraction", 0.75),
                    control_min_deg=item.get("control_min_deg", -25.0),
                    control_max_deg=item.get("control_max_deg", 25.0),
                )
                for item in data.get("surfaces", [])
            ],
            reference=ReferenceGeometry(**data.get("reference", {})),
            bodies=[BodyDefinition(**item) for item in data.get("bodies", [])],
            masses=[MassItem(**item) for item in data.get("masses", [])],
            cases=[FlightCase(**item) for item in data.get("cases", [])],
            airfoils={key: AirfoilDefinition(**value) for key, value in data.get("airfoils", {}).items()},
            propulsion=(
                PropulsionSetup(**{
                    **{key: value for key, value in data["propulsion"].items() if key != "propulsors"},
                    "propulsors": [
                        PropulsorSetup(**item)
                        for item in data["propulsion"].get("propulsors", [])
                    ],
                })
                if data.get("propulsion") is not None else None
            ),
            structure=StructuralSetup(**data.get("structure", {})),
            motors={key: catalog.Motor(**value) for key, value in data.get("motors", {}).items()},
            batteries={key: catalog.Battery(**value) for key, value in data.get("batteries", {}).items()},
            escs={key: catalog.ESC(**value) for key, value in data.get("escs", {}).items()},
            propellers={
                key: PropellerDefinition(
                    **{**value, "points": [PropellerPoint(**point) for point in value.get("points", [])]}
                )
                for key, value in data.get("propellers", {}).items()
            },
            notes=data.get("notes", ""),
            format_version=data.get("format_version", FORMAT_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "AircraftProject":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path) -> "AircraftProject":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def blank_project() -> AircraftProject:
    """A modest editable starting geometry, not a reference aircraft."""
    return AircraftProject(
        name="Untitled aircraft",
        notes="Starter geometry for a new student design; replace every assumed value.",
        surfaces=[
            LiftingSurface("Main wing", "horizontal", "wing", "fixed", True, [
                SurfaceStation(0.00, 0.00, 0.00, 0.36, 2.0, "naca2412"),
                SurfaceStation(0.05, 0.45, 0.02, 0.30, 1.0, "naca2412"),
                SurfaceStation(0.14, 0.80, 0.05, 0.18, -1.0, "naca2412"),
            ]),
            LiftingSurface("Horizontal tail", "horizontal", "tail", "whole_surface", True, [
                SurfaceStation(0.72, 0.00, 0.04, 0.18, 0.0, "naca0012"),
                SurfaceStation(0.75, 0.25, 0.04, 0.11, 0.0, "naca0012"),
            ]),
            LiftingSurface("Vertical tail", "vertical", "fin", "fixed", False, [
                SurfaceStation(0.68, 0.00, 0.04, 0.22, 0.0, "naca0012"),
                SurfaceStation(0.78, 0.00, 0.28, 0.10, 0.0, "naca0012"),
            ]),
        ],
        bodies=[BodyDefinition("fuselage", 0.90, width=0.11, height=0.13, x_nose=-0.12)],
        masses=[
            MassItem(
                "wing structure estimate", 0.24,
                distributed="surface_area", attached_to="Main wing",
            ),
            MassItem(
                "fuselage structure estimate", 0.18,
                distributed="body_volume", attached_to="fuselage",
            ),
            MassItem("payload, avionics, and installation hardware", 0.176, 0.10),
        ],
        cases=[FlightCase("Cruise", 12.0, altitude=1400.0, protuberance=0.10)],
    )


def example_project() -> AircraftProject:
    """A visibly multi-panel concept used to demonstrate the workbench."""
    project = blank_project()
    project.name = "Three-panel demonstrator"
    project.notes = (
        "Demonstration project with a constant-chord center section and two tapered "
        "outer panels. It is not a validated aircraft design."
    )
    project.surfaces[0].stations = [
        SurfaceStation(0.00, 0.00, 0.00, 0.38, 2.0, "naca2412"),
        SurfaceStation(0.00, 0.22, 0.00, 0.38, 2.0, "naca2412"),
        SurfaceStation(0.07, 0.58, 0.03, 0.27, 0.5, "sd7037"),
        SurfaceStation(0.16, 0.86, 0.07, 0.16, -1.5, "sd7037"),
    ]
    project.cases.extend([
        FlightCase("Takeoff", 9.0, altitude=1400.0, protuberance=0.10),
        FlightCase("Maneuver", 16.0, altitude=1400.0, load_factor=3.0, protuberance=0.10),
    ])
    return project
