"""Integrated preliminary analyses for :mod:`flightlab.project` designs.

The vortex lattice uses every station of every horizontal lifting surface.
Profile drag integrates local section data over those strips; empirical body
drag remains a component buildup. Equivalent trapezoids are retained only for
legacy derivative solvers that explicitly report that approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import airfoil, atmos, drag, loads as structural_loads, propulsion, stability, wing
from .project import AircraftProject, FlightCase, LiftingSurface
from .vlm import (
    Cosine,
    Freestream,
    Reference,
    Stability,
    Uniform,
    body_forces,
    far_field_drag,
    lifting_line_coefficients,
    lifting_line_geometry,
    stability_derivatives,
    steady_analysis,
    wing_to_grid,
)

__all__ = [
    "ProjectTrim",
    "TrimNotPossibleError",
    "DesignPoint",
    "ProjectDragBuildup",
    "AircraftPolar",
    "ProjectStructuralAnalysis",
    "PropulsionSystemPoint",
    "PropulsionAnalysis",
    "PropulsionDerivatives",
    "DynamicStability",
    "analyze",
    "trim",
    "neutral_point",
    "run_design_point",
    "profile_drag_buildup",
    "surface_section_cl_max",
    "aircraft_polar",
    "analyze_structure",
    "analyze_propulsion",
    "propulsion_derivatives",
    "analyze_dynamic_stability",
]

G0 = 9.80665


class TrimNotPossibleError(ValueError):
    """The requested flight case cannot be trimmed within the entered control limits."""


@dataclass(frozen=True)
class ProjectTrim:
    alpha: float
    trim_deflection: float
    mass: float
    x_cg: float
    x_np: float
    static_margin: float
    CL_required: float
    lift_residual: float
    moment_residual: float
    solution: wing.Solution

    @property
    def converged(self) -> bool:
        return abs(self.lift_residual) < 1e-6 and abs(self.moment_residual) < 1e-8


@dataclass(frozen=True)
class ProjectDragBuildup(drag.Buildup):
    """Profile/body drag whose surface rows include lift-dependent section drag."""

    @property
    def CD_profile_body(self) -> float:
        return self.f / self.S_ref

    def table(self) -> str:
        return super().table().replace("CD0 =", "CD_profile+body =")


@dataclass(frozen=True)
class DesignPoint:
    project_name: str
    case: FlightCase
    mass_properties: stability.MassProperties
    trim: ProjectTrim
    buildup: ProjectDragBuildup
    CD_total: float
    drag: float
    lift_to_drag: float
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AircraftPolar:
    """Whole-aircraft aerodynamic sweep at fixed speed and pitch-trim incidence."""

    alpha: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CD_profile: np.ndarray
    CD_i: np.ndarray
    Cm: np.ndarray
    LD: np.ndarray
    trim_deflection: float
    solutions: Tuple[wing.Solution, ...]


@dataclass(frozen=True)
class ProjectStructuralAnalysis:
    """One project-aware aerodynamic load case and preliminary spar result."""

    case: FlightCase
    surface: str
    load_factor: float
    design_speed: float
    alpha: float
    lift_slope_per_degree: float
    solution: wing.Solution
    span_load: structural_loads.SpanLoad
    sizing: Dict[str, float]
    deflection: Dict[str, object]
    cap_thickness: float
    EI: float


@dataclass(frozen=True)
class PropulsionAnalysis:
    """Shared-battery, multi-propulsor operating point and speed sweep."""

    operating_point: "PropulsionSystemPoint"
    speed: np.ndarray
    thrust_available: np.ndarray
    drag_required: np.ndarray
    current: np.ndarray
    rpm: np.ndarray
    efficiency_motor: np.ndarray
    efficiency_propeller: np.ndarray
    efficiency_esc: np.ndarray
    efficiency_total: np.ndarray
    power_electrical: np.ndarray
    power_shaft: np.ndarray
    power_useful: np.ndarray
    extrapolated: np.ndarray
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PropulsionSystemPoint:
    """One common battery-bus solution feeding all project propulsors."""

    propulsors: Tuple[propulsion.OperatingPoint, ...]
    bus_voltage: float
    current: float
    thrust: float
    power_electrical: float
    power_shaft: float
    power_useful: float
    efficiency_motor: float
    efficiency_propeller: float
    efficiency_esc: float
    efficiency_total: float

    @property
    def extrapolated(self) -> bool:
        return any(point.extrapolated for point in self.propulsors)

    def table(self) -> str:
        lines = [
            f"shared bus       {self.bus_voltage:10.3f} V",
            f"battery current {self.current:10.3f} A",
            f"total thrust    {self.thrust:10.3f} N",
            f"electrical in   {self.power_electrical:10.2f} W",
            f"shaft power     {self.power_shaft:10.2f} W",
            f"system efficiency {100 * self.efficiency_total:7.1f}%",
        ]
        for index, point in enumerate(self.propulsors, start=1):
            lines.extend(("", f"propulsor {index}", point.table()))
        return "\n".join(lines)


@dataclass(frozen=True)
class PropulsionDerivatives:
    """Fixed-throttle speed derivatives and throttle-control derivatives."""

    dT_dV: float
    dT_dthrottle: float
    dM_dV: float
    dM_dthrottle: float
    thrust: float
    pitch_moment: float


@dataclass(frozen=True)
class DynamicStability:
    """Linear modes from the project's mass model and equivalent surfaces."""

    longitudinal: stability.Modes
    lateral: stability.Modes
    derivatives: stability.Derivatives
    body_increments: Dict[str, float]
    propulsion_increments: Optional[PropulsionDerivatives]
    warnings: Tuple[str, ...] = ()


def _station_dihedral(surface: LiftingSurface) -> np.ndarray:
    """Rotate vertical-surface sections into the x-y plane.

    Horizontal-surface dihedral is already present in the stations' y/z path;
    rotating their absolute coordinates again would double-count it.
    """
    n = len(surface.stations)
    if surface.orientation == "vertical":
        return np.full(n, np.pi / 2.0)
    return np.zeros(n)


def _grid_for_surface(
    project: AircraftProject,
    surface: LiftingSurface,
    ns: int,
    nc: int,
    trim_deflection: float = 0.0,
):
    stations = surface.stations
    camber = [project.section(station.airfoil).camber_function() for station in stations]
    incidence_offset = trim_deflection if surface.trim_control == "whole_surface" else 0.0
    if surface.trim_control == "elevator" and trim_deflection:
        hinge = surface.control_hinge_fraction
        slope = np.tan(np.radians(trim_deflection))

        def deflected(base):
            def camber_line(x):
                x = np.asarray(x, dtype=float)
                # Positive elevator deflection is trailing-edge down.
                return base(x) - np.maximum(x - hinge, 0.0) * slope
            return camber_line

        camber = [deflected(base) for base in camber]
    return wing_to_grid(
        xle=[station.x_le for station in stations],
        yle=[station.y for station in stations],
        zle=[station.z for station in stations],
        chord=[station.chord for station in stations],
        theta=np.radians([station.twist_deg + incidence_offset for station in stations]),
        phi=_station_dihedral(surface),
        ns=ns,
        nc=nc,
        fc=camber,
        mirror=False,
        spacing_s=Cosine(),
        spacing_c=Uniform(),
    )


def _longitudinal_surfaces(project: AircraftProject) -> List[LiftingSurface]:
    surfaces = list(project.horizontal_surfaces)
    if any(not surface.symmetric for surface in surfaces):
        raise ValueError(
            "the longitudinal prototype requires horizontal surfaces to be symmetric; "
            "asymmetric and lateral cases will be added separately"
        )
    return surfaces


def _solve_system(
    project: AircraftProject,
    case: FlightCase,
    alpha: float,
    trim_deflection: float,
    ns: int,
    nc: int,
    x_ref: Optional[float],
    derivatives: bool = False,
):
    project.require_valid()
    surfaces = _longitudinal_surfaces(project)
    primary = project.primary_horizontal_surface
    S_ref, b_ref, c_ref = project.reference_quantities()
    x_ref = primary.aerodynamic_center_x if x_ref is None else float(x_ref)
    grids, ratios, names = [], [], []
    for surface in surfaces:
        surface_ns = max(8, int(round(ns * surface.span / b_ref)))
        control = trim_deflection if surface.trim_control != "fixed" else 0.0
        grid, ratio = _grid_for_surface(project, surface, surface_ns, nc, control)
        grids.append(grid)
        ratios.append(ratio)
        names.append(surface.name)
    reference = Reference(
        S_ref,
        c_ref,
        b_ref,
        [x_ref, 0.0, 0.0],
        case.speed,
    )
    fs = Freestream.from_degrees(case.speed, alpha=alpha)
    system = steady_analysis(
        grids,
        reference,
        fs,
        symmetric=True,
        ratios=ratios,
        derivatives=derivatives,
    )
    return system, surfaces, names, x_ref


def _solution(
    project: AircraftProject,
    case: FlightCase,
    alpha: float,
    system,
    names,
    x_ref: float,
) -> wing.Solution:
    air = atmos.at(case.altitude)
    CF, CM = body_forces(system, frame=Stability())
    CL, CY = float(CF[2]), float(CF[1])
    CD_i = float(far_field_drag(system))
    S_ref, b_ref, c_ref = project.reference_quantities()
    AR = b_ref**2 / S_ref
    e_inv = CL**2 / (np.pi * AR * CD_i) if CD_i > 0 else float("nan")

    r_ll, c_ll = lifting_line_geometry(system.grids)
    cf, cm = lifting_line_coefficients(system, r_ll, c_ll, frame=Stability())
    ys, chords, cls, cms, dss, slices = [], [], [], [], [], {}
    start = 0
    for i, name in enumerate(names):
        r_i = r_ll[i]
        y_edge = r_i[1, :]
        ds = np.linalg.norm(np.diff(r_i, axis=1), axis=0)
        chord = 0.5 * (c_ll[i][:-1] + c_ll[i][1:])
        ys.append(0.5 * (y_edge[:-1] + y_edge[1:]))
        chords.append(chord)
        cls.append(cf[i][2, :])
        cms.append(cm[i][1, :])
        dss.append(2.0 * ds)  # symmetric solve represents both sides
        slices[name] = slice(start, start + len(ds))
        start += len(ds)

    y = np.concatenate(ys)
    chord = np.concatenate(chords)
    return wing.Solution(
        alpha=float(alpha),
        beta=0.0,
        V=float(case.speed),
        altitude=float(case.altitude),
        air=air,
        CL=CL,
        CD_i=CD_i,
        CY=CY,
        Cl=float(CM[0]),
        Cm=float(CM[1]) + stability.body_pitching_moment(
            project.equivalent_aircraft()
        ) * np.radians(alpha),
        Cn=float(CM[2]),
        e_inv=float(e_inv),
        y=y,
        chord=chord,
        cl=np.concatenate(cls),
        cm_section=np.concatenate(cms),
        Re=air.reynolds(case.speed, chord),
        ds=np.concatenate(dss),
        surfaces=tuple(names),
        surface_slices=slices,
        reference={
            "area": S_ref,
            "mac": c_ref,
            "span": b_ref,
            "aspect_ratio": AR,
            "x_ref": x_ref,
        },
        _system=system,
    )


def analyze(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    alpha: Optional[float] = None,
    trim_deflection: float = 0.0,
    ns: int = 28,
    nc: int = 4,
    x_ref: Optional[float] = None,
) -> wing.Solution:
    """Analyze all horizontal lifting surfaces using their full station geometry."""
    case = project.case() if case is None else case
    alpha = case.alpha_deg if alpha is None else float(alpha)
    system, _, names, x_ref = _solve_system(
        project, case, alpha, trim_deflection, ns, nc, x_ref
    )
    return _solution(project, case, alpha, system, names, x_ref)


def neutral_point(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    alpha: Optional[float] = None,
    trim_deflection: float = 0.0,
    x_ref: Optional[float] = None,
    ns: int = 28,
    nc: int = 4,
) -> Dict[str, float]:
    """Lifting-surface neutral point and static slopes for a project."""
    case = project.case() if case is None else case
    alpha = case.alpha_deg if alpha is None else float(alpha)
    primary = project.primary_horizontal_surface
    S_ref, b_ref, c_ref = project.reference_quantities()
    x_ref = primary.aerodynamic_center_x if x_ref is None else float(x_ref)
    system, _, _, _ = _solve_system(
        project, case, alpha, trim_deflection, ns, nc, x_ref, derivatives=True
    )
    dCF, dCM = stability_derivatives(system)
    CL_alpha = float(dCF["alpha"][2])
    body_increment = stability.body_pitching_moment(project.equivalent_aircraft())
    Cm_alpha = float(dCM["alpha"][1]) + body_increment
    if abs(CL_alpha) < 1e-12:
        raise ValueError("neutral point is undefined because CL_alpha is zero")
    x_np = x_ref - Cm_alpha / CL_alpha * c_ref
    return {
        "x_np": float(x_np),
        "x_np_over_mac": float(
            (x_np - primary.aerodynamic_center_x + 0.25 * c_ref)
            / c_ref
        ),
        "CL_alpha": CL_alpha,
        "Cm_alpha": Cm_alpha,
        "body_Cm_alpha": body_increment,
        "x_ref": x_ref,
    }


def trim(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    ns: int = 28,
    nc: int = 4,
    guess: Tuple[float, float] = (3.0, -2.0),
) -> ProjectTrim:
    """Solve lift and pitching-moment balance using full station geometry."""
    case = project.case() if case is None else case
    if not project.trim_surfaces:
        raise TrimNotPossibleError(
            "no horizontal surface has a whole-surface or elevator pitch-trim control"
        )
    mass_properties = stability.mass_properties(project.components())
    q = atmos.at(case.altitude).q(case.speed)
    S_ref, _, c_ref = project.reference_quantities()
    CL_required = case.load_factor * mass_properties.mass * G0 / (q * S_ref)

    def solve(alpha, deflection):
        return analyze(
            project, case, alpha=alpha, trim_deflection=deflection,
            ns=ns, nc=nc, x_ref=mass_properties.x_cg,
        )

    alpha, deflection = map(float, guess)
    s00 = solve(alpha, deflection)
    s10 = solve(alpha + 1.0, deflection)
    s01 = solve(alpha, deflection + 1.0)
    jacobian = np.array([
        [s10.CL - s00.CL, s01.CL - s00.CL],
        [s10.Cm - s00.Cm, s01.Cm - s00.Cm],
    ])
    if abs(np.linalg.det(jacobian)) < 1e-14:
        raise TrimNotPossibleError(
            "trim response is singular; check that the selected pitch control has area, "
            "an effective hinge/incidence change, and a pitching-moment arm"
        )
    inverse = np.linalg.inv(jacobian)
    solution = s00
    for _ in range(12):
        residual = np.array([solution.CL - CL_required, solution.Cm])
        if abs(residual[0]) < 1e-10 and abs(residual[1]) < 1e-11:
            break
        step = inverse @ -residual
        alpha += float(step[0])
        deflection += float(step[1])
        solution = solve(alpha, deflection)
    else:
        raise TrimNotPossibleError(
            f"trim did not converge: lift residual {solution.CL - CL_required:.3e}, "
            f"moment residual {solution.Cm:.3e}"
        )

    lower = max(surface.control_min_deg for surface in project.trim_surfaces)
    upper = min(surface.control_max_deg for surface in project.trim_surfaces)
    if lower > upper:
        raise TrimNotPossibleError(
            "pitch-trim surfaces do not share an overlapping deflection range"
        )
    if not lower <= deflection <= upper:
        controls = ", ".join(surface.name for surface in project.trim_surfaces)
        raise TrimNotPossibleError(
            f"trim requires {deflection:.2f} deg from {controls}, outside the entered "
            f"{lower:.2f} to {upper:.2f} deg control limits"
        )

    np_data = neutral_point(
        project, case, alpha=alpha, trim_deflection=deflection,
        x_ref=mass_properties.x_cg, ns=ns, nc=nc,
    )
    return ProjectTrim(
        alpha=alpha,
        trim_deflection=deflection,
        mass=mass_properties.mass,
        x_cg=mass_properties.x_cg,
        x_np=np_data["x_np"],
        static_margin=(np_data["x_np"] - mass_properties.x_cg) / c_ref,
        CL_required=CL_required,
        lift_residual=solution.CL - CL_required,
        moment_residual=solution.Cm,
        solution=solution,
    )


def run_design_point(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    ns: int = 28,
    nc: int = 4,
) -> DesignPoint:
    """Run geometry, mass/CG, VLM trim, stability, and parasite drag together."""
    case = project.case() if case is None else case
    mass_properties = stability.mass_properties(project.components())
    trimmed = trim(project, case, ns=ns, nc=nc)
    buildup = profile_drag_buildup(project, trimmed.solution, case)
    CD_total = buildup.CD_profile_body + trimmed.solution.CD_i
    q = atmos.at(case.altitude).q(case.speed)
    S_ref, _, _ = project.reference_quantities()
    drag_force = CD_total * q * S_ref
    warnings = [
        "Surface profile drag integrates local airfoil cd(cl, Re) over every VLM strip; body drag uses empirical component correlations.",
        "Static body pitching effects use a slender-body correction; no coupled body-panel solution is used.",
    ]
    if any(surface.trim_control == "elevator" for surface in project.trim_surfaces):
        warnings.append(
            "Elevator deflection changes VLM camber aft of the hinge; profile drag still uses the "
            "clean section polar and does not include hinge-gap or control-surface drag."
        )
    warnings.extend(f"No drag geometry for {name}." for name in buildup.skipped)
    return DesignPoint(
        project_name=project.name,
        case=case,
        mass_properties=mass_properties,
        trim=trimmed,
        buildup=buildup,
        CD_total=float(CD_total),
        drag=float(drag_force),
        lift_to_drag=float(trimmed.solution.CL / CD_total),
        warnings=tuple(warnings),
    )


def _surface_strip_profile(project, surface, solution, case):
    """Return a drag.Row from local section data over one VLM surface."""
    view = solution.surface(surface.name)
    ds_one = view.ds / (2.0 if surface.symmetric else 1.0)
    strip_s = np.cumsum(ds_one) - 0.5 * ds_one
    station_s = np.r_[0.0, np.cumsum(surface.path_lengths())]
    interval = np.clip(np.searchsorted(station_s, strip_s, side="right") - 1, 0, len(surface.stations) - 2)
    denom = np.maximum(station_s[interval + 1] - station_s[interval], 1e-12)
    blend = (strip_s - station_s[interval]) / denom
    cd = np.empty_like(view.cl)
    perimeter = np.empty_like(view.cl)
    extrapolated = []
    for j in np.unique(interval):
        mask = interval == j
        left = surface.stations[j]
        right = surface.stations[j + 1]
        re_range = (max(1e4, 0.7 * float(np.min(view.Re[mask]))), 1.3 * float(np.max(view.Re[mask])))
        left_section = project.section(left.airfoil)
        right_section = project.section(right.airfoil)
        options = dict(n_crit=case.n_crit, xtr_upper=case.xtr_upper, xtr_lower=case.xtr_lower)
        left_table = airfoil.table(left_section, Re=re_range, **options)
        right_table = airfoil.table(right_section, Re=re_range, **options)
        wl = 1.0 - blend[mask]
        wr = blend[mask]
        cd[mask] = wl * left_table.cd(view.cl[mask], view.Re[mask]) + wr * right_table.cd(view.cl[mask], view.Re[mask])
        lp = project._section_properties(left_section)[3]
        rp = project._section_properties(right_section)[3]
        perimeter[mask] = wl * lp + wr * rp
        if np.any(left_table.out_of_range(view.Re[mask])) or np.any(right_table.out_of_range(view.Re[mask])):
            extrapolated.append(f"station interval {j + 1} Reynolds range clamped")
    f = float(np.sum(cd * view.chord * view.ds))
    swet = float(np.sum(perimeter * view.chord * view.ds))
    representative_re = float(np.average(view.Re, weights=view.chord * view.ds))
    return drag.Row(
        surface.name, "surface profile", swet, surface.mac, representative_re,
        float("nan"), float("nan"), f, float("nan"), "; ".join(extrapolated),
    )


def surface_section_cl_max(
    project: AircraftProject,
    surface: LiftingSurface,
    solution: wing.Solution,
    case: Optional[FlightCase] = None,
) -> np.ndarray:
    """Local section ``cl_max`` at every VLM strip's airfoil and Reynolds number."""
    case = project.case() if case is None else case
    view = solution.surface(surface.name)
    ds_one = view.ds / (2.0 if surface.symmetric else 1.0)
    strip_s = np.cumsum(ds_one) - 0.5 * ds_one
    station_s = np.r_[0.0, np.cumsum(surface.path_lengths())]
    interval = np.clip(
        np.searchsorted(station_s, strip_s, side="right") - 1,
        0,
        len(surface.stations) - 2,
    )
    denom = np.maximum(station_s[interval + 1] - station_s[interval], 1e-12)
    blend = (strip_s - station_s[interval]) / denom
    cl_max = np.empty_like(view.cl)
    options = dict(
        n_crit=case.n_crit,
        xtr_upper=case.xtr_upper,
        xtr_lower=case.xtr_lower,
    )
    for station_index in np.unique(interval):
        mask = interval == station_index
        left = surface.stations[station_index]
        right = surface.stations[station_index + 1]
        re_range = (
            max(1e4, 0.7 * float(np.min(view.Re[mask]))),
            1.3 * float(np.max(view.Re[mask])),
        )
        left_table = airfoil.table(project.section(left.airfoil), Re=re_range, **options)
        right_table = airfoil.table(project.section(right.airfoil), Re=re_range, **options)
        left_weight = 1.0 - blend[mask]
        right_weight = blend[mask]
        cl_max[mask] = (
            left_weight * left_table.cl_max(view.Re[mask])
            + right_weight * right_table.cl_max(view.Re[mask])
        )
    return cl_max


def _unloaded_surface_profile(project, surface, case):
    """Profile-drag row for a surface omitted from the longitudinal VLM, e.g. a fin."""
    air = atmos.at(case.altitude)
    f = swet = re_weight = area_weight = 0.0
    notes = []
    for j, (left, right, ds) in enumerate(
        zip(surface.stations[:-1], surface.stations[1:], surface.path_lengths())
    ):
        for xi, gw in ((0.1127016654, 5 / 18), (0.5, 8 / 18), (0.8872983346, 5 / 18)):
            chord = left.chord + xi * (right.chord - left.chord)
            re = float(air.reynolds(case.speed, chord))
            re_range = (max(1e4, 0.7 * re), 1.3 * re)
            ls, rs = project.section(left.airfoil), project.section(right.airfoil)
            options = dict(n_crit=case.n_crit, xtr_upper=case.xtr_upper, xtr_lower=case.xtr_lower)
            lt = airfoil.table(ls, Re=re_range, **options)
            rt = airfoil.table(rs, Re=re_range, **options)
            cd = (1.0 - xi) * float(lt.cd(0.0, re)) + xi * float(rt.cd(0.0, re))
            perimeter = (
                (1.0 - xi) * project._section_properties(ls)[3]
                + xi * project._section_properties(rs)[3]
            )
            multiplier = 2.0 if surface.symmetric else 1.0
            strip = multiplier * ds * gw
            f += cd * chord * strip
            swet += perimeter * chord * strip
            re_weight += re * chord * strip
            area_weight += chord * strip
    return drag.Row(
        surface.name, "surface profile (zero side force)", swet, surface.mac,
        re_weight / max(area_weight, 1e-12), float("nan"), float("nan"), f,
        float("nan"), "; ".join(notes),
    )


def profile_drag_buildup(
    project: AircraftProject,
    solution: wing.Solution,
    case: Optional[FlightCase] = None,
) -> ProjectDragBuildup:
    """Station-resolved lifting-surface profile drag plus empirical body drag."""
    case = project.case() if case is None else case
    handbook = drag.buildup(
        project.equivalent_aircraft(), case.speed, altitude=case.altitude,
        interference=0.0, protuberance=0.0, f_other=0.0, cooling=0.0,
    )
    body_names = {body.name for body in project.bodies}
    rows = []
    for surface in project.surfaces:
        if surface.name in solution.surfaces:
            rows.append(_surface_strip_profile(project, surface, solution, case))
        else:
            rows.append(_unloaded_surface_profile(project, surface, case))
    rows.extend(row for row in handbook.rows if row.name in body_names)
    return ProjectDragBuildup(
        rows=tuple(rows), S_ref=project.reference_quantities()[0],
        interference=case.interference, protuberance=case.protuberance,
        f_other=case.f_other + case.cooling, V=case.speed,
        altitude=case.altitude, mach=float(atmos.at(case.altitude).mach(case.speed)),
        skipped=handbook.skipped,
    )


def aircraft_polar(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    alpha=None,
    trim_deflection: Optional[float] = None,
    ns: int = 28,
    nc: int = 4,
) -> AircraftPolar:
    """Sweep the complete lifting-surface model and add handbook parasite drag.

    ``CL``, ``CD_i`` and ``Cm`` come from the full station VLM geometry.
    Surface profile/body drag comes from strip and component integration at the case speed. The tail
    pitch-trim incidence is held fixed through the sweep, as it is for an aircraft polar.
    """
    case = project.case() if case is None else case
    alpha = np.linspace(-6.0, 14.0, 17) if alpha is None else np.asarray(alpha, dtype=float)
    if trim_deflection is None:
        trim_deflection = trim(project, case, ns=ns, nc=nc).trim_deflection
    mp = stability.mass_properties(project.components())
    solutions = tuple(
        analyze(
            project, case, alpha=float(value), trim_deflection=float(trim_deflection),
            ns=ns, nc=nc, x_ref=mp.x_cg,
        )
        for value in np.atleast_1d(alpha)
    )
    cl = np.array([item.CL for item in solutions])
    cdi = np.array([item.CD_i for item in solutions])
    cd0 = np.array([
        profile_drag_buildup(project, item, case).CD_profile_body for item in solutions
    ])
    cd = cd0 + cdi
    return AircraftPolar(
        alpha=np.asarray(alpha, dtype=float), CL=cl, CD=cd, CD_profile=cd0,
        CD_i=cdi, Cm=np.array([item.Cm for item in solutions]),
        LD=np.divide(cl, cd, out=np.full_like(cl, np.nan), where=cd > 0),
        trim_deflection=float(trim_deflection), solutions=solutions,
    )


def analyze_structure(
    project: AircraftProject,
    case: FlightCase,
    *,
    surface: str,
    load_factor: float,
    speed: float,
    ns: int = 36,
    nc: int = 4,
    spar_height: float = 0.03,
    allowable_stress: float = 300e6,
    ultimate_factor: float = 1.5,
    elastic_modulus: float = 70e9,
    cap_width: float = 0.02,
) -> ProjectStructuralAnalysis:
    """Run one direct project load case through preliminary two-cap spar sizing."""
    project.require_valid()
    lifting_surface = project.surface_named(surface)
    if lifting_surface is None or lifting_surface.orientation != "horizontal":
        raise ValueError("choose a horizontal structural lifting surface")
    if load_factor <= 0:
        raise ValueError("structural design load factor must be positive")
    if speed <= 0:
        raise ValueError("structural design speed must be positive")
    if cap_width <= 0:
        raise ValueError("available cap width must be positive")
    if elastic_modulus <= 0:
        raise ValueError("cap elastic modulus must be positive")

    mass_properties = stability.mass_properties(project.components())
    mass = mass_properties.mass
    reference_area = project.reference_quantities()[0]
    load_case = replace(case, speed=float(speed), load_factor=float(load_factor))
    zero = analyze(
        project, load_case, alpha=0.0, trim_deflection=0.0,
        ns=ns, nc=nc, x_ref=mass_properties.x_cg,
    )
    one = analyze(
        project, load_case, alpha=1.0, trim_deflection=0.0,
        ns=ns, nc=nc, x_ref=mass_properties.x_cg,
    )
    lift_slope_per_degree = one.CL - zero.CL
    if abs(lift_slope_per_degree) < 1e-8:
        raise ValueError("the lifting-surface model has essentially zero lift-curve slope")
    target_cl = load_factor * mass * G0 / (zero.q * reference_area)
    alpha = (target_cl - zero.CL) / lift_slope_per_degree
    solution = analyze(
        project, load_case, alpha=alpha, trim_deflection=0.0,
        ns=ns, nc=nc, x_ref=mass_properties.x_cg,
    )
    span = structural_loads.span_load(
        project.equivalent_aircraft(), mass=mass, n=load_factor, V=speed,
        altitude=case.altitude, ns=ns, solution=solution, surface=surface,
    )
    sizing = structural_loads.spar_sizing(
        abs(span.root_moment), height=spar_height,
        sigma_allow=allowable_stress, safety_factor=ultimate_factor,
    )
    cap_thickness = sizing["cap_area"] / cap_width
    cap_EI = elastic_modulus * sizing["cap_area"] * spar_height**2 / 2.0
    deflection = structural_loads.tip_deflection(span, EI=cap_EI)
    return ProjectStructuralAnalysis(
        case=load_case,
        surface=surface,
        load_factor=float(load_factor),
        design_speed=float(speed),
        alpha=float(alpha),
        lift_slope_per_degree=float(lift_slope_per_degree),
        solution=solution,
        span_load=span,
        sizing=sizing,
        deflection=deflection,
        cap_thickness=float(cap_thickness),
        EI=float(cap_EI),
    )


def _propulsion_system_point(
    project: AircraftProject,
    case: FlightCase,
    speed: float,
    throttle_offset: float = 0.0,
) -> PropulsionSystemPoint:
    """Close every propulsor on one battery bus, including total-current sag."""
    setup = project.propulsion
    if setup is None:
        raise ValueError("this project has no electric propulsion setup")
    battery = project.battery()
    open_voltage = propulsion.battery_voltage(battery, 0.0, setup.state_of_charge)
    bus_voltage = open_voltage
    points = ()
    for _ in range(40):
        points = tuple(
            propulsion.operating_point(
                project.motor(propulsor), project.propeller(propulsor).model(), battery,
                float(speed),
                throttle=float(np.clip(propulsor.throttle + throttle_offset, 0.01, 1.0)),
                altitude=case.altitude, soc=setup.state_of_charge,
                esc=project.esc(propulsor), supply_voltage=bus_voltage,
            )
            for propulsor in setup.propulsors
        )
        power = sum(point.power_electrical for point in points)
        current = power / max(bus_voltage, 1e-9)
        target = propulsion.battery_voltage(battery, current, setup.state_of_charge)
        updated = 0.5 * (bus_voltage + target)
        if abs(updated - bus_voltage) < 1e-7:
            bus_voltage = updated
            break
        bus_voltage = updated
    else:
        raise ValueError("shared battery-bus voltage did not converge")

    # Recompute once at the converged common bus voltage.
    points = tuple(
        propulsion.operating_point(
            project.motor(propulsor), project.propeller(propulsor).model(), battery,
            float(speed),
            throttle=float(np.clip(propulsor.throttle + throttle_offset, 0.01, 1.0)),
            altitude=case.altitude, soc=setup.state_of_charge,
            esc=project.esc(propulsor), supply_voltage=bus_voltage,
        )
        for propulsor in setup.propulsors
    )
    power_electrical = float(sum(point.power_electrical for point in points))
    power_shaft = float(sum(point.power_shaft for point in points))
    power_useful = float(sum(point.power_useful for point in points))
    motor_input = float(sum(
        point.power_electrical * point.efficiency_esc for point in points
    ))
    thrust = float(sum(
        point.thrust
        * np.cos(np.radians(propulsor.pitch_deg))
        * np.cos(np.radians(propulsor.yaw_deg))
        for propulsor, point in zip(setup.propulsors, points)
    ))
    return PropulsionSystemPoint(
        propulsors=points, bus_voltage=float(bus_voltage),
        current=power_electrical / max(bus_voltage, 1e-9), thrust=thrust,
        power_electrical=power_electrical, power_shaft=power_shaft,
        power_useful=power_useful,
        efficiency_motor=power_shaft / motor_input if motor_input > 0 else 0.0,
        efficiency_propeller=power_useful / power_shaft if power_shaft > 0 else 0.0,
        efficiency_esc=motor_input / power_electrical if power_electrical > 0 else 0.0,
        efficiency_total=power_useful / power_electrical if power_electrical > 0 else 0.0,
    )


def analyze_propulsion(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    speed=None,
) -> PropulsionAnalysis:
    """Match all propulsors on their shared battery bus to aircraft drag."""
    case = project.case() if case is None else case
    setup = project.propulsion
    if setup is None:
        raise ValueError("this project has no electric propulsion setup")
    project.require_valid()
    battery = project.battery()
    if speed is None:
        speed = np.linspace(max(1.0, 0.45 * case.speed), 1.8 * case.speed, 19)
    speed = np.asarray(speed, dtype=float)
    points = tuple(_propulsion_system_point(project, case, value) for value in speed)
    at_case = _propulsion_system_point(project, case, case.speed)
    mp = stability.mass_properties(project.components())
    air = atmos.at(case.altitude)
    S = project.reference_quantities()[0]
    design = run_design_point(project, case)
    zero = analyze(project, case, alpha=0.0, trim_deflection=design.trim.trim_deflection, x_ref=mp.x_cg)
    one = analyze(project, case, alpha=1.0, trim_deflection=design.trim.trim_deflection, x_ref=mp.x_cg)
    cl_per_degree = one.CL - zero.CL
    drag_required = []
    for value in speed:
        q = air.q(float(value))
        cl = case.load_factor * mp.mass * G0 / (q * S)
        alpha = (cl - zero.CL) / cl_per_degree
        local_case = replace(case, speed=float(value))
        solution = analyze(
            project, local_case, alpha=alpha,
            trim_deflection=design.trim.trim_deflection, x_ref=mp.x_cg,
        )
        buildup = profile_drag_buildup(project, solution, local_case)
        cd = buildup.CD_profile_body + solution.CD_i
        drag_required.append(cd * q * S)
    warnings = []
    if any(point.extrapolated for point in points):
        warnings.append("One or more propeller points extrapolate beyond measured advance-ratio coverage.")
    if at_case.current > battery.current_max:
        warnings.append(
            f"Shared battery current {at_case.current:.1f} A exceeds its "
            f"{battery.current_max:.1f} A continuous rating."
        )
    for propulsor, point in zip(setup.propulsors, at_case.propulsors):
        motor = project.motor(propulsor)
        esc = project.esc(propulsor)
        if point.current > motor.current_max:
            warnings.append(
                f"{propulsor.name}: motor current {point.current:.1f} A exceeds "
                f"the {motor.current_max:.1f} A limit."
            )
        if point.current > esc.current_max:
            warnings.append(
                f"{propulsor.name}: current {point.current:.1f} A exceeds the "
                f"{esc.current_max:.1f} A ESC limit."
            )
    return PropulsionAnalysis(
        operating_point=at_case, speed=speed,
        thrust_available=np.array([point.thrust for point in points]),
        drag_required=np.asarray(drag_required),
        current=np.array([point.current for point in points]),
        rpm=np.array([[unit.rpm for unit in point.propulsors] for point in points]),
        efficiency_motor=np.array([point.efficiency_motor for point in points]),
        efficiency_propeller=np.array([point.efficiency_propeller for point in points]),
        efficiency_esc=np.array([point.efficiency_esc for point in points]),
        efficiency_total=np.array([point.efficiency_total for point in points]),
        power_electrical=np.array([point.power_electrical for point in points]),
        power_shaft=np.array([point.power_shaft for point in points]),
        power_useful=np.array([point.power_useful for point in points]),
        extrapolated=np.array([point.extrapolated for point in points], dtype=bool),
        warnings=tuple(warnings),
    )


def propulsion_derivatives(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
) -> PropulsionDerivatives:
    """Differentiate total thrust and pitch moment on the shared battery bus."""
    case = project.case() if case is None else case
    setup = project.propulsion
    if setup is None:
        raise ValueError("this project has no electric propulsion setup")
    mp = stability.mass_properties(project.components())

    def loads(V, throttle_offset):
        system = _propulsion_system_point(
            project, case, float(V), throttle_offset=float(throttle_offset)
        )
        forward_total = moment_total = 0.0
        for propulsor, point in zip(setup.propulsors, system.propulsors):
            pitch = np.radians(propulsor.pitch_deg)
            yaw = np.radians(propulsor.yaw_deg)
            forward = point.thrust * np.cos(pitch) * np.cos(yaw)
            vertical = point.thrust * np.sin(pitch) * np.cos(yaw)
            x = propulsor.x - mp.x_cg
            z = propulsor.z - mp.z_cg
            forward_total += forward
            # x is aft and z is up; positive result is nose-up.
            moment_total += -z * forward - x * vertical
        return float(forward_total), float(moment_total)

    dv = max(0.1, 0.01 * case.speed)
    t_lo, m_lo = loads(max(0.0, case.speed - dv), 0.0)
    t_hi, m_hi = loads(case.speed + dv, 0.0)
    dT_dV = (t_hi - t_lo) / (2.0 * dv)
    dM_dV = (m_hi - m_lo) / (2.0 * dv)
    dt = 0.02
    throttle_lo = max(-dt, max(0.01 - item.throttle for item in setup.propulsors))
    throttle_hi = min(dt, min(1.0 - item.throttle for item in setup.propulsors))
    if throttle_hi - throttle_lo < 1e-8:
        raise ValueError("collective throttle derivative is unavailable at the entered limits")
    tt_lo, mt_lo = loads(case.speed, throttle_lo)
    tt_hi, mt_hi = loads(case.speed, throttle_hi)
    span = throttle_hi - throttle_lo
    thrust, moment = loads(case.speed, 0.0)
    return PropulsionDerivatives(
        dT_dV=dT_dV, dT_dthrottle=(tt_hi - tt_lo) / span,
        dM_dV=dM_dV, dM_dthrottle=(mt_hi - mt_lo) / span,
        thrust=thrust, pitch_moment=moment,
    )


def analyze_dynamic_stability(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    ns: int = 22,
    nc: int = 4,
) -> DynamicStability:
    """Compute longitudinal and lateral linear modes for a project.

    This first workbench implementation uses equivalent single-trapezoid
    surfaces because the existing mirrored derivative solver consumes the
    legacy aircraft adapter. Area, span, MAC, placement, sweep, twist, mass,
    CG, and inertia are retained. Body and propulsion increments are then
    added with the documented empirical models; station breaks are not retained.
    """
    case = project.case() if case is None else case
    project.require_valid()
    aircraft = project.equivalent_aircraft()
    mp = stability.mass_properties(project.components())
    trimmed = stability.trim(
        aircraft, case.speed, case.altitude, mass=mp.mass, x_cg=mp.x_cg,
        ns=ns, nc=nc, include_body=True,
    )
    derivatives = stability.derivatives(
        aircraft, case.speed, case.altitude, alpha=trimmed.alpha,
        x_cg=mp.x_cg, tail_incidence_deg=trimmed.tail_incidence,
        ns=ns, nc=nc, lateral=True,
    )
    body = stability.body_derivatives(aircraft, mp.x_cg)
    design = run_design_point(project, case, ns=ns, nc=nc)
    drag_slope = aircraft_polar(
        project, case,
        alpha=[design.trim.alpha - 0.25, design.trim.alpha + 0.25],
        trim_deflection=design.trim.trim_deflection, ns=ns, nc=nc,
    )
    cd_alpha = float((drag_slope.CD[1] - drag_slope.CD[0]) / np.radians(0.5))
    corrected = replace(
        derivatives,
        CD=design.CD_total,
        CD_alpha=cd_alpha,
        Cm=derivatives.Cm + body["Cm_alpha"] * np.radians(trimmed.alpha),
        Cm_alpha=derivatives.Cm_alpha + body["Cm_alpha"],
        CY_beta=derivatives.CY_beta + body["CY_beta"],
        Cl_beta=derivatives.Cl_beta + body["Cl_beta"],
        Cn_beta=derivatives.Cn_beta + body["Cn_beta"],
        CY_p=derivatives.CY_p + body["CY_p"],
        CY_r=derivatives.CY_r + body["CY_r"],
        Cl_p=derivatives.Cl_p + body["Cl_p"],
        Cn_p=derivatives.Cn_p + body["Cn_p"],
        Cl_r=derivatives.Cl_r + body["Cl_r"],
        Cn_r=derivatives.Cn_r + body["Cn_r"],
    )
    power = propulsion_derivatives(project, case) if project.propulsion is not None else None
    longitudinal = stability.longitudinal_modes(
        aircraft, case.speed, case.altitude, mass=mp.mass, Iyy=mp.Iyy,
        x_cg=mp.x_cg, derivs=corrected,
        thrust_dT_dV=power.dT_dV if power else 0.0,
        thrust_dM_dV=power.dM_dV if power else 0.0,
    )
    lateral = stability.lateral_modes(
        aircraft, case.speed, case.altitude, mass=mp.mass,
        Ixx=mp.Ixx, Izz=mp.Izz, Ixz=mp.Ixz, x_cg=mp.x_cg,
        derivs=corrected,
    )
    warnings = [
        "Dynamic derivatives use equivalent trapezoidal lifting surfaces; station breaks are not yet retained.",
        "Body increments use slender-body and strip-crossflow correlations, not a coupled body-panel solution.",
        "Propulsion speed derivatives assume fixed throttle; nonlinear controls and propeller gyroscopic effects are omitted.",
    ]
    if any(surface.trim_control == "elevator" for surface in project.trim_surfaces):
        warnings.append(
            "The dynamic-mode adapter does not yet retain elevator hinge geometry; elevator trim is "
            "used by the integrated design point but the derivative model uses an equivalent full tail."
        )
    return DynamicStability(
        longitudinal=longitudinal, lateral=lateral, derivatives=corrected,
        body_increments=body, propulsion_increments=power,
        warnings=tuple(warnings),
    )
