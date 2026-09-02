"""Local browser workbench for designing and analyzing a FlightLab aircraft.

Run with ``flightlab workbench`` or ``python -m flightlab workbench``.  Panel
is deliberately optional; importing the numerical package does not import it.
"""

from __future__ import annotations

from dataclasses import asdict
from html import escape
import io
import math
from pathlib import Path
import re
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import panel as pn

from . import catalog, drag, foil, loads, stability
from .project import (
    AirfoilDefinition,
    AircraftProject,
    BodyDefinition,
    FlightCase,
    LiftingSurface,
    MassItem,
    PropellerDefinition,
    PropellerPoint,
    PropulsorSetup,
    PropulsionSetup,
    ReferenceGeometry,
    SurfaceStation,
    blank_project,
    example_project,
)
from .project_analysis import (
    TrimNotPossibleError,
    _grid_for_surface,
    analyze as analyze_project,
    analyze_structure,
    aircraft_polar,
    analyze_dynamic_stability,
    analyze_propulsion,
    propulsion_derivatives,
    run_design_point,
    surface_section_cl_max,
)

pn.extension("tabulator", notifications=True, sizing_mode="stretch_width")

ORIENTATION_OPTIONS = ["horizontal", "vertical"]
PURPOSE_OPTIONS = ["wing", "tail", "canard", "fin", "other"]
TRIM_CONTROL_OPTIONS = ["fixed", "whole_surface", "elevator"]
REFERENCE_MODES = ["surface", "selected_surfaces", "manual"]
DRAG_MODELS = [
    "streamlined_body", "bluff_round_member", "faired_member", "streamlined_strut",
]

STATION_TOOLTIPS = {
    "x_le": "Leading-edge x coordinate; positive x is aft [m].",
    "y": "Station y coordinate; positive y is to the aircraft's right [m].",
    "z": "Station z coordinate; positive z is up [m].",
    "chord": "Local section chord [m].",
    "twist_deg": "Local geometric incidence; positive rotates the nose up [deg].",
    "airfoil": "Bundled, generated NACA four-digit, or imported project airfoil.",
}
MASS_TOOLTIPS = {
    "mass": "Known total component mass [kg]; leave blank only for a density-derived model.",
    "x": "Point location or reference location for a distributed item [m aft].",
    "y": "Point location or reference location [m right].",
    "z": "Point location or reference location [m up].",
    "distributed": "How this mass is placed for CG and inertia; point is a concentrated mass.",
    "span": "Full span of a span-distributed known mass [m].",
    "attached_to": "Surface or body whose geometry distributes this mass.",
    "density": "Material density used when mass is blank [kg/m³].",
    "skin_thickness": "Shell/skin thickness used by surface_area [m].",
}
CASE_TOOLTIPS = {
    "speed": "True airspeed for this operating condition [m/s].",
    "altitude": "Geometric altitude for the atmosphere model [m].",
    "load_factor": "Required lift divided by weight; 1.0 is steady level flight.",
    "alpha_deg": "Initial angle-of-attack guess [deg]. Integrated trim solves for alpha; this is not the answer.",
    "interference": "Fractional drag-area markup for component interference; 0.05 means +5%.",
    "protuberance": "Fractional drag-area markup for exposed hardware and excrescences; 0.10 means +10%.",
    "f_other": "Additional dimensional drag area not represented by a component [m²].",
    "cooling": "Cooling-flow drag coefficient on aircraft reference area.",
    "transition": "Natural leaves transition free; forced uses the entered upper/lower x/c locations.",
    "n_crit": "Boundary-layer disturbance level for the airfoil model; 9 is smooth, low-turbulence flow.",
    "xtr_upper": "Forced upper-surface transition location x/c; 1.0 means no forced trip.",
    "xtr_lower": "Forced lower-surface transition location x/c; 1.0 means no forced trip.",
}

PYTHON_GUIDE = r"""
## From workbench model to design study

Save the `.flightlab.json` file beside your script. Treat it as the unchanged
baseline, make a fresh copy for each candidate, validate the copy, then collect
named results. FlightLab uses SI units and human-facing angles are in degrees.

### Find and change inputs

```python
from copy import deepcopy
from flightlab.project import AircraftProject

baseline = AircraftProject.load("my-aircraft.flightlab.json")
print([surface.name for surface in baseline.surfaces])
print([body.name for body in baseline.bodies])
print([item.name for item in baseline.masses])
print([case.name for case in baseline.cases])

candidate = deepcopy(baseline)
wing = candidate.surface_named("Main wing")
fuselage = candidate.body_named("fuselage")
if wing is None or fuselage is None:
    raise KeyError("expected geometry is missing")

wing.stations[-1].chord = 0.12
wing.stations[-1].twist_deg = -2.0
fuselage.length = 0.95
candidate.case("Cruise").speed = 14.0
candidate.require_valid()
```

The editable input model is organized as follows:

| Input | Where it lives |
|---|---|
| lifting-surface geometry and airfoils | `project.surfaces[*].stations` |
| fuselages, pods, booms, struts | `project.bodies` |
| point and geometry-attached masses | `project.masses` |
| speed, altitude, load factor, drag assumptions | `project.cases` |
| battery and installed propulsors | `project.propulsion` |
| spar geometry and material properties | `project.structure` |

Use `surface_named`, `body_named`, `mass_named`, `propulsor_named`, and `case`
instead of depending on list positions.

### Use the right analysis

| Question | Function |
|---|---|
| trim, mass/CG, total and component drag | `run_design_point` |
| whole-aircraft alpha sweep | `aircraft_polar` |
| span load and preliminary spar sizing | `analyze_structure` |
| battery–motor–propeller speed sweep | `analyze_propulsion` |
| longitudinal and lateral modes | `analyze_dynamic_stability` |

The **Current project script** tab imports and demonstrates all five using the
file name, case, and numerical controls selected in this workbench.

### Read named outputs

For `result = run_design_point(...)`, commonly needed values are:

| Quantity | Expression | Units |
|---|---|---|
| trim angle | `result.trim.alpha` | deg |
| control deflection | `result.trim.trim_deflection` | deg |
| lift coefficient | `result.trim.solution.CL` | — |
| induced drag coefficient | `result.trim.solution.CD_i` | — |
| total drag coefficient | `result.CD_total` | — |
| total drag | `result.drag` | N |
| lift-to-drag ratio | `result.lift_to_drag` | — |
| mass and CG | `result.mass_properties.mass`, `.x_cg` | kg, m |
| drag components | `result.buildup.rows` or `.table()` | rows/text |
| model cautions | `result.warnings` | strings |

Other result objects expose NumPy arrays such as `polar.CL`, `polar.CD`,
`structure.span_load.moment`, `power.thrust_available`, and mode tables. Use
`dataclasses.fields(result)` and `help(type(result))` to discover every field.

### Reusable sweep pattern

```python
from copy import deepcopy
import numpy as np
from flightlab.project_analysis import run_design_point

rows = []
for value in np.linspace(0.10, 0.20, 11):
    candidate = deepcopy(baseline)       # changes do not leak between trials
    wing = candidate.surface_named("Main wing")
    if wing is None:
        raise KeyError("Main wing not found")
    wing.stations[-1].chord = float(value)
    candidate.require_valid()
    result = run_design_point(candidate, candidate.case("Cruise"), ns=28, nc=4)
    rows.append({
        "tip chord [m]": value,
        "drag [N]": result.drag,
        "trim alpha [deg]": result.trim.alpha,
        "L/D": result.lift_to_drag,
    })
```

Keep `ns` and `nc` fixed while comparing candidates. Copy the project for
physical changes; use `dataclasses.replace(case, speed=...)` for a temporary
flight condition that should not alter the saved project.

`candidate.require_valid()` is a preflight check. It raises `ValueError` with
all validation errors; warnings are allowed. Project analyses also validate,
but checking immediately after edits identifies a bad candidate before an
expensive sweep. `project.validate()` returns the error/warning records when a
script needs to display or filter them, and `project.save()` does not validate.

For atmosphere, performance, airfoil, and other focused calculations, run
`flightlab.show_tools()`, `flightlab.show_tools("performance")`, and
`flightlab.example("performance")`, or use `help()` on a function.

The repository's [complete Python workflow guide](https://github.com/byuflowlab/flightlab/blob/main/docs/python-workflows.md)
adds constrained and multi-parameter studies, all result fields, plotting,
saving modified projects, model limits, and common mistakes.
"""


def _records(frame: pd.DataFrame):
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _coerce_records(frame: pd.DataFrame, schema, table_name: str):
    """Convert editable-table values with errors that identify row and field."""
    rows = []
    for row_number, raw in enumerate(_records(frame), start=1):
        row = {}
        for field, converter, optional in schema:
            value = raw.get(field)
            if optional and (value is None or (isinstance(value, str) and not value.strip())):
                row[field] = None
                continue
            try:
                if converter is bool:
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if lowered not in {"true", "false", "yes", "no", "1", "0"}:
                            raise ValueError
                        value = lowered in {"true", "yes", "1"}
                    else:
                        value = bool(value)
                elif converter is str:
                    value = str(value).strip()
                    if not value:
                        raise ValueError
                else:
                    value = converter(value)
            except (TypeError, ValueError):
                expected = "a number" if converter in {float, int} else converter.__name__
                raise ValueError(
                    f"{table_name} row {row_number}, {field}: expected {expected}; got {value!r}"
                ) from None
            row[field] = value
        rows.append(row)
    return rows


def _set_axes_equal_3d(ax):
    """Give x, y, and z one physical scale in a Matplotlib 3-D axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = limits.mean(axis=1)
    radius = max(0.5 * np.max(limits[:, 1] - limits[:, 0]), 1e-6)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _display_grid_for_surface(surface: LiftingSurface, ns: int, nc: int) -> np.ndarray:
    """Build a body-axis panel grid for a lifting-surface preview."""
    stations = surface.stations
    if len(stations) < 2:
        raise ValueError("a panel preview needs at least two stations")
    distances = np.concatenate(([0.0], np.cumsum(surface.path_lengths())))
    if distances[-1] <= 0.0:
        raise ValueError("surface stations must span a nonzero distance")
    eta = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ns + 1)))
    locations = eta * distances[-1]

    def interpolate(attribute):
        return np.interp(
            locations, distances, [getattr(item, attribute) for item in stations]
        )

    x_le = interpolate("x_le")[:, None]
    y_le = interpolate("y")[:, None]
    z_le = interpolate("z")[:, None]
    chord = interpolate("chord")[:, None]
    twist = np.radians(interpolate("twist_deg"))[:, None]
    chord_fraction = np.linspace(0.0, 1.0, nc + 1)[None, :]
    chordwise = chord * chord_fraction
    x = x_le + chordwise * np.cos(twist)
    if surface.orientation == "vertical":
        y = y_le + chordwise * np.sin(twist)
        z = np.broadcast_to(z_le, x.shape).copy()
    else:
        y = np.broadcast_to(y_le, x.shape).copy()
        z = z_le - chordwise * np.sin(twist)
    return np.stack((x, y, z))


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return (stem or "aircraft") + ".flightlab.json"


def _table(frame, height=260, editors=None, configuration=None, header_tooltips=None):
    tooltips = header_tooltips or {}
    return pn.widgets.Tabulator(
        frame,
        show_index=False,
        selectable="checkbox",
        height=height,
        editors=editors or {},
        configuration=configuration or {"layout": "fitColumns"},
        header_tooltips=tooltips,
        titles={name: f"{name} ⓘ" for name in tooltips},
    )


class Workbench:
    """Stateful Panel view over an :class:`AircraftProject`."""

    def __init__(self, project: Optional[AircraftProject] = None):
        self.project = project or example_project()
        self._updating = False
        self._last_airfoil_result = None
        self._last_airfoil_alpha = None
        self._last_airfoil_context = None
        self._last_result = None
        self._last_polar = None
        self._last_stall = None
        self._analysis_cache = {}
        self._last_loads_result = None
        self._last_propulsion_result = None

        self.status = pn.pane.Alert("Ready.", alert_type="light")
        self.project_name = pn.widgets.TextInput(label="Project name")
        self.project_notes = pn.widgets.TextAreaInput(label="Design notes", height=110)
        self.validation = pn.pane.HTML()

        self.blank_button = pn.widgets.Button(label="New starter design", icon="file-plus")
        self.example_button = pn.widgets.Button(label="Load multi-panel example", icon="plane")
        self.project_upload = pn.widgets.FileInput(label="Open project", accept=".json,.flightlab")
        self.project_filename = pn.widgets.TextInput(
            label="Project file name", placeholder="aircraft.flightlab.json"
        )
        self.project_download = pn.widgets.FileDownload(
            label="Download project", icon="download", callback=self._download_project
        )

        self.surface_select = pn.widgets.Select(label="Lifting surface")
        self.surface_name = pn.widgets.TextInput(label="Surface name")
        self.surface_orientation = pn.widgets.Select(label="Orientation", options=ORIENTATION_OPTIONS)
        self.surface_purpose = pn.widgets.Select(label="Purpose (descriptive)", options=PURPOSE_OPTIONS)
        self.surface_trim_control = pn.widgets.Select(
            label="Pitch-trim control", options=TRIM_CONTROL_OPTIONS
        )
        self.surface_control_hinge = pn.widgets.FloatInput(
            label="Elevator hinge x/c", value=0.75, start=0.05, end=0.95, step=0.05
        )
        self.surface_control_min = pn.widgets.FloatInput(
            label="Minimum deflection [deg]", value=-25.0, step=1.0
        )
        self.surface_control_max = pn.widgets.FloatInput(
            label="Maximum deflection [deg]", value=25.0, step=1.0
        )
        self.surface_symmetric = pn.widgets.Checkbox(label="Mirror across centerline")
        self.reference_mode = pn.widgets.Select(label="Coefficient reference", options=REFERENCE_MODES)
        self.reference_surface = pn.widgets.Select(label="Reference surface")
        self.reference_surfaces = pn.widgets.MultiChoice(label="Surfaces included in Sref")
        self.reference_area = pn.widgets.FloatInput(label="Manual Sref [m²]", value=1.0, step=0.01)
        self.reference_span = pn.widgets.FloatInput(label="Manual bref [m]", value=1.0, step=0.01)
        self.reference_chord = pn.widgets.FloatInput(label="Manual cref [m]", value=0.2, step=0.01)
        self.station_table = _table(
            pd.DataFrame(), height=285,
            editors={name: {"type": "number"} for name in ("x_le", "y", "z", "chord", "twist_deg")},
            header_tooltips=STATION_TOOLTIPS,
        )
        self.add_station_button = pn.widgets.Button(label="Add station", icon="plus")
        self.delete_station_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.add_surface_button = pn.widgets.Button(label="Add surface", icon="plus")
        self.delete_surface_button = pn.widgets.Button(label="Delete surface", icon="trash")

        self.body_table = _table(
            pd.DataFrame(), height=250,
            editors={
                **{name: {"type": "number"} for name in ("length", "width", "height", "diameter", "x_nose", "y", "z", "count", "cone_fraction")},
                "drag_model": {"type": "list", "values": DRAG_MODELS},
            }, configuration={"layout": "fitDataTable"},
        )
        self.add_body_button = pn.widgets.Button(label="Add body/component", icon="plus")
        self.delete_body_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.mass_table = _table(
            pd.DataFrame(), height=275,
            editors={
                **{name: {"type": "number"} for name in ("mass", "x", "y", "z", "span", "density", "skin_thickness")},
                "distributed": {"type": "list", "values": ["point", "span", "surface_area", "surface_volume", "body_volume"]},
            }, configuration={"layout": "fitDataTable"}, header_tooltips=MASS_TOOLTIPS,
        )
        self.add_mass_button = pn.widgets.Button(label="Add mass item", icon="plus")
        self.delete_mass_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.case_table = _table(
            pd.DataFrame(), height=285,
            editors={
                **{name: {"type": "number"} for name in (
                    "speed", "altitude", "load_factor", "alpha_deg", "interference",
                    "protuberance", "f_other", "cooling",
                    "n_crit", "xtr_upper", "xtr_lower",
                )},
                "transition": {"type": "list", "values": ["natural", "forced"]},
            }, configuration={"layout": "fitDataTable"}, header_tooltips=CASE_TOOLTIPS,
        )
        self.add_case_button = pn.widgets.Button(label="Add flight case", icon="plus")
        self.delete_case_button = pn.widgets.Button(label="Delete selected", icon="trash")

        self.geometry_plot = pn.pane.Matplotlib(height=760, tight=True, format="svg")
        self.surface_geometry_plot = pn.pane.Matplotlib(height=650, tight=True, format="svg")
        self.show_mass_components = pn.widgets.Checkbox(
            label="Show mass components (CG is always shown)", value=False,
        )
        self.show_panel_mesh = pn.widgets.Checkbox(
            label="Show lifting-surface panels", value=False,
        )
        self.mass_marker_legend = pn.pane.HTML(visible=False)
        self.geometry_summary = pn.pane.HTML()
        self.surface_summary_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=210)
        self.body_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=230)
        self.mass_summary = pn.pane.HTML()
        self.mass_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=240)
        self.mass_geometry_plot = pn.pane.Matplotlib(height=400, tight=True, format="svg")

        self.airfoil_select = pn.widgets.Select(label="Airfoil")
        self.airfoil_upload = pn.widgets.FileInput(label="Import .dat", accept=".dat")
        self.naca_code = pn.widgets.TextInput(
            label="Add NACA four-digit section", placeholder="0009", width=220,
            description="Enter four digits, such as 0009 or 2412. Coordinates are generated analytically.",
        )
        self.add_naca_button = pn.widgets.Button(label="Add NACA section", icon="plus")
        self.airfoil_re = pn.widgets.FloatInput(label="Reynolds number", value=5e5, step=5e4)
        self.airfoil_alpha_min = pn.widgets.FloatInput(label="Minimum α [deg]", value=-6.0)
        self.airfoil_alpha_max = pn.widgets.FloatInput(label="Maximum α [deg]", value=16.0)
        self.airfoil_transition = pn.widgets.FloatInput(
            label="Forced transition x/c", value=1.0, step=0.05, start=0.01, end=1.0
        )
        self.airfoil_plots = pn.widgets.CheckBoxGroup(
            label="Plots",
            options=[
                "Airfoil shape", "Lift curve", "Drag polar", "Pitching moment",
                "Lift-to-drag", "Transition",
            ],
            value=["Airfoil shape", "Lift curve", "Drag polar", "Pitching moment", "Lift-to-drag"],
            inline=True,
        )
        self.run_airfoil_button = pn.widgets.Button(
            label="Run airfoil analysis", color="primary", icon="player-play"
        )
        self.airfoil_download = pn.widgets.FileDownload(
            label="Download airfoil CSV", icon="download",
            callback=self._download_airfoil_csv, disabled=True,
        )
        self.airfoil_plot = pn.pane.Matplotlib(height=620, tight=True, format="svg")
        self.airfoil_metrics = pn.pane.HTML()
        self.airfoil_diagnostics = pn.pane.Alert(alert_type="light")

        self.analysis_case = pn.widgets.Select(label="Flight case")
        self.analysis_ns = pn.widgets.IntInput(label="Reference-span panels", value=28, start=12, end=100)
        self.analysis_nc = pn.widgets.IntInput(label="Chordwise panels", value=4, start=2, end=12)
        self.run_analysis_button = pn.widgets.Button(
            label="Run integrated design point", color="primary", icon="player-play"
        )
        self.analysis_download = pn.widgets.FileDownload(
            label="Download polar CSV", icon="download",
            callback=self._download_analysis_csv, disabled=True,
        )
        self.analysis_span_download = pn.widgets.FileDownload(
            label="Download spanwise CSV", icon="download",
            callback=self._download_spanwise_csv, disabled=True,
        )
        self.analysis_metrics = pn.pane.HTML()
        self.analysis_plots = pn.pane.Matplotlib(height=820, tight=True, format="svg")
        self.drag_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=260)
        self.analysis_warnings = pn.pane.Alert(alert_type="warning", visible=False)

        self.loads_case = pn.widgets.Select(label="Flight case atmosphere")
        self.loads_surface = pn.widgets.Select(label="Structural lifting surface")
        self.loads_mode = pn.widgets.Select(
            label="Load-case method",
            options=["Direct RC design case", "Maneuver/gust V–n envelope"],
            value="Direct RC design case",
        )
        self.loads_cl_max = pn.widgets.FloatInput(
            label="Maximum lift coefficient CLmax", value=1.4, start=0.1, step=0.05
        )
        self.loads_cl_min = pn.widgets.FloatInput(
            label="Minimum lift coefficient CLmin", value=-0.8, end=-0.01, step=0.05
        )
        self.loads_n_pos = pn.widgets.FloatInput(
            label="Positive limit load factor", value=3.8, start=0.1, step=0.1
        )
        self.loads_n_neg = pn.widgets.FloatInput(
            label="Negative limit load factor", value=-1.5, end=-0.01, step=0.1
        )
        self.loads_v_max = pn.widgets.FloatInput(
            label="Maximum/dive speed [m/s] (0 = 1.4 × case)", value=0.0, start=0.0, step=1.0
        )
        self.loads_gust = pn.widgets.FloatInput(
            label="Sharp-edged gust speed [m/s]", value=15.24, start=0.0, step=0.5
        )
        self.loads_factor = pn.widgets.FloatInput(
            label="Structural design load factor", value=3.8, start=0.1, step=0.1
        )
        self.loads_design_speed = pn.widgets.FloatInput(
            label="Structural design speed [m/s] (0 = flight-case speed)",
            value=0.0, start=0.0, step=1.0,
        )
        self.loads_ns = pn.widgets.IntInput(
            label="Spanwise panels", value=36, start=12, end=100
        )
        self.loads_spar_height = pn.widgets.FloatInput(
            label="Distance between spar-cap centroids [m]", value=0.03, start=0.001, step=0.005
        )
        self.loads_allowable = pn.widgets.FloatInput(
            label="Cap allowable stress [MPa]", value=300.0, start=0.1, step=10.0
        )
        self.loads_ultimate_factor = pn.widgets.FloatInput(
            label="Limit-to-ultimate factor", value=1.5, start=1.0, step=0.05
        )
        self.loads_modulus = pn.widgets.FloatInput(
            label="Cap elastic modulus [GPa]", value=70.0, start=0.1, step=1.0
        )
        self.loads_cap_width = pn.widgets.FloatInput(
            label="Available cap width [m]", value=0.02, start=0.001, step=0.005
        )
        self.run_loads_button = pn.widgets.Button(
            label="Run loads and spar sizing", color="primary", icon="player-play"
        )
        self.loads_download = pn.widgets.FileDownload(
            label="Download span-load CSV", icon="download",
            callback=self._download_loads_csv, disabled=True,
        )
        self.loads_metrics = pn.pane.HTML()
        self.loads_plots = pn.pane.Matplotlib(height=680, tight=True, format="svg")
        self.loads_warnings = pn.pane.Alert(alert_type="warning", visible=False)

        self.battery_select = pn.widgets.Select(label="Battery", options=sorted(catalog.BATTERIES))
        self.state_of_charge = pn.widgets.FloatInput(label="Battery state of charge", value=0.9, start=0.0, end=1.0, step=0.05)
        self.battery_x = pn.widgets.FloatInput(label="Battery x [m]", value=0.02, step=0.01)
        self.battery_y = pn.widgets.FloatInput(label="Battery y [m]", value=0.0, step=0.01)
        self.battery_z = pn.widgets.FloatInput(label="Battery z [m]", value=0.0, step=0.01)
        self.include_propulsion_masses = pn.widgets.Checkbox(
            label="Include battery and propulsor hardware in mass properties", value=True
        )
        self.propulsor_table = _table(
            pd.DataFrame(), height=150,
            editors={
                "motor": {"type": "list", "values": sorted(catalog.MOTORS)},
                "propeller": {"type": "list", "values": sorted(catalog.PROPELLERS)},
                "esc": {"type": "list", "values": sorted(catalog.ESCS)},
                **{name: {"type": "number"} for name in (
                    "throttle", "x", "y", "z", "pitch_deg", "yaw_deg",
                )},
            }, configuration={"layout": "fitDataTable"},
        )
        self.add_propulsor_button = pn.widgets.Button(label="Add propulsor", icon="plus")
        self.delete_propulsor_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.motor_table = _table(pd.DataFrame(), height=205, configuration={"layout": "fitDataTable"})
        self.battery_table = _table(pd.DataFrame(), height=175, configuration={"layout": "fitDataTable"})
        self.esc_table = _table(pd.DataFrame(), height=145, configuration={"layout": "fitDataTable"})
        self.propeller_table = _table(pd.DataFrame(), height=220, configuration={"layout": "fitDataTable"})
        self.propeller_data_select = pn.widgets.Select(label="Propeller coefficient dataset")
        self.propeller_data_table = _table(
            pd.DataFrame(columns=["rpm", "J", "CT", "CP"]), height=260,
            editors={name: {"type": "number"} for name in ("rpm", "J", "CT", "CP")},
        )
        self.propeller_data_upload = pn.widgets.FileInput(
            label="Import coefficient CSV", accept=".csv,.txt"
        )
        self.add_motor_button = pn.widgets.Button(label="Add motor", icon="plus")
        self.delete_motor_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.add_battery_button = pn.widgets.Button(label="Add battery", icon="plus")
        self.delete_battery_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.add_esc_button = pn.widgets.Button(label="Add ESC", icon="plus")
        self.delete_esc_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.add_propeller_button = pn.widgets.Button(label="Add propeller", icon="plus")
        self.delete_propeller_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.add_propeller_point_button = pn.widgets.Button(label="Add coefficient point", icon="plus")
        self.delete_propeller_point_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.run_propulsion_button = pn.widgets.Button(label="Run propulsion analysis", color="primary", icon="player-play")
        self.propulsion_speed_min = pn.widgets.FloatInput(
            label="Sweep minimum speed [m/s] (0 = automatic)", value=0.0, start=0.0, step=1.0
        )
        self.propulsion_speed_max = pn.widgets.FloatInput(
            label="Sweep maximum speed [m/s] (0 = automatic)", value=0.0, start=0.0, step=1.0
        )
        self.propulsion_speed_points = pn.widgets.IntInput(
            label="Sweep points", value=19, start=5, end=201
        )
        self.propulsion_download = pn.widgets.FileDownload(
            label="Download speed sweep CSV", icon="download",
            callback=self._download_propulsion_csv, disabled=True,
        )
        self.propulsion_metrics = pn.pane.HTML()
        self.propulsor_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=150)
        self.propulsion_plot = pn.pane.Matplotlib(height=560, tight=True, format="svg")
        self.propulsion_warnings = pn.pane.Alert(alert_type="warning", visible=False)

        self.run_dynamics_button = pn.widgets.Button(label="Run dynamic stability", color="primary", icon="player-play")
        self.mode_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=280)
        self.derivative_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=300)
        self.dynamics_plot = pn.pane.Matplotlib(height=450, tight=True, format="svg")
        self.dynamics_warnings = pn.pane.Alert(alert_type="warning", visible=False)
        self.python_guide = pn.pane.Markdown(PYTHON_GUIDE)
        self.python_output = pn.pane.Markdown()

        self._set_widget_descriptions()
        self._wire_callbacks()
        self._load_project(self.project)
        self._loads_mode_changed()
        self.run_airfoil()

    # -- wiring and state -------------------------------------------------

    def _set_widget_descriptions(self):
        """Attach concise hover help to the controls students meet most often."""
        descriptions = {
            "project_name": "Aircraft/project name stored inside the project file.",
            "project_notes": "Free-form assumptions, provenance, and design notes saved with the project.",
            "project_upload": "Open a .json or .flightlab project from this computer.",
            "project_filename": "Name used for the downloaded project file; your browser chooses the download location.",
            "surface_select": "Choose the lifting surface edited by the controls and station table below.",
            "surface_name": "Unique name used to attach masses and identify analysis results.",
            "reference_mode": "Choose how coefficient reference area, span, and chord are obtained.",
            "reference_surface": "Surface that supplies Sref, bref, and cref.",
            "reference_surfaces": "Surfaces whose areas are summed for the aircraft coefficient reference.",
            "reference_area": "Manual coefficient reference area Sref [m²].",
            "reference_span": "Manual coefficient reference span bref [m].",
            "reference_chord": "Manual coefficient reference chord cref [m].",
            "surface_orientation": "Horizontal surfaces enter longitudinal analysis; vertical surfaces enter directional models.",
            "surface_purpose": "Descriptive role; it does not change the solver.",
            "surface_trim_control": "Geometry the trim solver may deflect to balance pitching moment.",
            "surface_symmetric": "Reflect this stored half-surface across the aircraft centerline.",
            "surface_control_hinge": "Chord fraction measured aft from the leading edge.",
            "surface_control_min": "Minimum incidence/elevator deflection available to the trim solver [deg].",
            "surface_control_max": "Maximum incidence/elevator deflection available to the trim solver [deg].",
            "airfoil_select": "Airfoil used by the standalone section analysis.",
            "airfoil_upload": "Import a Selig-style airfoil coordinate file into this project.",
            "naca_code": "Four digits such as 0009 or 2412; coordinates are generated analytically.",
            "airfoil_re": "Section Reynolds number for this standalone airfoil sweep.",
            "airfoil_transition": "Forced transition x/c; 1.0 applies no artificial trip and represents natural transition.",
            "airfoil_alpha_min": "Lowest section angle of attack in the airfoil sweep [deg].",
            "airfoil_alpha_max": "Highest section angle of attack in the airfoil sweep [deg].",
            "analysis_case": "Flight condition used for atmosphere, required lift, drag, and trim.",
            "analysis_ns": "Spanwise panels assigned to the reference-span surface; other horizontal surfaces scale with span, with at least eight.",
            "analysis_nc": "Chordwise panels used on every lifting surface in the VLM solve.",
            "loads_case": "Flight case supplying atmosphere; direct mode also uses its speed when design speed is zero.",
            "loads_surface": "Horizontal lifting surface whose span load and spar are evaluated.",
            "loads_mode": "Use a direct speed/load-factor case for RC sizing, or opt into a full maneuver/gust envelope.",
            "loads_ns": "Spanwise panels used for the structural surface's aerodynamic load distribution.",
            "loads_cl_max": "Positive stall boundary used to construct the maneuver envelope.",
            "loads_cl_min": "Negative stall boundary used to construct the maneuver envelope.",
            "loads_n_pos": "Positive maneuver-envelope limit load factor.",
            "loads_n_neg": "Negative maneuver-envelope limit load factor.",
            "loads_v_max": "Envelope maximum speed; zero uses 1.4 times the selected case speed.",
            "loads_gust": "Sharp-edged vertical gust increment used for the gust lines [m/s].",
            "loads_design_speed": "Zero uses the selected flight-case speed in direct mode or the positive corner speed in envelope mode.",
            "loads_factor": "Positive limit load factor used to size the selected lifting surface and spar caps.",
            "loads_spar_height": "Vertical separation between the tension and compression cap centroids [m].",
            "loads_allowable": "Allowable normal stress used for each spar cap [MPa].",
            "loads_ultimate_factor": "Multiplier from limit bending moment to ultimate sizing moment.",
            "loads_modulus": "Elastic modulus used in the cap-only deflection estimate [GPa].",
            "loads_cap_width": "Available width used to convert required cap area into thickness [m].",
            "battery_select": "Shared battery definition feeding every propulsor row.",
            "include_propulsion_masses": "Add catalog battery, motor, ESC, and propeller masses at their entered locations.",
            "state_of_charge": "Fraction of usable battery charge remaining; affects open-circuit voltage.",
            "battery_x": "Battery x location used for aircraft CG and inertia [m].",
            "battery_y": "Battery y location used for aircraft CG and inertia [m].",
            "battery_z": "Battery z location used for aircraft CG and inertia [m].",
            "propulsion_speed_min": "Lowest speed in the thrust/drag sweep; zero with a zero maximum selects an automatic range.",
            "propulsion_speed_max": "Highest speed in the thrust/drag sweep; zero with a zero minimum selects an automatic range.",
            "propulsion_speed_points": "Number of points in the propulsion/airframe speed sweep.",
            "propeller_data_select": "Propeller whose measured coefficient rows are shown below.",
            "propeller_data_upload": "Import measured rpm, J, CT, and CP coefficient rows.",
        }
        for name, description in descriptions.items():
            widget = getattr(self, name)
            if "description" in widget.param:
                widget.description = description

    def _wire_callbacks(self):
        self.blank_button.on_click(lambda _: self._load_project(blank_project()))
        self.example_button.on_click(lambda _: self._load_project(example_project()))
        self.project_upload.param.watch(self._open_project, "value")
        self.project_name.param.watch(self._project_metadata_changed, "value")
        self.project_notes.param.watch(self._project_metadata_changed, "value")
        self.project_filename.param.watch(self._project_filename_changed, "value")

        self.surface_select.param.watch(self._surface_selected, "value")
        self.surface_name.param.watch(self._surface_metadata_changed, "value")
        self.surface_orientation.param.watch(self._surface_metadata_changed, "value")
        self.surface_purpose.param.watch(self._surface_metadata_changed, "value")
        self.surface_trim_control.param.watch(self._surface_metadata_changed, "value")
        self.surface_control_hinge.param.watch(self._surface_metadata_changed, "value")
        self.surface_control_min.param.watch(self._surface_metadata_changed, "value")
        self.surface_control_max.param.watch(self._surface_metadata_changed, "value")
        self.surface_symmetric.param.watch(self._surface_metadata_changed, "value")
        for widget in (
            self.reference_mode, self.reference_surface, self.reference_surfaces, self.reference_area,
            self.reference_span, self.reference_chord,
        ):
            widget.param.watch(self._reference_changed, "value")
        self.station_table.param.watch(self._stations_changed, "value")
        self.add_station_button.on_click(self._add_station)
        self.delete_station_button.on_click(self._delete_station)
        self.add_surface_button.on_click(self._add_surface)
        self.delete_surface_button.on_click(self._delete_surface)

        self.body_table.param.watch(self._bodies_changed, "value")
        self.mass_table.param.watch(self._masses_changed, "value")
        self.case_table.param.watch(self._cases_changed, "value")
        self.add_body_button.on_click(self._add_body)
        self.delete_body_button.on_click(self._delete_body)
        self.add_mass_button.on_click(self._add_mass)
        self.delete_mass_button.on_click(self._delete_mass)
        self.add_case_button.on_click(self._add_case)
        self.delete_case_button.on_click(self._delete_case)

        self.airfoil_upload.param.watch(self._airfoil_uploaded, "value")
        self.add_naca_button.on_click(self._add_naca_section)
        self.run_airfoil_button.on_click(self.run_airfoil)
        self.run_analysis_button.on_click(self.run_integrated_analysis)
        self.run_loads_button.on_click(self.run_loads_analysis)
        self.run_propulsion_button.on_click(self.run_propulsion_analysis)
        self.run_dynamics_button.on_click(self.run_dynamic_stability)
        self.motor_table.param.watch(self._motors_changed, "value")
        self.battery_table.param.watch(self._batteries_changed, "value")
        self.esc_table.param.watch(self._escs_changed, "value")
        self.propeller_table.param.watch(self._propellers_changed, "value")
        self.propeller_data_select.param.watch(self._propeller_data_selected, "value")
        self.propeller_data_table.param.watch(self._propeller_points_changed, "value")
        self.propeller_data_upload.param.watch(self._propeller_data_uploaded, "value")
        self.add_propeller_point_button.on_click(self._add_propeller_point)
        self.delete_propeller_point_button.on_click(self._delete_propeller_point)
        self.add_motor_button.on_click(lambda _: self._append_table_row(
            self.motor_table, asdict(catalog.Motor("motor_new", "New motor", 1000, 0.1, 1.0, 20, 0.05))
        ))
        self.delete_motor_button.on_click(lambda _: self._delete_table_row(self.motor_table))
        self.add_battery_button.on_click(lambda _: self._append_table_row(
            self.battery_table, asdict(catalog.Battery("battery_new", "New battery", 3, 1, 2.0, 0.15, 30, 0.01))
        ))
        self.delete_battery_button.on_click(lambda _: self._delete_table_row(self.battery_table))
        self.add_esc_button.on_click(lambda _: self._append_table_row(
            self.esc_table, asdict(catalog.ESC("esc_new", "New ESC", 30, 0.03))
        ))
        self.delete_esc_button.on_click(lambda _: self._delete_table_row(self.esc_table))
        self.add_propeller_button.on_click(lambda _: self._append_table_row(
            self.propeller_table,
            {key: value for key, value in asdict(
                PropellerDefinition("prop_new", "New propeller", 0.02, diameter=0.254, pitch=0.127)
            ).items() if key != "points"},
        ))
        self.delete_propeller_button.on_click(lambda _: self._delete_table_row(self.propeller_table))
        self.propulsor_table.param.watch(self._propulsors_changed, "value")
        self.add_propulsor_button.on_click(lambda _: self._append_table_row(
            self.propulsor_table,
            asdict(PropulsorSetup(name=f"Propulsor {len(self.project.propulsion.propulsors) + 1}")),
        ))
        self.delete_propulsor_button.on_click(lambda _: self._delete_table_row(self.propulsor_table))
        for widget in (
            self.battery_select, self.state_of_charge,
            self.battery_x, self.battery_y, self.battery_z,
            self.include_propulsion_masses,
        ):
            widget.param.watch(self._propulsion_changed, "value")
        for widget in (
            self.airfoil_select, self.airfoil_re, self.airfoil_alpha_min,
            self.airfoil_alpha_max, self.airfoil_transition,
        ):
            widget.param.watch(self._airfoil_inputs_changed, "value")
        self.analysis_case.param.watch(self._analysis_case_changed, "value")
        self.analysis_ns.param.watch(self._analysis_resolution_changed, "value")
        self.analysis_nc.param.watch(self._analysis_resolution_changed, "value")
        self.show_mass_components.param.watch(lambda _: self._refresh_geometry(), "value")
        self.show_panel_mesh.param.watch(lambda _: self._refresh_geometry(), "value")
        self.loads_mode.param.watch(self._loads_mode_changed, "value")
        for widget in (
            self.loads_case,
            self.loads_cl_max, self.loads_cl_min,
            self.loads_n_pos, self.loads_n_neg, self.loads_v_max, self.loads_gust,
            self.loads_factor, self.loads_design_speed, self.loads_ns,
        ):
            widget.param.watch(self._analysis_inputs_changed, "value")
        for widget in (
            self.loads_surface, self.loads_spar_height, self.loads_allowable,
            self.loads_ultimate_factor, self.loads_modulus, self.loads_cap_width,
        ):
            widget.param.watch(self._structure_changed, "value")
        for widget in (
            self.propulsion_speed_min, self.propulsion_speed_max,
            self.propulsion_speed_points,
        ):
            widget.param.watch(self._analysis_inputs_changed, "value")

    def _load_project(self, project: AircraftProject):
        self._updating = True
        self.project = project
        self.project_name.value = project.name
        self.project_notes.value = project.notes
        self.project_filename.value = _safe_filename(project.name)
        names = [surface.name for surface in project.surfaces]
        self.surface_select.options = names
        self.surface_select.value = names[0] if names else None
        self.reference_surface.options = names
        self.reference_surfaces.options = names
        self.reference_mode.value = project.reference.mode
        self.reference_surface.value = project.reference.surface if project.reference.surface in names else (names[0] if names else None)
        self.reference_surfaces.value = [name for name in project.reference.surfaces if name in names]
        self.reference_area.value = project.reference.area or 1.0
        self.reference_span.value = project.reference.span or 1.0
        self.reference_chord.value = project.reference.chord or 0.2
        self.body_table.value = pd.DataFrame([asdict(body) for body in project.bodies])
        self.mass_table.value = pd.DataFrame([
            {**asdict(item), "distributed": item.distributed or "point"}
            for item in project.masses
        ])
        self.case_table.value = pd.DataFrame([self._case_record(case) for case in project.cases])
        case_names = [case.name for case in project.cases]
        self.analysis_case.options = case_names
        self.analysis_case.value = case_names[0] if case_names else None
        self.loads_case.options = case_names
        self.loads_case.value = case_names[0] if case_names else None
        horizontal_names = [surface.name for surface in project.horizontal_surfaces]
        self.loads_surface.options = horizontal_names
        try:
            default_structural_surface = project.primary_horizontal_surface.name
        except ValueError:
            default_structural_surface = horizontal_names[0] if horizontal_names else None
        structural_surface = (
            project.structure.surface
            if project.structure.surface in horizontal_names
            else default_structural_surface
        )
        project.structure.surface = structural_surface or ""
        self.loads_surface.value = structural_surface
        self.loads_spar_height.value = project.structure.spar_height
        self.loads_allowable.value = project.structure.allowable_stress / 1e6
        self.loads_ultimate_factor.value = project.structure.ultimate_factor
        self.loads_modulus.value = project.structure.elastic_modulus / 1e9
        self.loads_cap_width.value = project.structure.cap_width
        self.project_download.filename = self.project_filename.value
        self._refresh_airfoil_options()
        self._load_propulsion_library_tables()
        setup = project.propulsion or PropulsionSetup()
        self.battery_select.value = setup.battery
        self.state_of_charge.value = setup.state_of_charge
        self.battery_x.value = setup.battery_x
        self.battery_y.value = setup.battery_y
        self.battery_z.value = setup.battery_z
        self.include_propulsion_masses.value = setup.include_component_masses
        self.propulsor_table.value = pd.DataFrame([asdict(item) for item in setup.propulsors])
        self._updating = False
        self._update_reference_visibility()
        self._show_selected_surface()
        self._refresh_all("Project loaded.")

    def _refresh_all(self, message="Project updated."):
        deleted_cached_cases = self._invalidate_export_results()
        self._refresh_validation()
        self._refresh_geometry()
        self._refresh_attachment_options()
        self._refresh_propulsion_options()
        self._refresh_component_summaries()
        self._refresh_generated_python()
        if not self.project_filename.value.strip():
            self.project_filename.value = _safe_filename(self.project.name)
        self._project_filename_changed(None)
        if deleted_cached_cases:
            message += " All cached flight-case results were deleted."
        self.status.object = message
        self.status.alert_type = "light"

    def _analysis_inputs_changed(self, _):
        if self._updating:
            return
        self._last_loads_result = None
        self._last_propulsion_result = None
        self.loads_download.disabled = True
        self.propulsion_download.disabled = True
        self._refresh_generated_python()

    def _structure_changed(self, _):
        """Persist physical spar/material choices with the aircraft project."""
        if self._updating:
            return
        setup = self.project.structure
        setup.surface = self.loads_surface.value or ""
        setup.spar_height = float(self.loads_spar_height.value)
        setup.allowable_stress = float(self.loads_allowable.value) * 1e6
        setup.ultimate_factor = float(self.loads_ultimate_factor.value)
        setup.elastic_modulus = float(self.loads_modulus.value) * 1e9
        setup.cap_width = float(self.loads_cap_width.value)
        self._last_loads_result = None
        self.loads_download.disabled = True
        self._refresh_generated_python()

    def _analysis_resolution_changed(self, _):
        if self._updating:
            return
        self._invalidate_export_results()
        self._clear_analysis_display(
            "Panel counts changed; all cached flight-case results were deleted. "
            "Rerun each case you want to compare."
        )
        self._refresh_generated_python()

    def _analysis_case_changed(self, _):
        if self._updating:
            return
        self._last_propulsion_result = None
        self.propulsion_download.disabled = True
        cached = self._analysis_cache.get(self.analysis_case.value)
        if cached is None:
            self._last_result = None
            self._last_polar = None
            self._last_stall = None
            self.analysis_download.disabled = True
            self.analysis_span_download.disabled = True
            self._clear_analysis_display(
                f"No cached analysis for {self.analysis_case.value}; run this case once."
            )
        else:
            self._display_analysis_result(*cached, cached=True)
        self._refresh_generated_python()

    def _loads_mode_changed(self, _=None):
        envelope_mode = self.loads_mode.value == "Maneuver/gust V–n envelope"
        for widget in (
            self.loads_cl_max, self.loads_cl_min, self.loads_n_pos,
            self.loads_n_neg, self.loads_v_max, self.loads_gust,
        ):
            widget.visible = envelope_mode
        suffix = "positive corner" if envelope_mode else "flight-case speed"
        self.loads_design_speed.name = f"Structural design speed [m/s] (0 = {suffix})"
        if not self._updating:
            self._analysis_inputs_changed(None)

    def _airfoil_inputs_changed(self, _):
        if self._updating:
            return
        self._last_airfoil_result = None
        self._last_airfoil_alpha = None
        self._last_airfoil_context = None
        self.airfoil_download.disabled = True

    def _project_metadata_changed(self, _):
        if self._updating:
            return
        self.project.name = self.project_name.value
        self.project.notes = self.project_notes.value
        self._refresh_all()

    def _project_filename_changed(self, _):
        if self._updating:
            return
        filename = self.project_filename.value.strip()
        if not filename:
            filename = _safe_filename(self.project.name)
        if not filename.lower().endswith((".json", ".flightlab")):
            filename += ".flightlab.json"
        self.project_download.filename = filename

    def _download_project(self):
        return io.BytesIO((self.project.to_json() + "\n").encode("utf-8"))

    def _invalidate_export_results(self):
        """Prevent downloads from silently describing an earlier project state."""
        deleted_cached_cases = bool(self._analysis_cache)
        self._last_airfoil_result = None
        self._last_airfoil_alpha = None
        self._last_airfoil_context = None
        self._last_result = None
        self._last_polar = None
        self._last_stall = None
        self._analysis_cache.clear()
        self._last_loads_result = None
        self._last_propulsion_result = None
        self.airfoil_download.disabled = True
        self.analysis_download.disabled = True
        self.analysis_span_download.disabled = True
        self.loads_download.disabled = True
        self.propulsion_download.disabled = True
        return deleted_cached_cases

    def _download_airfoil_csv(self):
        """Export the most recent airfoil sweep with model diagnostics."""
        if self._last_airfoil_result is None or self._last_airfoil_alpha is None:
            raise ValueError("run the airfoil analysis before downloading its sweep")
        result = self._last_airfoil_result
        alpha = self._last_airfoil_alpha
        context = self._last_airfoil_context
        count = len(alpha)
        frame = pd.DataFrame({
            "airfoil": np.repeat(context["airfoil"], count),
            "reynolds_number": np.repeat(context["reynolds_number"], count),
            "forced_transition_x_c": np.repeat(context["transition"], count),
            "alpha_deg": alpha,
            "cl": result["cl"],
            "cd": result["cd"],
            "cm_about_c4": result["cm"],
            "upper_transition_x_c": result["top_xtr"],
            "lower_transition_x_c": result["bot_xtr"],
            "model_confidence": result["confidence"],
        })
        return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

    def _download_analysis_csv(self):
        """Export the most recent integrated-analysis polar with its case metadata."""
        if self._last_result is None or self._last_polar is None:
            raise ValueError("run the integrated analysis before downloading its polar")
        result = self._last_result
        polar = self._last_polar
        case = result.case
        count = len(polar.alpha)
        frame = pd.DataFrame({
            "project": np.repeat(result.project_name, count),
            "case": np.repeat(case.name, count),
            "reference_speed_m_s": np.repeat(case.speed, count),
            "altitude_m": np.repeat(case.altitude, count),
            "mass_kg": np.repeat(result.mass_properties.mass, count),
            "alpha_deg": polar.alpha,
            "CL": polar.CL,
            "CD": polar.CD,
            "CD_profile_body": polar.CD_profile,
            "CD_induced": polar.CD_i,
            "Cm_about_cg": polar.Cm,
            "L_over_D": polar.LD,
            "trim_control_deg": np.repeat(polar.trim_deflection, count),
        })
        return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

    def _download_spanwise_csv(self):
        """Export station-level results from the most recent integrated analysis."""
        if self._last_result is None:
            raise ValueError("run the integrated analysis before downloading spanwise results")
        result = self._last_result
        solution = result.trim.solution
        frames = []
        for surface_name in solution.surfaces:
            view = solution.surface(surface_name)
            surface = self.project.surface_named(surface_name)
            section_limit = surface_section_cl_max(
                self.project, surface, solution, result.case
            )
            stall_distribution = (
                self._last_stall["distributions"].get(surface_name)
                if self._last_stall is not None else None
            )
            stall_section_cl = (
                stall_distribution["cl"]
                if stall_distribution is not None else np.full_like(view.cl, np.nan)
            )
            count = len(view.y)
            frames.append(pd.DataFrame({
                "project": np.repeat(result.project_name, count),
                "case": np.repeat(result.case.name, count),
                "surface": np.repeat(surface_name, count),
                "mass_kg": np.repeat(result.mass_properties.mass, count),
                "alpha_deg": np.repeat(solution.alpha, count),
                "y_m": view.y,
                "strip_width_m": view.ds,
                "chord_m": view.chord,
                "section_cl": view.cl,
                "section_cl_max": section_limit,
                "section_cl_at_estimated_stall": stall_section_cl,
                "span_loading_c_cl_m": view.ccl,
                "section_cm": view.cm_section,
                "reynolds_number": view.Re,
            }))
        return io.BytesIO(pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8"))

    def _download_loads_csv(self):
        """Export the structural span-load and deflection distributions."""
        if self._last_loads_result is None:
            raise ValueError("run loads and spar sizing before downloading the span load")
        result = self._last_loads_result
        span = result["span_load"]
        deflection = result["deflection"]
        count = len(span.y)
        frame = pd.DataFrame({
            "project": np.repeat(self.project.name, count),
            "case": np.repeat(result["case"], count),
            "surface": np.repeat(result["surface"], count),
            "load_factor": np.repeat(result["load_factor"], count),
            "y_m": span.y,
            "net_aerodynamic_load_N_per_m": span.lift,
            "shear_N": span.shear,
            "bending_moment_N_m": span.moment,
            "deflection_m": deflection["deflection"],
        })
        return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

    def _download_propulsion_csv(self):
        """Export the most recent propulsion/airframe speed sweep."""
        if self._last_propulsion_result is None:
            raise ValueError("run the propulsion analysis before downloading its speed sweep")
        result = self._last_propulsion_result
        case = self.project.case(self.analysis_case.value)
        count = len(result.speed)
        data = {
            "project": np.repeat(self.project.name, count),
            "case": np.repeat(case.name, count),
            "altitude_m": np.repeat(case.altitude, count),
            "load_factor": np.repeat(case.load_factor, count),
            "speed_true_m_s": result.speed,
            "thrust_available_N": result.thrust_available,
            "drag_required_N": result.drag_required,
            "battery_current_A": result.current,
            "power_electrical_W": result.power_electrical,
            "power_shaft_W": result.power_shaft,
            "power_useful_W": result.power_useful,
            "efficiency_motor": result.efficiency_motor,
            "efficiency_propeller": result.efficiency_propeller,
            "efficiency_esc": result.efficiency_esc,
            "efficiency_total": result.efficiency_total,
            "outside_propeller_data": result.extrapolated,
        }
        for index, propulsor in enumerate(self.project.propulsion.propulsors):
            data[f"rpm_{index + 1}_{propulsor.name}"] = result.rpm[:, index]
        frame = pd.DataFrame(data)
        return io.BytesIO(frame.to_csv(index=False).encode("utf-8"))

    def _open_project(self, event):
        if not event.new:
            return
        try:
            self._load_project(AircraftProject.from_json(event.new.decode("utf-8")))
        except Exception as exc:
            self._error(f"Could not open project: {exc}")

    # -- surfaces ---------------------------------------------------------

    def _current_surface(self):
        return next(
            (surface for surface in self.project.surfaces if surface.name == self.surface_select.value),
            None,
        )

    def _surface_selected(self, _):
        if not self._updating:
            self._show_selected_surface()

    def _show_selected_surface(self):
        surface = self._current_surface()
        if surface is None:
            return
        self._updating = True
        self.surface_name.value = surface.name
        self.surface_orientation.value = surface.orientation
        self.surface_purpose.value = surface.purpose
        self.surface_trim_control.value = surface.trim_control
        self.surface_control_hinge.value = surface.control_hinge_fraction
        self.surface_control_min.value = surface.control_min_deg
        self.surface_control_max.value = surface.control_max_deg
        self.surface_control_hinge.visible = surface.trim_control == "elevator"
        limits_visible = surface.trim_control != "fixed"
        self.surface_control_min.visible = limits_visible
        self.surface_control_max.visible = limits_visible
        self.surface_symmetric.value = surface.symmetric
        self.station_table.value = pd.DataFrame([asdict(station) for station in surface.stations])
        self._updating = False
        self._refresh_surface_geometry()

    def _surface_metadata_changed(self, _):
        if self._updating:
            return
        surface = self._current_surface()
        if surface is None:
            return
        old_name = surface.name
        surface.name = self.surface_name.value.strip() or old_name
        surface.orientation = self.surface_orientation.value
        surface.purpose = self.surface_purpose.value
        surface.trim_control = self.surface_trim_control.value
        surface.control_hinge_fraction = float(self.surface_control_hinge.value)
        surface.control_min_deg = float(self.surface_control_min.value)
        surface.control_max_deg = float(self.surface_control_max.value)
        surface.symmetric = self.surface_symmetric.value
        self._updating = True
        names = [item.name for item in self.project.surfaces]
        self.surface_select.options = names
        self.surface_select.value = surface.name
        self.reference_surface.options = names
        self.reference_surfaces.options = names
        if self.project.reference.surface == old_name:
            self.project.reference.surface = surface.name
            self.reference_surface.value = surface.name
        if old_name in self.project.reference.surfaces:
            self.project.reference.surfaces = [
                surface.name if name == old_name else name
                for name in self.project.reference.surfaces
            ]
            self.reference_surfaces.value = self.project.reference.surfaces
        self._updating = False
        self.surface_control_hinge.visible = surface.trim_control == "elevator"
        limits_visible = surface.trim_control != "fixed"
        self.surface_control_min.visible = limits_visible
        self.surface_control_max.visible = limits_visible
        self._refresh_all(f"Updated {surface.name}.")

    def _reference_changed(self, _):
        if self._updating:
            return
        self.project.reference = ReferenceGeometry(
            mode=self.reference_mode.value,
            surface=self.reference_surface.value or "",
            surfaces=list(self.reference_surfaces.value),
            area=float(self.reference_area.value),
            span=float(self.reference_span.value),
            chord=float(self.reference_chord.value),
        )
        self._update_reference_visibility()
        self._refresh_all("Coefficient reference updated.")

    def _update_reference_visibility(self):
        mode = self.reference_mode.value
        self.reference_surface.visible = mode == "surface"
        self.reference_surfaces.visible = mode == "selected_surfaces"
        for widget in (self.reference_area, self.reference_span, self.reference_chord):
            widget.visible = mode == "manual"

    def _stations_changed(self, event):
        if self._updating or event.new is None:
            return
        surface = self._current_surface()
        if surface is None:
            return
        try:
            schema = [
                ("x_le", float, False), ("y", float, False), ("z", float, False),
                ("chord", float, False), ("twist_deg", float, False),
                ("airfoil", str, False),
            ]
            surface.stations = [
                SurfaceStation(**row)
                for row in _coerce_records(event.new, schema, f"{surface.name} station")
            ]
            self._refresh_all(f"Updated {surface.name} stations.")
        except Exception as exc:
            self._error(f"Station edit is incomplete: {exc}")

    def _add_station(self, _):
        surface = self._current_surface()
        if surface is None:
            return
        last = surface.stations[-1] if surface.stations else SurfaceStation(0, 0, 0, 0.2)
        if surface.orientation == "vertical":
            new = SurfaceStation(last.x_le + 0.03, last.y, last.z + 0.15, last.chord * 0.8, last.twist_deg, last.airfoil)
        else:
            new = SurfaceStation(last.x_le + 0.03, last.y + 0.20, last.z, last.chord * 0.8, last.twist_deg, last.airfoil)
        surface.stations.append(new)
        self._show_selected_surface()
        self._refresh_all(f"Added a station to {surface.name}.")

    def _delete_station(self, _):
        surface = self._current_surface()
        selected = sorted(set(self.station_table.selection))
        if surface is None or not selected:
            self._error("Select at least one station row to delete.")
            return
        if len(surface.stations) - len(selected) < 2:
            self._error("A lifting surface needs at least two stations.")
            return
        selected_set = set(selected)
        surface.stations = [
            station for index, station in enumerate(surface.stations)
            if index not in selected_set
        ]
        self.station_table.selection = []
        self._show_selected_surface()
        count = len(selected)
        noun = "station" if count == 1 else "stations"
        self._refresh_all(f"Deleted {count} {noun} from {surface.name}.")

    def _add_surface(self, _):
        index = len(self.project.surfaces) + 1
        surface = LiftingSurface(
            f"Surface {index}", "horizontal", "other", "fixed", True,
            [SurfaceStation(0.5, 0, 0, 0.2), SurfaceStation(0.55, 0.3, 0, 0.12)],
        )
        self.project.surfaces.append(surface)
        self._updating = True
        self.surface_select.options = [item.name for item in self.project.surfaces]
        self.reference_surface.options = [item.name for item in self.project.surfaces]
        self.reference_surfaces.options = [item.name for item in self.project.surfaces]
        self.surface_select.value = surface.name
        self._updating = False
        self._show_selected_surface()
        self._refresh_all("Added a lifting surface.")

    def _delete_surface(self, _):
        surface = self._current_surface()
        if surface is None:
            return
        self.project.surfaces.remove(surface)
        names = [item.name for item in self.project.surfaces]
        self._updating = True
        self.surface_select.options = names
        self.surface_select.value = names[0] if names else None
        self.reference_surface.options = names
        self.reference_surfaces.options = names
        if self.project.reference.surface == surface.name:
            self.project.reference.surface = names[0] if names else ""
            self.reference_surface.value = self.project.reference.surface or None
        if surface.name in self.project.reference.surfaces:
            self.project.reference.surfaces.remove(surface.name)
            self.reference_surfaces.value = self.project.reference.surfaces
        self._updating = False
        self._show_selected_surface()
        self._refresh_all(f"Deleted {surface.name}.")

    # -- row-table editors ------------------------------------------------

    def _bodies_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("name", str, False), ("length", float, False),
                ("width", float, True), ("height", float, True),
                ("diameter", float, True), ("x_nose", float, True),
                ("y", float, False), ("z", float, False),
                ("count", int, False), ("drag_model", str, False),
                ("cone_fraction", float, False),
            ]
            self.project.bodies = [BodyDefinition(**row) for row in _coerce_records(event.new, schema, "Body")]
            self._refresh_all("Body geometry updated.")
        except Exception as exc:
            self._error(f"Body edit is incomplete: {exc}")

    def _masses_changed(self, event):
        if self._updating:
            return
        try:
            self._apply_mass_frame(event.new)
        except Exception as exc:
            self._error(f"Mass edit is incomplete: {exc}")

    def _apply_mass_frame(self, frame):
        schema = [
            ("name", str, False), ("mass", float, True), ("x", float, False),
            ("y", float, False), ("z", float, False),
            ("distributed", str, True), ("span", float, True),
            ("attached_to", str, True), ("density", float, True),
            ("skin_thickness", float, True),
        ]
        rows = _coerce_records(frame, schema, "Mass")
        for row in rows:
            row["distributed"] = row["distributed"] or "point"
            row["attached_to"] = row["attached_to"] or ""
        self.project.masses = [MassItem(**row) for row in rows]
        self._refresh_all("Mass model updated.")

    def _cases_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("name", str, False), ("speed", float, False), ("altitude", float, False),
                ("load_factor", float, False), ("alpha_deg", float, False),
                ("interference", float, False), ("protuberance", float, False),
                ("f_other", float, False), ("cooling", float, False),
                ("transition", str, False),
                ("n_crit", float, False), ("xtr_upper", float, False),
                ("xtr_lower", float, False),
            ]
            rows = _coerce_records(event.new, schema, "Flight case")
            cases = []
            for row in rows:
                transition = row.pop("transition").lower()
                if transition not in {"natural", "forced"}:
                    raise ValueError("transition must be 'natural' or 'forced'")
                if transition == "natural":
                    row["xtr_upper"] = row["xtr_lower"] = 1.0
                cases.append(FlightCase(**row))
            self.project.cases = cases
            names = [case.name for case in self.project.cases]
            self._updating = True
            self.case_table.value = pd.DataFrame([
                self._case_record(case) for case in self.project.cases
            ])
            self.analysis_case.options = names
            if self.analysis_case.value not in names:
                self.analysis_case.value = names[0] if names else None
            self.loads_case.options = names
            if self.loads_case.value not in names:
                self.loads_case.value = names[0] if names else None
            self._updating = False
            self._refresh_all("Flight cases updated.")
        except Exception as exc:
            self._error(f"Flight-case edit is incomplete: {exc}")

    def _append_table_row(self, table, row):
        frame = table.value.copy()
        table.value = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    def _delete_table_row(self, table):
        if not table.selection:
            self._error("Select at least one row to delete.")
            return
        selected = sorted(set(table.selection))
        frame = table.value.drop(table.value.index[selected]).reset_index(drop=True)
        if table is self.mass_table:
            # Apply the new frame explicitly. Some browser/Tabulator versions
            # coalesce the shorter DataFrame assignment and its value watcher,
            # leaving derived tables and plots on the previous component list.
            self._updating = True
            table.value = frame
            table.selection = []
            self._updating = False
            self._apply_mass_frame(frame)
        else:
            table.value = frame
            table.selection = []

    def _add_body(self, _):
        self._append_table_row(self.body_table, asdict(BodyDefinition("new body", 0.5, diameter=0.08)))

    def _delete_body(self, _):
        self._delete_table_row(self.body_table)

    def _add_mass(self, _):
        self._append_table_row(
            self.mass_table,
            {**asdict(MassItem("new component", 0.05, 0.25)), "distributed": "point"},
        )

    def _delete_mass(self, _):
        self._delete_table_row(self.mass_table)

    def _add_case(self, _):
        self._append_table_row(self.case_table, self._case_record(FlightCase("New case", 12.0)))

    def _delete_case(self, _):
        self._delete_table_row(self.case_table)

    # -- airfoils ---------------------------------------------------------

    @staticmethod
    def _case_record(case):
        row = asdict(case)
        row["transition"] = (
            "natural"
            if math.isclose(case.xtr_upper, 1.0) and math.isclose(case.xtr_lower, 1.0)
            else "forced"
        )
        ordered = [
            "name", "speed", "altitude", "load_factor", "alpha_deg",
            "interference", "protuberance", "transition", "n_crit",
            "xtr_upper", "xtr_lower", "f_other", "cooling",
        ]
        return {name: row[name] for name in ordered}

    def _refresh_airfoil_options(self):
        station_airfoils = {
            station.airfoil for surface in self.project.surfaces for station in surface.stations
        }
        names = sorted(
            set(foil.available()) | set(self.project.airfoils) | station_airfoils
            | {"naca0009", "naca0012", "naca2412"}
        )
        current = self.airfoil_select.value
        self.airfoil_select.options = names
        self.airfoil_select.value = current if current in names else "naca2412"
        editors = dict(self.station_table.editors)
        editors["airfoil"] = {"type": "list", "values": names}
        self.station_table.editors = editors

    def _add_naca_section(self, _):
        code = re.sub(r"\s+", "", self.naca_code.value.lower())
        code = code.removeprefix("naca")
        if not re.fullmatch(r"\d{4}", code):
            self._error("Enter exactly four digits for a NACA four-digit section, such as 0009.")
            return
        key = f"naca{code}"
        try:
            foil.load(key)
        except Exception as exc:
            self._error(f"Could not generate {key.upper()}: {exc}")
            return
        names = sorted(set(self.airfoil_select.options) | {key})
        self.airfoil_select.options = names
        self.airfoil_select.value = key
        editors = dict(self.station_table.editors)
        editors["airfoil"] = {"type": "list", "values": names}
        self.station_table.editors = editors
        self.status.object = f"Added analytic section {key.upper()}; choose it in the station table."
        self.status.alert_type = "success"

    # -- editable propulsion component library ---------------------------

    def _load_propulsion_library_tables(self):
        self.motor_table.value = pd.DataFrame([asdict(item) for item in self.project.motors.values()])
        self.battery_table.value = pd.DataFrame([asdict(item) for item in self.project.batteries.values()])
        self.esc_table.value = pd.DataFrame([asdict(item) for item in self.project.escs.values()])
        self.propeller_table.value = pd.DataFrame([
            {key: value for key, value in asdict(item).items() if key != "points"}
            for item in self.project.propellers.values()
        ])
        keys = list(self.project.propellers)
        self.propeller_data_select.options = keys
        self.propeller_data_select.value = keys[0] if keys else None
        self._show_propeller_points()

    def _refresh_propulsion_options(self):
        setup = self.project.propulsion
        self._updating = True
        battery_keys = list(self.project.batteries)
        desired = setup.battery if setup else None
        self.battery_select.options = battery_keys
        self.battery_select.value = desired if desired in self.project.batteries else (
            battery_keys[0] if battery_keys else None
        )
        if setup is not None and self.battery_select.value is not None:
            setup.battery = self.battery_select.value
        editors = dict(self.propulsor_table.editors)
        editors["motor"] = {"type": "list", "values": list(self.project.motors)}
        editors["propeller"] = {"type": "list", "values": list(self.project.propellers)}
        editors["esc"] = {"type": "list", "values": list(self.project.escs)}
        self.propulsor_table.editors = editors
        self._updating = False

    @staticmethod
    def _optional_text(row, name):
        return row[name] or ""

    def _motors_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("key", str, False), ("name", str, False), ("Kv_rpm", float, False),
                ("resistance", float, False), ("current_no_load", float, False),
                ("current_max", float, False), ("mass", float, False),
                ("no_load_voltage", float, False), ("cells_min", int, False),
                ("cells_max", int, False), ("provisional", bool, False), ("notes", str, True),
            ]
            rows = _coerce_records(event.new, schema, "Motor")
            values = [catalog.Motor(**{**row, "notes": self._optional_text(row, "notes")}) for row in rows]
            self.project.motors = {item.key: item for item in values}
            if len(self.project.motors) != len(values):
                raise ValueError("motor keys must be unique")
            self._refresh_propulsion_options()
            self._refresh_all("Motor library updated.")
        except Exception as exc:
            self._error(f"Motor edit is incomplete: {exc}")

    def _batteries_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("key", str, False), ("name", str, False), ("cells_series", int, False),
                ("cells_parallel", int, False), ("capacity_ah", float, False),
                ("mass", float, False), ("c_rating", float, False),
                ("cell_resistance", float, False), ("cell_voltage_nominal", float, False),
                ("cell_voltage_full", float, False), ("cell_voltage_empty", float, False),
                ("provisional", bool, False), ("notes", str, True),
            ]
            rows = _coerce_records(event.new, schema, "Battery")
            values = [catalog.Battery(**{**row, "notes": self._optional_text(row, "notes")}) for row in rows]
            self.project.batteries = {item.key: item for item in values}
            if len(self.project.batteries) != len(values):
                raise ValueError("battery keys must be unique")
            self._refresh_propulsion_options()
            self._refresh_all("Battery library updated.")
        except Exception as exc:
            self._error(f"Battery edit is incomplete: {exc}")

    def _escs_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("key", str, False), ("name", str, False), ("current_max", float, False),
                ("mass", float, False), ("efficiency", float, False),
                ("provisional", bool, False),
            ]
            values = [catalog.ESC(**row) for row in _coerce_records(event.new, schema, "ESC")]
            self.project.escs = {item.key: item for item in values}
            if len(self.project.escs) != len(values):
                raise ValueError("ESC keys must be unique")
            self._refresh_propulsion_options()
            self._refresh_all("ESC library updated.")
        except Exception as exc:
            self._error(f"ESC edit is incomplete: {exc}")

    def _propellers_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("key", str, False), ("name", str, False), ("mass", float, False),
                ("data", str, True), ("diameter", float, True), ("pitch", float, True),
                ("notes", str, True),
            ]
            rows = _coerce_records(event.new, schema, "Propeller")
            old = self.project.propellers
            values = []
            for row in rows:
                key = row["key"]
                values.append(PropellerDefinition(
                    **{**row, "data": self._optional_text(row, "data"),
                       "notes": self._optional_text(row, "notes"),
                       "points": old[key].points if key in old else []}
                ))
            self.project.propellers = {item.key: item for item in values}
            if len(self.project.propellers) != len(values):
                raise ValueError("propeller keys must be unique")
            self._updating = True
            keys = list(self.project.propellers)
            current = self.propeller_data_select.value
            self.propeller_data_select.options = keys
            self.propeller_data_select.value = current if current in keys else (keys[0] if keys else None)
            self._updating = False
            self._show_propeller_points()
            self._refresh_propulsion_options()
            self._refresh_all("Propeller library updated.")
        except Exception as exc:
            self._error(f"Propeller edit is incomplete: {exc}")

    def _show_propeller_points(self):
        definition = self.project.propellers.get(self.propeller_data_select.value)
        self._updating = True
        self.propeller_data_table.value = pd.DataFrame(
            [asdict(point) for point in definition.points] if definition else [],
            columns=["rpm", "J", "CT", "CP"],
        )
        self._updating = False

    def _propeller_data_selected(self, _):
        if not self._updating:
            self._show_propeller_points()

    def _propeller_points_changed(self, event):
        if self._updating or self.propeller_data_select.value not in self.project.propellers:
            return
        try:
            schema = [(name, float, False) for name in ("rpm", "J", "CT", "CP")]
            self.project.propellers[self.propeller_data_select.value].points = [
                PropellerPoint(**row) for row in _coerce_records(event.new, schema, "Propeller data")
            ]
            self._refresh_all("Measured propeller coefficients updated.")
        except Exception as exc:
            self._error(f"Propeller coefficient edit is incomplete: {exc}")

    def _propeller_data_uploaded(self, event):
        if not event.new or self.propeller_data_select.value not in self.project.propellers:
            return
        try:
            frame = pd.read_csv(io.BytesIO(event.new))
            normalized = {str(column).strip().lower(): column for column in frame.columns}
            required = {"rpm", "j", "ct", "cp"}
            if not required <= set(normalized):
                raise ValueError("CSV needs columns rpm, J, CT, and CP")
            selected = frame[[normalized[name] for name in ("rpm", "j", "ct", "cp")]].copy()
            selected.columns = ["rpm", "J", "CT", "CP"]
            self.propeller_data_table.value = selected
        except Exception as exc:
            self._error(f"Could not import propeller data: {exc}")

    def _add_propeller_point(self, _):
        self._append_table_row(
            self.propeller_data_table, {"rpm": 5000.0, "J": 0.2, "CT": 0.1, "CP": 0.05}
        )

    def _delete_propeller_point(self, _):
        self._delete_table_row(self.propeller_data_table)

    def _propulsion_changed(self, _):
        if self._updating:
            return
        setup = self.project.propulsion or PropulsionSetup()
        setup.battery = self.battery_select.value
        setup.state_of_charge = float(self.state_of_charge.value)
        setup.battery_x = float(self.battery_x.value)
        setup.battery_y = float(self.battery_y.value)
        setup.battery_z = float(self.battery_z.value)
        setup.include_component_masses = bool(self.include_propulsion_masses.value)
        self.project.propulsion = setup
        self._refresh_all("Propulsion system updated.")

    def _propulsors_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("name", str, False), ("motor", str, False),
                ("propeller", str, False), ("esc", str, False),
                ("throttle", float, False),
                ("x", float, False), ("y", float, False), ("z", float, False),
                ("pitch_deg", float, False), ("yaw_deg", float, False),
            ]
            propulsors = [
                PropulsorSetup(**row)
                for row in _coerce_records(event.new, schema, "Propulsor")
            ]
            setup = self.project.propulsion or PropulsionSetup()
            setup.propulsors = propulsors
            self.project.propulsion = setup
            self._refresh_all("Propulsor arrangement updated.")
        except Exception as exc:
            self._error(f"Propulsor edit is incomplete: {exc}")

    def _refresh_attachment_options(self):
        names = [surface.name for surface in self.project.surfaces] + [body.name for body in self.project.bodies]
        editors = dict(self.mass_table.editors)
        editors["attached_to"] = {"type": "list", "values": [""] + names}
        self.mass_table.editors = editors
        horizontal = [surface.name for surface in self.project.horizontal_surfaces]
        current = self.loads_surface.value
        self.loads_surface.options = horizontal
        if current not in horizontal:
            try:
                current = self.project.primary_horizontal_surface.name
            except ValueError:
                current = horizontal[0] if horizontal else None
            self.loads_surface.value = current

    def _airfoil_uploaded(self, event):
        if not event.new:
            return
        try:
            filename = self.airfoil_upload.filename or "custom.dat"
            section = foil.from_dat_text(event.new.decode("utf-8", errors="replace"), Path(filename).stem)
            key = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem).lower()
            self.project.airfoils[key] = AirfoilDefinition.from_section(section)
            self._refresh_airfoil_options()
            self.airfoil_select.value = key
            self._refresh_all(f"Imported airfoil {key!r}.")
            self.run_airfoil()
        except Exception as exc:
            self._error(f"Could not import airfoil: {exc}")

    def run_airfoil(self, _=None):
        self.run_airfoil_button.loading = True
        try:
            section = self.project.section(self.airfoil_select.value)
            alpha = np.linspace(self.airfoil_alpha_min.value, self.airfoil_alpha_max.value, 121)
            result = foil.aero(
                section,
                alpha,
                self.airfoil_re.value,
                xtr_upper=self.airfoil_transition.value,
                xtr_lower=self.airfoil_transition.value,
                model_size="xlarge",
            )
            self._last_airfoil_result = result
            self._last_airfoil_alpha = alpha
            self._last_airfoil_context = {
                "airfoil": self.airfoil_select.value,
                "reynolds_number": float(self.airfoil_re.value),
                "transition": float(self.airfoil_transition.value),
            }
            section_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.airfoil_select.value).lower()
            transition_stem = f"xtr_{self.airfoil_transition.value:.3g}".replace(".", "p")
            self.airfoil_download.filename = (
                f"{section_stem}_re_{self.airfoil_re.value:.6g}_{transition_stem}.csv"
            )
            self.airfoil_download.disabled = False
            selected = self.airfoil_plots.value or ["Lift curve"]
            cols = 2
            rows = math.ceil(len(selected) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(10.5, max(3.2, 3.15 * rows)))
            axes = np.atleast_1d(axes).ravel()
            cl, cd, cm = result["cl"], result["cd"], result["cm"]
            for ax, plot_name in zip(axes, selected):
                if plot_name == "Airfoil shape":
                    ax.plot(section.x, section.z)
                    ax.axis("equal")
                    ax.set(xlabel="x/c", ylabel="z/c")
                elif plot_name == "Lift curve":
                    ax.plot(alpha, cl)
                    ax.set(xlabel="angle of attack [deg]", ylabel="$c_l$")
                elif plot_name == "Drag polar":
                    ax.plot(cd, cl)
                    ax.set(xlabel="$c_d$", ylabel="$c_l$")
                elif plot_name == "Pitching moment":
                    ax.plot(alpha, cm)
                    ax.set(xlabel="angle of attack [deg]", ylabel="$c_m$ about c/4")
                elif plot_name == "Lift-to-drag":
                    ax.plot(alpha, cl / cd)
                    ax.set(xlabel="angle of attack [deg]", ylabel="$c_l/c_d$")
                elif plot_name == "Transition":
                    ax.plot(alpha, result["top_xtr"], label="upper")
                    ax.plot(alpha, result["bot_xtr"], label="lower")
                    ax.set(xlabel="angle of attack [deg]", ylabel="transition x/c")
                    ax.legend()
                ax.set_title(plot_name)
                ax.grid(alpha=0.25)
            for ax in axes[len(selected):]:
                ax.set_visible(False)
            fig.suptitle(f"{section.name} — Re = {self.airfoil_re.value:.3g}, NeuralFoil xlarge")
            fig.tight_layout()
            self._replace_figure(self.airfoil_plot, fig)
            best = int(np.nanargmax(cl / cd))
            self.airfoil_metrics.object = self._metric_cards([
                ("Maximum cₗ in sweep", f"{np.max(cl):.3f}"),
                ("Minimum c_d", f"{np.min(cd):.5f}"),
                ("Best cₗ/c_d", f"{cl[best] / cd[best]:.1f}"),
                ("α at best cₗ/c_d", f"{alpha[best]:.2f}°"),
                ("c_m at best cₗ/c_d", f"{cm[best]:.4f}"),
            ])
            minimum_confidence = float(np.min(result["confidence"]))
            self.airfoil_diagnostics.object = (
                f"Model coverage diagnostic: minimum NeuralFoil confidence = "
                f"{minimum_confidence:.3f}. This measures similarity to the training domain, "
                "not agreement with experiment."
            )
            self.airfoil_diagnostics.alert_type = "warning" if minimum_confidence < 0.5 else "light"
        except Exception as exc:
            self._last_airfoil_result = None
            self._last_airfoil_alpha = None
            self._last_airfoil_context = None
            self.airfoil_download.disabled = True
            self.airfoil_metrics.object = ""
            self.airfoil_diagnostics.object = f"Airfoil analysis failed: {type(exc).__name__}: {exc}"
            self.airfoil_diagnostics.alert_type = "danger"
        finally:
            self.run_airfoil_button.loading = False

    # -- validation, geometry, and analysis ------------------------------

    def _refresh_validation(self):
        issues = self.project.validate()
        if not issues:
            self.validation.object = (
                "<div style='padding:10px;border-left:4px solid #2f855a;background:#edf9f1'>"
                "Project geometry and inputs pass the current checks.</div>"
            )
            return
        items = "".join(
            f"<li><b>{escape(issue.level.title())}:</b> {escape(issue.message)}</li>"
            for issue in issues
        )
        self.validation.object = f"<div style='padding:10px;background:#fff8df'><ul>{items}</ul></div>"

    def _refresh_geometry(self):
        fig = plt.figure(figsize=(10.8, 8.6))
        ax3d = fig.add_subplot(221, projection="3d")
        ax_plan = fig.add_subplot(222)
        ax_side = fig.add_subplot(223)
        ax_front = fig.add_subplot(224)
        colors = {"wing": "#2563a6", "tail": "#3b8554", "canard": "#7b55a3", "fin": "#a64b35", "other": "#6b7280"}
        for surface in self.project.surfaces:
            stations = surface.stations
            color = colors.get(surface.purpose, "0.4")
            for sign in ([1, -1] if surface.symmetric else [1]):
                y = np.array([station.y * sign for station in stations])
                xle = np.array([station.x_le for station in stations])
                z = np.array([station.z for station in stations])
                chord = np.array([station.chord for station in stations])
                ax3d.plot(xle, y, z, color=color)
                ax3d.plot(xle + chord, y, z, color=color)
                for i in range(len(stations)):
                    ax3d.plot([xle[i], xle[i] + chord[i]], [y[i], y[i]], [z[i], z[i]], color=color, alpha=0.55)
                ax_plan.plot(xle, y, color=color)
                ax_plan.plot(xle + chord, y, color=color)
                ax_side.plot(xle, z, color=color)
                ax_side.plot(xle + chord, z, color=color)
                ax_front.plot(y, z, color=color)
                for i in range(len(stations)):
                    ax_plan.plot([xle[i], xle[i] + chord[i]], [y[i], y[i]], color=color, alpha=0.55)
                    ax_side.plot([xle[i], xle[i] + chord[i]], [z[i], z[i]], color=color, alpha=0.55)

            if self.show_panel_mesh.value:
                try:
                    _, b_ref, _ = self.project.reference_quantities()
                    surface_ns = max(8, int(round(self.analysis_ns.value * surface.span / b_ref)))
                    if surface.orientation == "vertical":
                        grid = _display_grid_for_surface(
                            surface, surface_ns, int(self.analysis_nc.value)
                        )
                    else:
                        grid, _ = _grid_for_surface(
                            self.project, surface, surface_ns, int(self.analysis_nc.value)
                        )
                    for sign in ([1, -1] if surface.symmetric else [1]):
                        mesh = grid.copy()
                        mesh[1] *= sign
                        for i in range(mesh.shape[1]):
                            ax3d.plot(*mesh[:, i, :], color="0.25", lw=0.35, alpha=0.65)
                            ax_plan.plot(mesh[0, i, :], mesh[1, i, :], color="0.25", lw=0.35, alpha=0.65)
                            ax_side.plot(mesh[0, i, :], mesh[2, i, :], color="0.25", lw=0.35, alpha=0.65)
                            ax_front.plot(mesh[1, i, :], mesh[2, i, :], color="0.25", lw=0.35, alpha=0.65)
                        for j in range(mesh.shape[2]):
                            ax3d.plot(*mesh[:, :, j], color="0.25", lw=0.35, alpha=0.65)
                            ax_plan.plot(mesh[0, :, j], mesh[1, :, j], color="0.25", lw=0.35, alpha=0.65)
                            ax_side.plot(mesh[0, :, j], mesh[2, :, j], color="0.25", lw=0.35, alpha=0.65)
                            ax_front.plot(mesh[1, :, j], mesh[2, :, j], color="0.25", lw=0.35, alpha=0.65)
                except Exception:
                    pass
        for body in self.project.bodies:
            x0 = body.x_nose or 0.0
            width = body.diameter or body.width or body.height or 0.01
            height = body.diameter or body.height or body.width or 0.01
            ax3d.plot([x0, x0 + body.length], [body.y, body.y], [body.z, body.z], color="0.35", lw=4)
            ax_plan.add_patch(Ellipse((x0 + body.length / 2, body.y), body.length, width, fill=False, edgecolor="0.35", lw=1.2))
            ax_side.add_patch(Ellipse((x0 + body.length / 2, body.z), body.length, height, fill=False, edgecolor="0.35", lw=1.2))
            ax_front.add_patch(Ellipse((body.y, body.z), width, height, fill=False, edgecolor="0.35", lw=1.2))
        try:
            components = self.project.components()
            mp = stability.mass_properties(components)
            ax3d.scatter([mp.x_cg], [mp.y_cg], [mp.z_cg], marker="X", s=90, color="black", edgecolor="white", depthshade=False)
            for ax, xx, yy in (
                (ax_plan, mp.x_cg, mp.y_cg),
                (ax_side, mp.x_cg, mp.z_cg),
                (ax_front, mp.y_cg, mp.z_cg),
            ):
                ax.scatter([xx], [yy], marker="X", s=70, color="black", edgecolor="white", zorder=9, label="CG" if ax is ax_plan else None)
                ax.annotate("CG", (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold")
            if self.show_mass_components.value:
                maximum_mass = max((component.mass for component in components), default=1.0)
                for index, component in enumerate(components, start=1):
                    size = 25.0 + 90.0 * np.sqrt(component.mass / maximum_mass)
                    ax3d.scatter([component.x], [component.y], [component.z], s=size, color="#d58b16", edgecolor="white", linewidth=0.6, depthshade=False)
                    for ax, xx, yy in (
                        (ax_plan, component.x, component.y),
                        (ax_side, component.x, component.z),
                        (ax_front, component.y, component.z),
                    ):
                        ax.scatter([xx], [yy], s=size, color="#d58b16", edgecolor="white", linewidth=0.6, zorder=7)
                        offset = (8 + 6 * ((index - 1) // 2)) * (1 if index % 2 else -1)
                        ax.annotate(
                            f"M{index}", (xx, yy), xytext=(5, offset),
                            textcoords="offset points", fontsize=7, color="#7a4b00",
                            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#7a4b00"},
                        )
                rows = "".join(
                    "<tr>"
                    f"<td style='padding:3px 10px'><b>M{index}</b></td>"
                    f"<td style='padding:3px 10px'>{escape(component.name)}</td>"
                    f"<td style='padding:3px 10px'>{component.mass:.4g} kg</td>"
                    "</tr>"
                    for index, component in enumerate(components, start=1)
                )
                self.mass_marker_legend.object = (
                    "<div style='overflow-x:auto'><b>Mass-marker key</b>"
                    f"<table>{rows}</table></div>"
                )
                self.mass_marker_legend.visible = True
            else:
                self.mass_marker_legend.visible = False
        except Exception:
            # Validation and the Mass tab report incomplete mass rows. Keep the
            # geometry itself usable while a student is in the middle of an edit.
            self.mass_marker_legend.visible = False
            pass

        setup = self.project.propulsion
        if setup is not None:
            for index, propulsor in enumerate(setup.propulsors):
                pitch = np.radians(propulsor.pitch_deg)
                yaw = np.radians(propulsor.yaw_deg)
                direction = np.array([
                    -np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch),
                ])
                try:
                    definition = self.project.propeller(propulsor)
                    diameter = definition.diameter or definition.model().diameter
                except Exception:
                    diameter = 0.01
                normal = direction / np.linalg.norm(direction)
                helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                basis_1 = np.cross(normal, helper)
                basis_1 /= np.linalg.norm(basis_1)
                basis_2 = np.cross(normal, basis_1)
                angle = np.linspace(0.0, 2.0 * np.pi, 81)
                center = np.array([propulsor.x, propulsor.y, propulsor.z])
                disk = center[:, None] + 0.5 * diameter * (
                    basis_1[:, None] * np.cos(angle) + basis_2[:, None] * np.sin(angle)
                )
                label = "propeller disk" if index == 0 else None
                ax3d.plot(*disk, color="#8b3fb0", lw=1.8)
                ax_plan.plot(disk[0], disk[1], color="#8b3fb0", lw=1.8, label=label)
                ax_side.plot(disk[0], disk[2], color="#8b3fb0", lw=1.8)
                ax_front.plot(disk[1], disk[2], color="#8b3fb0", lw=1.8)
                ax_front.annotate(f"P{index + 1}", (propulsor.y, propulsor.z), xytext=(5, 5), textcoords="offset points", fontsize=7, color="#6a2888")
        ax3d.set(xlabel="x aft [m]", ylabel="y right [m]", zlabel="z up [m]", title="3D")
        _set_axes_equal_3d(ax3d)
        ax_plan.set(xlabel="x aft [m]", ylabel="y right [m]", title="Planform (top)")
        ax_side.set(xlabel="x aft [m]", ylabel="z up [m]", title="Side")
        ax_front.set(xlabel="y right [m]", ylabel="z up [m]", title="Front")
        for ax in (ax_plan, ax_side, ax_front):
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
        ax_plan.legend(fontsize=7, loc="best")
        fig.tight_layout()
        self._replace_figure(self.geometry_plot, fig)
        self._refresh_surface_geometry()
        try:
            primary = self.project.primary_horizontal_surface
            S_ref, b_ref, c_ref = self.project.reference_quantities()
            mp = stability.mass_properties(self.project.components())
            htail = next((surface for surface in self.project.trim_surfaces if surface is not primary), None)
            vtail = max(self.project.vertical_surfaces, key=lambda surface: surface.area, default=None)
            wing_loading = mp.mass * 9.80665 / S_ref
            vh = (
                htail.area * (htail.aerodynamic_center_x - mp.x_cg) / (S_ref * c_ref)
                if htail else float("nan")
            )
            vv = (
                vtail.area * (vtail.aerodynamic_center_x - mp.x_cg) / (S_ref * b_ref)
                if vtail else float("nan")
            )
            self.geometry_summary.object = self._metric_cards([
                ("Vehicle mass", f"{mp.mass:.4g} kg"),
                ("CG (x, y, z)", f"{mp.x_cg:.3g}, {mp.y_cg:.3g}, {mp.z_cg:.3g} m"),
                ("Reference loading W/Sref", f"{wing_loading:.2f} N/m²"),
                ("Reference area Sref", f"{S_ref:.4f} m²"),
                ("Reference span bref", f"{b_ref:.4f} m"),
                ("Reference aspect ratio", f"{b_ref**2 / S_ref:.2f}"),
                ("Reference chord cref", f"{c_ref:.4f} m"),
                ("Horizontal-tail volume", f"{vh:.3f}" if np.isfinite(vh) else "—"),
                ("Vertical-tail volume", f"{vv:.3f}" if np.isfinite(vv) else "—"),
                ("Bodies / mass items", f"{len(self.project.bodies)} / {len(self.project.masses)}"),
            ])
            self.surface_summary_table.value = pd.DataFrame([
                {
                    "surface": surface.name, "orientation": surface.orientation,
                    "purpose": surface.purpose, "trim control": surface.trim_control,
                    "area [m²]": surface.area, "span/height [m]": surface.span,
                    "MAC [m]": surface.mac, "AC x [m]": surface.aerodynamic_center_x,
                    "stations": len(surface.stations),
                }
                for surface in self.project.surfaces
            ])
        except Exception as exc:
            self.geometry_summary.object = f"<p>Geometry summary unavailable: {escape(str(exc))}</p>"

    def _refresh_surface_geometry(self):
        """Show the selected lifting surface and its body-axis panel preview."""
        surface = self._current_surface()
        if surface is None or len(surface.stations) < 2:
            return
        fig = plt.figure(figsize=(10.4, 7.2))
        ax3d = fig.add_subplot(221, projection="3d")
        ax_plan = fig.add_subplot(222)
        ax_side = fig.add_subplot(223)
        ax_front = fig.add_subplot(224)
        axes_2d = (ax_plan, ax_side, ax_front)
        try:
            _, b_ref, _ = self.project.reference_quantities()
            ns = max(8, int(round(self.analysis_ns.value * surface.span / b_ref)))
        except Exception:
            ns = max(8, int(self.analysis_ns.value))
        nc = int(self.analysis_nc.value)
        grid = _display_grid_for_surface(surface, ns, nc)
        color = "#a64b35" if surface.orientation == "vertical" else "#2563a6"
        for sign in ([1, -1] if surface.symmetric else [1]):
            mesh = grid.copy()
            mesh[1] *= sign
            for index in range(mesh.shape[1]):
                ax3d.plot(*mesh[:, index, :], color=color, lw=0.6)
                ax_plan.plot(mesh[0, index, :], mesh[1, index, :], color=color, lw=0.6)
                ax_side.plot(mesh[0, index, :], mesh[2, index, :], color=color, lw=0.6)
                ax_front.plot(mesh[1, index, :], mesh[2, index, :], color=color, lw=0.6)
            for index in range(mesh.shape[2]):
                ax3d.plot(*mesh[:, :, index], color=color, lw=0.6)
                ax_plan.plot(mesh[0, :, index], mesh[1, :, index], color=color, lw=0.6)
                ax_side.plot(mesh[0, :, index], mesh[2, :, index], color=color, lw=0.6)
                ax_front.plot(mesh[1, :, index], mesh[2, :, index], color=color, lw=0.6)
        ax3d.set(xlabel="x aft [m]", ylabel="y right [m]", zlabel="z up [m]", title="3D panel preview")
        _set_axes_equal_3d(ax3d)
        ax_plan.set(xlabel="x aft [m]", ylabel="y right [m]", title="Planform (top)")
        ax_side.set(xlabel="x aft [m]", ylabel="z up [m]", title="Side")
        ax_front.set(xlabel="y right [m]", ylabel="z up [m]", title="Front")
        for axis in axes_2d:
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
        fig.suptitle(f"{surface.name} — {ns} spanwise × {nc} chordwise panels")
        fig.tight_layout()
        self._replace_figure(self.surface_geometry_plot, fig)

    def _refresh_mass_geometry(self, components, mass_properties):
        """Draw mass locations in the tab where students edit those masses."""
        fig, (ax_plan, ax_side) = plt.subplots(1, 2, figsize=(10.4, 4.4))
        colors = {"wing": "#2563a6", "tail": "#3b8554", "canard": "#7b55a3", "fin": "#a64b35", "other": "#6b7280"}
        for surface in self.project.surfaces:
            color = colors.get(surface.purpose, "0.5")
            for sign in ([1, -1] if surface.symmetric else [1]):
                x_le = np.array([station.x_le for station in surface.stations])
                chord = np.array([station.chord for station in surface.stations])
                y = sign * np.array([station.y for station in surface.stations])
                z = np.array([station.z for station in surface.stations])
                ax_plan.plot(x_le, y, color=color, alpha=0.65)
                ax_plan.plot(x_le + chord, y, color=color, alpha=0.65)
                ax_side.plot(x_le, z, color=color, alpha=0.65)
                ax_side.plot(x_le + chord, z, color=color, alpha=0.65)
        for body in self.project.bodies:
            x0 = body.x_nose or 0.0
            width = body.diameter or body.width or body.height or 0.01
            height = body.diameter or body.height or body.width or 0.01
            ax_plan.add_patch(Ellipse((x0 + body.length / 2, body.y), body.length, width, fill=False, edgecolor="0.45"))
            ax_side.add_patch(Ellipse((x0 + body.length / 2, body.z), body.length, height, fill=False, edgecolor="0.45"))
        maximum_mass = max((component.mass for component in components), default=1.0)
        for index, component in enumerate(components, start=1):
            size = 30.0 + 95.0 * np.sqrt(component.mass / maximum_mass)
            for axis, horizontal, vertical in (
                (ax_plan, component.x, component.y),
                (ax_side, component.x, component.z),
            ):
                axis.scatter([horizontal], [vertical], s=size, color="#d58b16", edgecolor="white", zorder=7)
                offset = (9 + 7 * ((index - 1) // 2)) * (1 if index % 2 else -1)
                axis.annotate(
                    f"M{index}", (horizontal, vertical), xytext=(5, offset),
                    textcoords="offset points", fontsize=8, color="#7a4b00",
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#7a4b00"},
                )
        for axis, horizontal, vertical in (
            (ax_plan, mass_properties.x_cg, mass_properties.y_cg),
            (ax_side, mass_properties.x_cg, mass_properties.z_cg),
        ):
            axis.scatter([horizontal], [vertical], marker="X", s=80, color="black", edgecolor="white", zorder=9)
            axis.annotate("CG", (horizontal, vertical), xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
        ax_plan.set(xlabel="x aft [m]", ylabel="y right [m]", title="Mass locations — planform")
        ax_side.set(xlabel="x aft [m]", ylabel="z up [m]", title="Mass locations — side")
        fig.tight_layout()
        self._replace_figure(self.mass_geometry_plot, fig)

    def _refresh_component_summaries(self):
        """Update live body-drag and independently entered mass results."""
        try:
            components = self.project.components()
            mp = stability.mass_properties(components)
            self.mass_summary.object = self._metric_cards([
                ("Total entered mass", f"{mp.mass:.4g} kg"),
                ("CG x", f"{mp.x_cg:.4g} m"), ("CG z", f"{mp.z_cg:.4g} m"),
                ("Ixx", f"{mp.Ixx:.4g} kg m²"), ("Iyy", f"{mp.Iyy:.4g} kg m²"),
                ("Izz", f"{mp.Izz:.4g} kg m²"), ("Ixz", f"{mp.Ixz:.4g} kg m²"),
            ])
            mass_models = {item.name: item.distributed or "point" for item in self.project.masses}
            mass_frame = pd.DataFrame([
                {
                    "marker": f"M{index}",
                    "component": component.name,
                    "model": mass_models.get(component.name, "propulsion library"),
                    "calculated mass [kg]": component.mass,
                    "CG x [m]": component.x,
                    "CG y [m]": component.y,
                    "CG z [m]": component.z,
                    "Ixx at CG [kg m²]": component.Ixx_cg,
                    "Iyy at CG [kg m²]": component.Iyy_cg,
                    "Izz at CG [kg m²]": component.Izz_cg,
                }
                for index, component in enumerate(components, start=1)
            ])
            # Clearing first forces Tabulator to remove stale browser-side rows
            # when a component list becomes shorter.
            self.mass_results.value = pd.DataFrame(columns=mass_frame.columns)
            self.mass_results.value = mass_frame
            self._refresh_mass_geometry(components, mp)
        except Exception as exc:
            self.mass_summary.object = f"Mass result unavailable: {escape(str(exc))}"
            self.mass_results.value = pd.DataFrame([{"result": f"Unavailable: {exc}"}])
        try:
            case = self.project.case()
            buildup = drag.buildup(
                self.project.equivalent_aircraft(), case.speed, altitude=case.altitude,
                interference=case.interference, protuberance=case.protuberance,
                f_other=case.f_other, cooling=case.cooling,
            )
            body_names = {body.name for body in self.project.bodies}
            rows = [row for row in buildup.rows if row.name in body_names]
            models = {body.name: body.drag_model for body in self.project.bodies}
            descriptions = {
                "streamlined_body": "skin friction × form factor on wetted area",
                "bluff_round_member": "round/bluff member: CD=0.90 on frontal area",
                "faired_member": "faired member: CD=0.25 on frontal area",
                "streamlined_strut": "streamlined strut: CD=0.10 on frontal area",
            }
            self.body_results.value = pd.DataFrame([
                {
                    "body": row.name, "selected model": models[row.name],
                    "correlation / reference": descriptions[models[row.name]],
                    "area used [m²]": row.S_wet,
                    "Re": row.Re, "form factor": row.FF, "drag area [m²]": row.f,
                    "CD on frontal area": row.cd_frontal,
                }
                for row in rows
            ])
        except Exception as exc:
            self.body_results.value = pd.DataFrame([{"result": f"Unavailable: {exc}"}])

    def _estimate_stall_from_polar(self, case, polar):
        """Find where the first local section reaches its airfoil clmax."""
        first = polar.solutions[0]
        limits = {}
        for name in first.surfaces:
            surface = self.project.surface_named(name)
            limits[name] = surface_section_cl_max(self.project, surface, first, case)
        margins = np.array([
            min(
                float(np.min(limits[name] - solution.surface(name).cl))
                for name in solution.surfaces
            )
            for solution in polar.solutions
        ])
        crossings = np.flatnonzero(margins <= 0.0)
        reached = bool(len(crossings))
        if reached:
            upper = int(crossings[0])
            lower = max(upper - 1, 0)
            if upper == lower:
                fraction = 0.0
            else:
                denominator = margins[lower] - margins[upper]
                fraction = margins[lower] / denominator if denominator > 0 else 1.0
        else:
            lower = upper = len(polar.solutions) - 1
            fraction = 0.0
        alpha = float(polar.alpha[lower] + fraction * (polar.alpha[upper] - polar.alpha[lower]))
        aircraft_cl = float(polar.CL[lower] + fraction * (polar.CL[upper] - polar.CL[lower]))
        distributions = {}
        critical = None
        critical_margin = float("inf")
        for name in first.surfaces:
            low_view = polar.solutions[lower].surface(name)
            high_view = polar.solutions[upper].surface(name)
            local_cl = low_view.cl + fraction * (high_view.cl - low_view.cl)
            local_margin = limits[name] - local_cl
            index = int(np.argmin(local_margin))
            if local_margin[index] < critical_margin:
                critical_margin = float(local_margin[index])
                critical = (name, index, float(low_view.y[index]))
            distributions[name] = {
                "y": low_view.y,
                "cl": local_cl,
                "cl_max": limits[name],
            }
        return {
            "reached": reached,
            "alpha": alpha,
            "CL": aircraft_cl,
            "margins": margins,
            "alpha_samples": polar.alpha,
            "distributions": distributions,
            "critical": critical,
        }

    def _clear_analysis_display(self, message=""):
        self.analysis_metrics.object = ""
        self.drag_table.value = pd.DataFrame()
        old = self.analysis_plots.object
        self.analysis_plots.object = None
        if old is not None:
            plt.close(old)
        self.analysis_warnings.object = message
        self.analysis_warnings.alert_type = "light"
        self.analysis_warnings.visible = bool(message)

    def _display_analysis_result(self, result, polar, stall, cached=False):
        self._last_result = result
        self._last_polar = polar
        self._last_stall = stall
        stem = _safe_filename(self.project.name).removesuffix(".flightlab.json")
        case_stem = _safe_filename(result.case.name).removesuffix(".flightlab.json")
        self.analysis_download.filename = f"{stem}_{case_stem}_polar.csv"
        self.analysis_download.disabled = False
        self.analysis_span_download.filename = f"{stem}_{case_stem}_spanwise.csv"
        self.analysis_span_download.disabled = False
        trim = result.trim
        stall_cl = f"{stall['CL']:.3f}" if stall["reached"] else f"> {stall['CL']:.3f}"
        stall_alpha = f"{stall['alpha']:.2f}°" if stall["reached"] else f"> {stall['alpha']:.2f}°"
        self.analysis_metrics.object = self._metric_cards([
            ("Mass", f"{result.mass_properties.mass:.3f} kg"),
            ("CG x", f"{trim.x_cg:.4f} m"),
            ("Trim α", f"{trim.alpha:.3f}°"),
            ("Trim CL", f"{trim.solution.CL:.4f}"),
            ("Total CD", f"{result.CD_total:.4f}"),
            ("Pitch-control deflection", f"{trim.trim_deflection:.3f}°"),
            ("Static margin", f"{100 * trim.static_margin:.1f}% MAC"),
            ("Span efficiency", f"{trim.solution.e_inv:.3f}"),
            ("Profile + body CD", f"{result.buildup.CD_profile_body:.4f}"),
            ("CDᵢ", f"{trim.solution.CD_i:.4f}"),
            ("L/D", f"{result.lift_to_drag:.1f}"),
            ("Estimated aircraft CLmax", stall_cl),
            ("Estimated stall α", stall_alpha),
            ("Best L/D in sweep", f"{np.nanmax(polar.LD):.1f}"),
        ])
        self._show_analysis_plots(result, polar, stall)
        self.drag_table.value = pd.DataFrame([
            {
                "component": row.name,
                "kind": row.kind,
                "drag area [m²]": row.f,
                "share [%]": 100 * row.f / result.buildup.f_components,
            }
            for row in result.buildup.rows
        ])
        warnings = list(result.warnings)
        warnings.append(
            "The CLmax estimate holds the trimmed pitch-control deflection fixed and combines linear VLM loading with NeuralFoil section limits; it does not model post-stall redistribution."
        )
        if not stall["reached"]:
            warnings.append(
                "No section reached its local clmax by the highest swept angle; the reported aircraft CLmax is a lower bound."
            )
        self.analysis_warnings.object = "\n".join(f"• {warning}" for warning in warnings)
        self.analysis_warnings.alert_type = "warning"
        self.analysis_warnings.visible = bool(warnings)
        verb = "Loaded cached" if cached else "Completed"
        self.status.object = f"{verb} integrated analysis for {result.case.name}."
        self.status.alert_type = "success"

    def run_integrated_analysis(self, _=None):
        self.run_analysis_button.loading = True
        self.status.object = "Running VLM, trim, stability, stall, mass, and drag analyses…"
        self.status.alert_type = "primary"
        case_name = self.analysis_case.value
        try:
            case = self.project.case(case_name)
            result = run_design_point(
                self.project, case, ns=self.analysis_ns.value, nc=self.analysis_nc.value
            )
            polar = aircraft_polar(
                self.project, case, alpha=np.linspace(-5.0, 20.0, 21),
                trim_deflection=result.trim.trim_deflection,
                ns=self.analysis_ns.value, nc=self.analysis_nc.value,
            )
            stall = self._estimate_stall_from_polar(case, polar)
            self._analysis_cache[case.name] = (result, polar, stall)
            self._display_analysis_result(result, polar, stall)
        except TrimNotPossibleError as exc:
            self._analysis_cache.pop(case_name, None)
            self._last_result = self._last_polar = self._last_stall = None
            self.analysis_download.disabled = True
            self.analysis_span_download.disabled = True
            self._clear_analysis_display()
            self.analysis_metrics.object = self._metric_cards([
                ("Trim status", "Not possible within entered control limits"),
            ])
            self.analysis_warnings.object = str(exc)
            self.analysis_warnings.alert_type = "danger"
            self.analysis_warnings.visible = True
            self.status.object = "The selected flight case cannot be trimmed with the entered control geometry and limits."
            self.status.alert_type = "danger"
        except Exception as exc:
            self._analysis_cache.pop(case_name, None)
            self._last_result = self._last_polar = self._last_stall = None
            self.analysis_download.disabled = True
            self.analysis_span_download.disabled = True
            self._clear_analysis_display()
            self.analysis_warnings.object = f"{type(exc).__name__}: {exc}"
            self.analysis_warnings.alert_type = "danger"
            self.analysis_warnings.visible = True
            self.status.object = "Analysis failed; the error is shown in the Analysis tab."
            self.status.alert_type = "danger"
        finally:
            self.run_analysis_button.loading = False

    def _show_analysis_plots(self, result, polar, stall):
        solution = result.trim.solution
        fig, axes = plt.subplots(3, 2, figsize=(10.8, 10.0))
        ax = axes[0, 0]
        ax.plot(polar.alpha, polar.CL)
        ax.plot(result.trim.alpha, result.trim.solution.CL, "o", label="trim")
        if stall["reached"]:
            ax.plot(stall["alpha"], stall["CL"], "s", label="first section at $c_{l,max}$")
        ax.set(xlabel="angle of attack [deg]", ylabel="$C_L$", title="Aircraft lift curve")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(polar.CD, polar.CL)
        ax.plot(result.CD_total, result.trim.solution.CL, "o", label="trim")
        ax.set(xlabel="$C_D$", ylabel="$C_L$", title="Aircraft drag polar")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.plot(polar.alpha, polar.LD)
        ax.plot(result.trim.alpha, result.lift_to_drag, "o", label="trim")
        ax.set(xlabel="angle of attack [deg]", ylabel="$L/D$", title="Aircraft lift-to-drag")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ax.plot(polar.alpha, polar.Cm)
        ax.axhline(0, color="0.45", lw=1)
        ax.plot(result.trim.alpha, result.trim.solution.Cm, "o", label="trim")
        ax.set(xlabel="angle of attack [deg]", ylabel="$C_m$ about CG", title="Aircraft pitching moment")
        ax.legend(fontsize=8)

        ax = axes[2, 0]
        for surface in self.project.horizontal_surfaces:
            if surface.name not in solution.surfaces:
                continue
            view = solution.surface(surface.name)
            line = ax.plot(
                np.r_[0.0, view.y], np.r_[view.ccl[0], view.ccl],
                label=f"{surface.name} actual",
            )[0]
            if surface.purpose == "wing":
                eta = np.clip(2.0 * np.abs(view.y) / surface.span, 0.0, 1.0)
                ellipse_shape = np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
                denominator = np.sum(ellipse_shape * view.ds)
                ellipse = ellipse_shape * (
                    np.sum(view.ccl * view.ds) / max(abs(denominator), 1e-30)
                )
                ax.plot(
                    np.r_[0.0, view.y], np.r_[ellipse[0], ellipse], "--",
                    color=line.get_color(), label=f"{surface.name} same-lift ellipse",
                )
        ax.set(
            xlabel="semispan panel center [m]", ylabel="$c c_l$ [m]",
            title="All-surface loading; per-wing same-lift ellipses",
        )
        ax.legend(fontsize=7, ncol=2)

        ax = axes[2, 1]
        for name, distribution in stall["distributions"].items():
            line = ax.plot(distribution["y"], distribution["cl"], label=f"{name} $c_l$")[0]
            ax.plot(
                distribution["y"], distribution["cl_max"], "--",
                color=line.get_color(), label=f"{name} $c_{{l,max}}$",
            )
        title = "Section lift at first local stall" if stall["reached"] else "Section lift at highest swept angle"
        ax.set(xlabel="semispan panel center [m]", ylabel="section coefficient", title=title)
        ax.legend(fontsize=7, ncol=2)

        for axis in axes.ravel():
            axis.grid(alpha=0.22)
        fig.suptitle(f"{result.project_name} — {result.case.name}")
        fig.tight_layout()
        self._replace_figure(self.analysis_plots, fig)

    def run_loads_analysis(self, _=None):
        """Run a project-aware maneuver/gust, span-load, and spar-cap analysis."""
        self.run_loads_button.loading = True
        self.status.object = "Running flight loads and spar sizing…"
        self.status.alert_type = "primary"
        try:
            self.project.require_valid()
            case = self.project.case(self.loads_case.value)
            surface = self.project.surface_named(self.loads_surface.value)
            if surface is None or surface.orientation != "horizontal":
                raise ValueError("choose a horizontal structural lifting surface")

            mass_properties = stability.mass_properties(self.project.components())
            mass = mass_properties.mass
            S_ref, _, c_ref = self.project.reference_quantities()
            aircraft = self.project.equivalent_aircraft()
            envelope_mode = self.loads_mode.value == "Maneuver/gust V–n envelope"
            envelope = None
            if envelope_mode:
                maximum_speed = None if self.loads_v_max.value <= 0 else self.loads_v_max.value
                envelope = loads.vn_diagram(
                    aircraft,
                    mass=mass,
                    CL_max=self.loads_cl_max.value,
                    CL_min=self.loads_cl_min.value,
                    altitude=case.altitude,
                    n_pos=self.loads_n_pos.value,
                    n_neg=self.loads_n_neg.value,
                    V_max=maximum_speed,
                    reference_area=S_ref,
                )

            design_n = float(self.loads_factor.value)
            if design_n <= 0:
                raise ValueError("structural design load factor must be positive")
            entered_design_speed = float(self.loads_design_speed.value)
            if entered_design_speed == 0.0:
                design_speed = float(envelope["V_A"]) if envelope_mode else float(case.speed)
            else:
                design_speed = entered_design_speed
            if design_speed <= 0.0:
                raise ValueError("structural design speed must be positive")
            if envelope_mode and design_speed > envelope["V_max"]:
                raise ValueError(
                    "structural design speed must be positive and no greater than "
                    "the maneuver-envelope maximum speed"
                )
            ns = int(self.loads_ns.value)
            nc = min(int(self.analysis_nc.value), 6)
            structural = analyze_structure(
                self.project, case, load_factor=design_n, speed=design_speed,
                ns=ns, nc=nc,
            )
            alpha = structural.alpha
            solution = structural.solution
            span = structural.span_load
            sizing = structural.sizing
            cap_thickness = structural.cap_thickness
            cap_EI = structural.EI
            deflection = structural.deflection

            if envelope_mode:
                gust_speed = np.linspace(0.0, envelope["V_max"], 160)
                lift_slope_per_radian = structural.lift_slope_per_degree * 180.0 / np.pi
                gust_positive = loads.gust_load_factor(
                    aircraft, gust_speed, gust=self.loads_gust.value, mass=mass,
                    CL_alpha=lift_slope_per_radian, altitude=case.altitude,
                    reference_area=S_ref, reference_chord=c_ref,
                )
                gust_negative = 2.0 - gust_positive

            self._last_loads_result = {
                "case": case.name,
                "mode": "envelope" if envelope_mode else "direct",
                "load_factor": design_n,
                "envelope": envelope,
                "span_load": span,
                "sizing": sizing,
                "deflection": deflection,
                "solution": solution,
                "surface": surface.name,
                "cap_thickness": cap_thickness,
                "EI": cap_EI,
            }
            stem = _safe_filename(self.project.name).removesuffix(".flightlab.json")
            surface_stem = _safe_filename(surface.name).removesuffix(".flightlab.json")
            self.loads_download.filename = f"{stem}_{surface_stem}_span_load.csv"
            self.loads_download.disabled = False
            metrics = [
                ("Aircraft mass", f"{mass:.4g} kg"),
                ("Reference W/S", f"{mass * loads.G0 / S_ref:.4g} N/m²"),
                ("Structural case", f"n={design_n:.3g} at {design_speed:.3g} m/s"),
                ("Solved angle of attack", f"{alpha:.3g}°"),
                (f"{surface.name} root shear", f"{span.root_shear:.4g} N"),
                ("Limit root moment", f"{abs(span.root_moment):.4g} N·m"),
                ("Ultimate root moment", f"{sizing['ultimate_moment']:.4g} N·m"),
                ("Required area, each cap", f"{1e6 * sizing['cap_area']:.4g} mm²"),
                ("Required cap thickness", f"{1e3 * cap_thickness:.4g} mm"),
                ("Cap-only EI", f"{cap_EI:.4g} N·m²"),
                ("Tip deflection", f"{1e3 * deflection['tip_deflection']:.4g} mm"),
                ("Tip / semispan", f"{100 * deflection['tip_over_semispan']:.3g}%"),
            ]
            if envelope_mode:
                metrics[2:2] = [
                    ("Stall speed", f"{envelope['V_stall']:.3g} m/s"),
                    ("Corner speed", f"{envelope['V_A']:.3g} m/s"),
                ]
            self.loads_metrics.object = self._metric_cards(metrics)

            if envelope_mode:
                fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
                ax = axes[0, 0]
                ax.plot(envelope["V_upper"], envelope["n_upper"], label="positive maneuver")
                ax.plot(envelope["V_lower"], envelope["n_lower"], label="negative maneuver")
                ax.plot(gust_speed, gust_positive, "--", label="positive gust")
                ax.plot(gust_speed, gust_negative, "--", label="negative gust")
                ax.axvline(envelope["V_A"], color="0.45", lw=1, label="corner speed")
                ax.set(xlabel="true airspeed [m/s]", ylabel="load factor n", title="Maneuver and gust envelope")
                ax.legend(fontsize=7)
                span_axis, internal_axis, deflection_axis = axes[0, 1], axes[1, 0], axes[1, 1]
                grid_axes = axes.ravel()
            else:
                fig, (span_axis, internal_axis, deflection_axis) = plt.subplots(1, 3, figsize=(11.2, 4.5))
                grid_axes = (span_axis, internal_axis, deflection_axis)

            ax = span_axis
            ax.plot(span.y, span.lift, color="#2563a6")
            ax.set(xlabel="semispan station [m]", ylabel="net aerodynamic load [N/m]", title=f"{surface.name} span load")

            ax = internal_axis
            shear_line = ax.plot(span.y, span.shear, color="#2f855a", label="shear")
            ax.set(xlabel="semispan station [m]", ylabel="shear [N]", title="Internal shear and bending moment")
            moment_axis = ax.twinx()
            moment_line = moment_axis.plot(span.y, span.moment, color="#a64b35", label="moment")
            moment_axis.set_ylabel("bending moment [N·m]")
            ax.legend(shear_line + moment_line, ["shear", "moment"], fontsize=8)

            ax = deflection_axis
            ax.plot(deflection["y"], 1e3 * deflection["deflection"], color="#7b55a3")
            ax.set(xlabel="semispan station [m]", ylabel="deflection [mm]", title="Cap-only beam deflection")
            for axis in grid_axes:
                axis.grid(alpha=0.22)
            moment_axis.grid(False)
            fig.suptitle(f"{self.project.name} — loads and preliminary spar caps")
            fig.tight_layout()
            self._replace_figure(self.loads_plots, fig)

            warnings = [
                "The structural VLM solve holds pitch-control deflection at zero and is linear about the solved load case.",
                "Distributed structural/fuel/battery mass is not yet applied as inertial relief, so the root moment is conservative when substantial mass lies in the wing.",
                "The two-cap beam omits web sizing, buckling, joints, local loads, fatigue, aeroelasticity, and material knockdowns. The deflection uses cap stiffness only.",
            ]
            if envelope_mode:
                warnings.insert(0, "The optional V–n envelope uses the aircraft reference area; the structural results are for the selected lifting surface.")
                if design_n > envelope["n_pos"]:
                    warnings.insert(0, "The structural design factor exceeds the entered positive limit load factor.")
            self.loads_warnings.object = "\n".join(f"• {item}" for item in warnings)
            self.loads_warnings.alert_type = "warning"
            self.loads_warnings.visible = True
            self.status.object = f"Completed loads and spar sizing for {surface.name}."
            self.status.alert_type = "success"
        except Exception as exc:
            self._last_loads_result = None
            self.loads_download.disabled = True
            self.loads_metrics.object = ""
            self.loads_warnings.object = f"{type(exc).__name__}: {exc}"
            self.loads_warnings.alert_type = "danger"
            self.loads_warnings.visible = True
            self.status.object = "Loads and spar sizing failed; see the Loads & structures tab."
            self.status.alert_type = "danger"
        finally:
            self.run_loads_button.loading = False

    def run_propulsion_analysis(self, _=None):
        self.run_propulsion_button.loading = True
        try:
            case = self.project.case(self.analysis_case.value)
            minimum_speed = float(self.propulsion_speed_min.value)
            maximum_speed = float(self.propulsion_speed_max.value)
            if minimum_speed == 0.0 and maximum_speed == 0.0:
                speed = None
            else:
                if minimum_speed <= 0.0 or maximum_speed <= minimum_speed:
                    raise ValueError(
                        "set both propulsion sweep limits with 0 < minimum < maximum, "
                        "or set both to zero for the automatic range"
                    )
                speed = np.linspace(
                    minimum_speed, maximum_speed, int(self.propulsion_speed_points.value)
                )
            result = analyze_propulsion(self.project, case, speed=speed)
            self._last_propulsion_result = result
            stem = _safe_filename(self.project.name).removesuffix(".flightlab.json")
            case_stem = _safe_filename(case.name).removesuffix(".flightlab.json")
            self.propulsion_download.filename = f"{stem}_{case_stem}_speed_sweep.csv"
            self.propulsion_download.disabled = False
            derivatives = propulsion_derivatives(self.project, case)
            point = result.operating_point
            self.propulsion_metrics.object = self._metric_cards([
                ("Thrust at case speed", f"{point.thrust:.3f} N"),
                ("Propulsors", str(len(point.propulsors))),
                ("Shared bus voltage", f"{point.bus_voltage:.2f} V"),
                ("Battery current", f"{point.current:.2f} A"),
                ("Total electrical power", f"{point.power_electrical:.1f} W"),
                ("Total shaft power", f"{point.power_shaft:.1f} W"),
                ("System efficiency", f"{100 * point.efficiency_total:.1f}%"),
                ("dT/dV", f"{derivatives.dT_dV:.3f} N/(m/s)"),
                ("dT/d collective throttle", f"{derivatives.dT_dthrottle:.2f} N"),
            ])
            self.propulsor_results.value = pd.DataFrame([
                {
                    "propulsor": setup.name,
                    "thrust [N]": unit.thrust,
                    "rpm": unit.rpm,
                    "current [A]": unit.current,
                    "motor voltage [V]": unit.voltage,
                    "advance ratio J": unit.J,
                    "motor efficiency": unit.efficiency_motor,
                    "propeller efficiency": unit.efficiency_prop,
                    "extrapolated": unit.extrapolated,
                }
                for setup, unit in zip(self.project.propulsion.propulsors, point.propulsors)
            ])
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
            ax = axes[0, 0]
            ax.plot(result.speed, result.thrust_available, label="thrust available")
            ax.plot(result.speed, result.drag_required, label="level-flight drag required")
            ax.axvline(case.speed, color="0.45", ls="--", lw=1)
            ax.set(ylabel="force [N]", title="Propulsion–airframe match")
            ax.legend(fontsize=8)
            ax = axes[0, 1]
            ax.plot(result.speed, 100 * result.efficiency_motor, label="motor")
            ax.plot(result.speed, 100 * result.efficiency_propeller, label="propeller")
            ax.plot(result.speed, 100 * result.efficiency_esc, label="ESC")
            ax.plot(result.speed, 100 * result.efficiency_total, lw=2.2, label="overall")
            ax.set(ylabel="efficiency [%]", title="Component efficiencies")
            ax.legend(fontsize=8)
            ax = axes[1, 0]
            ax.plot(result.speed, result.power_electrical, label="electrical in")
            ax.plot(result.speed, result.power_shaft, label="shaft")
            ax.plot(result.speed, result.power_useful, label="useful T·V")
            ax.set(xlabel="true airspeed [m/s]", ylabel="power [W]", title="Power flow")
            ax.legend(fontsize=8)
            ax = axes[1, 1]
            ax.plot(result.speed, result.current, label="battery current [A]")
            ax2 = ax.twinx()
            for index, propulsor in enumerate(self.project.propulsion.propulsors):
                ax2.plot(
                    result.speed, result.rpm[:, index],
                    label=f"{propulsor.name} rpm",
                )
            ax.set(xlabel="true airspeed [m/s]", ylabel="current [A]", title="Electrical and shaft loading")
            ax2.set_ylabel("rpm")
            lines = ax.lines + ax2.lines
            ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
            for axis in axes.ravel():
                axis.grid(alpha=0.25)
            fig.tight_layout()
            self._replace_figure(self.propulsion_plot, fig)
            warnings = list(result.warnings)
            for setup, unit in zip(self.project.propulsion.propulsors, point.propulsors):
                if unit.extrapolated_reason:
                    warnings.append(f"{setup.name}: {unit.extrapolated_reason}")
            self.propulsion_warnings.object = "\n".join(f"• {item}" for item in warnings)
            self.propulsion_warnings.visible = bool(warnings)
            self.propulsion_warnings.alert_type = "warning"
            self.status.object = f"Completed propulsion analysis for {case.name}."
            self.status.alert_type = "success"
        except Exception as exc:
            self._last_propulsion_result = None
            self.propulsion_download.disabled = True
            self.propulsion_warnings.object = f"{type(exc).__name__}: {exc}"
            self.propulsion_warnings.visible = True
            self.propulsion_warnings.alert_type = "danger"
            self.status.object = "Propulsion analysis failed; see the Propulsion tab."
            self.status.alert_type = "danger"
        finally:
            self.run_propulsion_button.loading = False

    def run_dynamic_stability(self, _=None):
        self.run_dynamics_button.loading = True
        try:
            case = self.project.case(self.analysis_case.value)
            result = analyze_dynamic_stability(
                self.project, case, ns=min(self.analysis_ns.value, 28), nc=self.analysis_nc.value
            )
            rows = []
            for family, modes in (("longitudinal", result.longitudinal), ("lateral", result.lateral)):
                for mode in modes:
                    rows.append({
                        "family": family, "mode": mode.name,
                        "real [1/s]": mode.real, "imag [1/s]": mode.imag,
                        "period [s]": mode.period if np.isfinite(mode.period) else None,
                        "damping ratio": mode.damping if mode.oscillatory else None,
                        "half/double time [s]": mode.time_to_half,
                        "stable": mode.stable,
                    })
            self.mode_table.value = pd.DataFrame(rows)
            derivative_rows = []
            for name, increment in result.body_increments.items():
                total = getattr(result.derivatives, name)
                derivative_rows.append({
                    "derivative": name, "lifting surfaces": total - increment,
                    "body increment": increment, "combined": total,
                    "units": "per radian" if name.endswith("alpha") or name.endswith("beta") else "per nondimensional rate",
                })
            power = result.propulsion_increments
            if power is not None:
                derivative_rows.extend([
                    {"derivative": "dT/dV", "lifting surfaces": None, "body increment": None, "combined": power.dT_dV, "units": "N/(m/s)"},
                    {"derivative": "dT/d throttle", "lifting surfaces": None, "body increment": None, "combined": power.dT_dthrottle, "units": "N"},
                    {"derivative": "dM/dV", "lifting surfaces": None, "body increment": None, "combined": power.dM_dV, "units": "N·m/(m/s)"},
                    {"derivative": "dM/d throttle", "lifting surfaces": None, "body increment": None, "combined": power.dM_dthrottle, "units": "N·m"},
                ])
            self.derivative_table.value = pd.DataFrame(derivative_rows)
            fig, ax = plt.subplots(figsize=(8.8, 4.5))
            for family, modes, marker in (
                ("longitudinal", result.longitudinal, "o"),
                ("lateral", result.lateral, "s"),
            ):
                for mode in modes:
                    ax.plot(mode.real, mode.imag, marker, label=f"{family}: {mode.name}")
                    if mode.imag > 0:
                        ax.plot(mode.real, -mode.imag, marker, color=ax.lines[-1].get_color())
            ax.axvline(0, color="0.35", lw=1)
            ax.set(xlabel="real eigenvalue [1/s]", ylabel="imaginary magnitude [1/s]", title="Linear dynamic modes")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            self._replace_figure(self.dynamics_plot, fig)
            self.dynamics_warnings.object = "\n".join(f"• {item}" for item in result.warnings)
            self.dynamics_warnings.visible = True
            self.dynamics_warnings.alert_type = "warning"
            self.status.object = f"Completed dynamic stability analysis for {case.name}."
            self.status.alert_type = "success"
        except Exception as exc:
            self.dynamics_warnings.object = f"{type(exc).__name__}: {exc}"
            self.dynamics_warnings.visible = True
            self.dynamics_warnings.alert_type = "danger"
            self.status.object = "Dynamic stability analysis failed; see the Dynamic stability tab."
            self.status.alert_type = "danger"
        finally:
            self.run_dynamics_button.loading = False

    def _refresh_generated_python(self):
        filename = self.project_filename.value.strip() or _safe_filename(self.project.name)
        case_name = self.analysis_case.value or (self.project.cases[0].name if self.project.cases else "Cruise")
        loads_v_max = None if self.loads_v_max.value <= 0 else float(self.loads_v_max.value)
        if self.loads_mode.value == "Maneuver/gust V–n envelope":
            load_case_setup = f'''# Optional maneuver/gust envelope.
envelope = loads.vn_diagram(
    project.equivalent_aircraft(), mass=result.mass_properties.mass,
    CL_max={self.loads_cl_max.value:.8g}, CL_min={self.loads_cl_min.value:.8g},
    altitude=case.altitude, n_pos={self.loads_n_pos.value:.8g},
    n_neg={self.loads_n_neg.value:.8g}, V_max={loads_v_max!r},
    reference_area=project.reference_quantities()[0],
)
design_speed = {self.loads_design_speed.value:.8g} or envelope["V_A"]'''
            envelope_print = 'print("corner speed [m/s] =", envelope["V_A"])'
        else:
            load_case_setup = f'''# Direct RC structural case; no certification-style V-n envelope.
design_speed = {self.loads_design_speed.value:.8g} or case.speed'''
            envelope_print = ""
        propulsion_min = float(self.propulsion_speed_min.value)
        propulsion_max = float(self.propulsion_speed_max.value)
        propulsion_points = int(self.propulsion_speed_points.value)
        if propulsion_min == 0.0 and propulsion_max == 0.0:
            propulsion_speed = (
                f"np.linspace(max(1.0, 0.45 * case.speed), "
                f"1.8 * case.speed, {propulsion_points})"
            )
        else:
            propulsion_speed = (
                f"np.linspace({propulsion_min:.8g}, {propulsion_max:.8g}, "
                f"{propulsion_points})"
            )
        code = f'''import numpy as np
import matplotlib.pyplot as plt

from flightlab import loads
from flightlab.project import AircraftProject
from flightlab.project_analysis import (
    aircraft_polar, analyze_dynamic_stability, analyze_propulsion,
    analyze_structure, run_design_point,
)

project = AircraftProject.load({filename!r})
case = project.case({case_name!r})
result = run_design_point(project, case, ns={self.analysis_ns.value}, nc={self.analysis_nc.value})

print(result.mass_properties.table())
print(result.buildup.table())
print("trim alpha [deg] =", result.trim.alpha)
print("pitch-control deflection [deg] =", result.trim.trim_deflection)
print("lift coefficient =", result.trim.solution.CL)
print("induced drag coefficient =", result.trim.solution.CD_i)
print("total drag coefficient =", result.CD_total)
print("total drag [N] =", result.drag)
print("static margin =", result.trim.static_margin)
print("L/D =", result.lift_to_drag)
for warning in result.warnings:
    print("WARNING:", warning)

# Whole-aircraft aerodynamics: full station geometry for CL, CDi, and Cm;
# local NeuralFoil cd(cl, Re) integrated over every surface strip, plus bodies.
polar = aircraft_polar(project, case, alpha=np.linspace(-5, 20, 21),
                       trim_deflection=result.trim.trim_deflection,
                       ns={self.analysis_ns.value}, nc={self.analysis_nc.value})
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(polar.alpha, polar.CL)
axes[0, 1].plot(polar.CD, polar.CL)
axes[1, 0].plot(polar.alpha, polar.LD)
axes[1, 1].plot(polar.alpha, polar.Cm)
plt.show()

# Spar surface, cap geometry, and material properties are saved in
# project.structure. The call below exposes the load case and numerical mesh.
{load_case_setup}
structure = analyze_structure(
    project, case, load_factor={self.loads_factor.value:.8g},
    speed=design_speed, ns={self.loads_ns.value},
    nc={min(self.analysis_nc.value, 6)},
)
{envelope_print}
print("root moment [N m] =", structure.span_load.root_moment)
print("required area of each cap [m^2] =", structure.sizing["cap_area"])
print("cap-only tip deflection [m] =", structure.deflection["tip_deflection"])

# Electric hardware and throttle are saved in project.propulsion; the requested
# speed sweep remains explicit.
propulsion_speed = {propulsion_speed}
power = analyze_propulsion(project, case, speed=propulsion_speed)
print(power.operating_point.table())

# Linear longitudinal and lateral modes at the explicit numerical resolution.
dynamics = analyze_dynamic_stability(
    project, case, ns={min(self.analysis_ns.value, 28)}, nc={self.analysis_nc.value}
)
print(dynamics.longitudinal.table())
print(dynamics.lateral.table())
print(dynamics.derivatives.table())
print("empirical body increments =", dynamics.body_increments)
print("propulsion derivatives =", dynamics.propulsion_increments)
'''
        self.python_output.object = (
            "Save the project beside your script, then use the same model directly. "
            "Physical configuration is read from the project; each call keeps the analysis request explicit.\n\n"
            f"```python\n{code}```"
        )

    # -- rendering helpers ------------------------------------------------

    @staticmethod
    def _metric_cards(items):
        cells = "".join(
            "<div style='padding:8px 12px;min-width:125px;border-right:1px solid #d9dee8'>"
            f"<div style='font-size:11px;color:#667085'>{escape(label)}</div>"
            f"<div style='font-size:17px;font-weight:650'>{escape(value)}</div></div>"
            for label, value in items
        )
        return f"<div style='display:flex;flex-wrap:wrap;border:1px solid #d9dee8'>{cells}</div>"

    @staticmethod
    def _replace_figure(pane, figure):
        old = pane.object
        pane.object = figure
        if old is not None and old is not figure:
            plt.close(old)

    def _error(self, message):
        self.status.object = message
        self.status.alert_type = "danger"

    # -- layout -----------------------------------------------------------

    def view(self):
        station_help = pn.pane.Markdown(
            "Each row is a real defining section. Coordinates are absolute body axes: "
            "**x aft, y right, z up**, in metres. Surfaces are linearly lofted between rows. "
            "The airfoil field is a selector containing bundled, NACA, and imported project airfoils."
        )
        role_help = pn.pane.Alert(
            "These fields have separate jobs. **Orientation** determines whether a surface enters the "
            "symmetric longitudinal solve or lateral-directional model. **Purpose** is a descriptive "
            "label only. **Pitch-trim control = whole_surface** rotates the complete surface; **elevator** "
            "deflects only the camber line aft of the entered hinge. Positive elevator deflection is "
            "trailing-edge down. The solver varies aircraft angle of attack and one shared control "
            "deflection while it solves lift = weight and pitching moment = 0. "
            "The aircraft-level coefficient reference is selected separately, so biplanes and tandem "
            "wings do not need to misuse a role label. If the required deflection is outside the entered "
            "limits—or the control has insufficient authority—the analysis reports that trim is not possible.",
            alert_type="light",
        )
        surface_controls = pn.Row(
            self.surface_select, self.surface_name, self.surface_orientation,
            self.surface_purpose, self.surface_trim_control,
            self.surface_symmetric, sizing_mode="stretch_width",
        )
        control_geometry = pn.Row(
            self.surface_control_hinge, self.surface_control_min, self.surface_control_max,
        )
        reference_controls = pn.Column(
            "### Aircraft coefficient reference",
            pn.pane.Markdown(
                "Select how S<sub>ref</sub>, b<sub>ref</sub>, and c<sub>ref</sub> are defined. "
                "**One surface** uses its "
                "area/span/MAC; **selected surfaces** sums the chosen areas, uses their largest span, and area-weights "
                "MAC; **manual** accepts any documented convention. These quantities normalize "
                "coefficients and do not decide which surfaces are analyzed."
            ),
            pn.Row(
                self.reference_mode, self.reference_surface, self.reference_surfaces, self.reference_area,
                self.reference_span, self.reference_chord,
            ),
        )
        surface_buttons = pn.Row(
            self.add_station_button, self.delete_station_button,
            pn.Spacer(width=20), self.add_surface_button, self.delete_surface_button,
        )

        airfoil_controls = pn.Column(
            pn.Row(self.airfoil_select, self.airfoil_upload),
            pn.Row(self.naca_code, self.add_naca_button),
            pn.Row(self.airfoil_re, self.airfoil_transition),
            pn.Row(self.airfoil_alpha_min, self.airfoil_alpha_max),
            self.airfoil_plots,
            pn.Row(self.run_airfoil_button, self.airfoil_download),
        )
        analysis_controls = pn.Column(
            pn.Row(self.analysis_case, self.analysis_ns, self.analysis_nc),
            pn.Row(self.run_analysis_button, self.analysis_download, self.analysis_span_download),
        )
        analysis_help = pn.pane.Alert(
            "Spanwise panels use cosine spacing. The entered count applies to a surface with the "
            "reference span; each other horizontal surface is scaled by its span and receives at "
            "least eight panels. The chordwise count applies to every lifting surface. Bodies are "
            "not panelled: they use the empirical drag correlations shown in the component table. "
            "Span-loading values are reported at panel centers. Results are cached by flight-case name. "
            "Any geometry or panel-count edit deletes every cached case, so stale results cannot be "
            "revisited. Every horizontal surface is plotted; each surface whose purpose is ‘wing’ also "
            "gets its own same-lift ellipse. For a biplane these are separate diagnostics, not a single "
            "whole-aircraft optimum. Aircraft CLmax is estimated where the first local section cl "
            "touches its airfoil clmax at the strip Reynolds number.",
            alert_type="light",
        )
        loads_help = pn.pane.Alert(
            "This tab uses the current project's total mass, aircraft reference area, selected atmosphere, "
            "and full-station VLM geometry. For an RC airplane, the default direct mode sizes a stated speed/load-factor "
            "case and does not require a certification-style V–n diagram. The optional envelope mode adds maneuver and "
            "sharp-edged-gust context; in that mode, a zero design speed selects the positive corner. In direct mode, "
            "zero selects the flight-case speed. Choose which horizontal surface carries the spar being sized. Material "
            "allowable, modulus, cap spacing, and cap width are saved aircraft-design inputs—not values inferred from geometry. "
            "The reported cap area is required for each of two equal caps.",
            alert_type="light",
        )
        loads_controls = pn.Column(
            pn.Row(self.loads_case, self.loads_surface, self.loads_mode, self.loads_ns),
            "### Structural design case",
            pn.Row(self.loads_cl_max, self.loads_cl_min),
            pn.Row(self.loads_n_pos, self.loads_n_neg),
            pn.Row(self.loads_v_max, self.loads_gust),
            pn.Row(self.loads_design_speed, self.loads_factor),
            "### Two-cap spar idealization",
            pn.Row(self.loads_spar_height, self.loads_allowable),
            pn.Row(self.loads_ultimate_factor, self.loads_modulus),
            pn.Row(self.loads_cap_width),
            pn.Row(self.run_loads_button, self.loads_download),
        )
        body_help = pn.pane.Markdown(
            "Bodies contribute **parasite-drag geometry**, not mass. Use **streamlined** for a fuselage "
            "or nacelle (skin friction × form factor on wetted area); **bluff_round_member** for a round/bluff "
            "member (C<sub>D</sub> = 0.90 on frontal area); **faired_member** for a faired member "
            "(C<sub>D</sub> = 0.25); and **streamlined_strut** for a streamlined strut "
            "(C<sub>D</sub> = 0.10). Enter `diameter` for a round "
            "section, or `width` and `height` for an elliptical/rectangular effective section. `x_nose` "
            "is the nose's body-axis x position. `cone_fraction` is the combined fraction of body length "
            "used for tapered nose/tail regions when estimating wetted area and volume. The result table "
            "states the area and correlation actually used."
        )
        mass_help = pn.pane.Alert(
            "Choose how each component's mass is distributed. If **mass** is entered, geometry only "
            "distributes that known total for CG and inertia. If mass is blank: `surface_volume` uses "
            "airfoil enclosed volume × density (for example, a solid foam wing); `surface_area` uses "
            "airfoil perimeter/wetted area × density × `skin_thickness` (a thin shell); and `body_volume` "
            "uses body volume × density. Build hybrid structures with multiple rows—for example foam core, "
            "skins, spar, and servos. Propulsion-library masses appear automatically when enabled in the "
            "Propulsion tab and should not be duplicated here. Fields irrelevant to the chosen model are ignored.",
            alert_type="light",
        )
        mass_model_guide = pn.widgets.Tabulator(
            pd.DataFrame([
                {"model": "point", "use for": "battery, payload, motor, equipment", "required if mass blank": "not allowed", "geometry fields used": "x, y, z"},
                {"model": "span", "use for": "spar/beam or known spanwise item", "required if mass blank": "not allowed", "geometry fields used": "x, y, z, span"},
                {"model": "surface_volume", "use for": "solid foam/core", "required if mass blank": "density", "geometry fields used": "attached surface + airfoil volume"},
                {"model": "surface_area", "use for": "skin or thin shell", "required if mass blank": "density + skin_thickness", "geometry fields used": "attached surface + airfoil perimeter"},
                {"model": "body_volume", "use for": "solid or uniformly distributed body", "required if mass blank": "density", "geometry fields used": "attached body volume"},
            ]),
            show_index=False, disabled=True, height=185,
        )
        propulsion_controls = pn.Column(
            pn.pane.Alert(
                "The battery and its location belong to the shared electrical system. Each propulsor "
                "row selects its own motor, ESC, propeller, throttle, thrust application point, and "
                "thrust direction. Multiple rows therefore represent one battery feeding multiple "
                "motors/propellers. When component masses are included, the battery and propulsor "
                "hardware automatically enter the aircraft mass properties at these same locations; "
                "do not duplicate them in the Mass tab.",
                alert_type="light",
            ),
            pn.Row(self.battery_select, self.state_of_charge, self.include_propulsion_masses),
            pn.Row(
                self.propulsion_speed_min, self.propulsion_speed_max,
                self.propulsion_speed_points,
            ),
            pn.Row(
                self.battery_x, self.battery_y, self.battery_z,
            ),
            pn.Row(self.run_propulsion_button, self.propulsion_download),
            "### Propulsors",
            self.propulsor_table,
            pn.Row(self.add_propulsor_button, self.delete_propulsor_button),
        )
        propulsion_library = pn.Column(
            pn.pane.Alert(
                "Catalog entries are copied into this project and are fully editable. Replace provisional "
                "motor resistance/no-load current and battery cell resistance with measured values here. "
                "For a measured propeller, add its diameter and pitch in metres, then import or edit rows "
                "with `rpm, J, CT, CP`; project points override the bundled dataset. All definitions are "
                "saved inside the `.flightlab.json` project.",
                alert_type="light",
            ),
            pn.Tabs(
                ("Motors", pn.Column(
                    self.motor_table, pn.Row(self.add_motor_button, self.delete_motor_button),
                )),
                ("Batteries", pn.Column(
                    self.battery_table, pn.Row(self.add_battery_button, self.delete_battery_button),
                )),
                ("ESCs", pn.Column(
                    self.esc_table, pn.Row(self.add_esc_button, self.delete_esc_button),
                )),
                ("Propellers", pn.Column(
                    self.propeller_table, pn.Row(self.add_propeller_button, self.delete_propeller_button),
                    self.propeller_data_select, self.propeller_data_upload, self.propeller_data_table,
                    pn.Row(self.add_propeller_point_button, self.delete_propeller_point_button),
                )),
                dynamic=True,
            ),
        )

        tabs = pn.Tabs(
            ("Aircraft", pn.Column(
                reference_controls, self.geometry_summary,
                pn.Row(self.show_mass_components, self.show_panel_mesh),
                self.geometry_plot, self.mass_marker_legend, "### Lifting-surface summary",
                self.surface_summary_table, self.validation,
            )),
            ("Airfoils", pn.Column(airfoil_controls, self.airfoil_metrics, self.airfoil_plot, "### Model diagnostics", self.airfoil_diagnostics)),
            ("Lifting surfaces", pn.Column(
                station_help, role_help, surface_controls, control_geometry,
                self.station_table, surface_buttons,
                pn.pane.Alert(
                    "The preview below shows only the selected lifting surface and its body-axis panel topology; masses, bodies, and propulsion markers are intentionally omitted.",
                    alert_type="light",
                ),
                self.surface_geometry_plot,
            )),
            ("Bodies", pn.Column(
                body_help, self.body_table, pn.Row(self.add_body_button, self.delete_body_button),
                "### Body drag results at the first flight case", self.body_results,
            )),
            ("Mass", pn.Column(
                mass_help, "### Which mass model should I use?", mass_model_guide,
                self.mass_table, pn.Row(self.add_mass_button, self.delete_mass_button),
                "### Mass locations", self.mass_geometry_plot,
                "### Calculated component properties", self.mass_results,
                "### Vehicle mass properties", self.mass_summary,
            )),
            ("Flight cases", pn.Column(
                "Cases share one aircraft but carry their own speed, altitude, load factor, drag markups, "
                "surface cleanliness (`n_crit`), and transition assumptions. **Natural** means no forced "
                "trip and sets upper/lower transition x/c to 1.0. **alpha_deg is only the initial guess** "
                "for integrated trim; the solver reports the trimmed angle of attack.",
                self.case_table, pn.Row(self.add_case_button, self.delete_case_button),
            )),
            ("Analysis", pn.Column(
                analysis_help, analysis_controls, self.analysis_metrics, self.analysis_warnings,
                self.analysis_plots, "### Component drag table", self.drag_table,
            )),
            ("Loads & structures", pn.Column(
                loads_help, loads_controls, self.loads_metrics,
                self.loads_warnings, self.loads_plots,
            )),
            ("Propulsion", pn.Column(
                "The analysis closes every motor–propeller torque match on one shared, sagging battery "
                "bus, then compares total forward thrust with whole-aircraft drag.",
                propulsion_controls, propulsion_library,
                self.propulsion_metrics, self.propulsion_warnings,
                "### Propulsor results at the selected flight case", self.propulsor_results,
                self.propulsion_plot,
            )),
            ("Dynamic stability", pn.Column(
                "Computes longitudinal (short-period/phugoid) and lateral-directional "
                "(Dutch-roll/roll/spiral) eigenmodes from the entered mass, CG, and inertias.",
                pn.Row(self.run_dynamics_button),
                self.dynamics_warnings,
                "### Derivative contributions", self.derivative_table,
                self.mode_table, self.dynamics_plot,
            )),
            ("Python", pn.Tabs(
                ("Guide", self.python_guide),
                ("Current project script", self.python_output),
                dynamic=True,
            )),
            dynamic=True,
        )
        sidebar = [
            "## Aircraft project",
            self.project_name,
            self.project_notes,
            pn.Column(self.blank_button, self.example_button),
            self.project_upload,
            self.project_filename,
            self.project_download,
            pn.pane.Markdown(
                "The file name is controlled here; the download folder or Save As dialog is controlled by your browser settings."
            ),
            self.status,
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Conventions**\n\n"
                "- SI units throughout\n"
                "- angles in degrees\n"
                "- x aft, y right, z up\n"
                "- NeuralFoil model: `xlarge`\n\n"
                "The VLM uses full station geometry. Bodies currently enter through "
                "handbook drag and mass models, not a 3D body-panel solver."
            ),
        ]
        return pn.template.FastListTemplate(
            title="FlightLab Aircraft Workbench",
            sidebar=sidebar,
            main=[tabs],
            accent_base_color="#1d4f7a",
            header_background="#173b5d",
        )


def create_workbench():
    """Create an independent workbench session for Panel's server."""
    return Workbench().view()


if __name__.startswith("bokeh") or __name__ == "__main__":  # panel serve support
    create_workbench().servable()
