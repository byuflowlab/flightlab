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
import numpy as np
import pandas as pd
import panel as pn

from . import catalog, drag, foil, stability
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
    aircraft_polar,
    analyze_dynamic_stability,
    analyze_propulsion,
    propulsion_derivatives,
    run_design_point,
)

pn.extension("tabulator", notifications=True, sizing_mode="stretch_width")

ORIENTATION_OPTIONS = ["horizontal", "vertical"]
PURPOSE_OPTIONS = ["wing", "tail", "canard", "fin", "other"]
TRIM_CONTROL_OPTIONS = ["fixed", "whole_surface", "elevator"]
REFERENCE_MODES = ["surface", "selected_surfaces", "manual"]
DRAG_MODELS = [
    "streamlined_body", "bluff_round_member", "faired_member", "streamlined_strut",
]


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


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return (stem or "aircraft") + ".flightlab.json"


def _table(frame, height=260, editors=None, configuration=None):
    return pn.widgets.Tabulator(
        frame,
        show_index=False,
        selectable=1,
        height=height,
        editors=editors or {},
        configuration=configuration or {"layout": "fitColumns"},
    )


class Workbench:
    """Stateful Panel view over an :class:`AircraftProject`."""

    def __init__(self, project: Optional[AircraftProject] = None):
        self.project = project or example_project()
        self._updating = False
        self._last_result = None

        self.status = pn.pane.Alert("Ready.", alert_type="light")
        self.project_name = pn.widgets.TextInput(label="Project name")
        self.project_notes = pn.widgets.TextAreaInput(label="Design notes", height=110)
        self.validation = pn.pane.HTML()

        self.blank_button = pn.widgets.Button(label="New starter design", icon="file-plus")
        self.example_button = pn.widgets.Button(label="Load multi-panel example", icon="plane")
        self.project_upload = pn.widgets.FileInput(label="Open project", accept=".json,.flightlab")
        self.project_download = pn.widgets.FileDownload(
            label="Save project", icon="download", callback=self._download_project
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
                "distributed": {"type": "list", "values": ["", "span", "surface_area", "surface_volume", "body_volume"]},
            }, configuration={"layout": "fitDataTable"},
        )
        self.add_mass_button = pn.widgets.Button(label="Add mass item", icon="plus")
        self.delete_mass_button = pn.widgets.Button(label="Delete selected", icon="trash")
        self.case_table = _table(
            pd.DataFrame(), height=285,
            editors={name: {"type": "number"} for name in (
                "speed", "altitude", "load_factor", "alpha_deg", "interference",
                "protuberance", "f_other", "cooling",
                "n_crit", "xtr_upper", "xtr_lower",
            )}, configuration={"layout": "fitDataTable"},
        )
        self.add_case_button = pn.widgets.Button(label="Add flight case", icon="plus")
        self.delete_case_button = pn.widgets.Button(label="Delete selected", icon="trash")

        self.geometry_plot = pn.pane.Matplotlib(height=500, tight=True, format="svg")
        self.surface_geometry_plot = pn.pane.Matplotlib(height=420, tight=True, format="svg")
        self.geometry_summary = pn.pane.HTML()
        self.surface_summary_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=210)
        self.body_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=230)
        self.mass_summary = pn.pane.HTML()
        self.mass_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=240)

        self.airfoil_select = pn.widgets.Select(label="Airfoil")
        self.airfoil_upload = pn.widgets.FileInput(label="Import .dat", accept=".dat")
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
        self.airfoil_plot = pn.pane.Matplotlib(height=620, tight=True, format="svg")
        self.airfoil_metrics = pn.pane.HTML()
        self.airfoil_diagnostics = pn.pane.Alert(alert_type="light")

        self.analysis_case = pn.widgets.Select(label="Flight case")
        self.analysis_ns = pn.widgets.IntInput(label="Wing spanwise panels", value=28, start=12, end=100)
        self.analysis_nc = pn.widgets.IntInput(label="Chordwise panels", value=4, start=2, end=12)
        self.run_analysis_button = pn.widgets.Button(
            label="Run integrated design point", color="primary", icon="player-play"
        )
        self.analysis_metrics = pn.pane.HTML()
        self.analysis_plots = pn.pane.Matplotlib(height=650, tight=True, format="svg")
        self.drag_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=260)
        self.analysis_warnings = pn.pane.Alert(alert_type="warning", visible=False)

        self.battery_select = pn.widgets.Select(label="Battery", options=sorted(catalog.BATTERIES))
        self.state_of_charge = pn.widgets.FloatInput(label="Battery state of charge", value=0.9, start=0.0, end=1.0, step=0.05)
        self.battery_x = pn.widgets.FloatInput(label="Battery x [m]", value=0.02, step=0.01)
        self.battery_y = pn.widgets.FloatInput(label="Battery y [m]", value=0.0, step=0.01)
        self.battery_z = pn.widgets.FloatInput(label="Battery z [m]", value=0.0, step=0.01)
        self.include_propulsion_masses = pn.widgets.Checkbox(
            label="Include battery and propulsor hardware in mass properties", value=True
        )
        self.propulsor_table = _table(
            pd.DataFrame(), height=260,
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
        self.delete_propulsor_button = pn.widgets.Button(label="Delete selected propulsor", icon="trash")
        self.motor_table = _table(pd.DataFrame(), height=250, configuration={"layout": "fitDataTable"})
        self.battery_table = _table(pd.DataFrame(), height=250, configuration={"layout": "fitDataTable"})
        self.esc_table = _table(pd.DataFrame(), height=210, configuration={"layout": "fitDataTable"})
        self.propeller_table = _table(pd.DataFrame(), height=230, configuration={"layout": "fitDataTable"})
        self.propeller_data_select = pn.widgets.Select(label="Propeller coefficient dataset")
        self.propeller_data_table = _table(
            pd.DataFrame(columns=["rpm", "J", "CT", "CP"]), height=260,
            editors={name: {"type": "number"} for name in ("rpm", "J", "CT", "CP")},
        )
        self.propeller_data_upload = pn.widgets.FileInput(
            label="Import coefficient CSV", accept=".csv,.txt"
        )
        self.add_motor_button = pn.widgets.Button(label="Add motor", icon="plus")
        self.delete_motor_button = pn.widgets.Button(label="Delete selected motor", icon="trash")
        self.add_battery_button = pn.widgets.Button(label="Add battery", icon="plus")
        self.delete_battery_button = pn.widgets.Button(label="Delete selected battery", icon="trash")
        self.add_esc_button = pn.widgets.Button(label="Add ESC", icon="plus")
        self.delete_esc_button = pn.widgets.Button(label="Delete selected ESC", icon="trash")
        self.add_propeller_button = pn.widgets.Button(label="Add propeller", icon="plus")
        self.delete_propeller_button = pn.widgets.Button(label="Delete selected propeller", icon="trash")
        self.add_propeller_point_button = pn.widgets.Button(label="Add coefficient point", icon="plus")
        self.delete_propeller_point_button = pn.widgets.Button(label="Delete selected point", icon="trash")
        self.run_propulsion_button = pn.widgets.Button(label="Run propulsion analysis", color="primary", icon="player-play")
        self.propulsion_metrics = pn.pane.HTML()
        self.propulsor_results = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=220)
        self.propulsion_plot = pn.pane.Matplotlib(height=560, tight=True, format="svg")
        self.propulsion_warnings = pn.pane.Alert(alert_type="warning", visible=False)

        self.run_dynamics_button = pn.widgets.Button(label="Run dynamic stability", color="primary", icon="player-play")
        self.mode_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=280)
        self.derivative_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=300)
        self.dynamics_plot = pn.pane.Matplotlib(height=450, tight=True, format="svg")
        self.dynamics_warnings = pn.pane.Alert(alert_type="warning", visible=False)
        self.python_output = pn.pane.Markdown()

        self._wire_callbacks()
        self._load_project(self.project)
        self.run_airfoil()

    # -- wiring and state -------------------------------------------------

    def _wire_callbacks(self):
        self.blank_button.on_click(lambda _: self._load_project(blank_project()))
        self.example_button.on_click(lambda _: self._load_project(example_project()))
        self.project_upload.param.watch(self._open_project, "value")
        self.project_name.param.watch(self._project_metadata_changed, "value")
        self.project_notes.param.watch(self._project_metadata_changed, "value")

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
        self.run_airfoil_button.on_click(self.run_airfoil)
        self.run_analysis_button.on_click(self.run_integrated_analysis)
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
        self.analysis_case.param.watch(lambda _: self._refresh_generated_python(), "value")
        self.analysis_ns.param.watch(lambda _: self._refresh_generated_python(), "value")
        self.analysis_nc.param.watch(lambda _: self._refresh_generated_python(), "value")

    def _load_project(self, project: AircraftProject):
        self._updating = True
        self.project = project
        self.project_name.value = project.name
        self.project_notes.value = project.notes
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
        self.mass_table.value = pd.DataFrame([asdict(item) for item in project.masses])
        self.case_table.value = pd.DataFrame([asdict(case) for case in project.cases])
        case_names = [case.name for case in project.cases]
        self.analysis_case.options = case_names
        self.analysis_case.value = case_names[0] if case_names else None
        self.project_download.filename = _safe_filename(project.name)
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
        self._refresh_validation()
        self._refresh_geometry()
        self._refresh_attachment_options()
        self._refresh_propulsion_options()
        self._refresh_component_summaries()
        self._refresh_generated_python()
        self.project_download.filename = _safe_filename(self.project.name)
        self.status.object = message
        self.status.alert_type = "light"

    def _project_metadata_changed(self, _):
        if self._updating:
            return
        self.project.name = self.project_name.value
        self.project.notes = self.project_notes.value
        self._refresh_all()

    def _download_project(self):
        return io.BytesIO((self.project.to_json() + "\n").encode("utf-8"))

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
        selected = self.station_table.selection
        if surface is None or not selected:
            self._error("Select one station row to delete.")
            return
        if len(surface.stations) <= 2:
            self._error("A lifting surface needs at least two stations.")
            return
        surface.stations.pop(selected[0])
        self._show_selected_surface()
        self._refresh_all(f"Deleted a station from {surface.name}.")

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
            schema = [
                ("name", str, False), ("mass", float, True), ("x", float, False),
                ("y", float, False), ("z", float, False),
                ("distributed", str, True), ("span", float, True),
                ("attached_to", str, True), ("density", float, True),
                ("skin_thickness", float, True),
            ]
            rows = _coerce_records(event.new, schema, "Mass")
            for row in rows:
                row["distributed"] = row["distributed"] or ""
                row["attached_to"] = row["attached_to"] or ""
            self.project.masses = [MassItem(**row) for row in rows]
            self._refresh_all("Mass model updated.")
        except Exception as exc:
            self._error(f"Mass edit is incomplete: {exc}")

    def _cases_changed(self, event):
        if self._updating:
            return
        try:
            schema = [
                ("name", str, False), ("speed", float, False), ("altitude", float, False),
                ("load_factor", float, False), ("alpha_deg", float, False),
                ("interference", float, False), ("protuberance", float, False),
                ("f_other", float, False), ("cooling", float, False),
                ("n_crit", float, False), ("xtr_upper", float, False),
                ("xtr_lower", float, False),
            ]
            self.project.cases = [FlightCase(**row) for row in _coerce_records(event.new, schema, "Flight case")]
            names = [case.name for case in self.project.cases]
            self._updating = True
            self.analysis_case.options = names
            if self.analysis_case.value not in names:
                self.analysis_case.value = names[0] if names else None
            self._updating = False
            self._refresh_all("Flight cases updated.")
        except Exception as exc:
            self._error(f"Flight-case edit is incomplete: {exc}")

    def _append_table_row(self, table, row):
        frame = table.value.copy()
        table.value = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    def _delete_table_row(self, table):
        if not table.selection:
            self._error("Select one row to delete.")
            return
        table.value = table.value.drop(table.value.index[table.selection[0]]).reset_index(drop=True)

    def _add_body(self, _):
        self._append_table_row(self.body_table, asdict(BodyDefinition("new body", 0.5, diameter=0.08)))

    def _delete_body(self, _):
        self._delete_table_row(self.body_table)

    def _add_mass(self, _):
        self._append_table_row(self.mass_table, asdict(MassItem("new component", 0.05, 0.25)))

    def _delete_mass(self, _):
        self._delete_table_row(self.mass_table)

    def _add_case(self, _):
        self._append_table_row(self.case_table, asdict(FlightCase("New case", 12.0)))

    def _delete_case(self, _):
        self._delete_table_row(self.case_table)

    # -- airfoils ---------------------------------------------------------

    def _refresh_airfoil_options(self):
        names = sorted(set(foil.available()) | set(self.project.airfoils) | {"naca0012", "naca2412"})
        current = self.airfoil_select.value
        self.airfoil_select.options = names
        self.airfoil_select.value = current if current in names else "naca2412"
        editors = dict(self.station_table.editors)
        editors["airfoil"] = {"type": "list", "values": names}
        self.station_table.editors = editors

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
        fig = plt.figure(figsize=(10.5, 5.1))
        ax3d = fig.add_subplot(121, projection="3d")
        ax2d = fig.add_subplot(122)
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
                ax2d.plot(xle, y, color=color)
                ax2d.plot(xle + chord, y, color=color)
                for i in range(len(stations)):
                    ax2d.plot([xle[i], xle[i] + chord[i]], [y[i], y[i]], color=color, alpha=0.55)
        for body in self.project.bodies:
            x0 = body.x_nose or 0.0
            ax3d.plot([x0, x0 + body.length], [body.y, body.y], [body.z, body.z], color="0.35", lw=4)
            ax2d.plot([x0, x0 + body.length], [body.y, body.y], color="0.35", lw=4, alpha=0.65)
        try:
            components = self.project.components()
            maximum_mass = max((component.mass for component in components), default=1.0)
            for index, component in enumerate(components):
                size = 28.0 + 125.0 * np.sqrt(component.mass / maximum_mass)
                label = "mass component" if index == 0 else None
                ax3d.scatter(
                    [component.x], [component.y], [component.z], s=size,
                    color="#d58b16", edgecolor="white", linewidth=0.6,
                    depthshade=False, label=label,
                )
                ax2d.scatter(
                    [component.x], [component.y], s=size, color="#d58b16",
                    edgecolor="white", linewidth=0.6, zorder=5, label=label,
                )
                short_name = component.name if len(component.name) <= 28 else component.name[:25] + "…"
                ax2d.annotate(
                    short_name, (component.x, component.y), xytext=(4, 4),
                    textcoords="offset points", fontsize=6.5, color="#7a4b00",
                )
        except Exception:
            # Validation and the Mass tab report incomplete mass rows. Keep the
            # geometry itself usable while a student is in the middle of an edit.
            pass

        setup = self.project.propulsion
        if setup is not None:
            try:
                _, b_ref, _ = self.project.reference_quantities()
                arrow_length = max(0.12 * b_ref, 0.12)
            except Exception:
                arrow_length = 0.15
            for index, propulsor in enumerate(setup.propulsors):
                pitch = np.radians(propulsor.pitch_deg)
                yaw = np.radians(propulsor.yaw_deg)
                direction = np.array([
                    -np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch),
                ])
                label = "propulsor / thrust" if index == 0 else None
                ax3d.scatter(
                    [propulsor.x], [propulsor.y], [propulsor.z], marker="D",
                    s=42, color="#8b3fb0", depthshade=False, label=label,
                )
                ax3d.quiver(
                    propulsor.x, propulsor.y, propulsor.z,
                    *(arrow_length * direction), color="#8b3fb0",
                    arrow_length_ratio=0.18, linewidth=1.8,
                )
                ax2d.scatter(
                    [propulsor.x], [propulsor.y], marker="D", s=42,
                    color="#8b3fb0", zorder=6, label=label,
                )
                ax2d.arrow(
                    propulsor.x, propulsor.y,
                    arrow_length * direction[0], arrow_length * direction[1],
                    color="#8b3fb0", width=0.002, head_width=0.025,
                    length_includes_head=True, zorder=5,
                )
                ax2d.annotate(
                    propulsor.name, (propulsor.x, propulsor.y), xytext=(4, -10),
                    textcoords="offset points", fontsize=7, color="#6a2888",
                )
        ax3d.set(xlabel="x aft [m]", ylabel="y right [m]", zlabel="z up [m]", title="3D lifting-surface stations")
        _set_axes_equal_3d(ax3d)
        ax2d.set(xlabel="x aft [m]", ylabel="y right [m]", title="Planform", aspect="equal")
        ax2d.grid(alpha=0.2)
        if self.project.masses or setup is not None:
            ax2d.legend(fontsize=7, loc="best")
        fig.tight_layout()
        old_geometry = self.geometry_plot.object
        old_surface = self.surface_geometry_plot.object
        self.geometry_plot.object = fig
        self.surface_geometry_plot.object = fig
        for old in (old_geometry, old_surface):
            if old is not None and old is not fig:
                plt.close(old)
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
                ("Reference area Sref", f"{S_ref:.4g} m²"),
                ("Reference span bref", f"{b_ref:.4g} m"),
                ("Reference aspect ratio", f"{b_ref**2 / S_ref:.2f}"),
                ("Reference chord cref", f"{c_ref:.4g} m"),
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
            self.mass_results.value = pd.DataFrame([
                {
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
                for component in components
            ])
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

    def run_integrated_analysis(self, _=None):
        self.run_analysis_button.loading = True
        self.status.object = "Running VLM, trim, stability, mass, and drag analyses…"
        self.status.alert_type = "primary"
        try:
            case = self.project.case(self.analysis_case.value)
            result = run_design_point(
                self.project, case, ns=self.analysis_ns.value, nc=self.analysis_nc.value
            )
            polar = aircraft_polar(
                self.project, case, alpha=np.linspace(-5.0, 14.0, 13),
                trim_deflection=result.trim.trim_deflection,
                ns=self.analysis_ns.value, nc=self.analysis_nc.value,
            )
            self._last_result = result
            trim = result.trim
            self.analysis_metrics.object = self._metric_cards([
                ("Mass", f"{result.mass_properties.mass:.3f} kg"),
                ("CG x", f"{trim.x_cg:.4f} m"),
                ("Trim α", f"{trim.alpha:.3f}°"),
                ("Pitch-control deflection", f"{trim.trim_deflection:.3f}°"),
                ("Static margin", f"{100 * trim.static_margin:.1f}% MAC"),
                ("Span efficiency", f"{trim.solution.e_inv:.3f}"),
                ("Profile + body CD", f"{result.buildup.CD_profile_body:.4f}"),
                ("CDᵢ", f"{trim.solution.CD_i:.4f}"),
                ("L/D", f"{result.lift_to_drag:.1f}"),
                ("Best L/D in sweep", f"{np.nanmax(polar.LD):.1f}"),
            ])
            self._show_analysis_plots(result, polar)
            self.drag_table.value = pd.DataFrame([
                {"component": row.name, "kind": row.kind, "drag area [m²]": row.f, "share [%]": 100 * row.f / result.buildup.f_components}
                for row in result.buildup.rows
            ])
            self.analysis_warnings.object = "\n".join(f"• {warning}" for warning in result.warnings)
            self.analysis_warnings.alert_type = "warning"
            self.analysis_warnings.visible = True
            self.status.object = f"Completed integrated analysis for {case.name}."
            self.status.alert_type = "success"
        except TrimNotPossibleError as exc:
            self.analysis_metrics.object = self._metric_cards([
                ("Trim status", "Not possible within entered control limits"),
            ])
            self.analysis_warnings.object = str(exc)
            self.analysis_warnings.alert_type = "danger"
            self.analysis_warnings.visible = True
            self.status.object = "The selected flight case cannot be trimmed with the entered control geometry and limits."
            self.status.alert_type = "danger"
        except Exception as exc:
            self.analysis_metrics.object = ""
            self.analysis_warnings.object = f"{type(exc).__name__}: {exc}"
            self.analysis_warnings.alert_type = "danger"
            self.analysis_warnings.visible = True
            self.status.object = "Analysis failed; the error is shown in the Results tab."
            self.status.alert_type = "danger"
        finally:
            self.run_analysis_button.loading = False

    def _show_analysis_plots(self, result, polar):
        solution = result.trim.solution
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.3))
        ax = axes[0, 0]
        ax.plot(polar.alpha, polar.CL)
        ax.plot(result.trim.alpha, result.trim.solution.CL, "o", label="trim")
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
        for name in solution.surfaces:
            view = solution.surface(name)
            ax.plot(view.y, view.ccl, label=name)
        ax.set(xlabel="semispan station [m]", ylabel="$c c_l$ [m]", title="Span loading")
        ax.legend(fontsize=8)

        ax = axes[2, 1]
        rows = result.buildup.rows
        ax.barh([row.name for row in rows], [row.f for row in rows], color="#356ea6")
        ax.invert_yaxis()
        ax.set(xlabel="drag area [m²]", title="Parasite-drag buildup")
        for axis in axes.ravel():
            axis.grid(alpha=0.22)
        fig.suptitle(f"{result.project_name} — {result.case.name}")
        fig.tight_layout()
        self._replace_figure(self.analysis_plots, fig)

    def run_propulsion_analysis(self, _=None):
        self.run_propulsion_button.loading = True
        try:
            case = self.project.case(self.analysis_case.value)
            result = analyze_propulsion(self.project, case)
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
        filename = _safe_filename(self.project.name)
        case_name = self.analysis_case.value or (self.project.cases[0].name if self.project.cases else "Cruise")
        code = f'''import numpy as np
import matplotlib.pyplot as plt

from flightlab.project import AircraftProject
from flightlab.project_analysis import (
    aircraft_polar, analyze_dynamic_stability, analyze_propulsion, run_design_point,
)

project = AircraftProject.load({filename!r})
case = project.case({case_name!r})
result = run_design_point(project, case, ns={self.analysis_ns.value}, nc={self.analysis_nc.value})

print(result.mass_properties.table())
print(result.buildup.table())
print("trim alpha [deg] =", result.trim.alpha)
print("pitch-control deflection [deg] =", result.trim.trim_deflection)
print("static margin =", result.trim.static_margin)
print("L/D =", result.lift_to_drag)

# Whole-aircraft aerodynamics: full station geometry for CL, CDi, and Cm;
# local NeuralFoil cd(cl, Re) integrated over every surface strip, plus bodies.
polar = aircraft_polar(project, case, alpha=np.linspace(-5, 14, 20),
                       trim_deflection=result.trim.trim_deflection,
                       ns={self.analysis_ns.value}, nc={self.analysis_nc.value})
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(polar.alpha, polar.CL)
axes[0, 1].plot(polar.CD, polar.CL)
axes[1, 0].plot(polar.alpha, polar.LD)
axes[1, 1].plot(polar.alpha, polar.Cm)
plt.show()

# Electric chain and thrust available versus airframe drag required.
power = analyze_propulsion(project, case)
print(power.operating_point.table())

# Linear longitudinal and lateral modes (currently equivalent surfaces).
dynamics = analyze_dynamic_stability(project, case)
print(dynamics.longitudinal.table())
print(dynamics.lateral.table())
print(dynamics.derivatives.table())
print("empirical body increments =", dynamics.body_increments)
print("propulsion derivatives =", dynamics.propulsion_increments)
'''
        self.python_output.object = (
            "Save the project beside your script or notebook, then use the same model directly:\n\n"
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
            pn.Row(self.airfoil_re, self.airfoil_alpha_min, self.airfoil_alpha_max, self.airfoil_transition),
            self.airfoil_plots,
            self.run_airfoil_button,
        )
        analysis_controls = pn.Row(
            self.analysis_case, self.analysis_ns, self.analysis_nc, self.run_analysis_button,
            sizing_mode="stretch_width",
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
            pn.Row(self.battery_x, self.battery_y, self.battery_z, self.run_propulsion_button),
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
                dynamic=False,
            ),
        )

        tabs = pn.Tabs(
            ("Aircraft", pn.Column(reference_controls, self.geometry_summary, self.geometry_plot, "### Lifting-surface summary", self.surface_summary_table, self.validation)),
            ("Airfoils", pn.Column(airfoil_controls, self.airfoil_metrics, self.airfoil_plot, "### Model diagnostics", self.airfoil_diagnostics)),
            ("Lifting surfaces", pn.Column(station_help, role_help, surface_controls, control_geometry, self.station_table, surface_buttons, self.surface_geometry_plot)),
            ("Bodies", pn.Column(
                body_help, self.body_table, pn.Row(self.add_body_button, self.delete_body_button),
                "### Body drag results at the first flight case", self.body_results,
            )),
            ("Mass", pn.Column(
                mass_help, "### Which mass model should I use?", mass_model_guide,
                self.mass_table, pn.Row(self.add_mass_button, self.delete_mass_button),
                "### Calculated component properties", self.mass_results,
                "### Vehicle mass properties", self.mass_summary,
            )),
            ("Flight cases", pn.Column(
                "Cases share one aircraft but carry their own speed, altitude, load factor, drag markups, "
                "surface cleanliness (`n_crit`), and optional forced-transition locations (`xtr_upper/lower`).",
                self.case_table, pn.Row(self.add_case_button, self.delete_case_button),
            )),
            ("Analysis", pn.Column(
                analysis_controls, self.analysis_metrics, self.analysis_warnings,
                self.analysis_plots, "### Component drag table", self.drag_table,
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
            ("Python", self.python_output),
            dynamic=True,
        )
        sidebar = [
            "## Aircraft project",
            self.project_name,
            self.project_notes,
            pn.Column(self.blank_button, self.example_button),
            self.project_upload,
            self.project_download,
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
