"""Optional Panel workbench smoke tests."""

import pytest


pytest.importorskip("panel")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from flightlab.workbench import Workbench, _coerce_records


def test_workbench_builds_and_runs_integrated_analysis():
    workbench = Workbench()
    editable_tables = (
        workbench.station_table, workbench.body_table, workbench.mass_table,
        workbench.case_table, workbench.propulsor_table, workbench.motor_table,
        workbench.battery_table, workbench.esc_table,
        workbench.propeller_table, workbench.propeller_data_table,
    )
    assert all(table.selectable == "checkbox" for table in editable_tables)
    view = workbench.view()
    workbench.run_integrated_analysis()

    assert view.title == "FlightLab Aircraft Workbench"
    assert workbench._last_result is not None
    assert workbench._last_result.trim.converged
    assert workbench._last_stall["reached"]
    assert workbench.status.alert_type == "success"
    # Lift curve, drag polar, L/D, and Cm all show an explicit trim marker.
    for axis in workbench.analysis_plots.object.axes[:4]:
        assert len(axis.lines) >= 2
    assert "surface_area" in workbench.mass_table.editors["distributed"]["values"]
    assert "Main wing" in workbench.mass_table.editors["attached_to"]["values"]
    assert workbench.surface_orientation.value == "horizontal"
    assert workbench.surface_trim_control.options == ["fixed", "whole_surface", "elevator"]
    assert workbench.reference_mode.value == "surface"
    assert "correlation / reference" in workbench.body_results.value.columns
    assert "calculated mass [kg]" in workbench.mass_results.value.columns
    assert "marker" in workbench.mass_results.value.columns
    assert "Trim CL" in workbench.analysis_metrics.object
    assert "Total CD" in workbench.analysis_metrics.object
    assert "M1000" in set(workbench.motor_table.value["key"])
    assert len(workbench.propulsor_table.value) == 1
    assert "Propulsion battery (B3S1300)" in set(workbench.mass_results.value["component"])
    # CG is always visible; individual mass markers are an optional uncluttering overlay.
    assert len(workbench.geometry_plot.object.axes[0].collections) == 1
    workbench.show_mass_components.value = True
    assert len(workbench.geometry_plot.object.axes[0].collections) > 1
    # The aircraft plot now has 3D, planform, side, and front views.
    assert len(workbench.geometry_plot.object.axes) == 4
    # Surface editing has its own selected-surface preview, without mass markers.
    assert workbench.surface_geometry_plot.object is not workbench.geometry_plot.object
    assert not workbench.surface_geometry_plot.object.axes[0].collections
    workbench.surface_select.value = "Vertical tail"
    vertical_side = workbench.surface_geometry_plot.object.axes[2]
    assert any(np.ptp(line.get_ydata()) > 0.1 for line in vertical_side.lines)
    # The Mass tab always has a labelled plan/side location view.
    assert len(workbench.mass_geometry_plot.object.axes) == 2
    assert workbench.mass_geometry_plot.object.axes[0].collections
    assert "naca0009" in workbench.airfoil_select.options
    assert "Reusable sweep pattern" in workbench.python_guide.object
    assert "body_named" in workbench.python_guide.object
    assert "total drag [N]" in workbench.python_output.object
    assert "point" in workbench.mass_table.editors["distributed"]["values"]
    assert set(workbench.case_table.value["transition"]) == {"natural"}
    propeller_line = next(
        line for line in workbench.geometry_plot.object.axes[1].lines
        if line.get_label() == "propeller disk"
    )
    expected_diameter = workbench.project.propeller(
        workbench.project.propulsion.propulsors[0]
    ).model().diameter
    assert np.ptp(propeller_line.get_ydata()) == pytest.approx(expected_diameter)
    outline_lines = len(workbench.geometry_plot.object.axes[1].lines)
    workbench.show_panel_mesh.value = True
    assert len(workbench.geometry_plot.object.axes[1].lines) > outline_lines
    assert len(workbench.analysis_plots.object.axes) == 6
    loading_axis = workbench.analysis_plots.object.axes[4]
    loading_labels = {line.get_label() for line in loading_axis.lines}
    assert {"Main wing actual", "Horizontal tail actual"} <= loading_labels

    # Completed cases are restored from cache when the selector cycles back.
    cruise_result = workbench._last_result
    workbench.analysis_case.value = "Takeoff"
    assert workbench._last_result is None
    workbench.run_integrated_analysis()
    assert workbench._last_result.case.name == "Takeoff"
    workbench.analysis_case.value = "Cruise"
    assert workbench._last_result is cruise_result
    assert "cached" in workbench.status.object.lower()
    workbench.analysis_ns.value += 2
    assert workbench._analysis_cache == {}
    assert workbench._last_result is None
    assert "deleted" in workbench.analysis_warnings.object.lower()
    workbench.project_filename.value = "my-analysis-copy"
    assert workbench.project_download.filename == "my-analysis-copy.flightlab.json"
    for widget_name in (
        "project_filename", "surface_select", "reference_mode", "airfoil_re",
        "analysis_case", "analysis_ns", "loads_mode", "loads_factor",
        "battery_select", "propulsion_speed_points",
    ):
        assert getattr(workbench, widget_name).description
    plt.close("all")


def test_mass_deletion_refreshes_every_derived_view():
    workbench = Workbench()
    workbench.show_mass_components.value = True
    removed_names = set(workbench.mass_table.value.iloc[[0, 2]]["name"])
    workbench.mass_table.selection = [0, 2]
    workbench._delete_mass(None)

    component_count = len(workbench.project.components())
    assert len(workbench.project.masses) == 1
    assert removed_names.isdisjoint(workbench.mass_results.value["component"])
    assert len(workbench.mass_results.value) == component_count
    # One collection per component plus the CG in both figures.
    assert len(workbench.geometry_plot.object.axes[0].collections) == component_count + 1
    assert len(workbench.mass_geometry_plot.object.axes[0].collections) == component_count + 1


def test_common_table_delete_removes_every_selected_row():
    workbench = Workbench()
    original = workbench.body_table.value.iloc[0].to_dict()
    second = {**original, "name": "second body"}
    workbench.body_table.value = pd.DataFrame([original, second])
    removed_names = set(workbench.body_table.value.iloc[[0, 1]]["name"])
    workbench.body_table.selection = [0, 1]
    workbench._delete_body(None)

    assert removed_names.isdisjoint(body.name for body in workbench.project.bodies)
    assert workbench.body_table.selection == []


def test_station_delete_handles_multiple_rows_and_preserves_minimum():
    workbench = Workbench()
    surface = workbench._current_surface()
    assert len(surface.stations) == 4
    workbench.station_table.selection = [1, 2]
    workbench._delete_station(None)

    assert len(surface.stations) == 2
    assert workbench.station_table.selection == []

    workbench.station_table.selection = [0, 1]
    workbench._delete_station(None)
    assert len(surface.stations) == 2
    assert "at least two stations" in workbench.status.object


def test_workbench_exposes_project_loads_and_spar_sizing():
    workbench = Workbench()
    view = workbench.view()
    workbench.loads_spar_height.value = 0.027
    workbench.loads_allowable.value = 420.0
    assert workbench.project.structure.spar_height == pytest.approx(0.027)
    assert workbench.project.structure.allowable_stress == pytest.approx(420e6)
    restored = type(workbench.project).from_json(workbench.project.to_json())
    assert restored.structure == workbench.project.structure
    workbench.run_loads_analysis()

    assert "Loads & structures" in view.main[0]._names
    assert workbench.status.alert_type == "success"
    result = workbench._last_loads_result
    assert result is not None
    assert result["mode"] == "direct"
    assert result["envelope"] is None
    assert result["surface"] == "Main wing"
    assert result["span_load"].root_moment > 0
    assert result["sizing"]["cap_area"] > 0
    assert result["deflection"]["tip_deflection"] > 0
    span_load = pd.read_csv(workbench._download_loads_csv())
    assert {
        "project", "case", "surface", "load_factor", "y_m",
        "net_aerodynamic_load_N_per_m", "shear_N", "bending_moment_N_m",
        "deflection_m",
    } <= set(span_load.columns)
    assert len(span_load) == len(result["span_load"].y)
    assert not workbench.loads_download.disabled
    assert len(workbench.loads_plots.object.axes) >= 4
    assert workbench.loads_mode.value == "Direct RC design case"
    assert not workbench.loads_cl_max.visible
    assert "required area of each cap" in workbench.python_output.object
    assert "project.structure" in workbench.python_output.object
    assert "spar_height=" not in workbench.python_output.object
    generated = workbench.python_output.object.split("```python\n", 1)[1].rsplit("```", 1)[0]
    compile(generated, "generated_flightlab_analysis.py", "exec")
    plt.close("all")


def test_workbench_exports_analysis_and_propulsion_sweeps_as_csv():
    workbench = Workbench()
    assert workbench.analysis_download.disabled
    assert workbench.propulsion_download.disabled

    airfoil = pd.read_csv(workbench._download_airfoil_csv())
    assert {
        "airfoil", "reynolds_number", "forced_transition_x_c", "alpha_deg",
        "cl", "cd", "cm_about_c4", "upper_transition_x_c",
        "lower_transition_x_c", "model_confidence",
    } <= set(airfoil.columns)
    assert len(airfoil) == 121
    assert not workbench.airfoil_download.disabled

    workbench.run_integrated_analysis()
    polar = pd.read_csv(workbench._download_analysis_csv())
    assert {
        "project", "case", "reference_speed_m_s", "altitude_m", "mass_kg",
        "alpha_deg", "CL", "CD", "CD_profile_body", "CD_induced",
        "Cm_about_cg", "L_over_D", "trim_control_deg",
    } <= set(polar.columns)
    assert len(polar) == 21
    assert polar["project"].nunique() == 1
    assert np.isfinite(polar[["CL", "CD", "L_over_D"]].to_numpy()).all()
    assert not workbench.analysis_download.disabled
    assert workbench.analysis_download.filename.endswith("_polar.csv")
    spanwise = pd.read_csv(workbench._download_spanwise_csv())
    assert {
        "project", "case", "surface", "mass_kg", "alpha_deg", "y_m",
        "strip_width_m", "chord_m", "section_cl", "span_loading_c_cl_m",
        "section_cl_max", "section_cl_at_estimated_stall",
        "section_cm", "reynolds_number",
    } <= set(spanwise.columns)
    assert {"Main wing", "Horizontal tail"} <= set(spanwise["surface"])
    assert not workbench.analysis_span_download.disabled

    workbench.propulsion_speed_min.value = 6.0
    workbench.propulsion_speed_max.value = 35.0
    workbench.propulsion_speed_points.value = 30
    assert "speed=propulsion_speed" in workbench.python_output.object
    assert "np.linspace(6, 35, 30)" in workbench.python_output.object
    assert "analyze_dynamic_stability(\n    project, case, ns=28, nc=4" in workbench.python_output.object
    workbench.run_propulsion_analysis()
    sweep = pd.read_csv(workbench._download_propulsion_csv())
    assert {
        "project", "case", "altitude_m", "speed_true_m_s",
        "thrust_available_N", "drag_required_N", "battery_current_A",
        "power_electrical_W", "power_shaft_W", "power_useful_W",
        "efficiency_total", "outside_propeller_data",
    } <= set(sweep.columns)
    assert len(sweep) == 30
    assert sweep["speed_true_m_s"].iloc[0] == pytest.approx(6.0)
    assert sweep["speed_true_m_s"].iloc[-1] == pytest.approx(35.0)
    assert np.all(np.diff(sweep["speed_true_m_s"]) > 0.0)
    assert (sweep["thrust_available_N"] > 0.0).all()
    assert not workbench.propulsion_download.disabled
    assert workbench.propulsion_download.filename.endswith("_speed_sweep.csv")

    workbench.project_name.value = "Changed after analysis"
    assert workbench.airfoil_download.disabled
    assert workbench.analysis_download.disabled
    assert workbench.analysis_span_download.disabled
    assert workbench.loads_download.disabled
    assert workbench.propulsion_download.disabled
    plt.close("all")


def test_table_coercion_accepts_numeric_strings_and_names_the_bad_field():
    rows = _coerce_records(
        pd.DataFrame([{"name": "wing", "chord": "0.42"}]),
        [("name", str, False), ("chord", float, False)],
        "Station",
    )
    assert rows == [{"name": "wing", "chord": 0.42}]

    with pytest.raises(ValueError, match=r"Station row 1, chord.*'wide'"):
        _coerce_records(
            pd.DataFrame([{"name": "wing", "chord": "wide"}]),
            [("name", str, False), ("chord", float, False)],
            "Station",
        )


def test_workbench_naca_generation_and_natural_transition_controls():
    workbench = Workbench()
    workbench.naca_code.value = "4415"
    workbench._add_naca_section(None)
    assert workbench.airfoil_select.value == "naca4415"
    assert "naca4415" in workbench.station_table.editors["airfoil"]["values"]

    frame = workbench.case_table.value.copy()
    frame.loc[0, ["transition", "xtr_upper", "xtr_lower"]] = ["forced", 0.4, 0.5]
    workbench.case_table.value = frame
    assert workbench.project.cases[0].xtr_upper == pytest.approx(0.4)
    assert workbench.project.cases[0].xtr_lower == pytest.approx(0.5)

    frame = workbench.case_table.value.copy()
    frame.loc[0, "transition"] = "natural"
    workbench.case_table.value = frame
    assert workbench.project.cases[0].xtr_upper == pytest.approx(1.0)
    assert workbench.project.cases[0].xtr_lower == pytest.approx(1.0)
    assert workbench.case_table.value.loc[0, "xtr_upper"] == pytest.approx(1.0)
    plt.close("all")


def test_flight_case_edits_do_not_redraw_case_independent_figures(monkeypatch):
    workbench = Workbench()
    old_body_re = float(workbench.body_results.value.loc[0, "Re"])
    table_events = []
    workbench.case_table.param.watch(lambda event: table_events.append(event), "value")

    def unexpected_refresh():
        raise AssertionError("a flight-case edit redrew case-independent figures")

    monkeypatch.setattr(workbench, "_refresh_geometry", unexpected_refresh)
    monkeypatch.setattr(workbench, "_refresh_component_summaries", unexpected_refresh)
    frame = workbench.case_table.value.copy()
    frame.loc[0, "speed"] *= 1.1
    workbench.case_table.value = frame

    assert workbench.project.cases[0].speed == pytest.approx(frame.loc[0, "speed"])
    assert float(workbench.body_results.value.loc[0, "Re"]) > old_body_re
    assert workbench.status.object == "Flight cases updated."
    # One event is the edit itself; a second would be a disruptive full-table reload.
    assert len(table_events) == 1
    plt.close("all")
