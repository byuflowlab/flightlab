"""Optional Panel workbench smoke tests."""

import pytest


pytest.importorskip("panel")

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from flightlab.workbench import Workbench, _coerce_records


def test_workbench_builds_and_runs_integrated_analysis():
    workbench = Workbench()
    view = workbench.view()
    workbench.run_integrated_analysis()

    assert view.title == "FlightLab Aircraft Workbench"
    assert workbench._last_result is not None
    assert workbench._last_result.trim.converged
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
    assert "M1000" in set(workbench.motor_table.value["key"])
    assert len(workbench.propulsor_table.value) == 1
    assert "Propulsion battery (B3S1300)" in set(workbench.mass_results.value["component"])
    # Mass markers plus the propulsor point/thrust vector are drawn with the geometry.
    assert len(workbench.geometry_plot.object.axes[0].collections) >= 3
    plt.close("all")


def test_workbench_exposes_project_loads_and_spar_sizing():
    workbench = Workbench()
    view = workbench.view()
    workbench.run_loads_analysis()

    assert "Loads & structures" in view.main[0]._names
    assert workbench.status.alert_type == "success"
    result = workbench._last_loads_result
    assert result is not None
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
    assert "required area of each cap" in workbench.python_output.object
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
    assert len(polar) == 13
    assert polar["project"].nunique() == 1
    assert np.isfinite(polar[["CL", "CD", "L_over_D"]].to_numpy()).all()
    assert not workbench.analysis_download.disabled
    assert workbench.analysis_download.filename.endswith("_polar.csv")
    spanwise = pd.read_csv(workbench._download_spanwise_csv())
    assert {
        "project", "case", "surface", "mass_kg", "alpha_deg", "y_m",
        "strip_width_m", "chord_m", "section_cl", "span_loading_c_cl_m",
        "section_cm", "reynolds_number",
    } <= set(spanwise.columns)
    assert {"Main wing", "Horizontal tail"} <= set(spanwise["surface"])
    assert not workbench.analysis_span_download.disabled

    workbench.propulsion_speed_min.value = 6.0
    workbench.propulsion_speed_max.value = 35.0
    workbench.propulsion_speed_points.value = 30
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
