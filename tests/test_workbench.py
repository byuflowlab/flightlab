"""Optional Panel workbench smoke tests."""

import pytest


pytest.importorskip("panel")

import matplotlib.pyplot as plt

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
