"""Integrated multi-station project analyses."""

from copy import deepcopy

import pytest

from flightlab import loads, stability
from flightlab.project import PropulsorSetup, example_project
from flightlab.project_analysis import (
    TrimNotPossibleError,
    aircraft_polar,
    analyze,
    analyze_dynamic_stability,
    analyze_propulsion,
    analyze_structure,
    run_design_point,
    surface_section_cl_max,
    trim,
)
from flightlab.propulsion import PropellerModel


def test_full_station_geometry_changes_vlm_even_with_same_root_and_tip():
    baseline = example_project()
    changed = deepcopy(baseline)
    # Keep the root and tip fixed but move an interior breakpoint aft.
    changed.reference_surface.stations[2].x_le += 0.15

    a = analyze(baseline, alpha=3.0)
    b = analyze(changed, alpha=3.0)

    assert b.CL != pytest.approx(a.CL, abs=1e-4)
    assert b.Cm != pytest.approx(a.Cm, abs=1e-4)


def test_span_load_can_select_a_named_surface_from_project_solution():
    project = example_project()
    solution = analyze(project, alpha=3.0, ns=18, nc=3)
    aircraft = project.equivalent_aircraft()
    wing = loads.span_load(
        aircraft, mass=project.total_mass(), V=project.case().speed,
        solution=solution, surface="Main wing",
    )
    tail = loads.span_load(
        aircraft, mass=project.total_mass(), V=project.case().speed,
        solution=solution, surface="Horizontal tail",
    )

    assert tail.y[-1] < wing.y[-1]
    assert tail.root_moment != pytest.approx(wing.root_moment)

    with pytest.raises(ValueError, match="unknown solution surface"):
        loads.span_load(
            aircraft, mass=project.total_mass(), V=project.case().speed,
            solution=solution, surface="missing",
        )


def test_design_point_integrates_mass_trim_stability_and_drag():
    result = run_design_point(example_project())

    assert result.trim.converged
    assert result.mass_properties.mass == pytest.approx(0.8)
    assert result.trim.static_margin > 0
    assert result.buildup.CD_profile_body > 0
    assert result.trim.solution.CD_i > 0
    assert result.CD_total == pytest.approx(
        result.buildup.CD_profile_body + result.trim.solution.CD_i
    )
    assert {row.name for row in result.buildup.rows} >= {
        "Main wing", "Horizontal tail", "Vertical tail", "fuselage"
    }


def test_named_cases_share_aircraft_but_change_required_lift():
    project = example_project()
    cruise = run_design_point(project, project.case("Cruise"))
    maneuver = run_design_point(project, project.case("Maneuver"))
    assert maneuver.trim.CL_required > cruise.trim.CL_required


def test_aircraft_polar_exposes_lift_drag_ld_and_pitching_moment():
    polar = aircraft_polar(example_project(), alpha=[-2, 0, 2], ns=14, nc=3)
    assert polar.CL.shape == (3,)
    assert polar.CD.shape == (3,)
    assert polar.LD.shape == (3,)
    assert polar.Cm.shape == (3,)
    assert polar.CL[2] > polar.CL[0]
    assert (polar.CD >= polar.CD_profile).all()


def test_project_section_limits_and_structural_case_use_station_geometry():
    project = example_project()
    solution = analyze(project, alpha=3.0, ns=16, nc=3)
    main_wing = project.surface_named("Main wing")
    local_limit = surface_section_cl_max(project, main_wing, solution, project.case())
    assert local_limit.shape == solution.surface("Main wing").cl.shape
    assert (local_limit > solution.surface("Main wing").cl).all()

    structure = analyze_structure(
        project, project.case(), surface="Main wing", load_factor=3.8,
        speed=project.case().speed, ns=16, nc=3,
    )
    assert structure.span_load.root_moment > 0
    assert structure.sizing["cap_area"] > 0
    assert structure.deflection["tip_deflection"] > 0


def test_propulsion_and_dynamic_stability_are_project_analyses():
    project = example_project()
    power = analyze_propulsion(project, speed=[8.0, 12.0, 16.0])
    assert power.thrust_available.shape == (3,)
    assert power.drag_required.shape == (3,)
    assert power.operating_point.current > 0
    assert power.efficiency_total.shape == (3,)
    assert power.power_electrical.shape == (3,)

    modes = analyze_dynamic_stability(project, ns=12, nc=3)
    assert len(modes.longitudinal) > 0
    assert len(modes.lateral) > 0
    assert modes.warnings
    assert modes.body_increments["Cm_alpha"] > 0
    assert modes.body_increments["Cn_beta"] != 0
    assert modes.propulsion_increments.dT_dV < 0


def test_multiple_propulsors_share_one_sagging_battery_bus():
    single = example_project()
    one = analyze_propulsion(single, speed=[12.0])

    twin = example_project()
    twin.propulsion.propulsors = [
        PropulsorSetup("Left", x=0.05, y=-0.30),
        PropulsorSetup("Right", x=0.05, y=0.30),
    ]
    two = analyze_propulsion(twin, speed=[12.0])

    assert len(two.operating_point.propulsors) == 2
    assert two.operating_point.current > one.operating_point.current
    assert two.operating_point.bus_voltage < one.operating_point.bus_voltage
    assert one.operating_point.thrust < two.operating_point.thrust < 2 * one.operating_point.thrust
    assert two.rpm.shape == (1, 2)


def test_empirical_body_pitch_derivative_moves_neutral_point_forward():
    aircraft = example_project().equivalent_aircraft()
    surfaces_only = stability.neutral_point(aircraft, 12.0, 1400.0, ns=14, nc=3)
    corrected = stability.neutral_point(
        aircraft, 12.0, 1400.0, ns=14, nc=3, include_body=True
    )
    assert corrected["body_increment"] > 0
    assert corrected["x_np"] < surfaces_only["x_np"]


def test_project_propeller_model_accepts_measured_coefficient_rows():
    points = []
    for rpm, scale in ((4000.0, 1.0), (6000.0, 0.98)):
        for J, CT, CP in ((0.0, 0.12, 0.055), (0.2, 0.105, 0.050), (0.5, 0.065, 0.035)):
            points.append({"rpm": rpm, "J": J, "CT": CT * scale, "CP": CP * scale})
    model = PropellerModel.from_points("measured", 0.254, 0.178, points)

    assert model.has_static
    assert model.J_range == pytest.approx((0.2, 0.5))
    assert model.CT(0.3, 80.0) > 0
    assert model.CP(0.3, 80.0) > 0


def test_elevator_geometry_changes_pitching_moment_and_trims_within_limits():
    project = example_project()
    tail = project.surface("tail")
    tail.trim_control = "elevator"
    tail.control_hinge_fraction = 0.75

    up = analyze(project, alpha=3.0, trim_deflection=-10.0, ns=14, nc=6)
    down = analyze(project, alpha=3.0, trim_deflection=10.0, ns=14, nc=6)
    assert down.Cm < up.Cm
    result = trim(project, ns=14, nc=6)
    assert tail.control_min_deg <= result.trim_deflection <= tail.control_max_deg


def test_trim_reports_required_deflection_outside_control_limits():
    project = example_project()
    tail = project.surface("tail")
    tail.trim_control = "elevator"
    tail.control_min_deg = -0.1
    tail.control_max_deg = 0.1

    with pytest.raises(TrimNotPossibleError, match="outside the entered"):
        trim(project, ns=14, nc=6)
