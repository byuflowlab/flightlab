"""Regression tests for the interfaces used directly by the draft-5 homework."""

import numpy as np
import pytest

import flightlab
from flightlab import (
    airfoil,
    atmos,
    catalog,
    drag,
    geom,
    loads,
    performance,
    plot,
    propulsion,
    stability,
    wing,
)
from flightlab.fleet import ASW27


def test_student_import_styles_work():
    """Both import styles shown in the README should expose real modules."""
    assert flightlab.__version__ == "0.9.0"
    assert flightlab.atmos is atmos
    assert all(
        module is not None
        for module in (
            airfoil,
            catalog,
            drag,
            geom,
            loads,
            performance,
            plot,
            propulsion,
            stability,
            wing,
        )
    )


def test_every_named_draft5_homework_interface_exists():
    required_callables = {
        atmos: ("at", "density", "eas_to_tas", "speed_of_sound"),
        airfoil: ("polar", "table"),
        geom: ("chord_at", "resolve"),
        wing: ("CL_max", "stall_speed", "trim_to_weight"),
        drag: ("buildup", "drag_divergence_mach", "polar", "wave_drag"),
        stability: (
            "derivatives", "mass_properties", "modes", "neutral_point",
            "tail_volume", "trim",
        ),
        propulsion: (
            "operating_point", "propeller_model", "rotor_hover", "sweep_speed",
            "turbofan_thrust",
        ),
        performance: (
            "endurance_breguet", "endurance_electric", "glide",
            "range_breguet", "range_electric", "speeds",
        ),
        loads: (
            "gust_load_factor", "span_load", "spar_sizing", "tip_deflection",
            "vn_diagram",
        ),
        plot: ("mode_response", "span_load", "span_loading", "stall_margin"),
    }
    for module, names in required_callables.items():
        for name in names:
            assert callable(getattr(module, name)), f"missing {module.__name__}.{name}"

    for name in ("RC1", "C172", "B787", "ASW27", "F16", "DC3", "JobyS4"):
        assert getattr(flightlab.fleet, name) is not None
    assert "B3S1300" in catalog.BATTERIES
    assert "P10x7" in catalog.PROPELLERS


def test_capability_browser_exposes_questions_inputs_outputs_and_examples(capsys):
    flightlab.show_tools("wings")
    text = capsys.readouterr().out
    assert "Finite-wing loading and stall" in text
    assert "Inputs:" in text
    assert "Named outputs:" in text
    assert "Model limits:" in text
    assert "wing.trim_to_weight" in text

    assert tuple(flightlab.TOPICS) == (
        "atmosphere", "airfoils", "wings", "drag", "stability",
        "propulsion", "performance", "loads",
    )
    assert "from flightlab import atmos" in flightlab.example("atmos")


@pytest.mark.parametrize("topic", flightlab.TOPICS)
def test_every_discovery_starter_example_executes(topic):
    """The discovery layer must never advertise stale or pseudocode APIs."""
    exec(flightlab.example(topic), {})


def test_hw8_gust_load_accepts_a_speed_sweep():
    V = np.linspace(0.0, 79.17, 40)
    result = loads.gust_load_factor(
        ASW27, V=V, gust=15.24, mass=500.0, CL_alpha=5.2, altitude=0.0
    )
    scalar_results = np.array(
        [
            loads.gust_load_factor(
                ASW27, V=float(v), gust=15.24, mass=500.0,
                CL_alpha=5.2, altitude=0.0,
            )
            for v in V
        ]
    )
    assert result.shape == V.shape
    assert result[0] == pytest.approx(1.0)
    assert result == pytest.approx(scalar_results)


def test_hw2_wave_drag_broadcasts_over_a_design_grid():
    thickness = np.linspace(0.09, 0.14, 6)[:, None]
    sweep = np.linspace(20.0, 40.0, 5)[None, :]
    result = drag.wave_drag(
        0.85, thickness=thickness, CL=0.50, sweep_c4_deg=sweep
    )
    assert result.shape == (6, 5)
    assert np.all(result >= 0.0)
    assert result[-1, 0] > result[0, -1]


def test_hw6_turbofan_model_uses_mach_and_accepts_arrays():
    mach = np.array([0.20, 0.50, 0.90])
    thrust = propulsion.turbofan_thrust(570e3, altitude=11_900.0, mach=mach)
    assert thrust.shape == mach.shape
    assert np.all(np.diff(thrust) < 0.0)


def test_hw6_measured_electrical_parameters_are_component_inputs():
    starter_motor = catalog.MOTORS["M1000"]
    starter_battery = catalog.BATTERIES["B3S1300"]
    measured_motor = starter_motor.with_measurements(
        resistance=0.085,
        current_no_load=0.78,
        no_load_voltage=11.1,
    )
    measured_battery = starter_battery.with_measurements(cell_resistance=0.009)

    assert measured_motor.resistance == pytest.approx(0.085)
    assert measured_motor.current_no_load == pytest.approx(0.78)
    assert measured_battery.cell_resistance == pytest.approx(0.009)
    assert not measured_motor.provisional
    assert not measured_battery.provisional
    assert starter_motor.provisional and starter_battery.provisional

    starter = propulsion.operating_point(
        starter_motor, "apce_10x7", starter_battery,
        V=9.1, altitude=1400.0, soc=0.9, esc="ESC30",
    )
    measured = propulsion.operating_point(
        measured_motor, "apce_10x7", measured_battery,
        V=9.1, altitude=1400.0, soc=0.9, esc="ESC30",
    )
    assert measured.current != pytest.approx(starter.current)

    with pytest.raises(ValueError, match="motor parameters must be positive"):
        starter_motor.with_measurements(resistance=-0.1, current_no_load=0.78)


def test_structural_inputs_fail_with_course_specific_messages():
    sl = loads.span_load(ASW27, mass=500.0, n=1.0, V=29.0, ns=20)
    with pytest.raises(ValueError, match="one value at each semispan station"):
        loads.span_load(
            ASW27, mass=500.0, n=1.0, V=29.0, ns=20,
            relief=np.ones(3),
        )
    with pytest.raises(ValueError, match="positive, finite"):
        loads.tip_deflection(sl, EI=-1.0)
