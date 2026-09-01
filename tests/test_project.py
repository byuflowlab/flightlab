"""General aircraft-project model and persistence."""

from dataclasses import replace

import numpy as np
import pytest

from flightlab import foil
from flightlab.project import (
    AircraftProject, AirfoilDefinition, MassItem, PropulsorSetup, ReferenceGeometry,
    example_project,
)
from flightlab import stability


def test_example_project_is_valid_and_multi_station():
    project = example_project()
    assert project.validate() == []
    assert len(project.reference_surface.stations) == 4
    assert project.reference_quantities()[0] > 0
    assert project.reference_quantities()[1] > 0
    assert project.reference_quantities()[2] > 0
    assert project.total_mass() == pytest.approx(0.8)


def test_project_json_round_trip_preserves_nested_design():
    original = example_project()
    original.airfoils["custom"] = AirfoilDefinition.from_section(foil.naca4("4412"))
    original.structure.surface = "Main wing"
    original.structure.spar_height = 0.027
    original.structure.allowable_stress = 420e6

    restored = AircraftProject.from_json(original.to_json())

    assert restored.to_dict() == original.to_dict()
    assert restored.reference_surface.stations[2].airfoil == "sd7037"
    assert restored.propulsion.propulsors[0].motor == original.propulsion.propulsors[0].motor
    assert restored.structure == original.structure
    assert restored.section("custom").thickness == pytest.approx(
        original.section("custom").thickness
    )


def test_uploaded_selig_dat_text_does_not_need_a_temporary_file():
    text = """tiny foil
1.0 0.0
0.5 0.08
0.0 0.0
0.5 -0.08
1.0 0.0
"""
    section = foil.from_dat_text(text, "tiny")
    assert section.name == "tiny foil"
    assert section.coordinates.shape == (5, 2)
    assert np.max(section.z) == pytest.approx(0.08)


def test_equivalent_aircraft_is_only_the_handbook_adapter():
    project = example_project()
    aircraft = project.equivalent_aircraft()
    assert aircraft.wing.area == pytest.approx(project.primary_horizontal_surface.area)
    assert aircraft.wing.span == pytest.approx(project.primary_horizontal_surface.span)
    assert aircraft.components == project.components()
    assert "Equivalent single trapezoid" in aircraft.wing.notes


def test_geometry_attached_mass_uses_surface_and_body_distribution():
    project = example_project()
    components = {component.name: component for component in project.components()}
    wing = components["wing structure estimate"]
    fuselage = components["fuselage structure estimate"]

    assert wing.mass == pytest.approx(0.24)
    assert wing.Ixx_cg > 0
    assert fuselage.mass == pytest.approx(0.18)
    assert fuselage.Iyy_cg > 0
    assert stability.mass_properties(project.components()).mass == pytest.approx(0.8)


def test_density_can_set_solid_surface_mass():
    project = example_project()
    project.masses = [
        MassItem(
            "solid wing", None, distributed="surface_volume",
            attached_to="Main wing", density=30.0,
        )
    ]
    component = project.components()[0]
    assert component.mass > 0
    assert component.Ixx_cg > 0


def test_reference_geometry_is_independent_of_surface_purpose():
    project = example_project()
    second_wing = project.surfaces[1]
    second_wing.purpose = "wing"
    project.reference = ReferenceGeometry(
        mode="selected_surfaces", surfaces=[project.surfaces[0].name, second_wing.name]
    )

    area, span, chord = project.reference_quantities()
    selected = [project.surfaces[0], second_wing]
    assert area == pytest.approx(sum(surface.area for surface in selected))
    assert span == pytest.approx(max(surface.span for surface in selected))
    assert chord > 0
    assert project.validate() == []


def test_measured_propulsion_components_are_owned_by_saved_project():
    project = example_project()
    project.motors["M1000"] = replace(
        project.motors["M1000"], resistance=0.087, current_no_load=0.73,
        provisional=False, notes="Measured on course thrust stand",
    )
    project.batteries["B3S1300"] = replace(
        project.batteries["B3S1300"], cell_resistance=0.009, provisional=False,
    )

    restored = AircraftProject.from_json(project.to_json())
    propulsor = restored.propulsion.propulsors[0]
    assert restored.motor(propulsor).resistance == pytest.approx(0.087)
    assert restored.motor(propulsor).provisional is False
    assert restored.battery().cell_resistance == pytest.approx(0.009)


def test_shared_battery_and_multiple_propulsors_supply_positioned_masses():
    project = example_project()
    project.propulsion.battery_x = 0.12
    project.propulsion.propulsors = [
        PropulsorSetup("Left propulsor", x=0.05, y=-0.30),
        PropulsorSetup("Right propulsor", x=0.05, y=0.30),
    ]
    components = {component.name: component for component in project.propulsion_components()}

    assert list(name for name in components if name.startswith("Propulsion battery")) == [
        "Propulsion battery (B3S1300)"
    ]
    assert components["Left propulsor hardware"].y == pytest.approx(-0.30)
    assert components["Right propulsor hardware"].y == pytest.approx(0.30)
