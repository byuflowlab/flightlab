"""Orientation is derived from the stations; every surface is eligible for every role."""

from copy import deepcopy

import pytest

from flightlab import drag
from flightlab.project import AircraftProject, LiftingSurface, SurfaceStation, blank_project, example_project


def test_vertical_is_read_from_the_stations_not_entered():
    project = blank_project()
    wing, tail, fin = project.surfaces
    assert not wing.is_vertical and not tail.is_vertical and fin.is_vertical
    assert not hasattr(fin, "orientation")
    # A bare surface with one station falls back to its purpose.
    assert LiftingSurface("stub", "fin", "fixed", False, [SurfaceStation(0, 0, 0, 0.1)]).is_vertical
    assert not LiftingSurface("stub", "other", "fixed", True, [SurfaceStation(0, 0, 0, 0.1)]).is_vertical


def test_files_saved_with_an_orientation_label_still_load():
    data = blank_project().to_dict()
    for item in data["surfaces"]:
        item["orientation"] = "vertical" if item["purpose"] == "fin" else "horizontal"
    restored = AircraftProject.from_dict(data)
    assert [surface.name for surface in restored.surfaces] == [s["name"] for s in data["surfaces"]]
    assert "orientation" not in restored.to_dict()["surfaces"][0]


def test_any_surface_may_be_reference_trim_or_spar_surface():
    project = example_project()
    fin = next(surface for surface in project.surfaces if surface.purpose == "fin")
    project.reference.mode = "surface"
    project.reference.surface = fin.name
    project.structure.surface = fin.name
    fin.trim_control = "whole_surface"
    errors = [issue.message for issue in project.validate() if issue.level == "error"]
    assert errors == []
    assert fin in project.trim_surfaces
    assert project.primary_surface is fin


def test_handbook_adapter_dihedral_and_fin_flag_come_from_geometry():
    project = blank_project()
    aircraft = project.equivalent_aircraft()
    assert aircraft.wing.vertical is False
    assert aircraft.vtail is not None and aircraft.vtail.vertical is True
    assert aircraft.vtail.dihedral_deg == pytest.approx(90.0)
    assert 0.0 < aircraft.wing.dihedral_deg < 10.0
    # A drag buildup still includes the fin's wetted area with no label anywhere.
    rows = {row.name for row in drag.buildup(aircraft, V=15.0).rows}
    assert {"wing", "htail", "vtail"} <= rows


def test_fin_slot_follows_purpose_when_no_surface_is_labelled_fin():
    project = blank_project()
    fin = project.surfaces[2]
    fin.purpose = "other"
    assert project.equivalent_aircraft().vtail is None
    fin.purpose = "fin"
    assert project.equivalent_aircraft().vtail is not None
