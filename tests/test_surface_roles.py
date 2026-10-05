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
    # A bare surface with one station has no direction yet.
    assert not LiftingSurface("stub", "fixed", False, [SurfaceStation(0, 0, 0, 0.1)]).is_vertical
    assert not hasattr(fin, "purpose")


def test_files_saved_with_orientation_and_purpose_labels_still_load():
    data = blank_project().to_dict()
    assert "purpose" not in data["surfaces"][0]
    for item, purpose in zip(data["surfaces"], ("wing", "tail", "fin")):
        item["orientation"] = "vertical" if purpose == "fin" else "horizontal"
        item["purpose"] = purpose
    restored = AircraftProject.from_dict(data)
    assert [surface.name for surface in restored.surfaces] == [s["name"] for s in data["surfaces"]]
    saved = restored.to_dict()["surfaces"][0]
    assert "orientation" not in saved and "purpose" not in saved


def test_any_surface_may_be_reference_trim_or_spar_surface():
    project = example_project()
    fin = next(surface for surface in project.surfaces if surface.is_vertical)
    project.reference.mode = "surface"
    project.reference.surface = fin.name
    project.structure.surface = fin.name
    fin.trim_control = "whole_surface"
    errors = [issue.message for issue in project.validate() if issue.level == "error"]
    assert errors == []
    assert fin in project.trim_surfaces
    assert project.primary_surface is fin


def test_handbook_adapter_carries_only_the_reference_surface_and_bodies():
    project = blank_project()
    aircraft = project.equivalent_aircraft()
    assert aircraft.wing.area == pytest.approx(project.reference_surface.area)
    assert 0.0 < aircraft.wing.dihedral_deg < 10.0
    assert aircraft.htail is None and aircraft.vtail is None
    rows = {row.name for row in drag.buildup(aircraft, V=15.0).rows}
    assert rows == {"wing", "fuselage"}


def test_primary_surface_is_the_reference_surface_else_the_first_listed():
    project = blank_project()
    assert project.primary_surface is project.reference_surface
    project.reference.mode = "manual"
    assert project.primary_surface is project.surfaces[0]
    project.surfaces.reverse()
    assert project.primary_surface is project.surfaces[0]
