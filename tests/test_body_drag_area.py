"""Bodies may charge an entered drag area instead of a correlation."""

import math

import numpy as np

from flightlab import drag
from flightlab.fleet import Aircraft, Body
from flightlab.project import BodyDefinition, example_project


def _gear_only(*bodies):
    return Aircraft(name="gear-only", label="gear", aircraft_class="rc", bodies=tuple(bodies))


def test_drag_area_body_is_charged_directly_and_reports_frontal_cd():
    leg = Body("gear leg", length=0.20, diameter=0.010, count=2,
               drag_model="drag_area", drag_area=0.0018)
    result = drag.buildup(_gear_only(leg), V=15.0)
    row, = result.rows
    assert row.kind == "drag_area"
    assert math.isclose(row.f, 2 * 0.0018)
    # frontal area = count * d * L = 0.004 m2, so CD on frontal area = 0.9
    assert math.isclose(row.cd_frontal, 0.90)
    assert np.isnan(row.FF) and np.isnan(row.cf)
    assert math.isclose(result.f_components, 0.0036)


def test_drag_area_body_without_cross_section_still_counts():
    pod = Body("camera pod", length=0.05, drag_model="drag_area", drag_area=0.0005)
    result = drag.buildup(_gear_only(pod), V=15.0)
    row, = result.rows
    assert math.isclose(row.f, 0.0005)
    assert np.isnan(row.cd_frontal)
    assert result.skipped == ()


def test_drag_area_body_with_no_value_is_skipped_not_silently_zero():
    pod = Body("pod", length=0.05, drag_model="drag_area")
    fuselage = Body("fuselage", length=0.9, diameter=0.1)
    result = drag.buildup(_gear_only(fuselage, pod), V=15.0)
    assert result.skipped == ("pod",)


def test_legacy_saved_body_models_convert_to_the_same_drag_area():
    saved = {"name": "gear leg", "length": 0.20, "diameter": 0.010, "count": 2,
             "drag_model": "bluff_round_member"}
    body = BodyDefinition.from_saved(saved)
    assert body.drag_model == "drag_area"
    assert math.isclose(body.drag_area, 0.90 * 0.010 * 0.20)  # per item
    legacy = Body("gear leg", length=0.20, diameter=0.010, count=2, drag_model="bluff_round_member")
    old = drag.buildup(_gear_only(legacy), V=15.0).f_components
    new = drag.buildup(_gear_only(body.to_body()), V=15.0).f_components
    assert math.isclose(old, new)
    for legacy_model, cd in (("faired_member", 0.25), ("streamlined_strut", 0.10)):
        converted = BodyDefinition.from_saved({**saved, "drag_model": legacy_model})
        assert math.isclose(converted.drag_area, cd * 0.010 * 0.20)
    unchanged = BodyDefinition.from_saved({"name": "fuselage", "length": 0.9, "diameter": 0.1})
    assert unchanged.drag_model == "streamlined_body" and unchanged.drag_area is None


def test_project_validation_requires_a_drag_area_for_that_model():
    project = example_project()
    project.bodies.append(BodyDefinition("wheel", 0.06, diameter=0.02, drag_model="drag_area"))
    messages = [issue.message for issue in project.validate() if issue.level == "error"]
    assert any("wheel" in message and "drag area" in message for message in messages)
    project.bodies[-1].drag_area = 0.0004
    assert not [issue for issue in project.validate() if issue.level == "error" and "wheel" in issue.message]
    project.bodies[-1].drag_model = "faired_member"
    assert any("unknown drag model" in issue.message for issue in project.validate())


def test_project_round_trip_keeps_drag_area_without_a_format_bump():
    project = example_project()
    project.bodies.append(BodyDefinition("wheel", 0.06, diameter=0.02, count=2,
                                         drag_model="drag_area", drag_area=0.0004))
    restored = type(project).from_dict(project.to_dict())
    wheel = restored.body_named("wheel")
    assert wheel.drag_model == "drag_area" and math.isclose(wheel.drag_area, 0.0004)
    rows = {row.name: row for row in drag.buildup(restored.equivalent_aircraft(), V=15.0).rows}
    assert math.isclose(rows["wheel"].f, 0.0008)
