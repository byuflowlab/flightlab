"""Keep the staged notebook sequence complete and mechanically runnable."""

import json
from pathlib import Path

import pytest


NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks"
SEQUENCE = {
    "01_flight_condition_and_drag.ipynb": "flight_condition_explorer",
    "02_airfoil_analysis.ipynb": "airfoil_explorer",
    "03_wing_design.ipynb": "wing_design_explorer",
    "04_stability_and_trim.ipynb": "stability_explorer",
    "05_complete_aircraft_drag.ipynb": "drag_explorer",
    "06_propulsion_matching.ipynb": "propulsion_explorer",
    "07_aircraft_performance.ipynb": "performance_explorer",
    "08_loads_and_structures.ipynb": "loads_explorer",
}


def _read_notebook(name):
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def test_numbered_notebook_sequence_is_complete():
    actual = {path.name for path in NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")}
    assert actual == set(SEQUENCE)


@pytest.mark.parametrize(("name", "explorer"), SEQUENCE.items())
def test_notebook_uses_its_task_explorer_and_has_valid_code(name, explorer):
    notebook = _read_notebook(name)
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert explorer in code
    compile(code, str(NOTEBOOK_DIR / name), "exec")


def test_reference_notebook_is_valid_json():
    notebook = _read_notebook("reference_toolbox.ipynb")
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
