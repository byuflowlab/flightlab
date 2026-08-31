"""Smoke tests for the optional task-focused notebook interfaces."""

import pytest


pytest.importorskip("ipywidgets")

import matplotlib.pyplot as plt

from flightlab import explorers


@pytest.mark.parametrize("name", explorers.__all__)
def test_explorer_builds_and_runs_its_default_case(name, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    app = getattr(explorers, name)()
    generated_code = app.children[1].children[1].children[1].value

    assert "did not produce a valid result" not in generated_code
    assert "from flightlab" in generated_code
    plt.close("all")
