"""Static checks for the no-install student launch path."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from flightlab.__main__ import launch_workbench


ROOT = Path(__file__).parents[1]
SETUP = ROOT / "student_setup"


def test_release_and_launcher_fallbacks_are_exact_git_commits():
    release = (SETUP / "release.txt").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", release)

    macos = (SETUP / "macos" / "Start FlightLab.command").read_text()
    windows = (SETUP / "windows" / "Start FlightLab.cmd").read_text()
    notebook = json.loads((ROOT / "notebooks" / "hw1_starter.ipynb").read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    fallback_patterns = (
        (macos, r'FLIGHTLAB_DEFAULT_COMMIT="([0-9a-f]{40})"'),
        (windows, r"FLIGHTLAB_DEFAULT_COMMIT=([0-9a-f]{40})"),
        (notebook_source, r'_fallback_commit = "([0-9a-f]{40})"'),
    )
    fallbacks = [re.search(pattern, source).group(1) for source, pattern in fallback_patterns]
    assert len(set(fallbacks)) == 1


def test_launchers_use_managed_python_and_the_controlled_release_channel():
    for relative_path in (
        Path("macos") / "Start FlightLab.command",
        Path("windows") / "Start FlightLab.cmd",
    ):
        source = (SETUP / relative_path).read_text()
        assert "student_setup/release.txt" in source
        assert "--python 3.12" in source
        assert "flightlab[workbench]" in source
        assert "flightlab workbench" in source
        assert "FLIGHTLAB_TEST_ONLY" in source


def test_workbench_launch_reports_the_silent_import_phase(monkeypatch, capsys):
    served = {}
    fake_panel = SimpleNamespace(
        serve=lambda app, **kwargs: served.update(app=app, kwargs=kwargs)
    )
    fake_workbench = SimpleNamespace(create_workbench=object())
    monkeypatch.setitem(sys.modules, "panel", fake_panel)
    monkeypatch.setitem(sys.modules, "flightlab.workbench", fake_workbench)

    launch_workbench(["--no-open"])

    assert "Preparing FlightLab (loading scientific libraries)..." in capsys.readouterr().out
    assert served["app"] is fake_workbench.create_workbench
    assert served["kwargs"]["show"] is False


def test_hw1_notebook_is_limited_to_problem_1b():
    notebook = json.loads((ROOT / "notebooks" / "hw1_starter.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "## 2. Sweep the pod shape" in source
    assert "## Problem 1a" not in source
    assert "## Problem 2a" not in source
    assert "## Problem 2b" not in source
    assert "workbench_results" not in source
    assert "brentq" not in source
    assert "from flightlab import atmos" not in source
