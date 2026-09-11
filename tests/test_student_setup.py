"""Static checks for the no-install student launch path."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from flightlab.__main__ import launch_workbench


ROOT = Path(__file__).parents[1]
SETUP = ROOT / "student_setup"
LAUNCHERS = (
    Path("macos") / "Start FlightLab.command",
    Path("windows") / "Start FlightLab.cmd",
    Path("linux") / "Start FlightLab.sh",
)


def test_release_and_launcher_fallbacks_are_exact_git_commits():
    release = (SETUP / "release.txt").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", release)

    macos = (SETUP / LAUNCHERS[0]).read_text()
    windows = (SETUP / LAUNCHERS[1]).read_text()
    linux = (SETUP / LAUNCHERS[2]).read_text()
    notebook = json.loads((ROOT / "notebooks" / "hw1_starter.ipynb").read_text())
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    fallback_patterns = (
        (macos, r'FLIGHTLAB_DEFAULT_COMMIT="([0-9a-f]{40})"'),
        (windows, r"FLIGHTLAB_DEFAULT_COMMIT=([0-9a-f]{40})"),
        (linux, r'FLIGHTLAB_DEFAULT_COMMIT="([0-9a-f]{40})"'),
        (notebook_source, r'_fallback_commit = "([0-9a-f]{40})"'),
    )
    fallbacks = [re.search(pattern, source).group(1) for source, pattern in fallback_patterns]
    assert len(set(fallbacks)) == 1


def test_launchers_use_managed_python_and_the_controlled_release_channel():
    for relative_path in LAUNCHERS:
        source = (SETUP / relative_path).read_text()
        assert "student_setup/release.txt" in source
        assert "--python 3.12" in source
        assert "flightlab[workbench]" in source
        assert "-m flightlab workbench" in source
        assert source.count("-m flightlab") == 2
        assert "FLIGHTLAB_TEST_ONLY" in source
        # The course build is installed once per promoted commit into a
        # FlightLab-only environment and then started directly; nothing is
        # re-resolved against PyPI on an ordinary launch.
        assert "installed.txt" in source
        assert "tool run" not in source
        # A failed update keeps the previously installed build usable.
        assert "previously installed build" in source


def test_every_launcher_ships_with_start_here_notes_and_a_download_build():
    for relative_path in LAUNCHERS:
        assert (SETUP / relative_path.parent / "START HERE.txt").exists()
    build = (SETUP / "build_downloads.sh").read_text()
    for name in ("macOS", "Windows", "Linux"):
        assert f"FlightLab-{name}.zip" in build
    workflow = (ROOT / ".github" / "workflows" / "student-launchers.yml").read_text()
    for runner in ("macos-latest", "windows-latest", "ubuntu-latest"):
        assert runner in workflow


def test_workbench_launch_reports_the_silent_import_phase(monkeypatch, capsys):
    served = {}
    fake_panel = SimpleNamespace(
        serve=lambda app, **kwargs: served.update(app=app, kwargs=kwargs)
    )

    class FakeWorkbench:
        instances = []

        def __init__(self):
            FakeWorkbench.instances.append(self)

        def view(self):
            return ("view", self)

    fake_workbench = SimpleNamespace(Workbench=FakeWorkbench)
    monkeypatch.setitem(sys.modules, "panel", fake_panel)
    monkeypatch.setitem(sys.modules, "flightlab.workbench", fake_workbench)

    launch_workbench(["--no-open"])

    out = capsys.readouterr().out
    assert "Preparing FlightLab (loading scientific libraries)..." in out
    assert "Building the workbench page" in out
    assert "Starting the local server" in out
    assert served["kwargs"]["show"] is False
    # The page built before the server started goes to the first browser tab,
    # so it appears as soon as the tab connects; later tabs get their own state.
    first = FakeWorkbench.instances[0]
    assert served["app"]() == ("view", first)
    _, second = served["app"]()
    assert second is not first
    assert len(FakeWorkbench.instances) == 2


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
