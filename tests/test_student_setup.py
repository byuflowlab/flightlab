"""Static checks for the no-install student launch path."""

import json
from pathlib import Path
import re


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
