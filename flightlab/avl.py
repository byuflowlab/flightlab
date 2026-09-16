"""Export a project to AVL and read AVL's answers back.

AVL (Drela and Youngren's Athena Vortex Lattice) is the reference the course
checks its own vortex lattice against.  This module writes a project as an
AVL geometry file and mass file, drives AVL in batch, and parses the total
forces, stability derivatives, and eigenmodes it prints.  The verification
tests use it to compare :func:`flightlab.project_analysis.derivatives` and the
dynamic modes with AVL on the same geometry.

AVL is not bundled.  Point :func:`find_avl` at an executable with the
``FLIGHTLAB_AVL`` environment variable or put ``avl`` on the ``PATH``.

Conventions that must match for the comparison to mean anything:

* Stations are written as AVL sections in body axes (x aft, y right, z up),
  which is also AVL's geometry frame.  A symmetric surface gets
  ``YDUPLICATE 0``; a centerline fin does not.
* Airfoils are written as coordinate files so AVL takes the same camber
  line the project uses.  AVL, like the project's lattice, uses only the
  camber line.
* Every pitch-trim surface carries one control named ``pitch``.  A
  ``whole_surface`` control has its hinge at x/c = 0 (AVL's all-moving
  surface); an ``elevator`` has it at the entered hinge fraction.  Positive
  deflection is trailing-edge down in both codes.
* Spanwise panels follow the project's rule (cosine over the whole surface,
  count scaled by span); chordwise panels are uniform in both.
* The mass file holds one lump: the project's total mass, centre of gravity,
  and inertias about it.  Applying it makes AVL take moments about the
  centre of gravity, as the project does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import atmos, stability
from .project import AircraftProject, FlightCase, LiftingSurface

G0 = 9.80665
PITCH_CONTROL = "pitch"


def find_avl() -> Optional[str]:
    """Path to an AVL executable, or ``None``."""
    candidate = os.environ.get("FLIGHTLAB_AVL")
    if candidate and Path(candidate).is_file():
        return candidate
    return shutil.which("avl")


def surface_panels(project: AircraftProject, surface: LiftingSurface, ns: int) -> int:
    """Spanwise panel count the project's own solves give ``surface``."""
    _, b_ref, _ = project.reference_quantities()
    return max(8, int(round(ns * surface.span / b_ref)))


def _airfoil_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".dat"


def write_geometry(
    project: AircraftProject,
    path,
    ns: int = 28,
    nc: int = 4,
    cd_profile: float = 0.0,
    x_ref: Optional[float] = None,
) -> Path:
    """Write ``project`` as an AVL geometry file, plus one coordinate file per airfoil.

    ``cd_profile`` becomes AVL's ``CDp`` header value, the constant profile
    drag AVL adds to its induced drag in the eigenmode model.
    """
    path = Path(path)
    S_ref, b_ref, c_ref = project.reference_quantities()
    if x_ref is None:
        try:
            x_ref = stability.mass_properties(project.components()).x_cg
        except Exception:
            x_ref = project.primary_surface.aerodynamic_center_x
    lines = [
        project.name,
        "0.0                 | Mach",
        "0  0  0.0           | iYsym  iZsym  Zsym",
        f"{S_ref:.8g}  {c_ref:.8g}  {b_ref:.8g}   | Sref  Cref  Bref",
        f"{x_ref:.8g}  0.0  0.0   | Xref  Yref  Zref",
        f"{cd_profile:.6g}              | CDp",
    ]
    written = set()
    for surface in project.surfaces:
        lines += [
            "#" + "-" * 60,
            "SURFACE",
            surface.name,
            f"{nc}  3.0  {surface_panels(project, surface, ns)}  1.0   | Nchord Cspace Nspan Sspace",
        ]
        if surface.symmetric:
            lines += ["YDUPLICATE", "0.0"]
        for station in surface.stations:
            lines += [
                "SECTION",
                # No trailing comment: AVL reads optional Nspan/Sspace from
                # whatever follows Ainc and rejects text there.
                f"{station.x_le:.8g}  {station.y:.8g}  {station.z:.8g}  {station.chord:.8g}  "
                f"{station.twist_deg:.8g}",
                "AFILE",
                _airfoil_filename(station.airfoil),
            ]
            if station.airfoil not in written:
                section = project.section(station.airfoil)
                coordinates = np.asarray(section.coordinates, dtype=float)
                with open(path.parent / _airfoil_filename(station.airfoil), "w") as handle:
                    handle.write(f"{station.airfoil}\n")
                    for x, z in coordinates:
                        handle.write(f"  {x: .7f}  {z: .7f}\n")
                written.add(station.airfoil)
            if surface.trim_control != "fixed":
                hinge = 0.0 if surface.trim_control == "whole_surface" else surface.control_hinge_fraction
                lines += ["CONTROL", f"{PITCH_CONTROL}  1.0  {hinge:.6g}  0. 0. 0.  1.0"]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_mass(project: AircraftProject, path, altitude: float = 0.0) -> Path:
    """Write the project's total mass, centre of gravity, and inertias as an AVL mass file."""
    path = Path(path)
    mp = stability.mass_properties(project.components())
    rho = float(atmos.at(altitude).density)
    path.write_text(
        "\n".join([
            f"# {project.name}: one lump with the project's mass properties about its CG",
            "Lunit = 1.0 m",
            "Munit = 1.0 kg",
            "Tunit = 1.0 s",
            f"g   = {G0}",
            f"rho = {rho:.6g}",
            # AVL 3.52 reads the optional inertia columns as Ixx Iyy Izz Ixy Ixz Iyz.
            "#  mass      x         y         z         Ixx        Iyy        Izz        Ixy   Ixz        Iyz",
            f"  {mp.mass:.8g}  {mp.x_cg:.8g}  {mp.y_cg:.8g}  {mp.z_cg:.8g}  "
            f"{mp.Ixx:.8g}  {mp.Iyy:.8g}  {mp.Izz:.8g}  0.0  {mp.Ixz:.8g}  0.0",
        ]) + "\n"
    )
    return path


@dataclass
class AvlRun:
    """What one AVL run case produced."""

    totals: Dict[str, float]
    derivatives: Dict[str, float]
    modes: List[complex] = field(default_factory=list)
    longitudinal_modes: List[complex] = field(default_factory=list)
    lateral_modes: List[complex] = field(default_factory=list)
    log: str = ""

    @property
    def alpha(self) -> float:
        return self.totals["Alpha"]

    @property
    def deflection(self) -> float:
        return self.totals.get(PITCH_CONTROL, 0.0)


_NUMBER = r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)"


def _parse_st(text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    totals: Dict[str, float] = {}
    derivatives: Dict[str, float] = {}
    in_derivatives = False
    for line in text.splitlines():
        if "Stability-axis derivatives" in line:
            in_derivatives = True
        if in_derivatives and "/" in line:
            # "Clb Cnr / Clr Cnb = ..." is a ratio, not a derivative named Cnb.
            continue
        for name, value in re.findall(r"([A-Za-z][A-Za-z0-9'_]*)\s*=\s*" + _NUMBER, line):
            key = name.replace("'", "")
            target = derivatives if in_derivatives else totals
            target[key] = float(value.replace("D", "E").replace("d", "e"))
    return totals, derivatives


def _parse_modes(log: str) -> Tuple[List[complex], List[complex], List[complex]]:
    """Eigenvalues from AVL's mode printout, split by the eigenvector's state content.

    Each mode block lists the components ``u w q the`` (longitudinal) and
    ``v p r phi`` (lateral); whichever set carries the eigenvector decides the
    family.  Returns ``(all, longitudinal, lateral)``.
    """
    modes, longitudinal, lateral = [], [], []
    blocks = re.split(r"(?=\s*mode\s+\d+:)", log)
    for block in blocks:
        head = re.match(r"\s*mode\s+\d+:\s+" + _NUMBER + r"\s+" + _NUMBER, block)
        if head is None:
            continue
        value = complex(float(head.group(1)), float(head.group(2)))
        modes.append(value)

        def size(names):
            total = 0.0
            for name in names:
                found = re.search(r"\b" + name + r"\s*:\s*" + _NUMBER + r"\s+" + _NUMBER, block)
                if found:
                    total += abs(complex(float(found.group(1)), float(found.group(2))))
            return total

        if size(("v", "p", "r", "phi")) > size(("u", "w", "q", "the")):
            lateral.append(value)
        else:
            longitudinal.append(value)
    return modes, longitudinal, lateral


def run(
    project: AircraftProject,
    case: Optional[FlightCase] = None,
    *,
    alpha: Optional[float] = None,
    CL: Optional[float] = None,
    deflection: Optional[float] = None,
    trim_pitch: bool = False,
    ns: int = 28,
    nc: int = 4,
    cd_profile: float = 0.0,
    modes: bool = False,
    x_ref: Optional[float] = None,
    trailing_leg_forces: bool = True,
    avl: Optional[str] = None,
    workdir=None,
) -> AvlRun:
    """Run one AVL case on ``project`` and return its forces, derivatives, and modes.

    Set the angle of attack with ``alpha`` or let AVL find it for a lift
    coefficient with ``CL``.  Set the pitch control with ``deflection`` or let
    AVL trim it for zero pitching moment with ``trim_pitch``.  By default the
    mass file is applied and moments are taken about the project's centre of
    gravity; pass ``x_ref`` to take them about that point instead, in which
    case no mass data is loaded and ``modes`` is unavailable.

    ``trailing_leg_forces`` switches on AVL's Kutta-Joukowski forces on the
    chordwise vortex legs lying on each surface.  AVL 3.51 turned them off
    by default; the project's lattice, like AVL before 3.51, includes them,
    and they matter for the rolling moment due to sideslip of a wing at
    angle of attack.
    """
    if x_ref is not None and modes:
        raise ValueError("eigenmodes need the mass file; leave x_ref as None")
    avl = avl or find_avl()
    if avl is None:
        raise FileNotFoundError("no AVL executable: set FLIGHTLAB_AVL or put avl on the PATH")
    case = project.case() if case is None else case
    if alpha is None and CL is None:
        alpha = case.alpha_deg
    rho = float(atmos.at(case.altitude).density)

    def _run(directory: Path) -> AvlRun:
        geometry = write_geometry(
            project, directory / "aircraft.avl", ns=ns, nc=nc, cd_profile=cd_profile, x_ref=x_ref,
        )
        commands = ["plop", "g", "", f"load {geometry.name}"]
        if x_ref is None:
            write_mass(project, directory / "aircraft.mass", altitude=case.altitude)
            commands += ["mass aircraft.mass", "mset 0"]
        commands += ["oper"]
        if trailing_leg_forces:
            commands += ["o", "t", ""]
        commands += ["m", f"v {case.speed:.8g}", f"d {rho:.8g}", ""]
        commands.append(f"a c {CL:.8g}" if CL is not None else f"a a {alpha:.8g}")
        if any(surface.trim_control != "fixed" for surface in project.surfaces):
            commands.append("d1 pm 0" if trim_pitch else f"d1 d1 {deflection or 0.0:.8g}")
        commands += ["x", "st st.txt", ""]
        if modes:
            commands += ["mode", "n", ""]
        commands.append("quit")
        completed = subprocess.run(
            [avl], input="\n".join(commands) + "\n", capture_output=True, text=True,
            cwd=directory, timeout=300,
        )
        log = completed.stdout + completed.stderr
        st_path = directory / "st.txt"
        if not st_path.exists():
            raise RuntimeError("AVL produced no stability output; log follows\n" + log[-3000:])
        totals, derivatives = _parse_st(st_path.read_text())
        all_modes, longitudinal, lateral = _parse_modes(log) if modes else ([], [], [])
        return AvlRun(totals, derivatives, all_modes, longitudinal, lateral, log)

    if workdir is not None:
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        return _run(directory)
    with tempfile.TemporaryDirectory(prefix="flightlab-avl-") as tmp:
        return _run(Path(tmp))
