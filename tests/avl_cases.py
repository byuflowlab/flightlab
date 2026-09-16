"""Geometries and comparison rules shared by the AVL verification test and its recorder.

The cases are chosen so both codes solve the same problem: no bodies (AVL's
body model differs), the tailplane raised clear of the wing's wake (both
lattices are unreliable inside it and place the wake sheet about a centimetre
apart), a small gap between fin and tailplane roots (a shared edge is a
singular configuration for any vortex lattice), and the geometry origin at
the centre of gravity (AVL takes its apparent inertia about the origin).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from flightlab import atmos, project_analysis as pa, stability
from flightlab.project import FlightCase, LiftingSurface, SurfaceStation, blank_project

G0 = 9.80665
REFERENCE = Path(__file__).parent / "data" / "avl_reference.json"
NS, NC = 28, 4
CD_PROFILE = 0.02  # AVL's constant profile drag for the eigenmode model

# Our derivative names against AVL's stability-axis names.
DERIVATIVES = (
    ("CL_alpha", "CLa"), ("Cm_alpha", "Cma"), ("CL_q", "CLq"), ("Cm_q", "Cmq"), ("CD_alpha", "CDa"),
    ("CY_beta", "CYb"), ("Cl_beta", "Clb"), ("Cn_beta", "Cnb"),
    ("Cl_p", "Clp"), ("Cn_p", "Cnp"), ("Cl_r", "Clr"), ("Cn_r", "Cnr"), ("CY_p", "CYp"), ("CY_r", "CYr"),
)


def simple_wing():
    """The AVL/VortexLattice.jl reference wing: 2 deg twist, no dihedral, alpha 1."""
    project = blank_project()
    project.name = "AVL simple wing"
    project.bodies, project.masses = [], []
    project.structure.surface = ""
    project.surfaces = [LiftingSurface("simple", "wing", "fixed", True, [
        SurfaceStation(0.0, 0.0, 0.0, 2.2, 2.0, "naca0012"),
        SurfaceStation(0.4, 7.5, 0.0, 1.8, 2.0, "naca0012"),
    ])]
    project.reference.mode = "manual"
    project.reference.area, project.reference.span, project.reference.chord = 30.0, 15.0, 2.0
    project.cases = [FlightCase("reference", 1.0, altitude=0.0)]
    return project


def verification_aircraft():
    """Blank-project wing, tail, and fin arranged for a clean two-code comparison."""
    project = blank_project()
    project.name = "AVL verification aircraft"
    project.bodies = []
    project.masses = [item for item in project.masses if item.attached_to != "fuselage"]
    project.masses[-1].x = 0.21
    for station in project.surfaces[1].stations:
        station.z = 0.15
    for station in project.surfaces[2].stations:
        station.z += 0.015
    mp = stability.mass_properties(project.components())
    for surface in project.surfaces:
        for station in surface.stations:
            station.x_le -= mp.x_cg
            station.z -= mp.z_cg
    for item in project.masses:
        item.x -= mp.x_cg
        item.z -= mp.z_cg
    project.propulsion.battery_x -= mp.x_cg
    for propulsor in project.propulsion.propulsors:
        propulsor.x -= mp.x_cg
    return project


CASES = {"simple_wing": simple_wing, "verification_aircraft": verification_aircraft}


def ours(name):
    """Everything the project computes for one case, in the shape of the AVL record."""
    project = CASES[name]()
    case = project.case()
    out = {}
    if name == "simple_wing":
        x_ref = 0.5
        d = pa.derivatives(project, case, alpha=1.0, x_ref=x_ref, ns=12, nc=1)
        out["derivatives"] = {avl_name: getattr(d, mine) for mine, avl_name in DERIVATIVES}
        return out
    mp = stability.mass_properties(project.components())
    S_ref, b_ref, c_ref = project.reference_quantities()
    trimmed = pa.trim(project, case, ns=NS, nc=NC)
    d = pa.derivatives(project, case, alpha=trimmed.alpha, trim_deflection=trimmed.trim_deflection,
                       x_ref=mp.x_cg, ns=NS, nc=NC)
    out["trim"] = {"alpha": trimmed.alpha, "deflection": trimmed.trim_deflection,
                   "x_np": trimmed.x_np, "CL": trimmed.solution.CL}
    out["derivatives"] = {avl_name: getattr(d, mine) for mine, avl_name in DERIVATIVES}
    added = pa.apparent_mass(project, case)
    out["apparent"] = {"Ixx": added.Ixx, "Iyy": added.Iyy, "Izz": added.Izz, "m_z": added.m_z}
    # AVL's eigenmode model: constant profile drag, no alpha-dot terms, apparent mass included.
    avl_like = replace(d, CD=d.CD + CD_PROFILE, CL_alphadot=0.0, Cm_alphadot=0.0)
    lon = stability.longitudinal_modes(None, case.speed, case.altitude, mass=mp.mass, Iyy=mp.Iyy, x_cg=mp.x_cg,
                                       derivs=avl_like, S_ref=S_ref, c_ref=c_ref, apparent=added)
    lat = stability.lateral_modes(None, case.speed, case.altitude, mass=mp.mass, Ixx=mp.Ixx, Izz=mp.Izz,
                                  Ixz=mp.Ixz, x_cg=mp.x_cg, derivs=avl_like, S_ref=S_ref, b_ref=b_ref, apparent=added)
    out["modes"] = {"longitudinal": sorted_modes([complex(m.real, m.imag) for m in lon]),
                    "lateral": sorted_modes([complex(m.real, m.imag) for m in lat])}
    return out


def sorted_modes(values):
    """Distinct eigenvalues as ``[real, |imag|]`` pairs, oscillatory last, fastest first."""
    unique = {(round(complex(v).real, 6), round(abs(complex(v).imag), 6)) for v in values if abs(complex(v)) > 1e-9}
    return [list(pair) for pair in sorted(unique, key=lambda z: (z[1] > 0, z[0]))]


def theirs(name, avl_path=None):
    """Run AVL on one case and return the same record shape."""
    from flightlab import avl

    project = CASES[name]()
    case = project.case()
    if name == "simple_wing":
        run = avl.run(project, case, alpha=1.0, ns=12, nc=1, x_ref=0.5, trailing_leg_forces=True, avl=avl_path)
        return {"derivatives": {k: run.derivatives[k] for _, k in DERIVATIVES}}
    mp = stability.mass_properties(project.components())
    S_ref, _, _ = project.reference_quantities()
    CL_required = mp.mass * G0 / (atmos.at(case.altitude).q(case.speed) * S_ref)
    run = avl.run(project, case, CL=CL_required, trim_pitch=True, ns=NS, nc=NC, cd_profile=CD_PROFILE,
                  modes=True, trailing_leg_forces=True, avl=avl_path)
    return {
        "trim": {"alpha": run.alpha, "deflection": run.deflection, "x_np": run.derivatives["Xnp"],
                 "CL": run.totals["CLtot"]},
        "derivatives": {k: run.derivatives[k] for _, k in DERIVATIVES},
        "modes": {"longitudinal": sorted_modes(run.longitudinal_modes),
                  "lateral": sorted_modes(run.lateral_modes)},
    }


def load_reference():
    return json.loads(REFERENCE.read_text())


def split_modes(modes):
    """Return (phugoid, short_period_roots, dutch_roll, roll_subsidence, spiral).

    ``modes`` is the ``{"longitudinal": [...], "lateral": [...]}`` record.  The
    phugoid is the slowest longitudinal root set and the short period the
    rest; the dutch roll is the lateral oscillatory pair, the roll subsidence
    the fastest lateral real root and the spiral the slowest.
    """
    lon = sorted([complex(r, i) for r, i in modes["longitudinal"]], key=abs)
    lat = [complex(r, i) for r, i in modes["lateral"]]
    phugoid, short = lon[0], lon[1:]
    pairs = [z for z in lat if z.imag > 0]
    reals = sorted([z.real for z in lat if z.imag == 0])
    dutch = max(pairs, key=abs) if pairs else None
    roll, spiral = reals[0], reals[-1]
    return phugoid, short, dutch, roll, spiral
