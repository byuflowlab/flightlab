"""Stability derivatives and dynamic modes use every lifting surface as drawn."""

from copy import deepcopy

import pytest

from flightlab.project import LiftingSurface, SurfaceStation, example_project
from flightlab.project_analysis import analyze_dynamic_stability, derivatives


def _ventral_fin():
    return LiftingSurface("Ventral fin", "other", "fixed", False, [
        SurfaceStation(0.70, 0.00, 0.02, 0.16, 0.0, "naca0012"),
        SurfaceStation(0.78, 0.00, -0.12, 0.08, 0.0, "naca0012"),
    ])


def test_derivatives_are_lateral_valid_and_weathercock_stable():
    project = example_project()
    d = derivatives(project, ns=10, nc=3)
    assert d.lateral_valid
    assert d.CL_alpha > 3.0
    assert d.Cm_alpha < 0.0
    assert d.Cn_beta > 0.0
    assert d.Cl_p < 0.0
    assert d.CL_alphadot > 0.0 and d.Cm_alphadot < 0.0


def test_a_fourth_surface_enters_the_derivative_solve():
    base = example_project()
    with_ventral = deepcopy(base)
    with_ventral.surfaces.append(_ventral_fin())
    d0 = derivatives(base, ns=10, nc=3)
    d1 = derivatives(with_ventral, ns=10, nc=3)
    # A second fin adds directional stiffness and yaw damping.
    assert d1.Cn_beta > d0.Cn_beta
    assert d1.Cn_r < d0.Cn_r
    # It sits below the CG, so it also changes the roll coupling.
    assert d1.Cl_beta != pytest.approx(d0.Cl_beta)


def test_purpose_labels_do_not_change_the_derivatives():
    labelled = example_project()
    relabelled = deepcopy(labelled)
    for surface in relabelled.surfaces:
        surface.purpose = "other"
    d0 = derivatives(labelled, ns=10, nc=3)
    d1 = derivatives(relabelled, ns=10, nc=3)
    for name in ("CL_alpha", "Cm_alpha", "Cn_beta", "Cl_beta", "Cl_p", "Cn_r"):
        assert getattr(d1, name) == pytest.approx(getattr(d0, name))


def test_dynamic_modes_include_the_fourth_surface_and_carry_no_adapter_caveats():
    base = example_project()
    with_ventral = deepcopy(base)
    with_ventral.surfaces.append(_ventral_fin())
    m0 = analyze_dynamic_stability(base, ns=10, nc=3)
    m1 = analyze_dynamic_stability(with_ventral, ns=10, nc=3)
    assert m1.derivatives.Cn_beta > m0.derivatives.Cn_beta
    assert len(m1.lateral) == 3 and len(m1.longitudinal) >= 2
    assert not any("equivalent" in item or "adapter" in item for item in m1.warnings)
    # Body increments sit on the project's own reference quantities.
    assert m0.body_increments["Cm_alpha"] > 0


def test_mirrored_solve_agrees_with_the_symmetric_pitch_solve():
    from flightlab import stability
    from flightlab.project_analysis import _solve_system
    from flightlab.vlm import stability_derivatives as vlm_derivatives

    project = example_project()
    case = project.case()
    x_cg = stability.mass_properties(project.components()).x_cg
    system, _, _, _ = _solve_system(project, case, 3.0, 0.0, 12, 3, x_cg, derivatives=True)
    dCF, dCM = vlm_derivatives(system)
    mirrored = derivatives(project, case, alpha=3.0, x_ref=x_cg, ns=12, nc=3)
    assert mirrored.CL_alpha == pytest.approx(float(dCF["alpha"][2]), rel=1e-6)
    assert mirrored.Cm_alpha == pytest.approx(float(dCM["alpha"][1]), rel=1e-6)
    assert mirrored.Cm_q == pytest.approx(float(dCM["q"][1]), rel=1e-6)


def test_an_isolated_fin_matches_the_same_panel_laid_flat():
    import numpy as np

    project = example_project()
    project.masses = []
    project.reference.mode = "manual"
    project.reference.area, project.reference.span, project.reference.chord = 0.06, 0.3, 0.2
    root = SurfaceStation(0.0, 0.0, 0.0, 0.2, 0.0, "naca0012")
    project.surfaces = [LiftingSurface("Fin", "fin", "fixed", False,
                                       [root, SurfaceStation(0.0, 0.0, 0.3, 0.2, 0.0, "naca0012")])]
    fin = derivatives(project, alpha=0.0, x_ref=0.05, ns=16, nc=4)
    project.surfaces = [LiftingSurface("Half wing", "wing", "fixed", False,
                                       [root, SurfaceStation(0.0, 0.3, 0.0, 0.2, 0.0, "naca0012")])]
    flat = derivatives(project, alpha=0.0, x_ref=0.05, ns=16, nc=4)
    assert -fin.CY_beta == pytest.approx(flat.CL_alpha, rel=1e-3)
    helmbold = 2 * np.pi * 1.5 / (2 + np.sqrt(1.5**2 + 4))
    assert -fin.CY_beta == pytest.approx(helmbold, rel=0.06)
