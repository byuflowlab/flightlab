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
    assert not any("equivalent" in item or "adapter" in item for item in m1.notes)
    assert m1.warnings == ()
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


def test_symmetric_lift_slope_converges_for_a_cambered_root_with_dihedral():
    """The blank wing's cambered root section, rotated by the root dihedral,
    used to sit a fraction of a millimetre off the symmetry plane and the lift
    slope wandered by ten per cent with the panel count."""
    import numpy as np
    from flightlab.project import blank_project
    from flightlab.project_analysis import analyze

    project = blank_project()
    project.surfaces = project.surfaces[:1]
    project.masses = []
    project.structure.surface = ""
    slopes = []
    for ns in (8, 16, 40):
        low = analyze(project, alpha=0.0, ns=ns, nc=4, x_ref=0.1).CL
        high = analyze(project, alpha=5.0, ns=ns, nc=4, x_ref=0.1).CL
        slopes.append((high - low) / np.radians(5.0))
    assert max(slopes) - min(slopes) < 0.01 * max(slopes)
    assert 4.0 < slopes[-1] < 4.5


def test_cruciform_tail_lateral_derivatives_converge_with_panel_count():
    """The fin and tailplane of the built-in projects share a root line; the
    finite core between surfaces keeps that junction from dominating."""
    from flightlab.project import blank_project

    project = blank_project()
    values = [derivatives(project, alpha=0.6, trim_deflection=-0.5, ns=ns, nc=4) for ns in (12, 28, 56)]
    for name in ("CY_beta", "Cn_beta", "Cn_r"):
        series = [getattr(v, name) for v in values]
        assert max(series) - min(series) < 0.03 * abs(series[-1]), (name, series)
    assert all(v.Cn_beta > 0 for v in values)


def test_zero_inertia_gives_a_plain_message_instead_of_a_crash():
    from flightlab.project import MassItem

    project = example_project()
    project.masses = [MassItem("everything", 0.75, x=0.05)]
    project.propulsion = None
    with pytest.raises(ValueError, match="no moment of inertia.*point mass"):
        analyze_dynamic_stability(project, ns=10, nc=3)


def test_pitch_stiffness_with_a_tail_near_the_wake_is_converged_at_workbench_paneling():
    """AVL without a vortex core needs 140 spanwise panels to settle the pitch
    stiffness of the blank project's low tail; the finite core between
    surfaces settles it here by 28, so a low tail needs no special treatment."""
    from flightlab.project import blank_project

    project = blank_project()
    project.bodies, project.masses = [], []
    project.structure.surface = ""
    project.surfaces = project.surfaces[:2]
    S_ref, b_ref, c_ref = blank_project().reference_quantities()
    project.reference.mode = "manual"
    project.reference.area, project.reference.span, project.reference.chord = S_ref, b_ref, c_ref
    coarse = derivatives(project, alpha=2.0, x_ref=0.12, ns=12, nc=4).Cm_alpha
    normal = derivatives(project, alpha=2.0, x_ref=0.12, ns=28, nc=4).Cm_alpha
    fine = derivatives(project, alpha=2.0, x_ref=0.12, ns=56, nc=4).Cm_alpha
    assert abs(normal - fine) < 0.005 * abs(fine)
    assert abs(coarse - fine) < 0.025 * abs(fine)


def test_split_real_pairs_report_the_equivalent_frequency_and_damping():
    """The starters' short period is just past critical damping; the two real
    roots report the second-order pair they came from, not bare time constants."""
    import numpy as np
    from flightlab.project import blank_project

    modes = analyze_dynamic_stability(blank_project(), ns=12, nc=3).longitudinal
    split = [m for m in modes if m.partner is not None]
    assert len(split) == 2 and all(m.name.startswith("short period") for m in split)
    first, second = split
    assert first.frequency == pytest.approx(second.frequency)
    assert first.damping == pytest.approx(second.damping)
    assert first.frequency == pytest.approx(np.sqrt(first.real * second.real), rel=1e-9)
    assert 1.0 < first.damping < 1.3
    assert not first.oscillatory
