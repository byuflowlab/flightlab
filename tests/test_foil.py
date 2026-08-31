"""Verification for ``flightlab.foil``.

Includes the rungs HW 3 stands on, so a NeuralFoil upgrade that moves the
headline result gets noticed here rather than in a student's write-up.
"""

import numpy as np
import pytest

from flightlab import foil, ref

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _slope_per_rad(section, Re, lo, hi, n=25, **kw):
    """Least-squares lift curve slope over a degree range, per radian."""
    a = np.linspace(lo, hi, n)
    cl = foil.aero(section, a, Re, **kw)["cl"]
    return np.polyfit(np.radians(a), cl, 1)[0]


# --- the symmetry rung ------------------------------------------------------


def test_symmetric_section_has_zero_lift_and_moment_at_zero_alpha():
    """One line, and it catches sign errors in the whole pipeline."""
    r = foil.aero("naca0012", alpha=0.0, Re=3e6)
    assert abs(float(r["cl"])) < 1e-6
    assert abs(float(r["cm"])) < 5e-3


def test_symmetric_section_is_antisymmetric_in_alpha():
    a = np.array([2.0, 5.0, 8.0])
    up = foil.aero("naca0012", a, 3e6)
    dn = foil.aero("naca0012", -a, 3e6)
    assert np.allclose(up["cl"], -dn["cl"], atol=1e-6)
    assert np.allclose(up["cd"], dn["cd"], rtol=1e-6)


# --- the analytic limit rung ------------------------------------------------


def test_naca2412_lift_slope_is_two_pi_at_high_reynolds():
    """HW 3's headline: 2*pi to three digits, and robust."""
    s = _slope_per_rad("naca2412", 3e6, -2.0, 8.0)
    assert s == pytest.approx(2.0 * np.pi, rel=0.01)


@pytest.mark.parametrize("model", ["medium", "large", "xlarge"])
def test_two_pi_result_is_robust_across_model_sizes(model):
    """The result the course leans on must not depend on the network size."""
    s = _slope_per_rad("naca2412", 3e6, -2.0, 8.0, model_size=model)
    assert s == pytest.approx(2.0 * np.pi, rel=0.015)


def test_two_pi_result_is_robust_across_coordinate_refinement():
    for n in (41, 81, 161):
        s = _slope_per_rad(foil.naca4("2412", n=n), 3e6, -2.0, 8.0)
        assert s == pytest.approx(2.0 * np.pi, rel=0.015)


def test_low_reynolds_lift_slope_depends_strongly_on_the_fit_range():
    """HW 3's most useful item, pinned down.

    Three fits to the same curve disagree by tens of percent, because the curve
    has a steep segment near zero lift and is simply not straight there.  The
    tool is not broken and it is not extrapolating -- it is answering a question
    that assumed a straight line, about a curve that is not one.
    """
    wide_hi_re = _slope_per_rad("naca2412", 3e6, -2.0, 8.0)
    fits = {
        (-2.0, 4.0): _slope_per_rad("naca2412", 1e5, -2.0, 4.0),
        (1.0, 4.0): _slope_per_rad("naca2412", 1e5, 1.0, 4.0),
        (-2.0, 8.0): _slope_per_rad("naca2412", 1e5, -2.0, 8.0),
    }
    values = np.array(list(fits.values()))
    spread = (values.max() - values.min()) / values.mean()
    assert spread > 0.25, f"expected the fits to disagree badly, got {fits}"
    # and the widest range lands suspiciously close to 2*pi
    assert fits[(-2.0, 8.0)] == pytest.approx(2.0 * np.pi, rel=0.04)
    # while the high-Re curve is genuinely straight there
    assert abs(wide_hi_re - 2 * np.pi) / (2 * np.pi) < 0.01


def test_recorded_lift_slopes_still_reproduce():
    """Guards against a silent NeuralFoil upgrade moving the reference numbers."""
    tol = ref.NEURALFOIL_NACA2412["tolerance"]
    ms = ref.NEURALFOIL_NACA2412["model_size"]
    for (re_key, lo, hi), expected in ref.NEURALFOIL_NACA2412[
        "cl_alpha_per_rad"
    ].items():
        Re = 3e6 if re_key == "Re3e6" else 1e5
        got = _slope_per_rad("naca2412", Re, lo, hi, model_size=ms)
        assert got == pytest.approx(expected, abs=tol), (re_key, lo, hi, got)


# --- the published-data rung ------------------------------------------------


def test_naca2412_against_published_data_including_where_it_disagrees():
    """HW 3 asks which of four quantities you would trust the tool on.

    The honest answer is two of them.  The lift slope lands inside the published
    range and the zero-lift angle is close.  Minimum drag comes out ~12% low and
    ``cl_max`` comes out ~6% high and sweep-dependent.  This test asserts that
    pattern rather than widening the reference ranges until the tool fits --
    which is the failure mode the whole assignment is about.
    """
    a = np.linspace(-8.0, 20.0, 113)
    r = foil.aero("naca2412", a, 3e6)
    cl, cd = r["cl"], r["cd"]

    # 1. lift curve slope: inside the published box
    lo, hi = ref.PUBLISHED_NACA2412["cl_alpha_per_deg"]
    slope_deg = _slope_per_rad("naca2412", 3e6, -2.0, 8.0) * np.pi / 180.0
    assert lo <= slope_deg <= hi

    # 2. zero-lift angle: just outside, by less than a quarter degree
    lo, hi = ref.PUBLISHED_NACA2412["alpha_L0_deg"]
    i = int(np.argmin(np.abs(cl)))
    alpha_L0 = np.interp(0.0, cl[i - 2 : i + 3], a[i - 2 : i + 3])
    assert lo - 0.35 <= alpha_L0 <= hi + 0.35

    # 3. minimum drag: the tool is optimistic, and by a reportable amount
    lo, hi = ref.PUBLISHED_NACA2412["cd_min"]
    assert cd.min() < lo, "expected NeuralFoil to under-predict cd_min"
    assert cd.min() > 0.8 * lo, "but not by more than about 20%"

    # 4. cl_max: the tool is high, and the value depends on the sweep range
    lo, hi = ref.PUBLISHED_NACA2412["cl_max"]
    assert cl.max() > hi, "expected NeuralFoil to over-predict cl_max"
    cl_short = foil.aero("naca2412", np.linspace(-8.0, 14.0, 89), 3e6)["cl"].max()
    assert cl_short < cl.max(), "cl_max should depend on how far you swept"


def test_recorded_published_comparison_still_holds():
    """Pins the recorded discrepancies so a tool upgrade is visible."""
    rec = ref.NEURALFOIL_VS_PUBLISHED_NACA2412
    a = np.linspace(-8.0, 20.0, 113)
    r = foil.aero("naca2412", a, 3e6)
    assert r["cd"].min() == pytest.approx(rec["cd_min"], abs=6e-4)
    assert r["cl"].max() == pytest.approx(rec["cl_max_swept_to_20deg"], abs=0.12)
    slope_deg = _slope_per_rad("naca2412", 3e6, -2.0, 8.0) * np.pi / 180.0
    assert slope_deg == pytest.approx(rec["cl_alpha_per_deg"], abs=3e-3)


def test_section_drag_never_falls_below_the_laminar_flat_plate():
    """HW 5's bounds rung, checked across the range the course actually uses."""
    for Re in (5e4, 1e5, 5e5, 1e6, 3e6, 1e7):
        for name in ("naca2412", "sd7037", "fx62k131", "clarky"):
            cd = foil.aero(name, np.linspace(-4.0, 10.0, 29), Re)["cd"]
            assert np.all(cd > ref.laminar_flat_plate_cf(Re)), (name, Re)


# --- confidence -------------------------------------------------------------


def test_confidence_is_returned_and_bounded():
    r = foil.aero("naca2412", np.linspace(-8.0, 20.0, 57), 1e5)
    assert "confidence" in r
    assert np.all(r["confidence"] >= 0.0)
    assert np.all(r["confidence"] <= 1.0)


def test_confidence_is_high_exactly_where_the_lift_slope_is_ambiguous():
    """The real lesson: a confidence metric does not validate your extraction.

    Through the region where the three lift-slope fits disagree by tens of
    percent, the surrogate reports high confidence -- because it has seen cases
    like this one.  That is a different question from whether the number you
    pulled out of its output means what you think it means.
    """
    r = foil.aero("naca2412", np.linspace(-2.0, 8.0, 41), 1e5)
    assert r["confidence"].min() > 0.90


def test_confidence_falls_off_near_stall():
    clean = foil.aero("naca2412", 4.0, 1e5)["confidence"]
    stalled = foil.aero("naca2412", 18.0, 1e5)["confidence"]
    assert float(stalled) < float(clean)


# --- transition and the drag bucket -----------------------------------------


def test_tripping_the_boundary_layer_increases_drag():
    """HW 3's contamination sweep must move in the right direction."""
    clean = foil.aero("fx62k131", 2.0, 1.2e6, xtr_upper=0.7)["cd"]
    dirty = foil.aero("fx62k131", 2.0, 1.2e6, xtr_upper=0.05)["cd"]
    assert float(dirty) > float(clean)


def test_transition_sweep_is_monotone_and_vectorized():
    xtr = np.linspace(0.05, 0.7, 14)
    cd = foil.aero("fx62k131", 2.0, 1.2e6, xtr_upper=xtr)["cd"]
    assert cd.shape == xtr.shape
    assert cd[0] > cd[-1]  # forward transition is draggier


def test_laminar_section_has_a_drag_bucket_at_its_design_reynolds():
    """The FX 62-K-131's bucket is what HW 3's item 2 is about.

    Note that this section's zero-lift angle is near -6 degrees, so a sweep that
    starts at -4 never gets below the bucket at all.  Sweep wide enough to see
    both edges or you will not find it.
    """
    a = np.linspace(-12.0, 16.0, 113)
    r = foil.aero("fx62k131", a, 1.2e6)
    cl, cd = r["cl"], r["cd"]

    cd_min = cd.min()
    in_bucket = cd < 1.35 * cd_min
    assert in_bucket.sum() > 5
    cl_lo, cl_hi = cl[in_bucket].min(), cl[in_bucket].max()
    # a real bucket spans a usable range of lift coefficient
    assert cl_hi - cl_lo > 0.4
    # and drag steps up on both sides of it
    below = cl < cl_lo - 0.2
    above = cl > cl_hi + 0.2
    assert below.sum() > 2 and above.sum() > 2
    assert cd[below].min() > 1.5 * cd_min
    assert cd[above].min() > 1.5 * cd_min


def test_high_reynolds_section_does_worse_than_a_low_re_section_at_low_re():
    """HW 3's item 3: a good high-Re airfoil is a bad low-Re choice."""
    a = np.linspace(-2.0, 8.0, 41)
    laminar = foil.aero("naca633618", a, 1e5)
    lowre = foil.aero("sd7037", a, 1e5)
    # compare best l/d at RC-1's Reynolds number
    ld_lam = np.max(laminar["cl"] / laminar["cd"])
    ld_low = np.max(lowre["cl"] / lowre["cd"])
    assert ld_low > ld_lam


# --- API mechanics ----------------------------------------------------------


def test_broadcasting_over_alpha_and_reynolds():
    a = np.linspace(-4.0, 10.0, 15)
    Re = np.logspace(4.7, 6.0, 7)
    A, R = np.meshgrid(a, Re, indexing="ij")
    out = foil.aero("naca2412", A, R)
    for key in ("cl", "cd", "cm", "confidence"):
        assert out[key].shape == A.shape
        assert np.all(np.isfinite(out[key]))


def test_scalar_input_gives_scalar_shaped_output():
    out = foil.aero("naca2412", 3.0, 1e6)
    assert out["cl"].shape == ()


def test_polar_helper_returns_its_alpha():
    out = foil.polar("clarky", Re=2e5)
    assert "alpha" in out
    assert out["cl"].shape == out["alpha"].shape


def test_named_and_generated_naca_agree_closely():
    """A NACA 4-digit from the UIUC file and from the equations should match."""
    a = np.linspace(-2.0, 8.0, 21)
    from_file = foil.aero("naca2412", a, 1e6)["cl"]
    generated = foil.aero(foil.naca4("2412", n=121), a, 1e6)["cl"]
    assert np.max(np.abs(from_file - generated)) < 0.03


def test_coordinates_can_be_passed_directly():
    coords = foil.load("sd7037").coordinates
    r = foil.aero(coords, 4.0, 2e5)
    assert np.isfinite(float(r["cl"]))


# --- compressibility --------------------------------------------------------


def test_mach_zero_applies_no_correction():
    a = np.array([0.0, 4.0])
    assert np.allclose(
        foil.aero("naca2412", a, 3e6, mach=0.0)["cl"],
        foil.aero("naca2412", a, 3e6)["cl"],
    )


def test_prandtl_glauert_steepens_the_lift_curve():
    mach = 0.6
    base = foil.aero("naca2412", 4.0, 3e6)
    comp = foil.aero("naca2412", 4.0, 3e6, mach=mach)
    beta = np.sqrt(1.0 - mach**2)
    assert float(comp["cl"]) == pytest.approx(float(base["cl"]) / beta, rel=1e-9)
    # and does not touch cd -- there is no drag rise in this correction
    assert float(comp["cd"]) == pytest.approx(float(base["cd"]), rel=1e-12)


def test_supersonic_mach_is_refused_rather_than_silently_wrong():
    with pytest.raises(ValueError, match="subsonic"):
        foil.aero("naca2412", 2.0, 3e6, mach=1.2)


def test_nonpositive_reynolds_is_refused():
    with pytest.raises(ValueError, match="Re"):
        foil.aero("naca2412", 2.0, -1.0)
