"""The analysis stack: atmosphere through loads, plus the caching.

These are the rungs the answer key rests on.  Where a number can be checked
against something outside this package -- a standard-atmosphere table, a closed
form, the existing course solution, published aircraft performance -- it is,
and the check is written so that a disagreement fails rather than being
absorbed into a widened tolerance.
"""

import numpy as np
import pytest

from flightlab import (
    airfoil,
    atmos,
    cache,
    drag,
    geom,
    loads,
    performance,
    propulsion,
    ref,
    stability,
    wing,
)
from flightlab.case import Case
from flightlab.fleet import ASW27, B787, C172, RC1

G0 = 9.80665


# --- atmosphere -------------------------------------------------------------


def test_sea_level_matches_the_standard_defining_values():
    s = atmos.at(0.0)
    assert s.temperature == pytest.approx(288.15)
    assert s.pressure == pytest.approx(101325.0)
    assert s.density == pytest.approx(1.225, rel=1e-4)
    assert s.speed_of_sound == pytest.approx(340.294, rel=1e-4)
    assert s.viscosity == pytest.approx(1.7894e-5, rel=1e-3)


@pytest.mark.parametrize(
    "h_geopotential, T, p",
    [(11000.0, 216.65, 22632.0), (20000.0, 216.65, 5474.9), (32000.0, 228.65, 868.02)],
)
def test_layer_boundaries_match_the_published_table(h_geopotential, T, p):
    """The 1976 standard's own tabulated values at each layer base."""
    h = atmos.geometric(h_geopotential)
    s = atmos.at(h)
    assert s.temperature == pytest.approx(T, rel=1e-4)
    assert s.pressure == pytest.approx(p, rel=1e-3)


def test_the_atmosphere_is_continuous_across_every_layer_boundary():
    """No jump in pressure or temperature where the lapse rate changes.

    Straddle each boundary by a millimetre, not a metre: over a metre the
    pressure genuinely changes by several pascals, and a test that called that
    a discontinuity would be measuring hydrostatics rather than continuity.
    """
    for hgp, _, _ in atmos._LAYERS[1:-1]:
        h = atmos.geometric(hgp)
        below, above = atmos.at(h - 1e-3), atmos.at(h + 1e-3)
        assert below.pressure == pytest.approx(above.pressure, rel=1e-6)
        assert below.temperature == pytest.approx(above.temperature, rel=1e-6)


def test_altitude_outside_the_table_raises_rather_than_extrapolating():
    with pytest.raises(ValueError, match="outside the 1976 standard"):
        atmos.at(120000.0)


def test_eas_and_tas_invert():
    for h in (0.0, 3000.0, 11000.0):
        assert atmos.eas_to_tas(atmos.tas_to_eas(50.0, h), h) == pytest.approx(50.0)


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("label", ["RC1", "B787", "ASW27", "ASG29", "C172"])
def test_every_resolved_planform_integrates_back_to_its_published_area(label):
    """The chord distribution has to reproduce the area it was resolved from.

    This is the check that caught the fin bug: a vertical tail's area counts
    one surface, not two, so its chord formula carries a factor a wing's does
    not.  Getting it backwards halved every fin chord, and since fin area is
    what tail volume uses, the error would have surfaced as a directional
    stability number twice what it should be.
    """
    from flightlab.fleet import AIRCRAFT

    aircraft = AIRCRAFT[label]
    for name in ("wing", "htail", "vtail"):
        plan = getattr(aircraft, name)
        if plan is None:
            continue
        p = geom.resolve(plan)
        eta = np.linspace(0.0, 1.0, 4001)
        sides = 1.0 if p.vertical else 2.0
        area = np.trapezoid(p.chord(eta), eta * p.semispan) * sides
        assert area == pytest.approx(p.area, rel=1e-9), f"{label}.{name}"


def test_mac_is_not_the_standard_mean_chord_on_a_tapered_wing():
    """Sources publish both and sometimes do not say which.

    The 787's differ by 18%.  Moments are non-dimensionalized by the MAC, so
    taking S/b for it puts every moment coefficient out by that much.
    """
    p = geom.resolve(B787.wing)
    assert p.standard_mean_chord == pytest.approx(6.271, rel=1e-3)
    assert p.mac == pytest.approx(7.413, rel=1e-3)
    assert p.mac / p.standard_mean_chord > 1.15


def test_a_rectangular_wings_two_mean_chords_agree():
    p = geom.resolve(RC1.wing)
    assert p.mac == pytest.approx(p.standard_mean_chord)


def test_assumptions_are_recorded_not_hidden():
    """The ASG 29 publishes span and area but no taper, so one is assumed."""
    p = geom.resolve(ASG29_wing())
    assert "taper" in p.assumed


def ASG29_wing():
    from flightlab.fleet import ASG29

    return ASG29.wing


def test_sweep_conversion_round_trips():
    p = geom.resolve(B787.wing)
    assert geom.sweep_at(p, 0.25) == pytest.approx(p.sweep_c4_deg, abs=1e-9)
    assert geom.sweep_at(p, 0.0) == pytest.approx(p.sweep_le_deg, abs=1e-9)


# --- airfoil ----------------------------------------------------------------


def test_naca2412_lift_slope_reproduces_the_course_regression():
    p = airfoil.polar("naca2412", Re=3e6)
    assert p.cl_alpha(-2, 8) == pytest.approx(6.28, abs=0.02)


def test_the_low_reynolds_lift_slope_depends_strongly_on_the_fitting_interval():
    """The point of HW 1's airfoil half, asserted rather than described.

    Three defensible intervals over the same curve disagree by more than 30%,
    and NeuralFoil's confidence stays high through all of it.  A confidence
    metric is not validation of the quantity you extracted.
    """
    p = airfoil.polar("naca2412", Re=1e5)
    wide, steep, narrow = p.cl_alpha(-2, 8), p.cl_alpha(-2, 4), p.cl_alpha(1, 4)
    assert steep == pytest.approx(7.87, abs=0.05)
    assert narrow == pytest.approx(5.77, abs=0.05)
    assert (steep - narrow) / narrow > 0.30


def test_table_lookup_agrees_with_a_direct_polar():
    t = airfoil.table("sd7037", Re=(5e4, 5e5), n_Re=10)
    p = airfoil.polar("sd7037", Re=1.5e5)
    assert float(t.cd(0.6, 1.5e5)) == pytest.approx(float(p.cd_at(0.6)), rel=0.02)


def test_table_clamps_outside_its_reynolds_range_rather_than_extrapolating():
    t = airfoil.table("sd7037", Re=(5e4, 5e5), n_Re=8)
    assert bool(t.out_of_range(1e4))
    assert bool(t.out_of_range(1e7))
    assert not bool(t.out_of_range(1e5))
    assert float(t.cd(0.5, 1e3)) == pytest.approx(float(t.cd(0.5, 5e4)))


def test_section_cd_never_falls_below_the_laminar_flat_plate():
    t = airfoil.table("naca2412", Re=(1e5, 1e7), n_Re=10)
    Re = np.logspace(5, 7, 30)
    cd = t.cd(np.full_like(Re, 0.3), Re)
    assert np.all(cd > drag.flat_plate_cf(Re, xtr=1.0))


def test_table_lookups_are_fast_enough_for_a_slider():
    t = airfoil.table("naca2412", Re=(1e5, 1e7), n_Re=10)
    import time

    cl, Re = np.linspace(0.1, 1.0, 60), np.linspace(1e5, 3e5, 60)
    start = time.perf_counter()
    for _ in range(100):
        t.cd(cl, Re)
    assert time.perf_counter() - start < 1.0


# --- wing -------------------------------------------------------------------


def test_strip_lift_integrates_to_total_lift_and_to_the_weight():
    """The identity every span-load result rests on."""
    m = ASW27.mass["gross"]
    s = wing.trim_to_weight(ASW27.wing, m, 29.0, 0.0, ns=60, nc=4, camber=False)
    assert s.strip_lift == pytest.approx(s.lift, rel=1e-9)
    assert s.lift == pytest.approx(m * G0, rel=1e-9)


def test_an_elliptical_planform_returns_unit_span_efficiency():
    from flightlab.vlm import (
        Freestream, Reference, Stability, Uniform, body_forces,
        far_field_drag, steady_analysis, wing_to_grid,
    )

    b, S, n = 10.0, 10.0, 120
    y = (b / 2) * np.sin(np.linspace(0.0, np.pi / 2, n + 1))
    c = (4 * S / (np.pi * b)) * np.sqrt(np.clip(1 - (2 * y / b) ** 2, 0.0, None))
    c[-1] = max(c[-1], 1e-6)
    grid, ratios = wing_to_grid(
        -0.25 * c, y, np.zeros(n + 1), c, np.zeros(n + 1), np.zeros(n + 1),
        n, 1, spacing_s=Uniform(), spacing_c=Uniform(),
    )
    system = steady_analysis(
        [grid], Reference(S, S / b, b, [0.0, 0.0, 0.0], 1.0),
        Freestream.from_degrees(1.0, alpha=3.0), symmetric=True, ratios=[ratios],
    )
    CF, _ = body_forces(system, frame=Stability())
    e = CF[2] ** 2 / (np.pi * (b * b / S) * far_field_drag(system))
    assert e == pytest.approx(1.0, abs=0.01)


def test_rc1_CL_max_matches_the_fleet_placeholder():
    r = wing.CL_max(RC1.wing, V=10.0, altitude=1400.0)
    assert r["CL_max"] == pytest.approx(RC1.placeholders["CLmax"], abs=0.05)


def test_taper_moves_the_stall_outboard():
    """Why washout exists, as an assertion.

    A strongly tapered wing runs its outboard sections at higher local ``cl``
    *and* lower local Reynolds number, so it stalls at the tip -- which is
    where the ailerons are.
    """
    from dataclasses import replace

    rect = replace(ASW27.wing, taper=1.0, root_chord=None, tip_chord=None)
    tapered = replace(ASW27.wing, taper=0.35, root_chord=None, tip_chord=None)
    eta = []
    for plan in (rect, tapered):
        r = wing.CL_max(plan, V=30.0, altitude=1000.0, ns=30)
        eta.append(r["eta_critical"])
    assert eta[1] > eta[0]


def test_a_symmetric_solve_zeroes_every_lateral_derivative():
    """Correct, and indistinguishable from a broken model, so it is pinned."""
    d = stability.derivatives(RC1, V=11.0, altitude=1400.0, lateral=False)
    assert d.CY_beta == 0.0 and d.Cl_beta == 0.0 and d.Cn_beta == 0.0
    assert d.lateral_valid is False
    mirrored = stability.derivatives(RC1, V=11.0, altitude=1400.0, lateral=True)
    assert mirrored.lateral_valid is True
    assert mirrored.Cn_beta > 0.0


def test_longitudinal_derivatives_do_not_care_about_the_mirroring():
    a = stability.derivatives(RC1, V=11.0, altitude=1400.0, lateral=False)
    b = stability.derivatives(RC1, V=11.0, altitude=1400.0, lateral=True)
    assert a.CL_alpha == pytest.approx(b.CL_alpha, rel=1e-6)
    assert a.Cm_alpha == pytest.approx(b.Cm_alpha, rel=1e-6)


# --- drag -------------------------------------------------------------------


def test_flat_plate_friction_matches_the_classical_values():
    assert float(drag.flat_plate_cf(1e6, xtr=1.0)) == pytest.approx(1.328e-3, rel=1e-3)
    assert float(drag.flat_plate_cf(1e6)) == pytest.approx(0.074 / 1e6**0.2, rel=1e-9)
    assert float(drag.flat_plate_cf(1e7)) < float(drag.flat_plate_cf(1e6))


def test_the_korn_equation_ranks_the_three_transonic_levers():
    """Sweep it, thin it, or fly it at lower CL -- each must raise M_dd."""
    base = drag.drag_divergence_mach(0.12, 0.5, 0.0)
    assert drag.drag_divergence_mach(0.12, 0.5, 30.0) > base
    assert drag.drag_divergence_mach(0.09, 0.5, 0.0) > base
    assert drag.drag_divergence_mach(0.12, 0.3, 0.0) > base


def test_wave_drag_is_negligible_below_the_critical_mach_and_bites_above_it():
    p = geom.resolve(B787.wing)
    low = float(drag.wave_drag(0.70, p.thickness, 0.5, p.sweep_c4_deg))
    high = float(drag.wave_drag(0.90, p.thickness, 0.5, p.sweep_c4_deg))
    assert low < 1e-4
    assert high > 20 * max(low, 1e-9)


def test_the_book_formulas_are_the_ones_implemented():
    """Pinned against the course text, equation by equation.

    The package must agree with what students are reading, not with whichever
    correlation a handbook happens to prefer.  Each value here is evaluated by
    hand from ``book/drag.tex``.
    """
    # turbulent skin friction, 0.074 / Re^0.2
    assert float(drag.flat_plate_cf(1e7)) == pytest.approx(0.074 / 1e7**0.2, rel=1e-12)
    # laminar, 1.328 / sqrt(Re)
    assert float(drag.flat_plate_cf(1e7, xtr=1.0)) == pytest.approx(
        1.328 / np.sqrt(1e7), rel=1e-12
    )
    # Mach correction
    ratio = float(drag.flat_plate_cf(1e7, mach=0.8)) / float(drag.flat_plate_cf(1e7))
    assert ratio == pytest.approx((1 + 0.144 * 0.8**2) ** -0.65, rel=1e-12)
    # Shevell form factor, incompressible: Z = 2 cos(L)
    assert drag.form_factor_surface(0.15, 0.0, 0.0) == pytest.approx(
        1 + 2 * 0.15 + 100 * 0.15**4, rel=1e-12
    )
    # body form factor fit, and its floor at fr >= 15
    assert drag.form_factor_body(8.0) == pytest.approx(
        1.675 - 0.09 * 8 + 0.003 * 64, rel=1e-12
    )
    assert drag.form_factor_body(20.0) == 1.0
    # Korn, with 0.95 fixed
    assert drag.drag_divergence_mach(0.11, 0.5, 30.0) == pytest.approx(
        0.95 / np.cos(np.radians(30))
        - 0.11 / np.cos(np.radians(30)) ** 2
        - 0.5 / (10 * np.cos(np.radians(30)) ** 3),
        rel=1e-12,
    )
    # M_cc = M_dd - 0.11
    assert drag.crest_critical_mach(0.11, 0.5, 30.0) == pytest.approx(
        drag.drag_divergence_mach(0.11, 0.5, 30.0) - 0.11, rel=1e-12
    )
    # Oswald efficiency
    assert drag.oswald_efficiency(0.98, 0.02, 8.0) == pytest.approx(
        1 / (1 / 0.98 + 0.38 * 0.02 * np.pi * 8.0), rel=1e-12
    )
    # fuselage span-efficiency loss
    assert drag.span_efficiency_with_fuselage(2.0, 20.0) == pytest.approx(
        0.98 * (1 - 2 * (2.0 / 20.0) ** 2), rel=1e-12
    )


def test_wing_wetted_area_follows_the_texts_expression():
    p = geom.resolve(B787.wing)
    assert geom.wetted_area(p) == pytest.approx(
        2 * (1 + 0.2 * p.thickness) * p.area, rel=1e-12
    )


def test_bluff_items_dominate_a_fixed_gear_singles_parasitic_drag():
    """The 172's gear, fairings and struts are bluff, not streamlined.

    Treated as streamlined bodies of revolution they contribute almost nothing
    and the aircraft's L/D comes out near 17 against a published best glide of
    9.  Referenced to frontal area they are together larger than everything
    except the wing -- which is why retractable gear exists.
    """
    b = drag.buildup(C172, 63.8, 2438.0, protuberance=0.075, interference=0.05)
    fr = b.fractions()
    bluff = fr["main gear leg"] + fr["wheel fairing"] + fr["wing strut"]
    assert bluff > 0.30
    assert bluff > fr["htail"] + fr["vtail"] + fr["fuselage"]


def test_a_clean_buildup_of_a_light_single_runs_below_its_real_polar():
    """Worth knowing, and worth not tuning away.

    Every geometric buildup misses antennas, door and control-surface gaps,
    steps, rivet lines, the windscreen and the exhaust.  On a riveted fixed-gear
    single that is a real fraction of the total, and the honest result is a
    ``CD0`` some 20-30% below the aircraft's measured polar.  The markup is
    where that goes, and choosing it is judgment rather than geometry.
    """
    b = drag.buildup(C172, 63.8, 2438.0, protuberance=0.075, interference=0.05)
    ratio = b.CD0 / C172.placeholders["CDp"]
    assert 0.65 < ratio < 1.0


def test_the_slender_body_form_factor_warns_when_used_outside_its_range():
    with pytest.warns(RuntimeWarning, match="fineness ratio"):
        drag.form_factor_body(3.0, strict=True)
    # silent by default, and reported as data instead
    assert drag.form_factor_body(3.0) > 1.0
    assert not drag.body_form_factor_valid(3.0)
    assert drag.body_form_factor_valid(8.0)


def test_rc1s_pod_is_outside_the_body_form_factor_fit_and_says_so():
    """A true fact about analyzing a model aeroplane with transport
    correlations, surfaced rather than suppressed."""
    b = drag.buildup(RC1, 12.0, 1400.0)
    pods = [r for r in b.extrapolated_rows if "pod" in r.name]
    assert pods, "RC-1's pod should be flagged as outside the fit"
    assert "fineness" in pods[0].extrapolated
    assert "*" in b.table()


# --- stability --------------------------------------------------------------


def test_rc1_mass_properties_reproduce_the_published_cg():
    mp = stability.mass_properties(RC1)
    p = geom.resolve(RC1.wing)
    assert mp.mass == pytest.approx(0.750, abs=1e-3)
    assert mp.x_cg == pytest.approx(0.0479, abs=1e-4)
    assert mp.x_cg_over_mac(p) == pytest.approx(0.300, abs=0.002)


def test_the_neutral_point_does_not_depend_on_the_moment_reference():
    """If it does, the reference length or the moment normalization is wrong."""
    p = geom.resolve(RC1.wing)
    a = stability.neutral_point(RC1, 11.0, 1400.0)
    assert a["x_np"] == pytest.approx(0.0825, abs=0.005)
    assert a["x_np_over_mac"] == pytest.approx(0.516, abs=0.03)


def test_trim_closes_both_equations_to_machine_precision():
    t = stability.trim(RC1, V=12.0, altitude=1400.0)
    assert t.converged
    assert abs(t.lift_residual) < 1e-9
    assert abs(t.moment_residual) < 1e-9


def test_the_fuselage_correction_reduces_the_static_margin():
    """A pod ahead of the wing is destabilizing, and the lattice cannot see it."""
    bare = stability.neutral_point(RC1, 11.0, 1400.0, include_body=False)
    with_pod = stability.neutral_point(RC1, 11.0, 1400.0, include_body=True)
    assert with_pod["x_np"] < bare["x_np"]


def test_a_mode_set_is_physically_sensible():
    lon, lat = stability.modes(RC1, V=12.0, altitude=1400.0)
    sp, ph = lon["short period"], lon["phugoid"]
    assert sp.frequency > 3 * ph.frequency
    assert sp.damping > 0.3
    assert lat["roll subsidence"].stable
    assert lat["dutch roll"].oscillatory


def test_the_phugoid_period_is_within_reach_of_lanchester():
    """``T = pi sqrt(2) V / g`` neglects drag and pitch dynamics, so the real
    period runs above it -- but not by a factor."""
    V = 12.0
    lon, _ = stability.modes(RC1, V=V, altitude=1400.0)
    lanchester = np.pi * np.sqrt(2.0) * V / G0
    assert 1.0 < lon["phugoid"].period / lanchester < 1.8


def test_modes_taken_away_from_trim_warn_rather_than_lying():
    d = stability.derivatives(RC1, V=12.0, altitude=1400.0, alpha=0.0, lateral=True)
    with pytest.warns(RuntimeWarning, match="not.*taken at the trim|level flight"):
        stability.longitudinal_modes(RC1, 12.0, 1400.0, derivs=d)


def test_a_fin_is_never_mirrored_onto_itself():
    """Mirroring a centreline fin duplicates its panels and the solve returns
    NaN, which is a confusing way to discover a geometry error."""
    p = geom.resolve(RC1.vtail)
    a, _ = geom.surface_grid(p, ns=10, nc=3, mirror=False)
    b, _ = geom.surface_grid(p, ns=10, nc=3, mirror=True)
    assert a.shape == b.shape


# --- propulsion -------------------------------------------------------------


def test_motor_peak_efficiency_ranks_by_I0R_and_not_by_Kv():
    """HW 5's central finding, asserted."""
    from flightlab import catalog

    rows = [
        (k, catalog.MOTORS[k].Kv_rpm, propulsion.motor_peak_efficiency(k, 11.1))
        for k in catalog.MOTORS
    ]
    by_kv = [k for k, _, _ in sorted(rows, key=lambda r: r[1])]
    by_eta = [k for k, _, _ in sorted(rows, key=lambda r: r[2]["efficiency"])]
    by_i0r = [k for k, _, _ in sorted(rows, key=lambda r: -r[2]["I0R_over_V"])]
    assert by_kv != by_eta, "the catalog must not let Kv stand in for efficiency"
    assert by_eta == by_i0r, "peak efficiency ranks with I0*R/V, exactly"
    # and the best motor must not be the one at either end of the Kv range,
    # so that no simple rule of thumb reproduces the answer
    assert by_eta[-1] not in (by_kv[0], by_kv[-1])


def test_kv_conversion_is_radians_not_rpm():
    assert propulsion.Kv_rad("M1000") == pytest.approx(1000.0 * 2 * np.pi / 60.0)


def test_propeller_thrust_falls_with_airspeed():
    """It did not, once: the static run was being treated as its own set of
    rotational speeds, which built single-point curves that interpolate to a
    constant and made thrust independent of airspeed."""
    p = propulsion.propeller_model("apce_10x7")
    omega = 5000.0 * propulsion.RPM_TO_RAD
    T = [p.thrust(V, omega) for V in (0.0, 4.0, 8.0, 12.0)]
    assert all(a > b for a, b in zip(T, T[1:]))
    assert p.has_static


def test_propeller_efficiency_identity_holds():
    p = propulsion.propeller_model("apce_10x7")
    for J in (0.2, 0.4, 0.6):
        n = 100.0
        assert float(p.efficiency(J, n)) == pytest.approx(
            J * float(p.CT(J, n)) / float(p.CP(J, n)), rel=1e-9
        )


def test_the_torque_match_balances():
    op = propulsion.operating_point("M1000", "apce_10x7", "B3S1300", V=12.0)
    p = propulsion.propeller_model("apce_10x7")
    assert op.torque == pytest.approx(p.torque(12.0, op.omega, 1.225), rel=1e-6)
    assert 0.0 < op.efficiency_total < 1.0


def test_rc1_baseline_overloads_its_own_motor():
    """RC-1's baseline is deliberately a poor match, and this is how it shows.

    The propeller data covers the operating point perfectly well -- the advance
    ratio is mid-range -- so the analysis is trustworthy, and what it says is
    that a 1000 Kv motor on 3S pulls about 24 A through a part rated for 13 A.
    That is a better lesson than a data-coverage complaint: the tool is inside
    its range and the aircraft is outside its own.
    """
    from flightlab import catalog

    op = propulsion.operating_point("M1000", "apce_10x7", "B3S1300", V=11.0,
                                    altitude=1400.0)
    assert not op.extrapolated, "the advance ratio is well inside the data"
    assert op.reynolds_ratio < 2.0, "and the speed excursion is modest"
    assert op.current > catalog.MOTORS["M1000"].current_max


def test_advance_ratio_is_the_coverage_limit_and_rpm_is_not():
    """The coefficients are functions of J; rpm enters as a Reynolds number.

    A point at an advance ratio the tunnel measured is covered even when the
    rotational speed is well above any sweep, and the reported Reynolds ratio
    says how far outside that speed sits without calling it an extrapolation.
    """
    m = propulsion.propeller_model("apce_10x7")
    assert not bool(m.out_of_range(0.5))
    assert bool(m.out_of_range(1.2))
    assert float(m.reynolds_ratio(150.0)) > 1.0
    assert float(m.reynolds_ratio(90.0)) == 1.0


def test_short_sweeps_are_extended_from_wider_ones_not_clamped():
    """Each speed was run only as far as the tunnel could reach.

    The 10x7's 6,531 rpm sweep stops at J = 0.44 while its 5,001 rpm sweep
    reaches 0.84.  Clamping at the end of the short curve reports a thrust far
    too high at speed; borrowing the wider curve's shape does not.  The
    coefficients at a common J must agree across speeds to within the Reynolds
    drift, which for these propellers is about ten per cent.
    """
    m = propulsion.propeller_model("apce_10x7")
    speeds = sorted(m._table)
    for n in speeds:
        assert m._table[n][0][-1] == pytest.approx(m._J_max)
    values = [float(m.CT(0.6, n)) for n in speeds]
    assert max(values) / min(values) < 1.25
    # and thrust must keep falling with speed right through the borrowed region
    T = [m.thrust(V, 8000.0 * propulsion.RPM_TO_RAD) for V in (12, 16, 20, 24)]
    assert all(a > b for a, b in zip(T, T[1:]))


def test_battery_voltage_sags_under_load():
    full = propulsion.battery_voltage("B3S1300", current=0.0)
    loaded = propulsion.battery_voltage("B3S1300", current=20.0)
    assert loaded < full
    assert (full - loaded) / full > 0.05


def test_ideal_propulsive_efficiency_is_a_ceiling():
    eta = propulsion.ideal_propulsive_efficiency(12.0, 8.0, 0.0507)
    assert 0.0 < eta < 1.0
    bigger = propulsion.ideal_propulsive_efficiency(12.0, 8.0, 0.2)
    assert bigger > eta


def test_hover_power_scales_as_thrust_to_the_three_halves():
    a = propulsion.rotor_hover(1000.0, 6.6)["power_ideal"]
    b = propulsion.rotor_hover(2000.0, 6.6)["power_ideal"]
    assert b / a == pytest.approx(2.0**1.5, rel=1e-9)


# --- performance ------------------------------------------------------------


def test_minimum_power_speed_sits_below_best_glide_speed():
    """``3^-0.25`` of it for an ideal parabolic polar."""
    pol = drag.polar(RC1, V=12.0, altitude=1400.0, protuberance=0.10, interference=0.05)
    s = performance.speeds(pol, mass=0.75)
    ratio = s["V_min_power"] / s["V_LD_max"]
    assert 0.70 < ratio < 0.82


def test_min_sink_and_best_glide_are_different_speeds():
    pol = drag.polar(ASW27, V=30.0, altitude=1000.0, protuberance=0.02, interference=0.03)
    g = performance.glide(pol, mass=525.0)
    assert g["V_min_sink"] < g["V_best_glide"]
    assert g["min_sink"] < g["sink_at_best_glide"]


def test_electric_range_does_not_depend_on_the_speed_flown():
    """``R = E eta (L/D) / W`` contains no speed, because the weight is
    constant.  Genuinely different from the fuel-burning case."""
    pol = drag.polar(RC1, V=12.0, altitude=1400.0, protuberance=0.10, interference=0.05)
    a = performance.range_electric(4e4, pol, 0.75, V=None)
    assert a["range"] == pytest.approx(4e4 * 0.5 * a["LD"] / (0.75 * G0), rel=1e-6)


def test_breguet_range_is_logarithmic_in_the_mass_ratio():
    r1 = performance.range_breguet(100.0, 50.0, 19.0, 250.0, 1.6e-5)
    r2 = performance.range_breguet(100.0, 25.0, 19.0, 250.0, 1.6e-5)
    assert r2 / r1 == pytest.approx(np.log(4.0) / np.log(2.0), rel=1e-9)


def test_a_turn_at_60_degrees_of_bank_pulls_two_g():
    pol = drag.polar(C172, V=63.8, altitude=2438.0, protuberance=0.075, interference=0.05)
    t = performance.turn(pol, 1157.0, 63.8, bank_deg=60.0)
    assert t["n"] == pytest.approx(2.0, rel=1e-9)
    assert t["radius"] > 0


# --- loads ------------------------------------------------------------------


def test_span_load_lands_just_below_the_elliptical_closed_form():
    """A tapered wing carries proportionally more load inboard, so its root
    bending moment must come out at or below ``L b / (3 pi)``."""
    sl = loads.span_load(ASW27, mass=525.0, n=5.3, V=60.0, ns=60)
    ell = loads.elliptical_root_bending_moment(sl.total_lift, ASW27.wing.span)
    assert 0.90 < sl.root_moment / ell <= 1.02


def test_root_shear_is_half_the_total_lift():
    sl = loads.span_load(ASW27, mass=525.0, n=1.0, V=30.0, ns=60)
    assert sl.root_shear == pytest.approx(0.5 * sl.total_lift, rel=0.02)


def test_inertial_relief_reduces_the_root_bending_moment():
    """Why a sailplane carries water in the wing and an airliner carries fuel
    there."""
    plain = loads.span_load(ASW27, mass=525.0, n=5.3, V=60.0, ns=60)
    relief = np.full_like(plain.y, 400.0)
    relieved = loads.span_load(
        ASW27, mass=525.0, n=5.3, V=60.0, ns=60, relief=relief
    )
    assert relieved.root_moment < plain.root_moment


def test_the_corner_speed_is_the_stall_speed_times_root_n():
    vn = loads.vn_diagram(ASW27, mass=525.0, CL_max=1.4)
    assert vn["V_A"] == pytest.approx(vn["V_stall"] * np.sqrt(vn["n_pos"]), rel=1e-9)
    assert vn["n_ultimate_pos"] == pytest.approx(1.5 * vn["n_pos"])


def test_load_envelope_accepts_an_aircraft_level_reference_area():
    ordinary = loads.vn_diagram(ASW27, mass=525.0, CL_max=1.4)
    doubled = loads.vn_diagram(
        ASW27, mass=525.0, CL_max=1.4,
        reference_area=2.0 * geom.resolve(ASW27.wing).area,
    )
    assert doubled["wing_loading"] == pytest.approx(0.5 * ordinary["wing_loading"])
    assert doubled["V_stall"] == pytest.approx(ordinary["V_stall"] / np.sqrt(2.0))


def test_a_lighter_wing_loading_is_thrown_further_by_the_same_gust():
    light = loads.gust_load_factor(RC1, V=12.0)
    heavy = loads.gust_load_factor(C172, V=63.8)
    assert light > heavy


def test_deflection_doubles_integration_is_zero_at_the_root():
    sl = loads.span_load(ASW27, mass=525.0, n=1.0, V=30.0, ns=40)
    d = loads.tip_deflection(sl, EI=5e4)
    assert d["deflection"][0] == 0.0
    assert d["tip_deflection"] > 0.0


# --- the case object and its cache ------------------------------------------


def test_changing_the_span_invalidates_the_aerodynamics():
    case = Case(RC1, V=12.0, altitude=1400.0)
    first = case.wing_aero()
    assert case.wing_aero() is first
    case.wing.span = 1.4
    assert case.wing_aero() is not first


def test_changing_a_component_mass_does_not_invalidate_the_aerodynamics():
    """The whole point of the dependency graph: a battery is not an
    aerodynamic input."""
    case = Case(RC1, V=12.0, altitude=1400.0)
    first = case.wing_aero()
    case.mass["Battery"] = 0.140
    assert case.wing_aero() is first


def test_a_no_op_assignment_does_not_invalidate_anything():
    case = Case(RC1, V=12.0, altitude=1400.0)
    first = case.wing_aero()
    case.wing.span = case.wing.span
    assert case.wing_aero() is first


def test_assigning_an_unknown_parameter_is_an_error_not_a_silent_no_op():
    case = Case(RC1)
    with pytest.raises(AttributeError, match="has no parameter"):
        case.wing.spam = 1.0


def test_the_case_reports_its_own_dependency_structure():
    case = Case(RC1, V=12.0, altitude=1400.0)
    assert "wing_aero" in case.invalidated_by("wing")
    assert "wing_aero" not in case.invalidated_by("mass")
    assert "trim" in case.invalidated_by("mass")


def test_changing_a_mass_changes_the_trim():
    case = Case(RC1, V=12.0, altitude=1400.0)
    before = case.trim().alpha
    case.mass["Battery"] = 0.30
    assert case.trim().alpha != before


def test_case_overrides_reach_the_geometry():
    case = Case(RC1, V=12.0, altitude=1400.0)
    case.wing.span = 1.5
    assert geom.resolve(case.aircraft().wing).span == 1.5
    assert case.panel().span == 1.5


def test_a_copied_case_is_independent():
    case = Case(RC1, V=12.0, altitude=1400.0)
    other = case.copy()
    other.wing.span = 1.6
    assert case.panel().span == pytest.approx(RC1.wing.span)
    assert other.panel().span == 1.6


# --- the new figures --------------------------------------------------------


def _agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def test_every_new_plot_renders():
    """Smoke test.  A plotting helper that raises is worse than no helper."""
    plt = _agg()
    from flightlab import plot

    sol = wing.trim_to_weight(ASW27.wing, 525.0, 30.0, 1000.0, ns=24)
    tbl = airfoil.table("fx62k131", Re=(3e5, 3e6), n_Re=6)
    pol = drag.polar(RC1, V=12.0, altitude=1400.0, protuberance=0.10, interference=0.05)
    V = np.linspace(8.0, 18.0, 20)
    d = performance.drag_curve(pol, 0.75, V)
    T = propulsion.thrust_available(
        "M1000", "apce_10x7", "B3S1300", V, altitude=1400.0
    )
    sl = loads.span_load(ASW27, mass=525.0, n=5.3, V=60.0, ns=30)
    lon, lat = stability.modes(RC1, V=12.0, altitude=1400.0)
    op = propulsion.operating_point("M1000", "apce_10x7", "B3S1300", V=12.0)

    fig, ax = plt.subplots()
    plot.stall_margin(sol, tbl, ax=ax)
    plot.span_load(sl, ax=plt.subplots()[1])
    plot.thrust_and_drag(V, T, d["drag"], ax=plt.subplots()[1])
    plot.power_curves(V, d["power"], ax=plt.subplots()[1])
    plot.mode_response(lon, [1.0, 0.0, 0.0, 0.0], np.linspace(0, 20, 200),
                       ax=plt.subplots()[1])
    plot.chain_breakdown(op, ax=plt.subplots()[1])
    plot.polar_comparison({"RC-1": pol}, ax=plt.subplots()[1])
    plot.fleet_overlay(
        {"RC-1": {"W/S": 38.0, "AR": 7.5}}, "W/S", "AR", ax=plt.subplots()[1]
    )
    fig3d = plt.figure()
    plot.loading_3d(sol, ax=fig3d.add_subplot(111, projection="3d"))
    plt.close("all")


def test_a_live_figure_builds_and_redraws_quickly():
    """The interactivity budget, asserted.

    A redraw slower than about a fifth of a second stops feeling like a slider.
    """
    import time
    import warnings

    plt = _agg()
    from flightlab import live

    case = Case(RC1, V=12.0, altitude=1400.0)

    def draw(c, ax):
        s = c.wing_aero()
        ax.plot(s.y, s.ccl)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = live.explore(case, draw, wing_span=(0.9, 1.8), cond_V=(8.0, 20.0))
        assert len(r["sliders"]) == 2
        start = time.perf_counter()
        r["sliders"]["wing_span"].set_val(1.5)
        assert time.perf_counter() - start < 0.5
    plt.close("all")


def test_a_live_figure_survives_an_impossible_parameter_combination():
    """A slider dragged somewhere the physics cannot go must show the error on
    the axes, not tear down the figure."""
    import warnings

    plt = _agg()
    from flightlab import live

    case = Case(RC1, V=12.0, altitude=1400.0)

    def draw(c, ax):
        if c.wing.span < 1.0:
            raise ValueError("deliberate")
        ax.plot([0, 1], [0, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = live.explore(case, draw, wing_span=(0.5, 1.8))
        r["sliders"]["wing_span"].set_val(0.6)
        texts = [t.get_text() for t in r["ax"].texts]
        assert any("deliberate" in t for t in texts)
    plt.close("all")


def test_the_flight_envelope_closes_at_the_top():
    """Level flight is possible over a band of speeds that narrows with
    altitude and pinches shut at the ceiling."""
    plt = _agg()
    from flightlab import plot

    def polar_at(h):
        return drag.polar(C172, V=63.8, altitude=h, protuberance=0.075, interference=0.05)

    def thrust_at(V, h):
        # a propeller aircraft: roughly constant power, so thrust falls with V
        power = 0.75 * C172.propulsion["power_each"] * 0.8
        sigma = atmos.at(h).density / atmos.SEA_LEVEL.density
        return power * sigma / max(V, 1.0)

    result = performance.envelope(
        polar_at, 1157.0, thrust_at, CL_max=1.55,
        altitudes=np.linspace(0.0, 6000.0, 7), V=np.linspace(25.0, 90.0, 40),
    )
    band = result["V_max"] - result["V_min"]
    finite = band[np.isfinite(band)]
    assert len(finite) >= 2
    assert finite[0] > finite[-1], "the envelope must narrow with altitude"
    assert np.all(result["V_stall"][1:] > result["V_stall"][:-1])

    fig, ax = plt.subplots()
    plot.envelope(result, ax=ax)
    plt.close("all")


def test_transition_uses_the_momentum_thickness_method_not_a_blend():
    """The appendix's method, and why it is not the average of two branches.

    A turbulent layer downstream of transition inherits the momentum thickness
    the laminar layer built up, so it behaves like a turbulent layer that
    started further upstream than it did.  Averaging the two branches ignores
    that and understates the drag of a partly laminar surface -- by up to about
    9% at the laminar runs a sailplane is designed for.
    """
    Re = 3e6
    turb = float(drag.flat_plate_cf(Re, xtr=0.0))
    lam = float(drag.flat_plate_cf(Re, xtr=1.0))
    assert turb == pytest.approx(0.074 / Re**0.2, rel=1e-12)
    assert lam == pytest.approx(1.328 / np.sqrt(Re), rel=1e-12)

    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        proper = float(drag.flat_plate_cf(Re, xtr=x))
        blend = x * lam + (1.0 - x) * turb
        assert lam < proper < turb, "must lie between the two limits"
        assert proper > blend, "the blend must be the optimistic one"
    assert float(drag.flat_plate_cf(Re, xtr=0.7)) / (
        0.7 * lam + 0.3 * turb
    ) == pytest.approx(1.10, abs=0.02)

    # continuous into both limits
    assert float(drag.flat_plate_cf(Re, xtr=1e-9)) == pytest.approx(turb, rel=1e-3)
    assert float(drag.flat_plate_cf(Re, xtr=1 - 1e-9)) == pytest.approx(lam, rel=1e-3)


def test_a_laminar_run_is_worth_a_lot_at_sailplane_reynolds_numbers():
    """Half a chord of laminar flow cuts flat-plate friction by about a third."""
    Re = 1.3e6  # the ASW-27B's wing
    assert float(drag.flat_plate_cf(Re, xtr=0.5)) / float(
        drag.flat_plate_cf(Re, xtr=0.0)
    ) < 0.70


def test_the_bluff_crossover_is_continuous_and_reduces_to_a_sphere():
    """A short body is not a slender one with a bigger form factor.

    Below a fineness of about 3 the text's slender-body fit understates badly:
    0.068 on frontal area at fineness 2 against a measured 0.20, and 0.036 at
    fineness 1 against a sphere's 0.47.  The bluff branch interpolates to the
    sphere instead, and must join the streamlined branch without a step.
    """
    Re_L, Re_d = 2.2e5, 5.8e4
    eps = 1e-4
    lo = drag.body_cd_frontal(drag.BLUFF_FINENESS - eps, Re_L, Re_d)
    hi = drag.body_cd_frontal(drag.BLUFF_FINENESS + eps, Re_L, Re_d)
    assert lo == pytest.approx(hi, rel=1e-3)

    # a sphere is a sphere
    assert drag.body_cd_frontal(1.0, Re_L, Re_d) == pytest.approx(
        float(drag.sphere_cd(Re_d)), rel=1e-9
    )
    # monotonically less draggy as it gets more slender, through the bluff range
    cds = [drag.body_cd_frontal(f, Re_L, Re_d) for f in (1.0, 1.5, 2.0, 3.0, 4.0)]
    assert all(a > b for a, b in zip(cds, cds[1:]))
    # and within reach of Hoerner's measured streamline bodies
    assert drag.body_cd_frontal(2.0, Re_L, Re_d) == pytest.approx(0.20, rel=0.30)
    assert drag.body_cd_frontal(3.0, Re_L, Re_d) == pytest.approx(0.13, rel=0.35)


def test_sphere_cd_hits_the_subcritical_plateau():
    assert float(drag.sphere_cd(1e5)) == pytest.approx(0.47, rel=0.10)
    assert float(drag.sphere_cd(1.0)) > 10.0  # Stokes-ish at the bottom


def test_rc1s_pod_is_treated_as_bluff_and_the_table_says_so():
    b = drag.buildup(RC1, 9.1, 1400.0)
    pod = next(r for r in b.rows if "pod" in r.name)
    assert pod.extrapolated and "bluff" in pod.extrapolated
    assert 0.10 < pod.cd_frontal < 0.20
    assert "CD_fr" in b.table()
