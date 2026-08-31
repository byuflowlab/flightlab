"""Consistency checks on the provided data: fleet, airfoils, propellers, catalog.

The whole reason these tables are provided is that transcription errors look
exactly like physics errors.  So the tables get checked.
"""

import numpy as np
import pytest

from flightlab import catalog, fleet, foil, props, ref


# --- fleet ------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(fleet.AIRCRAFT))
def test_wing_area_is_consistent_with_the_chords(label):
    """Where root and tip chords are given, they must reproduce the area."""
    a = fleet.AIRCRAFT[label]
    w = a.wing
    if w is None or w.root_chord is None or w.tip_chord is None:
        pytest.skip(f"{label} has no chord breakdown")
    trapezoid = 0.5 * (w.root_chord + w.tip_chord) * w.span
    assert trapezoid == pytest.approx(w.area, rel=1e-6)


@pytest.mark.parametrize(
    "label,published_AR",
    [("RC1", 7.5), ("B787", 9.59), ("ASW27", 25.0), ("ASG29", 30.9),
     ("F16", 3.2), ("DC3", 10.55)],
)
def test_aspect_ratio_matches_published(label, published_AR):
    a = fleet.AIRCRAFT[label]
    assert a.wing.aspect_ratio == pytest.approx(published_AR, rel=2e-3)


def test_no_aircraft_stores_a_redundant_aspect_ratio():
    """``b^2/S`` is a definition, so it is computed and never stored.

    Storing it invites exactly the failure this test replaced: an early draft of
    the fleet page listed the ASG 29 at AR 30.4 alongside an 18.0 m span and a
    10.5 m^2 area, which give 30.86.  Deriving it makes that impossible.

    The F-16 is not an instance of this: its 9.96 m over the launcher rails and
    its 9.45 m reference span measure two different things, and both are kept
    on purpose.

    Two suffixes are allowed, and only because they name numbers that are *not*
    the modelled wing's ``b^2/S``:

    ``_actual``
        the real wing, where the course analyzes a simplified one (the DC-3).
    ``_published``
        what the manufacturer prints, where it disagrees with its own span and
        area (the Cessna 172S: 36 ft 1 in and 174 sq ft give 7.48, and Cessna
        publishes 7.32).

    Both must actually disagree with the derived value.  A stored figure that
    merely repeats ``b^2/S`` is redundant no matter what it is called, so this
    test checks the disagreement rather than trusting the suffix.
    """
    allowed = {"aspect_ratio_actual", "aspect_ratio_published"}
    for label, a in fleet.AIRCRAFT.items():
        if a.wing is None:
            continue
        for key, value in a.published.items():
            if "aspect_ratio" not in key:
                continue
            assert key in allowed, (
                f"{label} stores a redundant aspect ratio in published[{key!r}]"
            )
            derived = a.wing.aspect_ratio
            assert abs(value - derived) / derived > 0.005, (
                f"{label}.published[{key!r}] = {value} agrees with the derived "
                f"b^2/S = {derived:.4f}, so it is redundant and should be "
                "deleted rather than kept under an exempt name"
            )


def test_c172_published_aspect_ratio_disagrees_with_its_own_span_and_area():
    """Cessna's own three numbers are not mutually consistent, and we keep it.

    The POH gives span 36 ft 1 in, area 174 sq ft, and aspect ratio 7.32.  The
    first two give 7.48.  This is the same exercise as the F-16's two spans,
    applied to a document thousands of people fly behind, and HW 1 asks
    students to find it.
    """
    a = fleet.C172
    assert a.wing.aspect_ratio == pytest.approx(7.483, rel=1e-3)
    assert a.published["aspect_ratio_published"] == 7.32


def test_asg29_aspect_ratio_follows_from_span_and_area():
    a = fleet.ASG29
    assert a.wing.span == 18.0
    assert a.wing.area == 10.5
    assert a.wing.aspect_ratio == pytest.approx(30.857, rel=1e-4)
    assert "aspect_ratio_quoted" not in a.published


def test_the_two_sailplanes_are_comparable_for_hw4():
    """HW 4's spine: three more metres of span, same manufacturer.

    Both aspect ratios must follow from their own span and area, or the
    comparison HW 4 asks for is measuring a transcription error.
    """
    for a in (fleet.ASW27, fleet.ASG29):
        assert a.wing.aspect_ratio == pytest.approx(
            a.wing.span**2 / a.wing.area, rel=1e-12
        )
    assert fleet.ASG29.wing.span - fleet.ASW27.wing.span == pytest.approx(3.0)
    assert fleet.ASG29.wing.aspect_ratio > fleet.ASW27.wing.aspect_ratio


@pytest.mark.parametrize("label", ["RC1", "B787", "ASW27", "F16", "DC3"])
def test_taper_matches_the_chords(label):
    w = fleet.AIRCRAFT[label].wing
    if w.root_chord is None or w.tip_chord is None or w.taper is None:
        pytest.skip(f"{label} has no chord breakdown")
    assert w.taper == pytest.approx(w.tip_chord / w.root_chord, rel=1e-6)


def test_rc1_component_table_sums_to_the_stated_mass():
    assert fleet.RC1.component_mass_total == pytest.approx(0.750, abs=1e-12)


def test_rc1_component_table_reproduces_the_published_cg():
    """HW 6's mass-properties sanity rung, checked on the data itself."""
    comps = fleet.RC1.components
    total = sum(c.mass for c in comps)
    x_cg = sum(c.mass * c.x for c in comps) / total
    assert x_cg == pytest.approx(fleet.RC1.propulsion["x_cg_published"], abs=5e-4)
    frac = x_cg / fleet.RC1.wing.mean_chord
    assert frac == pytest.approx(
        fleet.RC1.propulsion["x_cg_fraction_of_chord"], abs=2e-3
    )


def test_rc1_tail_boom_spans_the_gap_it_is_supposed_to():
    """The boom runs from the aft end of the pod to the tail leading edge."""
    pod, boom = fleet.RC1.bodies
    gap = fleet.RC1.htail.x_le - (pod.x_nose + pod.length)
    assert gap == pytest.approx(boom.length, abs=1e-9)


@pytest.mark.parametrize("label", ["RC1", "B787", "ASW27"])
def test_tail_areas_are_consistent(label):
    a = fleet.AIRCRAFT[label]
    for tail in (a.htail, a.vtail):
        if tail is None or tail.root_chord is None:
            continue
        assert tail.span * tail.root_chord == pytest.approx(tail.area, rel=1e-6)


def test_rc1_quarter_chord_positions_agree_with_leading_edges():
    for surf in (fleet.RC1.wing, fleet.RC1.htail, fleet.RC1.vtail):
        expected = surf.x_le + 0.25 * surf.root_chord
        assert surf.x_c4 == pytest.approx(expected, abs=1e-9)


def test_every_aircraft_names_its_sources():
    for label, a in fleet.ALL.items():
        assert a.sources, f"{label} has no sources"


def test_stand_in_sections_are_flagged_and_resolvable():
    """A stand-in must be detectable and its coordinates must actually load."""
    for label in ("ASW27", "F16"):
        w = fleet.AIRCRAFT[label].wing
        assert w.is_stand_in, f"{label}'s stand-in section is not flagged"
        sec = foil.load(w.section_file)
        assert len(sec.coordinates) > 20


def test_asw27_stand_in_thickness_is_close_to_the_real_section():
    """The substitution is only defensible if the thickness is comparable."""
    real = fleet.ASW27.wing.thickness  # 0.134, the DU 89-134/14
    stand_in = foil.load(fleet.ASW27.wing.section_file).thickness
    assert abs(stand_in - real) < 0.01


def test_dc3_simplified_planform_is_smaller_than_the_real_wing():
    """The 13% area gap HW 5 is built around."""
    a = fleet.DC3
    gap = 1.0 - a.wing.area / a.published["wing_area_actual"]
    assert 0.12 < gap < 0.14


def test_f16_reference_span_reconciles_the_published_aspect_ratio():
    """The data-sheet verification rung: 9.96 m is over the launcher rails."""
    a = fleet.F16
    assert a.wing.aspect_ratio == pytest.approx(3.2, rel=0.01)
    assert a.published["span_over_launchers"] > a.wing.span


def test_saturn_v_stage_masses_match_the_reference_targets():
    s_ic = fleet.SaturnV.stages[0]
    t = ref.SATURN_V_S_IC_TARGETS
    assert s_ic.propellant == t["propellant_mass"]
    assert s_ic.dry == t["dry_mass"]
    assert s_ic.height == t["height"]


def test_joby_has_no_invented_wing():
    """HW 7 needs mass, rotors and cruise L/D -- not a planform."""
    assert fleet.JobyS4.wing is None
    assert fleet.JobyS4.published["battery_energy_min"] > 0
    assert len(fleet.JobyS4.propulsion["mission"]) == 4


def test_estimated_fields_are_declared():
    """RC-1 is estimated throughout; the 787's core geometry is not."""
    assert fleet.RC1.estimated
    assert fleet.B787.is_estimated("wing.taper")
    assert not fleet.B787.is_estimated("wing.span")


# --- airfoils ---------------------------------------------------------------

#: Published thickness of each bundled section, from its own designation or
#: from the UIUC file header.
EXPECTED_THICKNESS = {
    "naca2412": 0.12,
    "naca64a210": 0.10,
    "naca633618": 0.18,
    "e212": 0.1055,
    "sd7037": 0.092,
    "fx62k131": 0.131,
    "clarky": 0.117,
    "s1223": 0.121,
}


@pytest.mark.parametrize("stem", sorted(EXPECTED_THICKNESS))
def test_bundled_airfoil_thickness_matches_its_designation(stem):
    """Catches a Selig/Lednicer format mix-up, which otherwise looks plausible."""
    sec = foil.load(stem)
    assert sec.thickness == pytest.approx(EXPECTED_THICKNESS[stem], abs=0.004)


@pytest.mark.parametrize("stem", sorted(EXPECTED_THICKNESS))
def test_bundled_airfoil_coordinates_are_sane(stem):
    sec = foil.load(stem)
    assert sec.x.min() == pytest.approx(0.0, abs=2e-3)
    assert sec.x.max() == pytest.approx(1.0, abs=2e-3)
    assert np.all(np.abs(sec.z) < 0.3)
    # ordered trailing edge -> leading edge -> trailing edge
    ile = int(np.argmin(sec.x))
    assert 0 < ile < len(sec.x) - 1


def test_every_bundled_airfoil_is_listed_and_loads():
    names = foil.available()
    assert set(EXPECTED_THICKNESS).issubset(set(names))
    for n in names:
        assert foil.load(n).coordinates.shape[1] == 2


@pytest.mark.parametrize("designation,t,camber", [
    ("naca0009", 0.09, 0.0), ("naca0012", 0.12, 0.0),
    ("naca2412", 0.12, 0.02), ("naca2215", 0.15, 0.02), ("naca2206", 0.06, 0.02),
])
def test_naca4_generation(designation, t, camber):
    sec = foil.naca4(designation)
    assert sec.thickness == pytest.approx(t, abs=2e-3)
    assert sec.camber == pytest.approx(camber, abs=2e-3)


def test_camber_function_is_usable_by_wing_to_grid():
    """The camber line is the only thing the vortex lattice sees of a section."""
    f = foil.load("naca2412").camber_function()
    x = np.linspace(0.0, 1.0, 11)
    z = f(x)
    assert z.shape == x.shape
    assert z[0] == pytest.approx(0.0, abs=2e-3)
    assert z[-1] == pytest.approx(0.0, abs=2e-3)
    assert z.max() == pytest.approx(0.02, abs=3e-3)  # 2% camber


def test_symmetric_section_has_zero_camber():
    assert foil.load("naca0012").camber < 1e-9


def test_load_rejects_an_unknown_section_clearly():
    with pytest.raises(FileNotFoundError, match="unknown section"):
        foil.load("not_an_airfoil")


# --- propellers -------------------------------------------------------------

CATALOG_PROPS = ["apce_8x6", "apce_9x6", "apce_10x5", "apce_10x7", "apce_11x7"]


def test_all_catalog_propellers_are_bundled():
    assert set(CATALOG_PROPS) == set(props.available())


@pytest.mark.parametrize("name", CATALOG_PROPS)
def test_propeller_dimensions_decode_from_the_name(name):
    p = props.load(name)
    dia_in, pitch_in = name.split("_")[1].split("x")
    assert p.diameter_in == pytest.approx(float(dia_in))
    assert p.pitch_in == pytest.approx(float(pitch_in))
    assert p.diameter == pytest.approx(float(dia_in) * 0.0254)


@pytest.mark.parametrize("name", CATALOG_PROPS)
def test_propeller_has_static_and_advance_ratio_data(name):
    p = props.load(name)
    assert p.static is not None and len(p.static) > 5
    assert len(p.runs) >= 5
    for r in p.runs:
        assert len(r) > 5
        assert np.all(np.diff(r.J) > 0)  # sorted
        assert np.all(r.CT > -0.05)
        assert np.all(r.CP > 0)


@pytest.mark.parametrize("name", CATALOG_PROPS)
def test_efficiency_identity_holds_in_the_data_files(name):
    """HW 8's 'identity, two ways': eta must equal J*CT/CP."""
    p = props.load(name)
    for r in p.runs:
        assert np.nanmax(np.abs(r.eta - r.eta_check)) < 0.01


@pytest.mark.parametrize("name", CATALOG_PROPS)
def test_measured_data_does_not_reach_zero_advance_ratio(name):
    """Static thrust lives at J=0 and the sweeps do not get there.

    HW 8 asks students to report what fraction of the flight envelope is
    extrapolation; this is why that question is not rhetorical.
    """
    lo, hi = props.load(name).J_range
    assert lo > 0.05
    assert hi < 1.1


def test_rpm_to_rad_per_second_conversion():
    r = props.load("apce_10x7").run(5000)
    assert r.omega == pytest.approx(r.rpm * 2 * np.pi / 60)
    assert r.n == pytest.approx(r.rpm / 60)
    assert r.omega / r.n == pytest.approx(2 * np.pi)


def test_static_thrust_of_the_rc1_propeller_is_a_few_hundred_grams():
    """HW 8's dimensional bound, applied to the data before any model."""
    p = props.load("apce_10x7")
    CT = np.interp(6000.0, p.static.rpm, p.static.CT)
    T = CT * 1.225 * (6000.0 / 60.0) ** 2 * p.diameter**4
    grams = T / 9.80665 * 1000.0
    assert 300 < grams < 900


def test_peak_measured_efficiency_is_plausible():
    for name in CATALOG_PROPS:
        _, J, _, _, eta = props.load(name).all_points()
        assert 0.55 < eta.max() < 0.90


def test_blade_geometry_is_present_where_uiuc_provides_it():
    p = props.load("apce_10x7")
    g = p.geometry
    assert g is not None
    assert np.all(np.diff(g.r_R) > 0)
    assert np.all(g.c_R > 0)
    assert g.beta_deg[0] > g.beta_deg[-1]  # twisted, root to tip
    # apce_8x6 has no geometry file in the UIUC database
    assert props.load("apce_8x6").geometry is None


# --- catalog ----------------------------------------------------------------


def test_catalog_is_the_stated_size():
    assert len(catalog.MOTORS) == 4
    assert len(catalog.PROPELLERS) == 5
    assert len(catalog.BATTERIES) == 3
    assert len(list(catalog.combinations())) == 60


def test_kv_unit_conversion():
    m = catalog.MOTORS["M1000"]
    assert m.Kv == pytest.approx(1000.0 * 2 * np.pi / 60)
    assert m.Kt == pytest.approx(1.0 / m.Kv)
    # the factor everyone gets wrong
    assert m.Kv_rpm / m.Kv == pytest.approx(60 / (2 * np.pi), rel=1e-12)


def test_peak_efficiency_does_not_rank_with_kv():
    """HW 7's central finding, guaranteed to be visible in this catalog."""
    V = 11.1
    by_kv = sorted(catalog.MOTORS.values(), key=lambda m: m.Kv_rpm)
    by_eta = sorted(catalog.MOTORS.values(), key=lambda m: m.peak_efficiency(V))
    assert [m.key for m in by_kv] != [m.key for m in by_eta]
    # and it does rank with I0*R
    by_i0r = sorted(
        catalog.MOTORS.values(),
        key=lambda m: -m.current_no_load * m.resistance,
    )
    assert [m.key for m in by_i0r] == [m.key for m in by_eta]


def test_motor_peak_efficiency_matches_the_reference_closed_form():
    for m in catalog.MOTORS.values():
        assert m.peak_efficiency(11.1) == pytest.approx(
            ref.motor_peak_efficiency(m.current_no_load, m.resistance, 11.1)
        )


def test_battery_energy_and_voltage_bookkeeping():
    b = catalog.BATTERIES["B3S1300"]
    assert b.voltage_nominal == pytest.approx(11.1)
    assert b.capacity == pytest.approx(1.3 * 3600)
    assert b.energy_nominal / 3600 == pytest.approx(11.1 * 1.3, rel=1e-9)
    assert b.resistance == pytest.approx(0.012 * 3)
    assert 100 < b.specific_energy / 3600 < 180  # Wh/kg, plausible for LiPo


def test_rc1_baseline_matches_the_fleet_page():
    m, p, b = catalog.RC1_BASELINE
    assert catalog.MOTORS[m].Kv_rpm == 1000.0
    assert catalog.PROPELLERS[p].data == "apce_10x7"
    assert catalog.BATTERIES[b].cells_series == 3
    assert catalog.BATTERIES[b].capacity_ah == pytest.approx(1.300)
    # the pack mass must match the battery row of RC-1's component table
    row = next(c for c in fleet.RC1.components if c.name == "Battery")
    assert catalog.BATTERIES[b].mass == pytest.approx(row.mass)


def test_provisional_entries_are_declared():
    """The motor and battery electricals are placeholders and must say so."""
    flags = catalog.check_provisional()
    assert set(flags["motors"]) == set(catalog.MOTORS)
    assert set(flags["batteries"]) == set(catalog.BATTERIES)
    assert flags["propellers"] == []  # the propeller data is real and measured


def test_catalog_propellers_all_resolve_to_bundled_data():
    for entry in catalog.PROPELLERS.values():
        p = entry.load()
        assert p.diameter > 0
        assert len(p.runs) >= 5


# --- reference values -------------------------------------------------------


def test_laminar_flat_plate_is_a_lower_bound_and_falls_with_re():
    Re = np.array([1e4, 1e5, 1e6, 1e7])
    cf = ref.laminar_flat_plate_cf(Re)
    assert np.all(np.diff(cf) < 0)
    assert cf[1] == pytest.approx(1.328 / np.sqrt(1e5))


def test_elliptical_root_bending_moment_closed_form():
    L, b = 1000.0, 15.0
    assert ref.elliptical_root_bending_moment(L, b) == pytest.approx(
        L * b / (3 * np.pi)
    )


def test_ideal_propulsive_efficiency_is_bounded():
    eta = ref.ideal_propulsive_efficiency(T=5.0, rho=1.225, V=10.0, A=0.05)
    assert 0.0 < eta < 1.0
    # more disk area for the same thrust is more efficient
    assert ref.ideal_propulsive_efficiency(5.0, 1.225, 10.0, 0.5) > eta


def test_phugoid_period_scales_linearly_with_speed():
    """The dimensional check that catches a rate-derivative normalization error."""
    assert ref.phugoid_period_approx(20.0) == pytest.approx(
        2 * ref.phugoid_period_approx(10.0)
    )


@pytest.mark.parametrize("condition", ["cruise", "cruise_1p25"])
def test_supplied_rc1_longitudinal_eigenvalues(condition):
    model = ref.RC1_LONGITUDINAL_MATRICES[condition]
    got = np.sort_complex(np.linalg.eigvals(model["A"]))
    expected = np.sort_complex(np.asarray(model["reference_eigenvalues"]))
    assert got == pytest.approx(expected, rel=2e-7, abs=2e-7)


def test_rc1_supplied_phugoid_speed_scaling():
    models = ref.RC1_LONGITUDINAL_MATRICES
    p0 = min((x for x in models["cruise"]["reference_eigenvalues"] if x.imag > 0),
             key=lambda x: x.imag)
    p1 = min((x for x in models["cruise_1p25"]["reference_eigenvalues"] if x.imag > 0),
             key=lambda x: x.imag)
    period_ratio = (2 * np.pi / p1.imag) / (2 * np.pi / p0.imag)
    assert period_ratio == pytest.approx(1.25, rel=2e-3)


def test_b787_course_engine_model_and_validity_box():
    model = ref.B787_ENGINE_MODEL
    thrust = ref.b787_thrust_available(0.31, 0.85, altitude=11_900.0)
    assert 100_000.0 < thrust < 250_000.0
    assert model["weight_tsfc_per_second"] == pytest.approx(
        model["weight_tsfc_per_hour"] / 3600
    )
    assert model["mass_tsfc_kg_per_N_hour"] == pytest.approx(
        model["weight_tsfc_per_hour"] / ref.ATMOS_CONSTANTS["g0"]
    )
    with pytest.raises(ValueError):
        ref.b787_thrust_available(1.225, 0.10, altitude=0.0)


def test_optimal_battery_fraction_is_two_thirds():
    assert ref.OPTIMAL_BATTERY_MASS_FRACTION == pytest.approx(2 / 3)
    # and it really is the maximum of m_b/(m_f+m_b)^1.5
    mf = 1.0
    mb = np.linspace(0.05, 10.0, 20001)
    endurance = mb / (mf + mb) ** 1.5
    assert mb[np.argmax(endurance)] == pytest.approx(2.0 * mf, rel=2e-3)


def test_battery_soc_course_reference_is_well_formed():
    curve = ref.BATTERY_SOC_REFERENCE
    soc = curve["soc"]
    voltage = curve["open_circuit_voltage_per_cell"]
    resistance = curve["resistance_multiplier"]
    assert np.all(np.diff(soc) > 0)
    assert np.all(np.diff(voltage) > 0)
    assert np.all(resistance > 0)
    assert soc[0] == 0.0 and soc[-1] == 1.0


def test_atmosphere_sea_level_constants_are_self_consistent():
    """p = rho R T at sea level, to the precision of the defined constants."""
    a = ref.ATMOS_SL
    R = ref.ATMOS_CONSTANTS["R"]
    assert a["pressure"] == pytest.approx(a["density"] * R * a["temperature"], rel=2e-4)
    gamma = ref.ATMOS_CONSTANTS["gamma"]
    assert a["speed_of_sound"] == pytest.approx(
        np.sqrt(gamma * R * a["temperature"]), rel=1e-4
    )


def test_published_ranges_are_ordered_pairs():
    for name in ("CDP_WIDEBODY_CRUISE", "SWET_SREF_WIDEBODY",
                 "MOTOR_PEAK_EFFICIENCY_BAND", "CHAIN_EFFICIENCY_BAND"):
        lo, hi = getattr(ref, name)
        assert lo < hi
    for key, val in ref.PUBLISHED_NACA2412.items():
        if isinstance(val, tuple):
            lo, hi = val
            assert lo < hi, f"{key} range is not ordered"
