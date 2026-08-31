"""End-to-end checks that the provided pieces actually compose.

The important one is :func:`test_dc3_strip_integration_matches_course_solution`.
The draft flagged an open question about whether ``lifting_line_coefficients``
exposes per-span-station coefficients cleanly enough for HW 5's strip
integration, and whether local chord and local Reynolds number are available at
the *same* stations.  This file answers it: they are, and the resulting drag
reproduces the existing course solution.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from flightlab import catalog, foil, props, ref
from flightlab.fleet import DC3, RC1
from flightlab.vlm import (
    Cosine,
    Freestream,
    Reference,
    Stability,
    Uniform,
    body_forces,
    far_field_drag,
    lifting_line_coefficients,
    lifting_line_geometry,
    steady_analysis,
    wing_to_grid,
)


def atmosphere(h):
    """1976 standard atmosphere, troposphere only.

    Inlined here rather than provided: writing this is HW 1.  The tests need
    *an* atmosphere, and this is the shortest correct one.
    """
    T0, p0, L = 288.15, 101325.0, -0.0065
    R, g = ref.ATMOS_CONSTANTS["R"], ref.ATMOS_CONSTANTS["g0"]
    T = T0 + L * h
    p = p0 * (T / T0) ** (-g / (R * L))
    rho = p / (R * T)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    return T, p, rho, mu


def test_atmosphere_helper_reproduces_sea_level():
    T, p, rho, mu = atmosphere(0.0)
    assert T == pytest.approx(ref.ATMOS_SL["temperature"])
    assert p == pytest.approx(ref.ATMOS_SL["pressure"])
    assert rho == pytest.approx(ref.ATMOS_SL["density"], rel=2e-4)
    assert mu == pytest.approx(ref.ATMOS_SL["viscosity"], rel=1e-2)


def _solve_trapezoid(b, S, cr, ct, alpha, V, ns=60, nc=4, x_ref=None):
    grid, ratios = wing_to_grid(
        [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [cr, ct],
        [0.0, 0.0], [0.0, 0.0], ns, nc,
        spacing_s=Cosine(), spacing_c=Uniform(),
    )
    x_ref = 0.25 * cr if x_ref is None else x_ref
    r = Reference(S, S / b, b, [x_ref, 0.0, 0.0], V)
    fs = Freestream.from_degrees(V, alpha=alpha)
    system = steady_analysis([grid], r, fs, symmetric=True, ratios=[ratios])
    CF, _ = body_forces(system, frame=Stability())
    return system, CF[2], far_field_drag(system)


def test_dc3_strip_integration_matches_course_solution():
    """HW 5, both halves, on the DC-3's simplified trapezoidal wing.

    Reproduces the existing course solution's inviscid span efficiency, its
    induced drag, and its strip-integrated viscous drag -- the last from a
    different section solver than the one that produced the reference value, so
    agreement within a few percent is the expected outcome, not an exact match.
    """
    w = DC3.wing
    b, S, cr, ct = w.span, w.area, w.root_chord, w.tip_chord
    V = DC3.operating["cruise_speed"]
    _, _, rho, mu = atmosphere(DC3.operating["cruise_altitude"])
    q = 0.5 * rho * V**2
    W = DC3.mass["gross"] * ref.ATMOS_CONSTANTS["g0"]
    CL_req = W / (q * S)
    AR = w.aspect_ratio
    sol = ref.DC3_COURSE_SOLUTION

    # trim on alpha to carry the weight
    alpha = brentq(
        lambda a: _solve_trapezoid(b, S, cr, ct, a, V)[1] - CL_req,
        -2.0, 8.0, xtol=1e-9,
    )
    system, CL, CDi = _solve_trapezoid(b, S, cr, ct, alpha, V)
    assert CL == pytest.approx(CL_req, rel=1e-8)

    # --- inviscid: span efficiency and induced drag --------------------
    e_inv = CL**2 / (np.pi * AR * CDi)
    assert e_inv == pytest.approx(sol["e_inv"], abs=0.005)
    assert CDi * q * S == pytest.approx(sol["induced_drag"], rel=0.02)

    # --- per-station output HW 5 needs ---------------------------------
    r_ll, c_ll = lifting_line_geometry(system.grids)
    cf, _ = lifting_line_coefficients(system, r_ll, c_ll, frame=Stability())
    y = r_ll[0][1, :]
    ds = np.linalg.norm(np.diff(r_ll[0], axis=1), axis=0)
    chord = 0.5 * (c_ll[0][:-1] + c_ll[0][1:])
    y_mid = 0.5 * (y[:-1] + y[1:])
    cl_local = cf[0][2, :]
    Re_local = rho * V * chord / mu

    # everything the strip integral needs, at the same stations
    assert cl_local.shape == chord.shape == Re_local.shape == y_mid.shape
    assert np.all(chord > 0) and np.all(Re_local > 0)
    assert chord.max() == pytest.approx(cr, rel=0.01)
    assert chord.min() == pytest.approx(ct, rel=0.01)

    # the identity: strip lift equals total lift equals weight
    L_strip = 2.0 * np.sum(cl_local * chord * ds) * q
    assert L_strip == pytest.approx(W, rel=1e-8)

    # --- Method B: integrate section drag along the span ---------------
    # each station gets its own cl at its own local Reynolds number
    alphas = np.linspace(-6.0, 12.0, 181)
    cd = np.empty_like(chord)
    for i in range(len(chord)):
        section = "naca2215" if y_mid[i] < b / 4 else "naca2206"
        pol = foil.aero(section, alphas, Re_local[i])
        a_i = np.interp(cl_local[i], pol["cl"], alphas)
        cd[i] = np.interp(a_i, alphas, pol["cd"])
        # HW 5's bounds rung, asserted inside the loop as the assignment asks
        assert cd[i] > ref.laminar_flat_plate_cf(Re_local[i])

    Dv = 2.0 * np.sum(cd * q * chord * ds)

    # within a few percent of the course value and of XFLR5's independent one,
    # both of which came from XFOIL rather than from NeuralFoil
    assert Dv == pytest.approx(sol["viscous_drag_strip"], rel=0.08)
    assert Dv == pytest.approx(sol["viscous_drag_xflr5"], rel=0.08)
    # and comfortably below the handbook method, which is the point of HW 5
    assert Dv < 0.85 * sol["viscous_drag_handbook"]


def test_dc3_reference_area_bookkeeping_changes_the_coefficient():
    """HW 5's cheapest demonstration: a CD is meaningless without its area."""
    w = DC3.wing
    V = DC3.operating["cruise_speed"]
    _, _, rho, _ = atmosphere(DC3.operating["cruise_altitude"])
    q = 0.5 * rho * V**2
    W = DC3.mass["gross"] * ref.ATMOS_CONSTANTS["g0"]

    S_trap = w.area
    S_real = DC3.published["wing_area_actual"]
    system, CL, CDi = _solve_trapezoid(
        w.span, S_trap, w.root_chord, w.tip_chord, 4.0, V
    )
    D = CDi * q * S_trap  # a force, independent of the reference area
    CD_trap = D / (q * S_trap)
    CD_real = D / (q * S_real)
    assert CD_trap > CD_real
    assert CD_trap / CD_real == pytest.approx(S_real / S_trap, rel=1e-9)
    # the smaller wing flies at the higher CL for the same weight
    assert W / (q * S_trap) > W / (q * S_real)


def test_rc1_wing_and_tail_trim_and_neutral_point():
    """HW 6's shape, on the aircraft the students are building.

    Checks that a two-surface RC-1 model trims, that the neutral point lands
    aft of the published CG, and that the resulting static margin is in the
    range a hand-launched model needs.
    """
    from flightlab.vlm import stability_derivatives

    wing, tail = RC1.wing, RC1.htail
    V = RC1.operating["cruise_speed"]
    _, _, rho, _ = atmosphere(RC1.operating["field_altitude"])
    q = 0.5 * rho * V**2
    W = RC1.mass["gross"] * ref.ATMOS_CONSTANTS["g0"]
    x_cg = RC1.propulsion["x_cg_published"]

    def build(incidence_deg, alpha_deg):
        wg, wr = wing_to_grid(
            [wing.x_le, wing.x_le], [0.0, wing.span / 2], [0.0, 0.0],
            [wing.root_chord, wing.tip_chord], [0.0, 0.0], [0.0, 0.0],
            20, 3, spacing_s=Cosine(),
            fc=[foil.load("naca2412").camber_function()] * 2,
        )
        it = np.radians(incidence_deg)
        tg, tr = wing_to_grid(
            [tail.x_le, tail.x_le], [0.0, tail.span / 2], [tail.z, tail.z],
            [tail.root_chord, tail.tip_chord], [it, it], [0.0, 0.0],
            10, 2, spacing_s=Cosine(),
        )
        r = Reference(wing.area, wing.mean_chord, wing.span, [x_cg, 0.0, 0.0], V)
        fs = Freestream.from_degrees(V, alpha=alpha_deg)
        return steady_analysis(
            [wg, tg], r, fs, symmetric=[True, True], surface_id=[1, 1],
            ratios=[wr, tr],
        )

    CL_req = W / (q * wing.area)

    # trim: find the tail incidence and alpha that give L = W and Cm = 0
    def residuals(x):
        system = build(x[0], x[1])
        CF, CM = body_forces(system, frame=Stability())
        return CF[2] - CL_req, CM[1]

    from scipy.optimize import fsolve

    x = fsolve(residuals, [-2.0, 4.0], full_output=False, xtol=1e-10)
    rL, rM = residuals(x)
    assert abs(rL) < 1e-8, "lift did not converge to the weight"
    assert abs(rM) < 1e-8, "pitching moment did not converge to zero"

    system = build(x[0], x[1])
    dCF, dCM = stability_derivatives(system)
    CLa, Cma = dCF["alpha"][2], dCM["alpha"][1]
    assert CLa > 0
    assert Cma < 0, "adding a tail behind the wing must make this stable"

    c = wing.mean_chord
    x_np = x_cg - Cma / CLa * c
    static_margin = (x_np - x_cg) / c
    assert x_np > x_cg
    assert 0.02 < static_margin < 0.60

    # the vortex lattice sees lifting surfaces only, so this excludes the pod's
    # destabilizing contribution -- which is HW 6's "missing fuselage" item
    assert system.nsurf == 2


def test_rc1_propulsion_chain_pieces_are_consistent():
    """HW 8's inputs: the catalog, the measured data, and the momentum bound."""
    m = catalog.MOTORS["M1000"]
    b = catalog.BATTERIES["B3S1300"]
    p = catalog.PROPELLERS["P10x7"].load()

    V = RC1.operating["cruise_speed"]
    _, _, rho, _ = atmosphere(RC1.operating["field_altitude"])

    # static thrust from the measured static sweep, at a plausible RPM
    rpm = 7000.0
    CT = np.interp(rpm, p.static.rpm, p.static.CT)
    T_static = CT * rho * (rpm / 60.0) ** 2 * p.diameter**4
    assert 2.0 < T_static < 12.0  # newtons: a few hundred grams to a kilo

    # at cruise the operating advance ratio must be inside the measured range
    J = V / ((rpm / 60.0) * p.diameter)
    lo, hi = p.J_range
    assert lo < J < hi, f"J = {J:.3f} is outside the measured {lo:.3f}..{hi:.3f}"

    # measured efficiency must sit below the actuator-disk bound
    run = p.run(rpm)
    T = run.CT * rho * run.n**2 * p.diameter**4
    speeds = run.J * run.n * p.diameter
    ideal = ref.ideal_propulsive_efficiency(
        np.maximum(T, 1e-9), rho, np.maximum(speeds, 1e-3), p.disk_area
    )
    good = speeds > 2.0  # the bound is meaningless as V -> 0
    assert np.all(run.eta[good] < ideal[good])

    # the motor can supply the current the pack allows
    assert m.current_max <= b.current_max
    assert ref.MOTOR_PEAK_EFFICIENCY_BAND[0] < m.peak_efficiency(
        b.voltage_nominal
    ) < 0.90


def test_wing_clmax_from_the_critical_station():
    """HW 4's wing CL_max: raise alpha until a station reaches its own cl_max.

    Each station is checked at *its own* local Reynolds number, which is the
    part that makes this more than a scalar comparison.
    """
    w = RC1.wing
    V = RC1.operating["cruise_speed"]
    _, _, rho, mu = atmosphere(RC1.operating["field_altitude"])

    section = "naca2412"
    alphas = np.linspace(-4.0, 20.0, 121)

    def critical_margin(alpha_deg):
        system, CL, _ = _solve_trapezoid(
            w.span, w.area, w.root_chord, w.tip_chord, alpha_deg, V,
            ns=20, nc=2,
        )
        r_ll, c_ll = lifting_line_geometry(system.grids)
        cf, _ = lifting_line_coefficients(system, r_ll, c_ll, frame=Stability())
        chord = 0.5 * (c_ll[0][:-1] + c_ll[0][1:])
        cl = cf[0][2, :]
        Re = rho * V * chord / mu
        clmax = np.array(
            [foil.aero(section, alphas, Re_i)["cl"].max() for Re_i in Re]
        )
        return np.max(cl - clmax), CL

    lo, _ = critical_margin(2.0)
    hi, _ = critical_margin(16.0)
    assert lo < 0 < hi, "the critical station should stall between 2 and 16 deg"

    a_stall = brentq(lambda a: critical_margin(a)[0], 2.0, 16.0, xtol=1e-3)
    _, CL_max = critical_margin(a_stall)

    # a cambered section on a rectangular AR 7.5 wing at Re ~ 1e5
    assert 0.8 < CL_max < 1.6
    # and the placeholder RC-1 has been carrying is in the same neighbourhood
    assert abs(CL_max - RC1.placeholders["CLmax"]) < 0.5
