"""End-to-end checks that the provided pieces actually compose."""

import numpy as np
import pytest
from scipy.optimize import brentq

from flightlab import catalog, foil, props, ref
from flightlab.fleet import RC1
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
