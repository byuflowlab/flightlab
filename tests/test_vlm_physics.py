"""Physics verification for ``flightlab.vlm``.

These are the rungs the course itself stands on -- if any of these break, HW 4
and HW 5 stop working, and they break in ways that look like student error.
"""

import numpy as np
import pytest

from flightlab import ref
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


def _solve(xle, yle, zle, chord, ns, nc, S, b, alpha=3.0, spacing=None,
           twist=None):
    n = len(chord)
    twist = np.zeros(n) if twist is None else np.asarray(twist, dtype=float)
    grid, ratios = wing_to_grid(
        xle, yle, zle, chord, twist, np.zeros(n), ns, nc,
        spacing_s=Cosine() if spacing is None else spacing,
        spacing_c=Uniform(),
    )
    r = Reference(S, S / b, b, [0.0, 0.0, 0.0], 1.0)
    fs = Freestream.from_degrees(1.0, alpha=alpha)
    system = steady_analysis([grid], r, fs, symmetric=True, ratios=[ratios])
    return system, grid


def _elliptic(ns, nc=1, spacing=None, AR=8.0, S=1.0, alpha=3.0, nsec=60):
    """An elliptical planform, defined by many spanwise sections."""
    b = np.sqrt(AR * S)
    c0 = 4.0 * S / (np.pi * b)
    y = np.linspace(0.0, b / 2, nsec)
    ch = c0 * np.sqrt(np.maximum(0.0, 1.0 - (2.0 * y / b) ** 2))
    ch[-1] = max(ch[-1], 1e-8)  # a zero chord makes a degenerate panel
    xle = 0.25 * (c0 - ch)  # straight quarter-chord line
    return _solve(xle, y, np.zeros_like(y), ch, ns, nc, S, b, alpha, spacing)


def _rect(ns, nc=1, AR=25.0, S=9.0, alpha=3.0, spacing=None):
    b = np.sqrt(AR * S)
    c = S / b
    return _solve([0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [c, c], ns, nc, S, b,
                  alpha, spacing)


def _CL_CDi_e(system, AR):
    CF, _ = body_forces(system, frame=Stability())
    CL = CF[2]
    CDi = far_field_drag(system)
    return CL, CDi, CL**2 / (np.pi * AR * CDi)


# --- the analytic limit rung -----------------------------------------------


def test_elliptical_planform_returns_unit_span_efficiency():
    """HW 4's calibration case: an elliptical planform must give e_inv = 1.000."""
    system, _ = _elliptic(ns=120, spacing=Cosine())
    _, _, e = _CL_CDi_e(system, AR=8.0)
    assert e == pytest.approx(ref.ELLIPTICAL_E_INV, abs=5e-3)


def test_span_efficiency_converges_from_above():
    """Refinement drives e_inv monotonically toward 1 from above."""
    e = []
    for ns in (10, 20, 40, 80, 160):
        system, _ = _elliptic(ns=ns, spacing=Cosine())
        e.append(_CL_CDi_e(system, AR=8.0)[2])
    e = np.array(e)
    assert np.all(np.diff(e) < 0)  # monotone
    assert abs(e[-1] - 1.0) < abs(e[0] - 1.0) / 20  # and converging
    assert abs(e[-1] - 1.0) < 5e-3


def test_cosine_spacing_converges_faster_than_uniform():
    """The spacing comparison HW 4 asks students to plot and explain.

    Induced drag is set by the loading near the tip, so how panels are
    distributed there matters more than how many there are.
    """
    ns = 20
    e_cos = _CL_CDi_e(_elliptic(ns=ns, spacing=Cosine())[0], 8.0)[2]
    e_uni = _CL_CDi_e(_elliptic(ns=ns, spacing=Uniform())[0], 8.0)[2]
    assert abs(e_cos - 1.0) < abs(e_uni - 1.0)


def test_induced_drag_matches_closed_form():
    """D_i = L^2/(q pi b^2) for elliptical loading."""
    S, AR = 1.0, 8.0
    b = np.sqrt(AR * S)
    system, _ = _elliptic(ns=120, AR=AR, S=S, spacing=Cosine())
    CL, CDi, _ = _CL_CDi_e(system, AR)
    q = 0.5 * 1.0 * 1.0**2  # rho = 1, V = 1 inside the solver
    L = CL * q * S
    Di_closed = ref.elliptical_induced_drag(L, q, b)
    assert CDi * q * S == pytest.approx(Di_closed, rel=5e-3)


# --- the identity rung -----------------------------------------------------


def test_strip_lift_integrates_to_total_lift():
    """HW 4's and HW 5's shared identity, and it should hold to round-off."""
    S, AR = 9.0, 25.0
    system, grid = _rect(ns=40, AR=AR, S=S)
    CL, _, _ = _CL_CDi_e(system, AR)

    r, c = lifting_line_geometry(system.grids)
    cf, _ = lifting_line_coefficients(system, r, c, frame=Stability())
    ds = np.linalg.norm(np.diff(r[0], axis=1), axis=0)
    cs = 0.5 * (c[0][:-1] + c[0][1:])
    CL_strip = 2.0 * np.sum(cf[0][2, :] * cs * ds) / S  # x2 for the mirror half

    assert CL_strip == pytest.approx(CL, rel=1e-10)


def test_lifting_line_coefficients_expose_what_hw5_needs():
    """HW 5 needs local chord and station geometry at the same stations as c_l."""
    system, grid = _rect(ns=24, AR=10.55, S=79.75)
    r, c = lifting_line_geometry(system.grids)
    cf, cm = lifting_line_coefficients(system, r, c, frame=Stability())

    ns = system.surfaces[0].ns
    assert cf[0].shape == (3, ns)
    assert cm[0].shape == (3, ns)
    assert r[0].shape == (3, ns + 1)
    assert c[0].shape == (ns + 1,)
    # local chord at the station midpoints, which is what the strip integral uses
    cs = 0.5 * (c[0][:-1] + c[0][1:])
    assert np.all(cs > 0)
    assert np.all(np.isfinite(cf[0]))


def test_tapered_wing_local_chord_varies_as_specified():
    """A straight taper must return the chord distribution it was given."""
    b, cr, ct = 29.0, 4.4, 1.1
    S = 0.5 * (cr + ct) * b
    grid, ratios = wing_to_grid(
        [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [cr, ct],
        [0.0, 0.0], [0.0, 0.0], 30, 1, spacing_s=Cosine(),
    )
    r, c = lifting_line_geometry([grid])
    y = r[0][1, :]  # the stations' own y locations, cosine spaced
    expected = cr + (ct - cr) * (2.0 * y / b)
    assert np.max(np.abs(c[0] - expected)) < 1e-9 * cr
    assert c[0][0] == pytest.approx(cr, rel=1e-9)
    assert c[0][-1] == pytest.approx(ct, rel=1e-9)


# --- the second-method rung ------------------------------------------------


@pytest.mark.parametrize("AR,max_ratio", [(25.0, 0.06), (7.5, 0.12), (3.2, 0.20)])
def test_lifting_line_degrades_as_aspect_ratio_falls(AR, max_ratio):
    """Lifting line agrees with the VLM at high AR and fails at low AR."""
    alpha = 3.0
    system, _ = _rect(ns=60, AR=AR, S=9.0, alpha=alpha)
    CL, _, _ = _CL_CDi_e(system, AR)
    CLa_vlm = CL / np.radians(alpha)
    CLa_ll = ref.lifting_line_CL_alpha(2.0 * np.pi, AR, e=1.0)

    err = abs(CLa_vlm / CLa_ll - 1.0)
    assert err < max_ratio
    # lifting line always over-predicts here
    assert CLa_vlm < CLa_ll


def test_flat_plate_slope_approaches_two_pi_at_high_aspect_ratio():
    """A very high AR flat plate should approach the 2-D result."""
    alpha = 2.0
    system, _ = _rect(ns=80, AR=200.0, S=9.0, alpha=alpha)
    CL, _, _ = _CL_CDi_e(system, 200.0)
    assert CL / np.radians(alpha) == pytest.approx(2.0 * np.pi, rel=0.04)


# --- symmetry and consistency ----------------------------------------------


def test_symmetric_flag_matches_mirrored_geometry():
    """Two ways of modelling the same wing must agree."""
    xle, yle = [0.0, 0.4], [0.0, 7.5]
    chord, theta = [2.2, 1.8], [np.radians(2.0)] * 2
    ref_q = Reference(30.0, 2.0, 15.0, [0.5, 0.0, 0.0], 1.0)
    fs = Freestream.from_degrees(1.0, alpha=4.0)

    g1, r1 = wing_to_grid(xle, yle, [0.0, 0.0], chord, theta, [0.0, 0.0], 20, 2,
                          mirror=False, spacing_s=Cosine())
    s1 = steady_analysis([g1], ref_q, fs, symmetric=True, ratios=[r1])

    g2, r2 = wing_to_grid(xle, yle, [0.0, 0.0], chord, theta, [0.0, 0.0], 20, 2,
                          mirror=True, spacing_s=Cosine())
    s2 = steady_analysis([g2], ref_q, fs, symmetric=False, ratios=[r2])

    CF1, CM1 = body_forces(s1, frame=Stability())
    CF2, CM2 = body_forces(s2, frame=Stability())
    assert CF1 == pytest.approx(CF2, abs=2e-4)
    assert CM1[1] == pytest.approx(CM2[1], abs=2e-4)
    assert far_field_drag(s1) == pytest.approx(far_field_drag(s2), abs=2e-5)


def test_zero_alpha_flat_plate_gives_zero_lift():
    """A symmetric, untwisted flat wing at zero incidence lifts nothing."""
    system, _ = _rect(ns=20, alpha=0.0)
    CF, CM = body_forces(system, frame=Stability())
    assert CF[2] == pytest.approx(0.0, abs=1e-12)
    assert far_field_drag(system) == pytest.approx(0.0, abs=1e-12)


def test_lift_is_very_nearly_linear_in_alpha():
    """CL scales with sin(alpha), so it is linear only to small-angle order.

    Worth knowing before you fit a lift curve slope to VLM output: over 1 to 8
    degrees ``CL/sin(alpha)`` drifts by about 0.15%, because the force is a
    cross product with a freestream that is itself rotating.
    """
    a = np.array([1.0, 2.0, 4.0, 8.0])
    CL = np.array([_CL_CDi_e(_rect(ns=30, alpha=ai)[0], 25.0)[0] for ai in a])
    slope_sin = CL / np.sin(np.radians(a))
    assert np.ptp(slope_sin) / slope_sin.mean() < 3e-3
    # and monotone increasing in alpha
    assert np.all(np.diff(CL) > 0)


def test_induced_drag_is_very_nearly_quadratic_in_lift():
    """e_inv is almost independent of alpha.

    Not exactly: the Trefftz plane is normal to the freestream, so rotating
    alpha changes the projected trailing-edge geometry slightly.  Expect a
    few tenths of a percent over a normal alpha range, not zero.
    """
    e = np.array([_CL_CDi_e(_rect(ns=30, alpha=ai)[0], 25.0)[2]
                  for ai in (1.0, 2.0, 4.0, 6.0, 8.0)])
    assert np.ptp(e) / e.mean() < 5e-3


def test_washout_reduces_tip_loading():
    """Negative twist unloads the tip, which is why it buys stall margin."""
    S, AR = 9.0, 25.0
    b = np.sqrt(AR * S)
    out = {}
    for tw in (0.0, -6.0):
        system, _ = _solve([0.0, 0.0], [0.0, b / 2], [0.0, 0.0],
                           [S / b, S / b], 40, 1, S, b, alpha=4.0,
                           twist=[0.0, np.radians(tw)])
        r, c = lifting_line_geometry(system.grids)
        cf, _ = lifting_line_coefficients(system, r, c, frame=Stability())
        cl = cf[0][2, :]
        out[tw] = cl / cl.mean()  # normalized, so total lift drops out
    # the outboard third carries relatively less load with washout
    n = len(out[0.0])
    assert out[-6.0][-n // 3:].mean() < out[0.0][-n // 3:].mean()


# --- stability derivatives -------------------------------------------------


def test_neutral_point_is_ahead_of_a_wing_alone_quarter_chord_reference():
    """Cm_alpha must be negative when the reference point is aft of the NP."""
    from flightlab.vlm import stability_derivatives

    S, AR = 9.0, 7.5
    b = np.sqrt(AR * S)
    c = S / b
    grid, ratios = wing_to_grid(
        [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [c, c],
        [0.0, 0.0], [0.0, 0.0], 24, 4, spacing_s=Cosine(),
    )
    # the reference point must be AHEAD of the neutral point for Cm_alpha < 0
    r = Reference(S, c, b, [0.10 * c, 0.0, 0.0], 1.0)
    fs = Freestream.from_degrees(1.0, alpha=2.0)
    system = steady_analysis([grid], r, fs, symmetric=True, ratios=[ratios])
    dCF, dCM = stability_derivatives(system)

    CLa, Cma = dCF["alpha"][2], dCM["alpha"][1]
    assert CLa > 0
    assert Cma < 0
    # neutral point, aft of the reference, in chords
    x_np = r.r[0] - Cma / CLa * c
    assert 0.20 * c < x_np < 0.30 * c  # a thin wing's NP sits near c/4


def test_neutral_point_is_independent_of_the_moment_reference():
    """The NP is a property of the wing, not of where you took moments.

    This is the cheapest check that your neutral-point bookkeeping is right,
    and it catches a reference-point sign error immediately.
    """
    from flightlab.vlm import stability_derivatives

    S, AR = 9.0, 7.5
    b = np.sqrt(AR * S)
    c = S / b
    grid, ratios = wing_to_grid(
        [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [c, c],
        [0.0, 0.0], [0.0, 0.0], 24, 4, spacing_s=Cosine(),
    )
    fs = Freestream.from_degrees(1.0, alpha=2.0)
    xnp = []
    for frac in (0.05, 0.10, 0.25, 0.60):
        r = Reference(S, c, b, [frac * c, 0.0, 0.0], 1.0)
        system = steady_analysis([grid], r, fs, symmetric=True, ratios=[ratios])
        dCF, dCM = stability_derivatives(system)
        xnp.append(r.r[0] - dCM["alpha"][1] / dCF["alpha"][2] * c)
    xnp = np.array(xnp)
    assert np.ptp(xnp) / c < 2e-3


def test_pitch_damping_is_negative():
    """Cmq < 0 for any sane wing; a positive value is a sign error."""
    from flightlab.vlm import stability_derivatives

    system, _ = _rect(ns=30, AR=7.5, S=9.0, alpha=2.0)
    _, dCM = stability_derivatives(system)
    assert dCM["q"][1] < 0


def test_lateral_derivatives_need_mirrored_geometry_not_the_symmetric_flag():
    """``symmetric=True`` zeroes Cl and Cn *by construction*.

    The mirror image cancels the antisymmetric force and moment components
    exactly, so a roll or yaw derivative computed with the symmetric flag comes
    back as zero -- not small, exactly zero.  HW 6's dihedral and fin sweep must
    use a mirrored geometry.  This test exists so that behaviour is pinned down
    somewhere rather than discovered at 2 a.m.
    """
    from flightlab.vlm import stability_derivatives

    S, AR, alpha = 9.0, 7.5, 2.0
    b = np.sqrt(AR * S)
    c = S / b
    out = {}
    for mirror in (False, True):
        grid, ratios = wing_to_grid(
            [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [c, c],
            [0.0, 0.0], [0.0, 0.0], 30, 1,
            mirror=mirror, spacing_s=Cosine(),
        )
        r = Reference(S, c, b, [0.0, 0.0, 0.0], 1.0)
        fs = Freestream.from_degrees(1.0, alpha=alpha)
        system = steady_analysis(
            [grid], r, fs, symmetric=not mirror, ratios=[ratios]
        )
        _, dCM = stability_derivatives(system)
        out[mirror] = dCM

    assert out[False]["p"][0] == 0.0  # symmetric flag: exactly zero
    assert out[True]["p"][0] < -0.1  # mirrored: real roll damping
    # the longitudinal derivative agrees between the two, as it must
    assert out[False]["q"][1] == pytest.approx(out[True]["q"][1], rel=1e-6)


# --- derivatives against finite differences ---------------------------------


def _asymmetric_wing(alpha, beta, dihedral=6.0, twist=-3.0, ns=30, nc=3):
    """A mirrored wing with dihedral and washout, in sideslip.

    Asymmetric enough that the roll and yaw moments are nonzero, which is what
    makes the moment sign convention observable at all.
    """
    b, S = 15.0, 9.0
    grid, ratios = wing_to_grid(
        [0.0, 0.3], [0.0, b / 2], [0.0, 0.0], [1.5, 0.6],
        [0.0, np.radians(twist)], [np.radians(dihedral)] * 2,
        ns, nc, mirror=True, spacing_s=Cosine(),
    )
    r = Reference(S, S / b, b, [0.4, 0.0, 0.0], 30.0)
    fs = Freestream.from_degrees(30.0, alpha=alpha, beta=beta)
    return steady_analysis([grid], r, fs, symmetric=False, ratios=[ratios])


@pytest.mark.parametrize("alpha,beta", [(4.0, 0.0), (4.0, 10.0), (8.0, -6.0)])
def test_alpha_derivatives_match_a_finite_difference_of_body_forces(alpha, beta):
    """The analytic derivatives must agree with differencing the public output.

    This is the check that settles axis-convention questions without appealing
    to a convention: ``body_forces`` is the documented output, so differencing
    it is ground truth for ``stability_derivatives``.

    It matters for more than tidiness.  Upstream VortexLattice.jl 0.2.3 applies
    the roll/yaw sign convention to the moment *derivatives* but not to ``CM``
    itself inside ``body_forces_derivatives``, and ``CM`` is then used in the
    product rule when rotating into the stability frame.  With that mismatch
    ``Cl_alpha`` comes out ~3% wrong and ``Cn_alpha`` changes sign.  This port
    applies the convention to both, and this test is why.
    """
    from flightlab.vlm import stability_derivatives

    h = 0.02  # degrees
    CM_p = body_forces(_asymmetric_wing(alpha + h, beta), frame=Stability())[1]
    CM_m = body_forces(_asymmetric_wing(alpha - h, beta), frame=Stability())[1]
    CF_p = body_forces(_asymmetric_wing(alpha + h, beta), frame=Stability())[0]
    CF_m = body_forces(_asymmetric_wing(alpha - h, beta), frame=Stability())[0]
    dCM_fd = (CM_p - CM_m) / (2 * np.radians(h))
    dCF_fd = (CF_p - CF_m) / (2 * np.radians(h))

    dCF, dCM = stability_derivatives(_asymmetric_wing(alpha, beta))

    scale = max(np.max(np.abs(dCM_fd)), 1e-6)
    assert np.max(np.abs(dCM["alpha"] - dCM_fd)) < 1e-4 * scale
    scale_f = max(np.max(np.abs(dCF_fd)), 1e-6)
    assert np.max(np.abs(dCF["alpha"] - dCF_fd)) < 1e-4 * scale_f


@pytest.mark.parametrize("alpha,beta", [(4.0, 0.0), (5.0, 8.0)])
def test_beta_derivatives_match_a_finite_difference(alpha, beta):
    from flightlab.vlm import stability_derivatives

    h = 0.02
    CM_p = body_forces(_asymmetric_wing(alpha, beta + h), frame=Stability())[1]
    CM_m = body_forces(_asymmetric_wing(alpha, beta - h), frame=Stability())[1]
    dCM_fd = (CM_p - CM_m) / (2 * np.radians(h))

    _, dCM = stability_derivatives(_asymmetric_wing(alpha, beta))
    scale = max(np.max(np.abs(dCM_fd)), 1e-6)
    assert np.max(np.abs(dCM["beta"] - dCM_fd)) < 1e-3 * scale


def test_rate_derivatives_match_a_finite_difference():
    """Also pins the non-dimensional rate normalization, phat = p*b/(2V)."""
    from flightlab.vlm import stability_derivatives

    b, V = 15.0, 30.0
    base = _asymmetric_wing(4.0, 0.0)
    _, dCM = stability_derivatives(base)

    # dCM["p"] is with respect to the STABILITY-frame roll rate, so the
    # perturbation has to be applied there and mapped back to body rates:
    # Omega_body = R.T @ Omega_stability, whose first column is (ca, 0, sa).
    # Perturbing the body p directly instead is off by cos(alpha) and is the
    # classic way to "verify" a rate derivative and get half a percent of
    # nonsense.
    alpha = np.radians(4.0)
    axis = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    dphat = 0.005
    dp = dphat * 2.0 * V / b

    out = []
    for sign in (+1, -1):
        s = _asymmetric_wing(4.0, 0.0)
        fs = s.freestream.replace(Omega=sign * dp * axis)
        s2 = steady_analysis(
            s.grids, s.reference, fs, symmetric=False, ratios=s.ratios
        )
        out.append(body_forces(s2, frame=Stability())[1])
    dCM_fd = (out[0] - out[1]) / (2 * dphat)

    assert dCM["p"][0] == pytest.approx(dCM_fd[0], rel=2e-3)
    assert dCM["p"][0] < 0  # roll damping
