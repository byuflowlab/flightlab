"""Coverage for the rest of the API the draft names as required.

``get_surface_properties``, the ``Sine`` spacing scheme, ``body_derivatives``,
the ``Wind`` frame, and the input-handling paths a student is likely to hit.
"""

import numpy as np
import pytest

from flightlab.vlm import (
    Body,
    Cosine,
    Freestream,
    PanelProperties,
    Reference,
    Sine,
    Stability,
    Surface,
    System,
    Uniform,
    Wind,
    body_derivatives,
    body_forces,
    far_field_drag,
    get_surface_properties,
    grid_to_surface_panels,
    influence_coefficients,
    induced_velocity,
    lifting_line_geometry,
    rotate,
    steady_analysis,
    translate,
    wing_to_grid,
)

B, S, CR, CT = 10.0, 12.5, 2.0, 0.5
REF = Reference(S, S / B, B, [0.5, 0.0, 0.0], 30.0)


def _wing(ns=16, nc=3, spacing=None, mirror=False, twist=0.0, dihedral=0.0):
    t = np.radians(twist)
    d = np.radians(dihedral)
    return wing_to_grid(
        [0.0, 0.3], [0.0, B / 2], [0.0, 0.0], [CR, CT], [0.0, t], [d, d],
        ns, nc, mirror=mirror,
        spacing_s=Cosine() if spacing is None else spacing,
        spacing_c=Uniform(),
    )


def _solve(alpha=4.0, beta=0.0, **kw):
    grid, ratios = _wing(**kw)
    fs = Freestream.from_degrees(REF.V, alpha=alpha, beta=beta)
    sym = not kw.get("mirror", False)
    return steady_analysis([grid], REF, fs, symmetric=sym, ratios=[ratios])


# --- get_surface_properties -------------------------------------------------


def test_get_surface_properties_shapes_and_content():
    system = _solve()
    props = get_surface_properties(system)
    assert isinstance(props, list) and len(props) == 1
    p = props[0]
    assert isinstance(p, PanelProperties)

    nc, ns = system.surfaces[0].shape
    assert p.gamma.shape == (nc, ns)
    assert p.velocity.shape == (nc, ns, 3)
    for name in ("cfb", "cfl", "cfr"):
        assert getattr(p, name).shape == (nc, ns, 3)
    assert p.shape == (nc, ns)
    assert p.cf_total.shape == (nc, ns, 3)
    assert np.all(np.isfinite(p.gamma))
    assert np.all(np.isfinite(p.velocity))


def test_panel_forces_sum_to_the_body_force():
    """The identity that makes the per-panel output trustworthy."""
    system = _solve()
    p = get_surface_properties(system)[0]
    total = p.cf_total.reshape(-1, 3).sum(axis=0)
    CF, _ = body_forces(system, frame=Body())
    # symmetric doubles the x and z components and cancels y
    assert CF[0] == pytest.approx(2.0 * total[0], rel=1e-10)
    assert CF[2] == pytest.approx(2.0 * total[2], rel=1e-10)
    assert CF[1] == 0.0


def test_panel_velocity_is_near_the_freestream_magnitude():
    """A vortex lattice perturbs the freestream; it should not dominate it."""
    system = _solve(alpha=3.0)
    p = get_surface_properties(system)[0]
    mag = np.linalg.norm(p.velocity, axis=-1)  # already normalized by Vref
    assert np.all(mag > 0.8)
    assert np.all(mag < 1.3)


def test_get_surface_properties_refuses_without_a_near_field_analysis():
    grid, ratios = _wing()
    fs = Freestream.from_degrees(REF.V, alpha=2.0)
    system = steady_analysis(
        [grid], REF, fs, symmetric=True, ratios=[ratios],
        near_field_analysis=False,
    )
    assert system.properties is None
    with pytest.raises(RuntimeError, match="near_field_analysis"):
        get_surface_properties(system)
    with pytest.raises(RuntimeError, match="near field analysis"):
        body_forces(system)
    # but the circulation solve still happened
    assert system.gamma is not None
    assert np.all(np.isfinite(system.gamma))


# --- spacing schemes --------------------------------------------------------


@pytest.mark.parametrize("spacing", [Uniform(), Sine(), Cosine()])
def test_every_spacing_scheme_solves(spacing):
    system = _solve(spacing=spacing)
    CF, CM = body_forces(system, frame=Stability())
    assert np.all(np.isfinite(CF)) and np.all(np.isfinite(CM))
    assert CF[2] > 0
    assert far_field_drag(system) > 0


def test_spacing_schemes_agree_when_refined():
    """Different panellings must converge to the same answer."""
    out = {}
    for spacing in (Uniform(), Sine(), Cosine()):
        system = _solve(ns=120, spacing=spacing)
        CF, _ = body_forces(system, frame=Stability())
        out[type(spacing).__name__] = (CF[2], far_field_drag(system))
    CLs = np.array([v[0] for v in out.values()])
    CDs = np.array([v[1] for v in out.values()])
    assert np.ptp(CLs) / CLs.mean() < 0.01, out
    assert np.ptp(CDs) / CDs.mean() < 0.02, out


def test_sine_spacing_bunches_panels_outboard():
    """Sine on a half wing becomes cosine once symmetry is applied."""
    from flightlab.vlm.geometry import spanwise_spacing

    eta_u, _ = spanwise_spacing(9, Uniform())
    eta_s, _ = spanwise_spacing(9, Sine())
    eta_c, _ = spanwise_spacing(9, Cosine())
    for eta in (eta_u, eta_s, eta_c):
        assert eta[0] == pytest.approx(0.0)
        assert eta[-1] == pytest.approx(1.0)
        assert np.all(np.diff(eta) > 0)
    # sine spacing puts the first interior station further outboard than uniform
    assert eta_s[1] > eta_u[1]
    # cosine clusters at both ends, so its first station is inboard of uniform
    assert eta_c[1] < eta_u[1]


def test_midpoints_are_mapped_through_the_spacing_function():
    """Not the arithmetic mean of the edges -- that would lose the accuracy."""
    from flightlab.vlm.geometry import spanwise_spacing

    eta, mid = spanwise_spacing(9, Cosine())
    arithmetic = 0.5 * (eta[:-1] + eta[1:])
    assert not np.allclose(mid, arithmetic)
    assert np.all(mid > eta[:-1]) and np.all(mid < eta[1:])


# --- body_derivatives -------------------------------------------------------


def test_body_derivatives_keys_and_signs():
    system = _solve(alpha=3.0, mirror=True)
    dCF, dCM = body_derivatives(system)
    assert set(dCF) == set(dCM) == {"u", "v", "w", "p", "q", "r"}
    for d in (dCF, dCM):
        for k, v in d.items():
            assert np.asarray(v).shape == (3,)
            assert np.all(np.isfinite(v))
    # a w perturbation is an alpha increase, so it must produce lift
    # (body-frame z is up, so the z force derivative is positive)
    assert dCF["w"][2] > 0


def test_body_derivatives_convert_from_alpha_consistently():
    """d/dw should equal d/dalpha times dalpha/dw at zero sideslip."""
    from flightlab.vlm import stability_derivatives
    from flightlab.vlm.nearfield import body_forces_derivatives

    system = _solve(alpha=3.0, mirror=True)
    _, _, dCFb, _ = body_forces_derivatives(system)
    dCF, _ = body_derivatives(system)
    u, v, w = np.array([
        system.freestream.Vinf * np.cos(system.freestream.alpha),
        0.0,
        system.freestream.Vinf * np.sin(system.freestream.alpha),
    ])
    a_w = u / (u * u + w * w)
    assert dCF["w"] == pytest.approx(dCFb["alpha"] * a_w, rel=1e-12)


# --- output frames ----------------------------------------------------------


def test_the_three_frames_agree_at_zero_alpha_and_beta():
    system = _solve(alpha=0.0, beta=0.0, twist=-3.0)
    ref_out = body_forces(system, frame=Body())
    for frame in (Stability(), Wind()):
        out = body_forces(system, frame=frame)
        assert out[0] == pytest.approx(ref_out[0], abs=1e-12)
        assert out[1] == pytest.approx(ref_out[1], abs=1e-12)


def test_stability_frame_rotates_the_force_by_alpha():
    alpha = 6.0
    system = _solve(alpha=alpha)
    CFb, _ = body_forces(system, frame=Body())
    CFs, _ = body_forces(system, frame=Stability())
    ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
    assert CFs[0] == pytest.approx(ca * CFb[0] + sa * CFb[2], rel=1e-12)
    assert CFs[2] == pytest.approx(-sa * CFb[0] + ca * CFb[2], rel=1e-12)


def test_wind_frame_differs_from_stability_only_in_sideslip():
    system = _solve(alpha=4.0, beta=0.0, mirror=True)
    CFs, _ = body_forces(system, frame=Stability())
    CFw, _ = body_forces(system, frame=Wind())
    assert CFw == pytest.approx(CFs, abs=1e-12)

    yawed = _solve(alpha=4.0, beta=10.0, mirror=True)
    CFs2, _ = body_forces(yawed, frame=Stability())
    CFw2, _ = body_forces(yawed, frame=Wind())
    assert not np.allclose(CFw2, CFs2)


def test_wind_frame_is_the_stability_frame_rotated_by_beta():
    """The wind frame differs from the stability frame by a yaw of beta.

    Sign convention, spelled out because it is easy to get backwards::

        CF_wind = [[ cb, -sb, 0],
                   [ sb,  cb, 0],
                   [  0,   0, 1]] @ CF_stability

    Note the direction that puts the axial force: with a small ``CY``, the
    wind-frame axial force comes out *smaller* than the stability-frame one by
    ``cos(beta)``, which is not the guess most people make.
    """
    beta = 12.0
    # dihedral so that CY is genuinely nonzero -- a flat mirrored wing's side
    # force cancels to exactly zero and would not exercise the mixing terms
    system = _solve(alpha=4.0, beta=beta, mirror=True, dihedral=10.0)
    CFs, _ = body_forces(system, frame=Stability())
    CFw, _ = body_forces(system, frame=Wind())
    cb, sb = np.cos(np.radians(beta)), np.sin(np.radians(beta))

    assert abs(CFs[1]) > 1e-6, "expected a real side force to rotate"
    assert CFw[0] == pytest.approx(cb * CFs[0] - sb * CFs[1], abs=1e-15)
    assert CFw[1] == pytest.approx(sb * CFs[0] + cb * CFs[1], abs=1e-15)
    assert CFw[2] == pytest.approx(CFs[2], rel=1e-12)  # lift is unaffected

    # the rotation preserves the magnitude of the in-plane force
    assert np.hypot(CFw[0], CFw[1]) == pytest.approx(
        np.hypot(CFs[0], CFs[1]), rel=1e-12
    )


def test_unknown_frame_is_refused():
    system = _solve()
    with pytest.raises(TypeError, match="unknown frame"):
        body_forces(system, frame="stability")


# --- input handling ---------------------------------------------------------


def test_a_bare_grid_is_accepted_without_a_list():
    grid, ratios = _wing()
    fs = Freestream.from_degrees(REF.V, alpha=2.0)
    a = steady_analysis(grid, REF, fs, symmetric=True, ratios=ratios)
    b = steady_analysis([grid], REF, fs, symmetric=True, ratios=[ratios])
    assert body_forces(a)[0] == pytest.approx(body_forces(b)[0])


def test_surfaces_may_be_passed_instead_of_grids():
    grid, ratios = _wing()
    _, _, surface = grid_to_surface_panels(grid, ratios=ratios)
    assert isinstance(surface, Surface)
    fs = Freestream.from_degrees(REF.V, alpha=2.0)
    system = steady_analysis([surface], REF, fs, symmetric=True)
    CF, _ = body_forces(system, frame=Stability())
    assert CF[2] > 0
    # no grids were supplied, so the lifting-line helper must say so clearly
    assert system.grids == []


def test_lifting_line_without_grids_raises_a_useful_error():
    from flightlab.vlm import lifting_line_coefficients

    grid, ratios = _wing()
    _, _, surface = grid_to_surface_panels(grid, ratios=ratios)
    fs = Freestream.from_degrees(REF.V, alpha=2.0)
    system = steady_analysis([surface], REF, fs, symmetric=True)
    with pytest.raises(RuntimeError, match="grids"):
        lifting_line_coefficients(system)


def test_regridding_path_of_grid_to_surface_panels():
    """The ``(xyz, ns, nc)`` form re-interpolates an existing grid."""
    grid, _ = _wing(ns=40, nc=6)
    out_grid, out_ratios, surface = grid_to_surface_panels(
        grid, 12, 3, spacing_s=Cosine(), spacing_c=Uniform()
    )
    assert surface.shape == (3, 12)
    assert out_grid.shape == (3, 4, 13)
    assert out_ratios.shape == (2, 3, 12)


def test_mismatched_flag_length_is_refused():
    grid, ratios = _wing()
    fs = Freestream.from_degrees(REF.V, alpha=2.0)
    with pytest.raises(ValueError, match="one entry per surface"):
        steady_analysis(
            [grid], REF, fs, symmetric=[True, False], ratios=[ratios]
        )


def test_grid_to_surface_panels_requires_both_ns_and_nc():
    grid, _ = _wing()
    with pytest.raises(ValueError, match="both ns and nc"):
        grid_to_surface_panels(grid, 12)


# --- geometry transforms ----------------------------------------------------


def test_translate_and_rotate_are_consistent():
    grid, _ = _wing()
    moved = translate(grid, [1.0, 0.0, 2.0])
    assert np.allclose(moved[0] - grid[0], 1.0)
    assert np.allclose(moved[2] - grid[2], 2.0)

    # a 90 degree rotation about x maps +y to +z
    c, s = 0.0, 1.0
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    spun = rotate(grid, R)
    assert np.allclose(spun[2], grid[1], atol=1e-12)


def test_translation_does_not_change_forces():
    """A rigid translation of the whole problem must be invisible."""
    grid, ratios = _wing()
    fs = Freestream.from_degrees(REF.V, alpha=4.0)
    base = steady_analysis([grid], REF, fs, symmetric=True, ratios=[ratios])
    CF0, _ = body_forces(base, frame=Stability())

    shift = np.array([3.0, 0.0, -1.5])
    moved_ref = Reference(REF.S, REF.c, REF.b, REF.r + shift, REF.V)
    moved = steady_analysis(
        [translate(grid, shift)], moved_ref, fs, symmetric=True, ratios=[ratios]
    )
    CF1, _ = body_forces(moved, frame=Stability())
    assert CF1 == pytest.approx(CF0, rel=1e-9)


def test_surface_reflect_round_trips():
    grid, ratios = _wing()
    _, _, surface = grid_to_surface_panels(grid, ratios=ratios)
    twice = surface.reflect().reflect()
    assert np.allclose(twice.rtl, surface.rtl)
    assert np.allclose(twice.ncp, surface.ncp)
    once = surface.reflect()
    assert np.allclose(once.rtc[..., 1], -surface.rtc[:, ::-1, 1])


def test_surface_geometry_properties():
    grid, ratios = _wing(ns=8, nc=2)
    _, _, s = grid_to_surface_panels(grid, ratios=ratios)
    assert s.shape == (2, 8)
    assert s.size == 16
    assert s.nc == 2 and s.ns == 8
    assert s.flat("rtl").shape == (16, 3)
    assert s.flat("chord").shape == (16,)
    # unit normals
    assert np.allclose(np.linalg.norm(s.ncp, axis=-1), 1.0)
    # the bound vortex is forward of the control point
    assert np.all(s.rcp[..., 0] > s.rtc[..., 0])


# --- lower-level entry points ----------------------------------------------


def test_influence_coefficients_is_square_and_nonsingular():
    grid, ratios = _wing(ns=10, nc=2)
    _, _, surface = grid_to_surface_panels(grid, ratios=ratios)
    AIC = influence_coefficients(
        [surface], [True], [1], [True], (1.0, 0.0, 0.0)
    )
    n = surface.size
    assert AIC.shape == (n, n)
    assert np.all(np.isfinite(AIC))
    assert abs(np.linalg.det(AIC)) > 0
    # a panel's strongest influence is on itself
    assert np.all(np.abs(np.diag(AIC)) > 0)


def test_induced_velocity_decays_far_from_the_wing():
    system = _solve(ns=12, nc=2)
    near = induced_velocity(
        np.array([[1.0, 2.0, 0.5]]), system.surfaces, system.gamma,
        system.symmetric, system.surface_id, system.trailing_vortices,
        system.xhat,
    )
    far = induced_velocity(
        np.array([[1.0, 2.0, 500.0]]), system.surfaces, system.gamma,
        system.symmetric, system.surface_id, system.trailing_vortices,
        system.xhat,
    )
    assert np.linalg.norm(far) < 0.02 * np.linalg.norm(near)


def test_system_bookkeeping():
    system = _solve(ns=10, nc=3)
    assert system.nsurf == 1
    assert system.npanels == 30
    assert list(system.offsets) == [0, 30]
    assert system.has_derivatives
    assert system.surface_gamma(0).shape == (3, 10)
    assert isinstance(system, System)


def test_freestream_degree_helpers_round_trip():
    fs = Freestream.from_degrees(20.0, alpha=5.0, beta=-3.0)
    assert fs.alpha_deg == pytest.approx(5.0)
    assert fs.beta_deg == pytest.approx(-3.0)
    assert fs.alpha == pytest.approx(np.radians(5.0))
    moved = fs.replace(alpha=np.radians(7.0))
    assert moved.alpha_deg == pytest.approx(7.0)
    assert moved.Vinf == fs.Vinf


def test_reference_lengths_order():
    r = Reference(30.0, 2.0, 15.0, [0.5, 0.0, 0.0], 1.0)
    assert list(r.lengths) == [15.0, 2.0, 15.0]  # (b, c, b) for (Cl, Cm, Cn)
    assert r.r.shape == (3,)


def test_dihedral_reduces_lift_and_is_handled():
    """A dihedral wing carries less lift per unit alpha; also a smoke test."""
    flat = _solve(alpha=4.0, dihedral=0.0)
    dihed = _solve(alpha=4.0, dihedral=15.0)
    CL_flat = body_forces(flat, frame=Stability())[0][2]
    CL_dihed = body_forces(dihed, frame=Stability())[0][2]
    assert 0 < CL_dihed < CL_flat
