"""Validate ``flightlab.vlm`` against AVL.

These are the same reference cases and the same tolerances used by
VortexLattice.jl's own test suite, whose expected values come from AVL runs.
If this file passes, the port reproduces the Julia package, which reproduces
AVL.
"""

import numpy as np
import pytest

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
    stability_derivatives,
    steady_analysis,
    translate,
    wing_to_grid,
)

ZTOL = np.sqrt(np.finfo(float).eps)


def avl_normal_vector(dr, theta):
    """Normal vector as AVL constructs it, for the dihedral+twist cases."""
    dr = np.asarray(dr, dtype=float)
    st, ct = np.sin(theta), np.cos(theta)
    shat = np.array([0.0, -dr[2], dr[1]]) / np.hypot(dr[1], dr[2])
    chat = np.array([ct, -st * shat[1], -st * shat[2]])
    ncp = np.cross(chat, dr)
    return ncp / np.linalg.norm(ncp)


# --- the simple wing shared by several runs --------------------------------

XLE = [0.0, 0.4]
YLE = [0.0, 7.5]
ZLE = [0.0, 0.0]
CHORD = [2.2, 1.8]
THETA = [np.radians(2.0), np.radians(2.0)]
PHI = [0.0, 0.0]

SREF, CREF, BREF = 30.0, 2.0, 15.0
RREF = [0.50, 0.0, 0.0]


def simple_wing(spacing_s, mirror, chord=None, ns=12, nc=1):
    chord = CHORD if chord is None else chord
    return wing_to_grid(
        XLE, YLE, ZLE, chord, THETA, PHI, ns, nc,
        mirror=mirror, spacing_s=spacing_s, spacing_c=Uniform(),
    )


def chord_over_cos():
    """AVL measures chord in the x-direction; twist makes that differ."""
    return [c / np.cos(t) for c, t in zip(CHORD, THETA)]


@pytest.mark.parametrize("mirror,symmetric", [(False, True), (True, False)])
def test_run1_uniform_spacing(mirror, symmetric):
    grid, ratios = simple_wing(Uniform(), mirror, chord=chord_over_cos())
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0)

    system = steady_analysis(
        [grid], ref, fs, symmetric=symmetric, ratios=[ratios]
    )
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.24324, abs=1e-3)
    assert CD == pytest.approx(0.00243, abs=1e-5)
    assert CDiff == pytest.approx(0.00245, abs=1e-5)
    assert Cm == pytest.approx(-0.02252, abs=1e-4)
    assert CY == pytest.approx(0.0, abs=ZTOL)
    assert Cl == pytest.approx(0.0, abs=ZTOL)
    assert Cn == pytest.approx(0.0, abs=ZTOL)


def test_run2_cosine_spacing():
    grid, ratios = simple_wing(Cosine(), mirror=True, chord=chord_over_cos())
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0)

    system = steady_analysis([grid], ref, fs, symmetric=False, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.23744, abs=1e-3)
    assert CD == pytest.approx(0.00254, abs=1e-5)
    assert CDiff == pytest.approx(0.00243, abs=1e-5)
    assert Cm == pytest.approx(-0.02165, abs=1e-4)
    assert CY == pytest.approx(0.0, abs=ZTOL)
    assert Cl == pytest.approx(0.0, abs=ZTOL)
    assert Cn == pytest.approx(0.0, abs=ZTOL)


def test_run3_high_alpha():
    grid, ratios = simple_wing(Uniform(), mirror=False, chord=chord_over_cos())
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=8.0)

    system = steady_analysis([grid], ref, fs, symmetric=True, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.80348, abs=1e-3)
    assert CD == pytest.approx(0.02651, abs=1e-4)
    assert CDiff == pytest.approx(0.02696, abs=1e-5)
    assert Cm == pytest.approx(-0.07399, abs=1e-3)


def test_run10_sideslip():
    grid, ratios = simple_wing(Uniform(), mirror=True, chord=chord_over_cos())
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0, beta=15.0)

    system = steady_analysis([grid], ref, fs, symmetric=False, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.22695, abs=1e-3)
    assert CD == pytest.approx(0.00227, abs=1e-5)
    assert CDiff == pytest.approx(0.0022852, abs=1e-5)
    assert Cm == pytest.approx(-0.02101, abs=1e-4)
    assert CY == pytest.approx(0.0, abs=1e-5)
    assert Cl == pytest.approx(-0.00644, abs=1e-4)
    assert Cn == pytest.approx(0.00012, abs=2e-4)


def test_run6_wing_and_tail():
    """Three surfaces, finite core off, matching AVL's normal vectors."""
    xle, yle, zle = [0.0, 0.2], [0.0, 5.0], [0.0, 1.0]
    chord, theta, phi = [1.0, 0.6], [np.radians(2.0), np.radians(2.0)], [0.0, 0.0]

    xle_h, yle_h, zle_h = [0.0, 0.14], [0.0, 1.25], [0.0, 0.0]
    chord_h, theta_h, phi_h = [0.7, 0.42], [0.0, 0.0], [0.0, 0.0]

    xle_v, yle_v, zle_v = [0.0, 0.14], [0.0, 0.0], [0.0, 1.0]
    chord_v, theta_v, phi_v = [0.7, 0.42], [0.0, 0.0], [0.0, 0.0]

    chord = [c / np.cos(t) for c, t in zip(chord, theta)]

    ref = Reference(9.0, 0.9, 10.0, [0.5, 0.0, 0.0], 1.0)
    fs = Freestream.from_degrees(1.0, alpha=5.0)

    ncp = avl_normal_vector(
        [xle[1] - xle[0], yle[1] - yle[0], zle[1] - zle[0]], np.radians(2.0)
    )

    wgrid, wratio = wing_to_grid(
        xle, yle, zle, chord, theta, phi, 12, 1,
        spacing_s=Uniform(), spacing_c=Uniform(),
    )
    hgrid, hratio = wing_to_grid(
        xle_h, yle_h, zle_h, chord_h, theta_h, phi_h, 6, 1,
        spacing_s=Uniform(), spacing_c=Uniform(),
    )
    hgrid = translate(hgrid, [4.0, 0.0, 0.0])
    vgrid, vratio = wing_to_grid(
        xle_v, yle_v, zle_v, chord_v, theta_v, phi_v, 5, 1,
        spacing_s=Uniform(), spacing_c=Uniform(),
    )
    vgrid = translate(vgrid, [4.0, 0.0, 0.0])

    from flightlab.vlm import grid_to_surface_panels

    _, _, wing = grid_to_surface_panels(wgrid, ratios=wratio)
    # compared by vector norm, as the Julia suite does: the two constructions
    # differ slightly for a twisted panel with dihedral
    assert np.linalg.norm(wing.ncp - ncp, axis=-1).max() < 0.01
    wing = wing.set_normal(ncp)
    _, _, htail = grid_to_surface_panels(hgrid, ratios=hratio)
    _, _, vtail = grid_to_surface_panels(vgrid, ratios=vratio)

    system = steady_analysis(
        [wing, htail, vtail], ref, fs,
        symmetric=[True, True, False], surface_id=[1, 1, 1],
    )
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.60408, abs=1e-2)
    assert CD == pytest.approx(0.01058, abs=1e-4)
    assert CDiff == pytest.approx(0.010378, abs=1e-3)
    assert Cm == pytest.approx(-0.02778, abs=2e-3)
    assert CY == pytest.approx(0.0, abs=ZTOL)
    assert Cl == pytest.approx(0.0, abs=ZTOL)
    assert Cn == pytest.approx(0.0, abs=ZTOL)


def test_run8_chordwise_panels():
    grid, ratios = simple_wing(
        Uniform(), mirror=False, chord=chord_over_cos(), ns=12, nc=6
    )
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0)

    system = steady_analysis([grid], ref, fs, symmetric=True, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CDiff = far_field_drag(system)
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.24454, abs=1e-3)
    assert CD == pytest.approx(0.00247, abs=1e-5)
    assert CDiff == pytest.approx(0.00248, abs=1e-5)
    assert Cm == pytest.approx(-0.02091, abs=1e-4)


def test_run11_stability_derivatives():
    """Note: no chord/cos adjustment here, matching the Julia test."""
    grid, ratios = simple_wing(Uniform(), mirror=True)
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0)

    system = steady_analysis([grid], ref, fs, symmetric=False, ratios=[ratios])
    dCF, dCM = stability_derivatives(system)

    CDa, CYa, CLa = dCF["alpha"]
    Cla, Cma, Cna = dCM["alpha"]
    CDb, CYb, CLb = dCF["beta"]
    Clb, Cmb, Cnb = dCM["beta"]
    CDp, CYp, CLp = dCF["p"]
    Clp, Cmp, Cnp = dCM["p"]
    CDq, CYq, CLq = dCF["q"]
    Clq, Cmq, Cnq = dCM["q"]
    CDr, CYr, CLr = dCF["r"]
    Clr, Cmr, Cnr = dCM["r"]

    assert CLa == pytest.approx(4.638088, rel=0.01)
    assert CLb == pytest.approx(0.0, abs=ZTOL)
    assert CYa == pytest.approx(0.0, abs=ZTOL)
    assert CYb == pytest.approx(-0.000007, abs=1e-4)
    assert Cla == pytest.approx(0.0, abs=ZTOL)
    assert Clb == pytest.approx(-0.025749, abs=0.001)
    assert Cma == pytest.approx(-0.429247, rel=0.01)
    assert Cmb == pytest.approx(0.0, abs=ZTOL)
    assert Cna == pytest.approx(0.0, abs=ZTOL)
    assert Cnb == pytest.approx(0.000466, abs=1e-3)
    assert Clp == pytest.approx(-0.518725, rel=0.01)
    assert Clq == pytest.approx(0.0, abs=ZTOL)
    assert Clr == pytest.approx(0.064243, rel=0.01)
    assert Cmp == pytest.approx(0.0, abs=ZTOL)
    assert Cmq == pytest.approx(-0.517094, rel=0.01)
    assert Cmr == pytest.approx(0.0, abs=ZTOL)
    assert Cnp == pytest.approx(-0.019846, rel=0.01)
    assert Cnq == pytest.approx(0.0, abs=ZTOL)
    assert Cnr == pytest.approx(-0.000898, rel=0.01)


def test_run12_rotational_velocity():
    grid, ratios = simple_wing(Uniform(), mirror=True)
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    # non-dimensional roll rate phat = 0.05
    fs = Freestream.from_degrees(
        1.0, alpha=1.0, Omega=[2 * 1.0 * 0.05 / BREF, 0.0, 0.0]
    )

    system = steady_analysis([grid], ref, fs, symmetric=False, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CD, CY, CL = CF
    Cl, Cm, Cn = CM

    assert CL == pytest.approx(0.24323, abs=1e-3)
    assert CD == pytest.approx(0.00069, abs=1e-5)
    assert Cm == pytest.approx(-0.02251, abs=1e-4)
    assert CY == pytest.approx(0.00235, abs=2e-4)
    assert Cl == pytest.approx(-0.02594, abs=1e-4)
    assert Cn == pytest.approx(-0.00099, abs=2e-4)


def test_lifting_line_coefficients():
    grid, ratios = simple_wing(Uniform(), mirror=False, chord=chord_over_cos())
    ref = Reference(SREF, CREF, BREF, RREF, 1.0)
    fs = Freestream.from_degrees(1.0, alpha=1.0)

    system = steady_analysis([grid], ref, fs, symmetric=True, ratios=[ratios])
    r_ll, c_ll = lifting_line_geometry([grid])
    cf, cm = lifting_line_coefficients(system, r_ll, c_ll, frame=Stability())

    cl_avl = np.array([0.2618, 0.2646, 0.2661, 0.2664, 0.2654, 0.2628, 0.2584,
                       0.2513, 0.2404, 0.2233, 0.1952, 0.1434])
    cd_avl = np.array([0.0029, 0.0024, 0.0023, 0.0023, 0.0023, 0.0023, 0.0024,
                       0.0024, 0.0025, 0.0026, 0.0026, 0.0022])

    assert np.max(np.abs(cf[0][2, :] - cl_avl)) < 1e-3
    assert np.max(np.abs(cf[0][0, :] - cd_avl)) < 1e-4
    assert np.max(np.abs(cm[0][1, :])) < 1e-4
