"""Smoke tests for ``flightlab.plot``.

These check that every plotting entry point runs on realistic inputs and
returns axes.  They do not check what the pictures look like.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from flightlab import fleet, foil, plot, props
from flightlab.vlm import (
    Cosine,
    Freestream,
    Reference,
    Stability,
    body_forces,
    lifting_line_coefficients,
    lifting_line_geometry,
    steady_analysis,
    wing_to_grid,
    translate,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def solved():
    """A tapered wing plus a tail, solved -- the usual thing to plot."""
    b, cr, ct = 29.0, 4.4, 1.1
    S = 0.5 * (cr + ct) * b
    wgrid, wratio = wing_to_grid(
        [0.0, 0.0], [0.0, b / 2], [0.0, 0.0], [cr, ct],
        [0.0, np.radians(-3.0)], [0.0, 0.0], 24, 2, spacing_s=Cosine(),
    )
    hgrid, hratio = wing_to_grid(
        [0.0, 0.0], [0.0, 3.0], [0.0, 0.0], [2.0, 1.2],
        [0.0, 0.0], [0.0, 0.0], 8, 1, spacing_s=Cosine(),
    )
    hgrid = translate(hgrid, [16.0, 0.0, 1.0])
    ref_q = Reference(S, S / b, b, [1.2, 0.0, 0.0], 93.0)
    fs = Freestream.from_degrees(93.0, alpha=3.0)
    system = steady_analysis(
        [wgrid, hgrid], ref_q, fs, symmetric=[True, True],
        surface_id=[1, 1], ratios=[wratio, hratio],
    )
    return system, S, b


def test_geometry_3d(solved):
    system, _, _ = solved
    ax = plot.geometry_3d(system, labels=["wing", "tail"], show_cp=True)
    assert ax is not None
    assert ax.get_xlabel() == "x (m)"


def test_geometry_3d_accepts_bare_grids(solved):
    system, _, _ = solved
    ax = plot.geometry_3d(system.grids)
    assert ax is not None


def test_planform(solved):
    system, _, _ = solved
    ax = plot.planform(system, labels=["wing", "tail"])
    assert ax.get_aspect() == 1.0
    # planforms are drawn nose-up, so the x axis is inverted
    lo, hi = ax.get_ylim()
    assert lo > hi


def test_airfoil_single_and_multiple():
    ax = plot.airfoil(foil.load("naca2412"), camber=True)
    assert ax.get_xlabel() == "x/c"
    sections = [foil.load(n) for n in ("clarky", "e212", "sd7037", "s1223")]
    ax2 = plot.airfoil(sections)
    assert len(ax2.get_lines()) == 4


def test_span_loading_with_elliptical_overlay(solved):
    system, S, b = solved
    r, c = lifting_line_geometry(system.grids)
    cf, _ = lifting_line_coefficients(system, r, c, frame=Stability())
    y = r[0][1, :]
    ymid = 0.5 * (y[:-1] + y[1:])
    cs = 0.5 * (c[0][:-1] + c[0][1:])
    cbar = S / b
    ax = plot.span_loading(ymid, cf[0][2, :] * cs / cbar)
    # computed curve plus the elliptical reference
    assert len(ax.get_lines()) == 2


def test_span_loading_without_overlay():
    y = np.linspace(0.0, 7.5, 20)
    ax = plot.span_loading(y, np.cos(y / 7.5), elliptical=False)
    assert len(ax.get_lines()) == 1


def test_cl_distribution_with_scalar_and_array_clmax(solved):
    system, _, _ = solved
    r, c = lifting_line_geometry(system.grids)
    cf, _ = lifting_line_coefficients(system, r, c, frame=Stability())
    y = r[0][1, :]
    ymid = 0.5 * (y[:-1] + y[1:])
    cl = cf[0][2, :]

    ax = plot.cl_distribution(ymid, cl, cl_max=1.4)
    assert ax.get_ylabel() == "$c_l$"
    # a per-station cl_max, which is the honest version
    clmax = np.linspace(1.5, 1.2, len(cl))
    ax2 = plot.cl_distribution(ymid, cl, cl_max=clmax)
    assert ax2 is not None


def test_drag_polar_single_and_multiple():
    CL = np.linspace(0.0, 1.4, 60)
    CD1 = 0.02 + CL**2 / (np.pi * 10.0 * 0.95)
    CD2 = 0.024 + CL**2 / (np.pi * 10.0 * 0.85)
    ax = plot.drag_polar(CD1, CL, label="method B")
    assert ax.get_xlim()[0] == 0.0
    ax2 = plot.drag_polar(
        [CD1, CD2], [CL, CL], label=["strip integration", "handbook"],
        ld_reference=14.0,
    )
    assert ax2 is not None


def test_airfoil_polars():
    results = [foil.polar(n, Re=1e5) for n in ("sd7037", "e212", "naca2412")]
    a1, a2 = plot.airfoil_polars(results, labels=["SD7037", "E212", "NACA 2412"])
    assert a1.get_ylabel() == "$c_l$"
    assert a2.get_xlim()[1] > 0


def test_contour_with_constraints_and_optimum():
    taper = np.linspace(0.3, 1.0, 30)
    twist = np.linspace(-6.0, 2.0, 28)
    T, W = np.meshgrid(taper, twist, indexing="ij")
    e_inv = 0.98 - 0.4 * (T - 0.42) ** 2 - 0.004 * (W + 2.0) ** 2
    margin = 0.25 + 0.05 * W - 0.3 * (1.0 - T)
    ax = plot.contour(
        T, W, e_inv,
        xlabel="taper ratio", ylabel="twist (deg)", zlabel="$e_{inv}$",
        constraints=[
            dict(mask=margin < 0.10, label="stall margin < 0.10"),
            dict(mask=T > 0.9, label="taper > 0.9"),
        ],
        optimum=(0.42, -2.0, "best $e_{inv}$"),
    )
    assert ax.get_xlabel() == "taper ratio"
    assert ax.get_legend() is not None


def test_contour_without_constraints():
    x = np.linspace(0, 1, 12)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ax = plot.contour(X, Y, X * Y, zlabel="z")
    assert ax is not None


def test_convergence_two_schemes():
    n = np.array([5, 10, 20, 40, 80, 160])
    uni = 1.0 + 0.5 / n
    cos = 1.0 + 0.12 / n
    ax = plot.convergence(
        n, [uni, cos], labels=["uniform", "cosine"],
        reference=1.0, tol=0.002, ylabel="$e_{inv}$", xlabel="spanwise panels",
    )
    assert ax.get_xscale() == "log"


def test_breakdown():
    names = ["wing", "fuselage", "h tail", "v tail", "nacelles", "boom", "misc"]
    vals = np.array([2.9, 2.4, 0.55, 0.31, 0.62, 0.04, 0.35])
    ax = plot.breakdown(names, vals, title="787-8 cruise drag area")
    assert len(ax.get_yticklabels()) == len(names)


def test_vn_diagram():
    V = np.linspace(0.0, 90.0, 200)
    rho, WS, CLmax = 1.225, 545.0, 1.4
    n_stall = 0.5 * rho * V**2 * CLmax / WS
    n_pos = np.minimum(n_stall, 5.3)
    n_neg = np.maximum(-n_stall, -2.65)
    ax = plot.vn_diagram(
        V, n_pos, n_neg, V_dive=79.0, V_stall_pos=float(V[np.argmax(n_stall >= 1.0)]),
        gust_lines=[dict(V=[0.0, 60.0], n=[1.0, 3.4], label="+15 m/s gust")],
    )
    assert ax.get_ylabel() == "load factor $n$"


def test_eigenvalues():
    longitudinal = np.array([
        -0.02 + 0.18j, -0.02 - 0.18j,  # phugoid
        -2.1 + 3.0j, -2.1 - 3.0j,      # short period
    ])
    ax = plot.eigenvalues(
        longitudinal,
        mode_names=["phugoid", "phugoid", "short period", "short period"],
    )
    assert ax.get_xlabel().startswith("Re")


def test_eigenvalues_multiple_sets_and_unstable_root():
    a = np.array([-0.03 + 0.2j, -0.03 - 0.2j, +0.01 + 0j])
    b = np.array([-0.05 + 0.3j, -0.05 - 0.3j, -0.02 + 0j])
    ax = plot.eigenvalues([a, b], labels=["sea level", "11.9 km"])
    assert ax.get_legend() is not None


def test_prop_curves():
    p = props.load("apce_10x7")
    aT, aP, aE = plot.prop_curves(p)
    assert aE.get_ylim() == (0.0, 1.0)
    assert "APC" in aT.get_title()


def test_prop_curves_pooled():
    aT, _, _ = plot.prop_curves(props.load("apce_9x6"), show_runs=False)
    assert aT is not None


def test_wing_loading_map_from_the_fleet():
    """HW 1's design-space plot, driven straight off the fleet data."""
    labels, ws, v = [], [], []
    for label, a in fleet.AIRCRAFT.items():
        if a.wing is None:
            continue
        speed = a.operating.get("cruise_speed")
        if speed is None:
            continue
        labels.append(label)
        ws.append(a.wing_loading)
        v.append(speed)
    assert len(labels) >= 2
    ax = plot.wing_loading_map(
        ws, v, names=labels,
        segments=[dict(V=[29.0, 34.0], W_S=[355.0, 545.0],
                       label="ASW-27B filling ballast")],
    )
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_every_public_plot_function_is_covered():
    """A guard so a new plotting helper does not ship untested.

    Scans the whole test suite rather than this file alone: the draft-4
    additions take result objects, so they are exercised alongside the modules
    that produce them in ``test_analysis.py``.  What matters is that every
    name in ``plot.__all__`` is called somewhere, not which file calls it.
    """
    from pathlib import Path

    suite = "\n".join(
        f.read_text() for f in Path(__file__).parent.glob("test_*.py")
    )
    uncalled = [
        name for name in plot.__all__ if f"plot.{name}(" not in suite
    ]
    assert not uncalled, (
        f"these plotting helpers are exported but never called in any test: "
        f"{uncalled}"
    )
