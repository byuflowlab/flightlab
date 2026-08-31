"""``flightlab.plot`` -- plotting for geometry and results.

Every function takes plain arrays wherever it can, so it works on your own
numbers and not only on this package's objects.  Every function accepts an
``ax`` and returns it, and none of them call ``show()`` or ``savefig()`` -- what
you do with the figure is yours.

Two habits this module tries to enforce, because reports lose points on them
every year:

* **Set your axis limits by hand.**  A drag polar autoscaled to include
  post-stall drag crushes the low-drag region into the axis, which is the only
  part anybody is looking at.  :func:`drag_polar` clips to the useful range by
  default.
* **Plot both span distributions.**  Lift per unit span sets induced drag;
  local ``c_l`` sets where the wing stalls first.  They are not the same thing
  and they do not peak in the same place.  :func:`span_loading` draws the first
  with an elliptical reference; :func:`cl_distribution` draws the second with
  the section ``cl_max`` line.

Angles are in **degrees** at this interface.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "geometry_3d",
    "planform",
    "airfoil",
    "span_loading",
    "cl_distribution",
    "drag_polar",
    "airfoil_polars",
    "contour",
    "vn_diagram",
    "eigenvalues",
    "prop_curves",
    "convergence",
    "breakdown",
    "wing_loading_map",
    # draft-4 additions
    "loading_3d",
    "span_load",
    "stall_margin",
    "thrust_and_drag",
    "power_curves",
    "mode_response",
    "envelope",
    "chain_breakdown",
    "fleet_overlay",
    "polar_comparison",
]


def _ax(ax=None, figsize=(7.0, 4.5), projection=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    return ax


def _grids_from(obj):
    """Accept a System, a list of grids, or a single grid; return a grid list."""
    if hasattr(obj, "grids") and obj.grids:
        return list(obj.grids)
    arr = np.asarray(obj, dtype=float) if not isinstance(obj, (list, tuple)) else None
    if arr is not None and arr.ndim == 3:
        return [arr]
    return [np.asarray(g, dtype=float) for g in obj]


# --- geometry ---------------------------------------------------------------


def geometry_3d(surfaces, ax=None, color=None, alpha=0.35, edgecolor="0.3",
                show_cp=False, equal=True, labels=None):
    """Draw lifting-surface panels in three dimensions.

    Parameters
    ----------
    surfaces : System, sequence of ndarray, or ndarray
        A solved :class:`~flightlab.vlm.System`, a list of grids of shape
        ``(3, nc+1, ns+1)``, or one such grid.
    ax : Axes3D, optional
    color : str or sequence, optional
        One colour, or one per surface.
    alpha : float
        Panel face transparency.
    edgecolor : str
        Panel edge colour.  Set to ``"none"`` to hide the mesh.
    show_cp : bool
        Mark the control points.  Useful once, to see where the 3/4-chord
        points actually landed under your spacing scheme.
    equal : bool
        Force an equal aspect ratio.  A wing plotted without this looks like a
        different wing.
    labels : sequence of str, optional
        Legend labels, one per surface.

    Returns
    -------
    Axes3D
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    grids = _grids_from(surfaces)
    ax = _ax(ax, figsize=(8.0, 6.0), projection="3d")

    if color is None:
        color = [f"C{i}" for i in range(len(grids))]
    elif isinstance(color, str):
        color = [color] * len(grids)

    for k, grid in enumerate(grids):
        P = np.moveaxis(grid, 0, -1)  # (nc+1, ns+1, 3)
        quads = []
        for i in range(P.shape[0] - 1):
            for j in range(P.shape[1] - 1):
                quads.append([P[i, j], P[i, j + 1], P[i + 1, j + 1], P[i + 1, j]])
        coll = Poly3DCollection(
            quads, alpha=alpha, facecolor=color[k], edgecolor=edgecolor,
            linewidths=0.3,
        )
        ax.add_collection3d(coll)
        if labels is not None:
            ax.plot([], [], [], color=color[k], label=labels[k])

    if show_cp and hasattr(surfaces, "surfaces"):
        for surf in surfaces.surfaces:
            p = surf.flat("rcp")
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=4, c="k", depthshade=False)

    allpts = np.hstack([g.reshape(3, -1) for g in grids])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if equal:
        c = allpts.mean(axis=1)
        r = 0.5 * np.max(allpts.max(axis=1) - allpts.min(axis=1))
        r = max(r, 1e-9)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    if labels is not None:
        ax.legend(loc="upper right", fontsize=8)
    return ax


def planform(surfaces, ax=None, color=None, labels=None, show_c4=True):
    """Draw the planform (top view) of one or more lifting surfaces.

    Parameters
    ----------
    surfaces : System, sequence of ndarray, or ndarray
    ax : Axes, optional
    color : str or sequence, optional
    labels : sequence of str, optional
    show_c4 : bool
        Draw the quarter-chord line, which is where the bound vortices sit.

    Returns
    -------
    Axes
    """
    grids = _grids_from(surfaces)
    ax = _ax(ax, figsize=(7.0, 4.0))
    if color is None:
        color = [f"C{i}" for i in range(len(grids))]
    elif isinstance(color, str):
        color = [color] * len(grids)

    for k, grid in enumerate(grids):
        le, te = grid[:, 0, :], grid[:, -1, :]
        y = np.concatenate([le[1], te[1][::-1], le[1][:1]])
        x = np.concatenate([le[0], te[0][::-1], le[0][:1]])
        lbl = None if labels is None else labels[k]
        ax.fill(y, x, color=color[k], alpha=0.25, edgecolor=color[k], label=lbl)
        for i in range(grid.shape[1]):
            ax.plot(grid[1, i, :], grid[0, i, :], color=color[k], lw=0.3, alpha=0.5)
        if show_c4:
            c4 = le + 0.25 * (te - le)
            ax.plot(c4[1], c4[0], color=color[k], lw=1.2, ls="--")

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.invert_yaxis()  # nose up, the way a planform is normally drawn
    ax.set_aspect("equal")
    if labels is not None:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def airfoil(sections, ax=None, labels=None, camber=False):
    """Plot one or more airfoil sections to scale.

    Parameters
    ----------
    sections : Section or sequence of Section
        From :func:`flightlab.foil.load`.
    ax : Axes, optional
    labels : sequence of str, optional
        Defaults to each section's name.
    camber : bool
        Overlay the mean camber line, which is all an inviscid vortex lattice
        sees of the section.

    Returns
    -------
    Axes
    """
    if not isinstance(sections, (list, tuple)):
        sections = [sections]
    ax = _ax(ax, figsize=(7.0, 2.6))
    for k, s in enumerate(sections):
        lbl = s.name if labels is None else labels[k]
        ax.plot(s.x, s.z, lw=1.2, color=f"C{k}", label=lbl)
        if camber:
            xc, zc = s.camber_line()
            ax.plot(xc, zc, lw=0.8, ls="--", color=f"C{k}", alpha=0.7)
    ax.set_xlabel("x/c")
    ax.set_ylabel("z/c")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# --- span distributions -----------------------------------------------------


def span_loading(y, loading, ax=None, label="computed", elliptical=True,
                 normalize=True, color="C0"):
    """Plot lift per unit span with an elliptical reference.

    Parameters
    ----------
    y : array_like
        Spanwise station locations, m.
    loading : array_like
        Lift per unit span, or the normalized ``c*c_l/cbar``.  Any quantity
        proportional to local lift works; the elliptical overlay is scaled to
        carry the same total.
    ax : Axes, optional
    label : str
    elliptical : bool
        Draw the elliptical distribution carrying the same total lift.  This is
        the comparison that makes the plot mean something.
    normalize : bool
        Divide by the maximum of the elliptical reference, so the ideal case
        peaks at 1.
    color : str

    Returns
    -------
    Axes

    Notes
    -----
    This is the distribution that sets **induced drag**.  It is not the one
    that tells you where the wing stalls -- see :func:`cl_distribution`.
    """
    y = np.asarray(y, dtype=float)
    loading = np.asarray(loading, dtype=float)
    ax = _ax(ax)

    semi = np.max(np.abs(y))
    ref = None
    if elliptical:
        eta = np.clip(y / semi, -1.0, 1.0)
        ell = np.sqrt(np.maximum(0.0, 1.0 - eta**2))
        # scale the ellipse to carry the same integrated lift
        num = np.trapezoid(loading, y) if hasattr(np, "trapezoid") else np.trapz(loading, y)
        den = np.trapezoid(ell, y) if hasattr(np, "trapezoid") else np.trapz(ell, y)
        ref = ell * (num / den if den != 0 else 1.0)

    scale = np.max(ref) if (normalize and ref is not None and np.max(ref) > 0) else 1.0
    ax.plot(y, loading / scale, color=color, lw=1.6, label=label)
    if ref is not None:
        ax.plot(y, ref / scale, "k--", lw=1.0, label="elliptical, same total lift")

    ax.set_xlabel("y (m)")
    ax.set_ylabel("normalized span loading" if normalize else "lift per unit span")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def cl_distribution(y, cl, cl_max=None, ax=None, label="local $c_l$", color="C1"):
    """Plot the local lift coefficient distribution and the stall margin.

    Parameters
    ----------
    y : array_like
        Spanwise stations, m.
    cl : array_like
        Local section lift coefficient at each station.
    cl_max : float or array_like, optional
        Section ``cl_max``.  A scalar draws one line; an array (one value per
        station, evaluated at each station's own local Reynolds number) draws
        the curve, which is the honest version.
    ax : Axes, optional
    label : str
    color : str

    Returns
    -------
    Axes

    Notes
    -----
    This is the distribution that sets **where the wing stalls first**.  The
    station where ``cl`` comes closest to ``cl_max`` goes first, and if that is
    near the tip you lose the ailerons at the moment you most need them.
    """
    y = np.asarray(y, dtype=float)
    cl = np.asarray(cl, dtype=float)
    ax = _ax(ax)
    ax.plot(y, cl, color=color, lw=1.6, label=label)

    if cl_max is not None:
        clm = np.broadcast_to(np.asarray(cl_max, dtype=float), cl.shape)
        ax.plot(y, clm, "k--", lw=1.0, label=r"section $c_{l,max}$")
        ax.fill_between(y, cl, clm, where=cl >= clm, color="crimson", alpha=0.3,
                        label="stalled")
        i = int(np.argmax(cl - clm))
        ax.plot(y[i], cl[i], "o", color="crimson", ms=6, zorder=5)
        ax.annotate(
            f"critical station\ny = {y[i]:.3g} m\nmargin = {clm[i] - cl[i]:+.3f}",
            xy=(y[i], cl[i]), xytext=(8, -28), textcoords="offset points",
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7),
        )

    ax.set_xlabel("y (m)")
    ax.set_ylabel("$c_l$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# --- polars -----------------------------------------------------------------

def drag_polar(CD, CL, ax=None, label=None, color=None, clip=True,
               show_ld_max=True, ld_reference=None):
    """Plot a drag polar with axis limits that keep the low-drag region visible.

    Parameters
    ----------
    CD, CL : array_like
        Drag and lift coefficients.  Several polars may be passed as a list of
        arrays for each, with matching ``label``.
    ax : Axes, optional
    label : str or sequence of str, optional
    color : str or sequence, optional
    clip : bool
        Limit the x-axis to just past the drag at the highest useful lift,
        instead of autoscaling to include post-stall drag.
    show_ld_max : bool
        Mark the maximum lift-to-drag point and draw the tangent from the
        origin through it, which is what makes it visibly the maximum.
    ld_reference : float, optional
        A published ``L/D`` to overlay as a reference line.

    Returns
    -------
    Axes
    """
    ax = _ax(ax, figsize=(6.0, 5.0))
    CDs = CD if isinstance(CD, (list, tuple)) else [CD]
    CLs = CL if isinstance(CL, (list, tuple)) else [CL]
    labels = label if isinstance(label, (list, tuple)) else [label] * len(CDs)
    colors = color if isinstance(color, (list, tuple)) else [color] * len(CDs)

    xmax = 0.0
    for k, (cd, cl) in enumerate(zip(CDs, CLs)):
        cd = np.asarray(cd, dtype=float)
        cl = np.asarray(cl, dtype=float)
        c = colors[k] if colors[k] is not None else f"C{k}"
        ax.plot(cd, cl, color=c, lw=1.5, label=labels[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(cd > 0, cl / cd, -np.inf)
        i = int(np.nanargmax(ld))
        if show_ld_max:
            ax.plot([0.0, cd[i]], [0.0, cl[i]], color=c, lw=0.7, ls=":", alpha=0.8)
            ax.plot(cd[i], cl[i], "o", color=c, ms=6)
            ax.annotate(
                f"$(L/D)_{{max}}$ = {ld[i]:.1f}",
                xy=(cd[i], cl[i]), xytext=(10, -4), textcoords="offset points",
                fontsize=8, color=c,
            )
        xmax = max(xmax, float(cd[i]) * 2.5)

    if ld_reference is not None:
        ylim = ax.get_ylim()
        cl_line = np.linspace(0.0, max(ylim[1], 0.1), 50)
        ax.plot(cl_line / ld_reference, cl_line, "k-.", lw=0.9,
                label=f"published L/D = {ld_reference:g}")

    if clip and xmax > 0:
        ax.set_xlim(0.0, xmax)
    ax.set_xlabel("$C_D$")
    ax.set_ylabel("$C_L$")
    ax.grid(alpha=0.3)
    if any(l is not None for l in labels) or ld_reference is not None:
        ax.legend(fontsize=8)
    return ax


def airfoil_polars(results, labels=None, axes=None, alpha_key="alpha",
                   clip_cd=True):
    """Lift curve and drag polar side by side, for several sections at once.

    Parameters
    ----------
    results : dict or sequence of dict
        Output of :func:`flightlab.foil.polar` (which includes ``alpha``).
    labels : sequence of str, optional
    axes : sequence of two Axes, optional
    alpha_key : str
    clip_cd : bool
        Hold the drag axis to the low-drag region.

    Returns
    -------
    tuple of Axes
    """
    import matplotlib.pyplot as plt

    if isinstance(results, dict):
        results = [results]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    a1, a2 = axes

    cdmax = 0.0
    for k, r in enumerate(results):
        lbl = None if labels is None else labels[k]
        a1.plot(r[alpha_key], r["cl"], color=f"C{k}", lw=1.4, label=lbl)
        a2.plot(r["cd"], r["cl"], color=f"C{k}", lw=1.4, label=lbl)
        cdmax = max(cdmax, float(np.nanpercentile(r["cd"], 60)) * 3.0)

    a1.set_xlabel(r"$\alpha$ (deg)")
    a1.set_ylabel("$c_l$")
    a2.set_xlabel("$c_d$")
    a2.set_ylabel("$c_l$")
    if clip_cd and cdmax > 0:
        a2.set_xlim(0.0, cdmax)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        if labels is not None:
            a.legend(fontsize=8)
    return a1, a2


# --- sweeps -----------------------------------------------------------------


def contour(X, Y, Z, ax=None, levels=14, xlabel="", ylabel="", zlabel="",
            constraints=(), optimum=None, cmap="viridis", clabel=True):
    """Filled contour of a two-parameter sweep, with constraint shading.

    Parameters
    ----------
    X, Y : ndarray
        2-D coordinate arrays, as from :func:`numpy.meshgrid`.
    Z : ndarray
        The swept quantity.
    ax : Axes, optional
    levels : int or array_like
    xlabel, ylabel, zlabel : str
    constraints : sequence of dict
        Each entry shades the region a constraint rules out.  Keys:
        ``mask`` (bool array, True where **infeasible**), ``label``,
        ``color`` (default red), ``hatch`` (default ``"//"``), ``alpha``.
        Shading what is *not* allowed, rather than what is, keeps the feasible
        region legible when several constraints overlap.
    optimum : tuple, optional
        ``(x, y)`` to mark, or ``(x, y, text)``.
    cmap : str
    clabel : bool
        Label the contour lines.

    Returns
    -------
    Axes

    Examples
    --------
    ::

        plot.contour(TAPER, TWIST, e_inv,
                     xlabel="taper ratio", ylabel="twist (deg)",
                     zlabel="$e_{inv}$",
                     constraints=[dict(mask=stall_margin < 0.1,
                                       label="stall margin < 0.1")])
    """
    ax = _ax(ax, figsize=(7.5, 5.5))
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.9)
    lines = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.4,
                       alpha=0.5)
    if clabel:
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
    cb = ax.figure.colorbar(cs, ax=ax)
    if zlabel:
        cb.set_label(zlabel)

    default_colors = ["crimson", "darkorange", "purple", "teal"]
    for i, con in enumerate(constraints):
        mask = np.asarray(con["mask"], dtype=bool)
        color = con.get("color", default_colors[i % len(default_colors)])
        hatch = con.get("hatch", ["//", "\\\\", "xx", ".."][i % 4])
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=[hatch])
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[color],
                    alpha=con.get("alpha", 0.22))
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=[color],
                   linewidths=1.4)
        ax.plot([], [], color=color, lw=1.4, label=con.get("label", f"constraint {i+1}"))

    if optimum is not None:
        ax.plot(optimum[0], optimum[1], "*", color="white", ms=16,
                markeredgecolor="k", zorder=6)
        if len(optimum) > 2:
            ax.annotate(optimum[2], xy=optimum[:2], xytext=(10, 10),
                        textcoords="offset points", fontsize=8,
                        color="k",
                        bbox=dict(fc="white", alpha=0.7, lw=0))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if constraints:
        ax.legend(fontsize=8, loc="best")
    return ax


def convergence(counts, values, ax=None, labels=None, reference=None,
                xlabel="panels", ylabel="value", tol=None):
    """Plot a resolution study.

    Parameters
    ----------
    counts : array_like or sequence of array_like
        Panel counts.
    values : array_like or sequence of array_like
        The converging quantity.
    ax : Axes, optional
    labels : sequence of str, optional
        One per curve, e.g. ``["uniform", "cosine"]``.
    reference : float, optional
        The known answer, drawn as a horizontal line.  Include one whenever you
        have it: "the answer stopped changing" is weaker evidence than "the
        answer stopped changing *and* the same settings reproduce a known case".
    xlabel, ylabel : str
    tol : float, optional
        Shade a band of this half-width about ``reference``.

    Returns
    -------
    Axes
    """
    ax = _ax(ax)
    cs = counts if isinstance(counts, (list, tuple)) else [counts]
    vs = values if isinstance(values, (list, tuple)) else [values]
    if len(cs) == 1 and len(vs) > 1:
        cs = cs * len(vs)

    for k, (c, v) in enumerate(zip(cs, vs)):
        lbl = None if labels is None else labels[k]
        ax.plot(c, v, "o-", ms=4, lw=1.3, color=f"C{k}", label=lbl)

    if reference is not None:
        ax.axhline(reference, color="k", ls="--", lw=1.0,
                   label=f"known answer = {reference:g}")
        if tol is not None:
            ax.axhspan(reference - tol, reference + tol, color="k", alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    if labels is not None or reference is not None:
        ax.legend(fontsize=8)
    return ax


def breakdown(names, values, ax=None, unit="m$^2$", percent=True,
              title="", color="C0"):
    """Horizontal bar chart of a component breakdown.

    Built for component drag-area buildup, where working in drag area rather than
    in coefficients is the point: drag areas add, and coefficients normalized to
    different reference areas do not.

    Parameters
    ----------
    names : sequence of str
    values : array_like
        One value per name, in consistent units.
    ax : Axes, optional
    unit : str
    percent : bool
        Annotate each bar with its share of the total.
    title : str
    color : str

    Returns
    -------
    Axes
    """
    names = list(names)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ax = _ax(ax, figsize=(7.0, 0.45 * len(names) + 1.5))
    ypos = np.arange(len(names))
    ax.barh(ypos, values[order], color=color, alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order])
    total = values.sum()
    if percent and total > 0:
        for k, i in enumerate(order):
            ax.annotate(f"  {values[i]:.4g} ({100*values[i]/total:.1f}%)",
                        xy=(values[i], k), va="center", fontsize=8)
        ax.set_xlim(0, values.max() * 1.45)
    ax.set_xlabel(f"drag area ({unit})" if unit else "")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    return ax


# --- flight envelope and dynamics ------------------------------------------


def vn_diagram(V, n_pos, n_neg, ax=None, V_dive=None, gust_lines=(),
               V_stall_pos=None, labels=True):
    """Plot a V-n manoeuvre envelope.

    Parameters
    ----------
    V : array_like
        Speeds for the manoeuvre boundary, m/s.
    n_pos, n_neg : array_like
        Positive and negative load-factor boundaries at each speed, already
        limited by both ``CL_max`` and the structural limit.
    ax : Axes, optional
    V_dive : float, optional
        Design dive speed, where the envelope is cut off vertically.
    gust_lines : sequence of dict
        Each with keys ``V``, ``n``, and optionally ``label`` and ``style``.
    V_stall_pos : float, optional
        1-g stall speed, marked for reference.
    labels : bool

    Returns
    -------
    Axes
    """
    V = np.asarray(V, dtype=float)
    n_pos = np.asarray(n_pos, dtype=float)
    n_neg = np.asarray(n_neg, dtype=float)
    ax = _ax(ax, figsize=(7.0, 5.0))

    ax.plot(V, n_pos, color="C0", lw=1.6, label="positive limit" if labels else None)
    ax.plot(V, n_neg, color="C3", lw=1.6, label="negative limit" if labels else None)
    ax.fill_between(V, n_neg, n_pos, color="C0", alpha=0.12)

    if V_dive is not None:
        ax.axvline(V_dive, color="k", ls="--", lw=1.0,
                   label="$V_{dive}$" if labels else None)

    for g in gust_lines:
        ax.plot(g["V"], g["n"], g.get("style", "g-."), lw=1.0,
                label=g.get("label", "gust") if labels else None)

    if V_stall_pos is not None:
        ax.plot([V_stall_pos], [1.0], "ko", ms=5)
        ax.annotate("1 g stall", xy=(V_stall_pos, 1.0), xytext=(6, 8),
                    textcoords="offset points", fontsize=8)

    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("V (m/s)")
    ax.set_ylabel("load factor $n$")
    ax.grid(alpha=0.3)
    if labels:
        ax.legend(fontsize=8)
    return ax


def eigenvalues(eigs, ax=None, labels=None, mode_names=None, annotate=True,
                color=None):
    """Plot eigenvalues on the complex plane.

    Parameters
    ----------
    eigs : array_like or sequence of array_like
        Complex eigenvalues.  Pass several sets to compare aircraft or
        conditions.
    ax : Axes, optional
    labels : sequence of str, optional
        One per set.
    mode_names : sequence of str, optional
        One per eigenvalue in the first set, annotated on the plot.
    annotate : bool
        Label each point with its period and damping ratio.
    color : str or sequence, optional

    Returns
    -------
    Axes

    Notes
    -----
    Sanity bounds worth checking before you believe the plot: a phugoid period
    of order tens of seconds, a short period of order one second, and a factor
    of roughly 10 to 30 between them for a conventional airplane.  An inertia
    wrong by a factor of a thousand produces eigenvalues that look plausible.
    """
    sets = eigs if isinstance(eigs, (list, tuple)) else [eigs]
    ax = _ax(ax, figsize=(6.5, 5.5))
    colors = color if isinstance(color, (list, tuple)) else [color] * len(sets)

    for k, e in enumerate(sets):
        e = np.atleast_1d(np.asarray(e, dtype=complex))
        c = colors[k] if colors[k] is not None else f"C{k}"
        lbl = None if labels is None else labels[k]
        ax.plot(e.real, e.imag, "x", ms=9, mew=1.8, color=c, label=lbl)
        if annotate and k == 0:
            for i, val in enumerate(e):
                if val.imag < -1e-12:
                    continue  # annotate one of each conjugate pair
                txt = []
                if mode_names is not None and i < len(mode_names):
                    txt.append(mode_names[i])
                mag = abs(val)
                if mag > 1e-12:
                    txt.append(f"$\\zeta$={-val.real/mag:.3f}")
                if abs(val.imag) > 1e-9:
                    txt.append(f"T={2*np.pi/abs(val.imag):.2f} s")
                elif val.real != 0:
                    txt.append(f"$\\tau$={-1/val.real:.2f} s")
                ax.annotate("\n".join(txt), xy=(val.real, val.imag),
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=7.5)

    ax.axvline(0.0, color="k", lw=0.8)
    ax.axhline(0.0, color="0.6", lw=0.6)
    xl = ax.get_xlim()
    if xl[1] > 0:
        ax.axvspan(0.0, xl[1], color="crimson", alpha=0.07)
        ax.annotate("unstable", xy=(xl[1], ax.get_ylim()[1]),
                    xytext=(-52, -14), textcoords="offset points",
                    fontsize=8, color="crimson")
    ax.set_xlabel(r"Re($\lambda$)  (1/s)")
    ax.set_ylabel(r"Im($\lambda$)  (rad/s)")
    ax.grid(alpha=0.3)
    if labels is not None:
        ax.legend(fontsize=8)
    return ax


# --- propulsion -------------------------------------------------------------


def prop_curves(propeller, axes=None, show_runs=True, color_by_rpm=True):
    """Plot ``CT``, ``CP`` and efficiency against advance ratio.

    Parameters
    ----------
    propeller : Propeller
        From :func:`flightlab.props.load`.
    axes : sequence of three Axes, optional
    show_runs : bool
        Draw each RPM run separately rather than pooling them.  Worth doing
        once: the spread between runs at the same ``J`` is a Reynolds number
        effect, and it tells you whether pooling the runs is defensible.
    color_by_rpm : bool

    Returns
    -------
    tuple of Axes
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    aT, aP, aE = axes

    runs = propeller.runs
    if show_runs:
        cmap = plt.get_cmap("viridis")
        rpms = propeller.rpm_values
        lo, hi = float(rpms.min()), float(rpms.max())
        for r in runs:
            c = cmap((r.rpm - lo) / (hi - lo) if hi > lo else 0.5) if color_by_rpm else "C0"
            aT.plot(r.J, r.CT, "-", lw=1.1, color=c, label=f"{r.rpm:.0f} rpm")
            aP.plot(r.J, r.CP, "-", lw=1.1, color=c)
            aE.plot(r.J, r.eta, "-", lw=1.1, color=c)
    else:
        _, J, CT, CP, eta = propeller.all_points()
        o = np.argsort(J)
        aT.plot(J[o], CT[o], ".", ms=4)
        aP.plot(J[o], CP[o], ".", ms=4)
        aE.plot(J[o], eta[o], ".", ms=4)

    if propeller.static is not None:
        aT.plot([0.0] * len(propeller.static), propeller.static.CT, "k.", ms=4,
                label="static (J=0)")
        aP.plot([0.0] * len(propeller.static), propeller.static.CP, "k.", ms=4)

    lo, hi = propeller.J_range
    for a in (aT, aP, aE):
        a.set_xlabel("$J = V/(nD)$")
        a.grid(alpha=0.3)
        a.axvspan(a.get_xlim()[0], lo, color="crimson", alpha=0.08)
    aT.set_ylabel("$C_T$")
    aP.set_ylabel("$C_P$")
    aE.set_ylabel(r"$\eta$")
    aE.set_ylim(0.0, 1.0)
    aT.set_title(
        f"{propeller.manufacturer} {propeller.diameter_in:g}x"
        f"{propeller.pitch_in:g}  (measured J: {lo:.2f}-{hi:.2f})",
        fontsize=9,
    )
    aT.legend(fontsize=7, ncol=2)
    return aT, aP, aE


# --- fleet ------------------------------------------------------------------


def wing_loading_map(wing_loading, cruise_speed, names=None, ax=None,
                     cl_lines=(0.1, 0.3, 0.5, 1.0), density=1.225,
                     segments=()):
    """Plot aircraft on log-log wing loading versus cruise speed axes.

    In steady level flight ``W/S`` and cruise speed
    are not independent -- they are tied together by the cruise lift
    coefficient, and lines of constant ``C_L`` are straight on these axes.

    Parameters
    ----------
    wing_loading : array_like
        ``W/S`` in N/m^2.
    cruise_speed : array_like
        Cruise speed in m/s.
    names : sequence of str, optional
        Annotated beside each point.
    ax : Axes, optional
    cl_lines : sequence of float
        Constant-``C_L`` reference lines to overlay.
    density : float or array_like
        Density used for the reference lines, kg/m^3.  Pass an array to use
        each aircraft's own cruise density for its own point.
    segments : sequence of dict
        Lines joining two states of the same aircraft, e.g. a sailplane filling
        its water ballast.  Keys: ``W_S`` (pair), ``V`` (pair), ``label``.

    Returns
    -------
    Axes
    """
    ws = np.atleast_1d(np.asarray(wing_loading, dtype=float))
    v = np.atleast_1d(np.asarray(cruise_speed, dtype=float))
    ax = _ax(ax, figsize=(7.5, 5.5))

    vline = np.logspace(np.log10(max(v.min() * 0.5, 1e-3)),
                        np.log10(v.max() * 2.0), 60)
    rho_ref = float(np.mean(np.atleast_1d(density)))
    for cl in cl_lines:
        ax.plot(vline, 0.5 * rho_ref * vline**2 * cl, "k:", lw=0.8, alpha=0.6)
        ax.annotate(f"$C_L$={cl:g}", xy=(vline[-1], 0.5 * rho_ref * vline[-1]**2 * cl),
                    xytext=(-38, 3), textcoords="offset points", fontsize=7,
                    color="0.35")

    ax.plot(v, ws, "o", ms=8, color="C0", zorder=4)
    if names is not None:
        for i, nm in enumerate(names):
            ax.annotate(nm, xy=(v[i], ws[i]), xytext=(8, -3),
                        textcoords="offset points", fontsize=8.5)

    for seg in segments:
        ax.plot(seg["V"], seg["W_S"], "-", lw=2.0, color="C3", alpha=0.8,
                zorder=3, label=seg.get("label"))
        ax.plot(seg["V"], seg["W_S"], "o", ms=5, color="C3", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cruise speed (m/s)")
    ax.set_ylabel("wing loading $W/S$ (N/m$^2$)")
    ax.grid(alpha=0.3, which="both")
    if segments:
        ax.legend(fontsize=8)
    return ax


# =============================================================================
# Draft-4 additions: results the new analysis modules produce.
#
# The functions above take plain arrays.  These take result objects, because
# the objects already carry the axis labels, the units and the reference values
# a plot of them needs, and passing eight arrays by hand is how a figure ends
# up mislabelled.  Every one still accepts an ``ax`` and returns it.
# =============================================================================


def loading_3d(solution, ax=None, quantity="cl", cmap="viridis", show_bar=True,
               equal=True, edgecolor="0.4", linewidth=0.2):
    """Draw the wing in three dimensions with a span quantity painted on it.

    Parameters
    ----------
    solution : flightlab.wing.Solution
    quantity : {"cl", "ccl", "Re", "margin"}
        Which per-station quantity to colour by.  ``"ccl"`` is the span loading
        that sets induced drag; ``"cl"`` is the local section demand that sets
        where it stalls.  They peak in different places, and being able to see
        both on the same planform is most of the point of drawing it.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    ax = _ax(ax, projection="3d")
    values = {
        "cl": solution.cl,
        "ccl": solution.ccl,
        "Re": solution.Re,
        "margin": solution.cl,
    }[quantity]

    grids = _grids_from(solution._system) if solution._system is not None else []
    norm = colors.Normalize(float(np.min(values)), float(np.max(values)))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    start = 0
    for grid in grids:
        _, nc1, ns1 = grid.shape
        ns = ns1 - 1
        vals = values[start:start + ns]
        start += ns
        for j in range(ns):
            colour = mapper.to_rgba(vals[j] if j < len(vals) else vals[-1])
            for i in range(nc1 - 1):
                quad = np.array([
                    grid[:, i, j], grid[:, i + 1, j],
                    grid[:, i + 1, j + 1], grid[:, i, j + 1],
                ])
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                poly = Poly3DCollection([quad], facecolors=[colour],
                                        edgecolors=edgecolor, linewidths=linewidth)
                ax.add_collection3d(poly)

    if grids:
        pts = np.concatenate([g.reshape(3, -1) for g in grids], axis=1)
        _equalize_3d(ax, pts) if equal else None
    if show_bar:
        label = {"cl": "local $c_l$", "ccl": "$c\\,c_l$ (m)",
                 "Re": "local $Re$", "margin": "local $c_l$"}[quantity]
        plt.colorbar(mapper, ax=ax, shrink=0.65, label=label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return ax


def _equalize_3d(ax, pts):
    """Give a 3D axis equal aspect, which matplotlib will not do itself."""
    centre = pts.mean(axis=1)
    span = float(np.max(pts.max(axis=1) - pts.min(axis=1))) / 2.0
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - span, c + span)


def span_load(result, ax=None, show=("lift", "shear", "moment"), color="C0"):
    """Running lift, shear and bending moment against span station.

    Parameters
    ----------
    result : flightlab.loads.SpanLoad
    show : sequence of str
        Which curves to draw.  Drawn on twinned axes when more than one is
        asked for, because they differ by orders of magnitude.
    """
    ax = _ax(ax)
    y = result.y
    axes = [ax]
    curves = {
        "lift": (result.lift, "running lift (N/m)"),
        "shear": (result.shear, "shear (N)"),
        "moment": (result.moment, "bending moment (N m)"),
    }
    for i, key in enumerate(show):
        values, label = curves[key]
        target = ax if i == 0 else ax.twinx()
        if i > 1:
            target.spines["right"].set_position(("outward", 45 * (i - 1)))
        target.plot(y, values, color=f"C{i}", label=label)
        target.set_ylabel(label, color=f"C{i}")
        target.tick_params(axis="y", labelcolor=f"C{i}")
        axes.append(target)
    ax.set_xlabel("span station y (m)")
    ax.set_title(f"n = {result.n:.2f}, root moment {result.root_moment:.0f} N m")
    ax.grid(alpha=0.3)
    return ax


def stall_margin(solution, table, ax=None, color="C0"):
    """Local ``c_l`` against each station's own section ``c_l max``.

    The plot that says *where* a wing stalls, which is the actionable half of a
    stall calculation.  The two curves are not parallel: the section limit
    falls toward the tip because the local Reynolds number does, so a tapered
    wing is squeezed from both sides out there.
    """
    ax = _ax(ax)
    right = solution.y >= 0
    y = solution.y[right]
    order = np.argsort(y)
    y = y[order]
    cl = solution.cl[right][order]
    cl_max = np.asarray(table.cl_max(solution.Re[right][order]))

    ax.plot(y, cl, color=color, label="local $c_l$")
    ax.plot(y, cl_max, color="firebrick", ls="--",
            label="section $c_{l,max}$ at local $Re$")
    ax.fill_between(y, cl, cl_max, where=cl < cl_max, color="0.9")
    i = int(np.argmin(cl_max - cl))
    ax.plot([y[i]], [cl[i]], "o", color="firebrick", zorder=5)
    ax.annotate(
        f"first to stall\n$\\eta$ = {y[i] / y[-1]:.2f}",
        (y[i], cl[i]), textcoords="offset points", xytext=(8, -22),
        fontsize=8, color="firebrick",
    )
    ax.set_xlabel("span station y (m)")
    ax.set_ylabel("section lift coefficient")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def thrust_and_drag(V, thrust, drag_force, ax=None, labels=("thrust available",
                    "drag required"), mark_intersections=True):
    """Thrust available against drag required, with the level-flight speeds.

    Where the curves cross is where the aircraft can hold level flight.  The
    gap between them is excess thrust, and excess thrust times speed over
    weight is rate of climb -- so this one figure contains the top speed, the
    minimum level speed, and the best climb speed.
    """
    ax = _ax(ax)
    V = np.asarray(V, dtype=float)
    T = np.asarray(thrust, dtype=float)
    D = np.asarray(drag_force, dtype=float)
    ax.plot(V, T, color="C0", label=labels[0])
    ax.plot(V, D, color="C3", label=labels[1])
    ax.fill_between(V, D, T, where=T > D, color="C0", alpha=0.12,
                    label="excess thrust")

    if mark_intersections:
        sign = np.sign(T - D)
        for i in np.where(np.diff(sign) != 0)[0]:
            frac = (D[i] - T[i]) / ((T[i + 1] - T[i]) - (D[i + 1] - D[i]))
            v = V[i] + frac * (V[i + 1] - V[i])
            ax.axvline(v, color="0.5", ls=":", lw=0.9)
            ax.annotate(f"{v:.1f} m/s", (v, ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(3, -12), fontsize=8)
    ax.set_xlabel("true airspeed (m/s)")
    ax.set_ylabel("force (N)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def power_curves(V, power_required, power_available=None, ax=None,
                 mark_minimum=True):
    """Power required against speed, with the minimum-power point marked.

    The minimum of this curve is the best-endurance speed, and it sits *below*
    the best-glide speed -- which is the thing students most reliably get
    backwards, so the figure marks both.
    """
    ax = _ax(ax)
    V = np.asarray(V, dtype=float)
    P = np.asarray(power_required, dtype=float)
    ax.plot(V, P, color="C3", label="power required")
    if power_available is not None:
        ax.plot(V, np.asarray(power_available, dtype=float), color="C0",
                label="power available")
    if mark_minimum:
        i = int(np.argmin(P))
        ax.plot([V[i]], [P[i]], "o", color="C3")
        ax.annotate(f"min power\n{V[i]:.1f} m/s", (V[i], P[i]),
                    textcoords="offset points", xytext=(6, 10), fontsize=8)
        j = int(np.argmin(P / V))
        ax.plot([V[j]], [P[j]], "s", color="C2")
        ax.annotate(f"best glide\n{V[j]:.1f} m/s", (V[j], P[j]),
                    textcoords="offset points", xytext=(6, -24), fontsize=8,
                    color="C2")
    ax.set_xlabel("true airspeed (m/s)")
    ax.set_ylabel("power (W)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def mode_response(modes, x0, t, ax=None, states=None, normalize=False):
    """Free response of a linear mode set to an initial perturbation.

    Parameters
    ----------
    modes : flightlab.stability.Modes
    x0 : array_like
        Initial state perturbation.
    t : array_like
        Times, s.
    normalize : bool
        Scale each state to its own maximum, so a 1 m/s velocity and a
        0.02 rad attitude are both visible on one axis.  Off by default,
        because the relative sizes are usually the interesting part.
    """
    ax = _ax(ax)
    t = np.asarray(t, dtype=float)
    x = modes.simulate(x0, t)
    names = states or modes.states
    for i, name in enumerate(names):
        y = x[:, i]
        if normalize and np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        ax.plot(t, y, label=name)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state perturbation" + (" (normalized)" if normalize else ""))
    ax.legend(fontsize=8, ncol=len(names))
    ax.grid(alpha=0.3)
    return ax


def envelope(result, ax=None, fill=True):
    """The speed-altitude flight envelope.

    Parameters
    ----------
    result : dict
        From :func:`flightlab.performance.envelope`.
    """
    ax = _ax(ax)
    h = np.asarray(result["altitude"], dtype=float)
    lo = np.asarray(result["V_min"], dtype=float)
    hi = np.asarray(result["V_max"], dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi)

    if fill:
        ax.fill_betweenx(h[ok], lo[ok], hi[ok], color="C0", alpha=0.18,
                         label="level flight possible")
    ax.plot(lo[ok], h[ok], color="C0")
    ax.plot(hi[ok], h[ok], color="C0")
    ax.plot(np.asarray(result["V_stall"])[ok], h[ok], color="firebrick",
            ls="--", label="stall")
    if ok.any():
        top = h[ok].max()
        ax.axhline(top, color="0.5", ls=":", lw=0.9)
        ax.annotate(f"ceiling {top:.0f} m", (ax.get_xlim()[0], top),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("true airspeed (m/s)")
    ax.set_ylabel("altitude (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def chain_breakdown(op, ax=None):
    """Where the watts go, from the battery to the air.

    Parameters
    ----------
    op : flightlab.propulsion.OperatingPoint

    Notes
    -----
    Drawn as a cascade rather than a pie, because the losses are sequential:
    each stage passes on what the one before it did not waste, and a 70%
    motor behind a 95% ESC is not 165% of anything.
    """
    ax = _ax(ax)
    stages = [
        ("electrical\nin", op.power_electrical, "C0"),
        ("after ESC", op.power_electrical * op.efficiency_esc, "C0"),
        ("shaft", op.power_shaft, "C2"),
        ("useful\n(T x V)", op.power_useful, "C2"),
    ]
    names = [s[0] for s in stages]
    values = [s[1] for s in stages]
    ax.bar(names, values, color=[s[2] for s in stages], alpha=0.85)
    for i in range(len(values) - 1):
        lost = values[i] - values[i + 1]
        ax.annotate(
            f"-{lost:.0f} W", ((i + i + 1) / 2, values[i + 1] + lost / 2),
            ha="center", fontsize=8, color="firebrick",
        )
    ax.set_ylabel("power (W)")
    ax.set_title(
        f"chain efficiency {100 * op.efficiency_total:.1f}%  "
        f"at {op.V:.1f} m/s, {op.rpm:.0f} rpm"
        + ("  (extrapolated)" if op.extrapolated else "")
    )
    ax.grid(alpha=0.3, axis="y")
    return ax


def fleet_overlay(entries, x_key, y_key, ax=None, xlabel="", ylabel="",
                  annotate=True, logx=False, logy=False):
    """One quantity against another for several aircraft on shared axes.

    Parameters
    ----------
    entries : dict
        Label to a mapping that contains ``x_key`` and ``y_key``.
    """
    ax = _ax(ax)
    for i, (label, data) in enumerate(entries.items()):
        x, y = data[x_key], data[y_key]
        ax.plot(np.atleast_1d(x), np.atleast_1d(y), "o", color=f"C{i}",
                markersize=8, label=label)
        if annotate:
            ax.annotate(label, (np.atleast_1d(x)[0], np.atleast_1d(y)[0]),
                        textcoords="offset points", xytext=(8, 4), fontsize=8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.grid(alpha=0.3, which="both")
    return ax


def polar_comparison(polars, ax=None, mass=None, show_ld=False):
    """Several drag polars on one pair of axes.

    Parameters
    ----------
    polars : dict
        Label to :class:`flightlab.drag.Polar`.
    show_ld : bool
        Plot ``L/D`` against speed instead of the polar itself, which needs
        ``mass``.
    """
    ax = _ax(ax)
    for i, (label, pol) in enumerate(polars.items()):
        if show_ld:
            if mass is None:
                raise ValueError("show_ld needs a mass to convert CL to speed")
            V = pol.speed_for(mass)
            order = np.argsort(V)
            ax.plot(V[order], pol.LD[order], color=f"C{i}", label=label)
            ax.plot([pol.V_LD_max(mass)], [pol.LD_max], "o", color=f"C{i}")
        else:
            ax.plot(pol.CD, pol.CL, color=f"C{i}",
                    label=f"{label}  (L/D max {pol.LD_max:.1f})")
    if show_ld:
        ax.set_xlabel("true airspeed (m/s)")
        ax.set_ylabel("L / D")
    else:
        ax.set_xlabel("$C_D$")
        ax.set_ylabel("$C_L$")
        ax.set_xlim(left=0.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
