"""``flightlab.live`` -- figures with sliders, for design exploration.

Static plots are for the memo.  This is for the half hour before the memo,
when you are turning a parameter to find out which way it moves things.

    >>> from flightlab import live, plot
    >>> from flightlab.case import Case
    >>> from flightlab.fleet import RC1
    >>> case = Case(RC1, V=12.0, altitude=1400.0)
    >>> def draw_span(case, ax):
    ...     solution = case.wing_aero()
    ...     plot.span_loading(solution.y, solution.ccl, ax=ax)
    >>> live.explore(case, draw_span, wing_span=(0.9, 1.8),
    ...              wing_taper=(0.4, 1.0))          # doctest: +SKIP

Move a slider and the figure redraws.  Only the analyses that actually depend
on the parameter you moved are recomputed -- :mod:`flightlab.cache` sees to that --
so changing a mass while looking at a span loading costs nothing.

Speed
-----
Interactivity has a budget: a redraw slower than about a fifth of a second
stops feeling like a slider and starts feeling like a form submission.  So
:func:`explore` drops the panel counts to a coarse setting by default and says
so on the figure.  **The coarse setting is for looking, not for reporting.**
Span efficiency in particular is still moving at those panel counts, so read
trends from a live figure and take numbers from a converged one.

Requires an interactive matplotlib backend.  In a notebook that means
``%matplotlib widget`` (or ``%matplotlib qt``); the inline backend renders a
static image and the sliders will not respond.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from . import cache
from .case import Case

__all__ = ["explore", "compare", "COARSE"]


#: Panel counts and table resolution used by :func:`explore` unless overridden.
#: Chosen so a redraw of a wing-plus-tail case stays under about 100 ms.
COARSE = {"ns": 18, "nc": 4, "n_Re": 6}


def _interactive_backend() -> bool:
    import matplotlib

    return matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}


def explore(
    case: Case,
    draw: Callable,
    coarse: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.5),
    slider_height: float = 0.035,
    **parameters,
):
    """Open a figure whose parameters are sliders.

    Parameters
    ----------
    case : flightlab.case.Case
        The case to vary.  It is mutated as the sliders move, so pass
        ``case.copy()`` if you want to keep the original where it is.
    draw : callable
        ``draw(case, ax)`` -- draws the current state onto ``ax``.  A two-line
        function can obtain a named case result and pass it to a helper in
        :mod:`flightlab.plot`.
    coarse : bool
        Drop the panel counts to :data:`COARSE` for responsiveness, and label
        the figure to say so.
    figsize : tuple
    slider_height : float
        Vertical fraction of the figure each slider occupies.
    **parameters
        ``group_parameter=(low, high)`` or ``group_parameter=(low, high, step)``
        for each slider, e.g. ``wing_span=(0.9, 1.8)`` or
        ``cond_V=(8.0, 20.0, 0.5)``.  Group names are the ones in
        :meth:`flightlab.case.Case.groups`.

    Returns
    -------
    dict
        ``fig``, ``ax``, and ``sliders`` -- keep a reference to the returned
        dict, because matplotlib widgets stop responding once they are garbage
        collected.

    Examples
    --------
    Span loading against span and taper::

        def draw_span(case, ax):
            solution = case.wing_aero()
            plot.span_loading(solution.y, solution.ccl, ax=ax)

        live.explore(case, draw_span,
                     wing_span=(0.9, 1.8), wing_taper=(0.4, 1.0))

    Anything can be the drawing function::

        def draw(case, ax):
            pol = case.polar()
            ax.plot(pol.CD, pol.CL)
            ax.set_title(f"L/D max = {pol.LD_max:.1f}")

        live.explore(case, draw, solver_protuberance=(0.0, 0.3), cond_V=(8, 20))
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if not parameters:
        raise ValueError(
            "give at least one slider, as group_parameter=(low, high); "
            f"groups are {list(case.groups())}"
        )
    if not _interactive_backend():
        import warnings

        warnings.warn(
            "the current matplotlib backend is not interactive, so the "
            "sliders will render but not respond.  In a notebook use "
            "'%matplotlib widget'; from a script, a GUI backend such as "
            "MacOSX, QtAgg or TkAgg.",
            RuntimeWarning,
            stacklevel=2,
        )

    if coarse:
        case.solver.replace(**COARSE)

    n = len(parameters)
    bottom = 0.10 + n * (slider_height + 0.02)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.11, bottom, 0.85, 0.96 - bottom])

    sliders: Dict[str, "Slider"] = {}
    for i, (key, spec) in enumerate(parameters.items()):
        group, _, field = key.partition("_")
        if group not in case.groups() or not field:
            raise AttributeError(
                f"cannot make a slider for {key!r}: expected "
                f"<group>_<parameter> with group in {list(case.groups())}"
            )
        lo, hi, *rest = spec
        step = rest[0] if rest else None
        current = _current(case, group, field, lo, hi)
        sax = fig.add_axes([0.11, 0.06 + i * (slider_height + 0.02), 0.85, slider_height])
        sliders[key] = Slider(
            sax, key.replace("_", " ", 1), float(lo), float(hi),
            valinit=float(current), valstep=step,
        )

    def redraw(_=None):
        for key, sl in sliders.items():
            group, _, field = key.partition("_")
            target = case.groups()[group]
            if hasattr(target, "__setitem__") and not hasattr(target, field):
                target[field] = float(sl.val)
            else:
                setattr(target, field, float(sl.val))
        ax.clear()
        try:
            draw(case, ax)
        except Exception as exc:  # a bad parameter combination is not fatal
            ax.clear()
            ax.text(
                0.5, 0.5, f"{type(exc).__name__}:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="firebrick", wrap=True,
            )
            ax.set_axis_off()
        if coarse:
            ax.text(
                0.995, 0.01,
                f"coarse: ns={COARSE['ns']} nc={COARSE['nc']} - trends only",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.45",
            )
        fig.canvas.draw_idle()

    for sl in sliders.values():
        sl.on_changed(redraw)
    redraw()
    return {"fig": fig, "ax": ax, "sliders": sliders, "case": case}


def _current(case: Case, group: str, field: str, lo: float, hi: float) -> float:
    """The case's present value for a slider, or the midpoint if unset."""
    target = case.groups()[group]
    try:
        value = target[field] if hasattr(target, "__getitem__") else getattr(target, field)
    except (AttributeError, KeyError):
        value = None
    if value is None:
        if group in ("wing", "htail", "vtail"):
            from . import geom

            base = getattr(case.base, group)
            if base is not None:
                panel = geom.resolve(base)
                value = getattr(panel, field, None)
    if value is None:
        return 0.5 * (lo + hi)
    return float(np.clip(float(value), lo, hi))


def compare(
    cases: Dict[str, Case],
    draw: Callable,
    coarse: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.5),
    **parameters,
):
    """Like :func:`explore`, but with several cases drawn on the same axes.

    The sliders move every case at once, which is how a fleet comparison stays
    a comparison: the ASW-27B and the ASG 29 at the same speed and the same
    wing loading, with span the thing that differs.

    Parameters
    ----------
    cases : dict
        Label to :class:`flightlab.case.Case`.
    draw : callable
        ``draw(case, ax, label=...)``.
    """
    import matplotlib.pyplot as plt

    first = next(iter(cases.values()))

    def draw_all(_case, ax):
        for label, c in cases.items():
            for key, sl_value in _slider_state.items():
                group, _, field = key.partition("_")
                target = c.groups()[group]
                if hasattr(target, "__setitem__") and not hasattr(target, field):
                    target[field] = sl_value
                else:
                    setattr(target, field, sl_value)
            draw(c, ax, label=label)
        ax.legend(fontsize=8)

    _slider_state: Dict[str, float] = {}
    result = explore(first, draw_all, coarse=coarse, figsize=figsize, **parameters)

    def sync(_=None):
        for key, sl in result["sliders"].items():
            _slider_state[key] = float(sl.val)

    for sl in result["sliders"].values():
        sl.on_changed(sync)
    sync()
    result["cases"] = cases
    return result
