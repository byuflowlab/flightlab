"""``flightlab.drag`` -- component buildup, strip integration, and the complete polar.

Three ways of getting the drag of an aircraft, and they do not agree.  That
disagreement is the content of the drag week, so all three are here and none of
them is labelled correct.

**The buildup.**  Wetted area times skin friction times a form factor times an
interference factor, component by component, plus a markup for everything the
geometry does not resolve.  Cheap, requires no solver, and is what a designer
uses before there is a shape to analyze.  It carries two judgment calls -- the
markup and the lift-dependent viscous coefficient -- that a student can turn
and watch the answer move.

**Strip integration.**  Take the local ``cl``, chord and Reynolds number the
vortex lattice reports at every span station, look the section drag up at each
station's own condition, and integrate.  Uses real section data and knows about
the spanwise variation the buildup averages away.  It resolves only the wing.

**The Trefftz plane.**  Induced drag, which neither of the above computes, from
:mod:`flightlab.wing`.

    >>> from flightlab import drag
    >>> from flightlab.fleet import DC3
    >>> b = drag.buildup(DC3, V=93.0, altitude=3000.0)
    >>> round(b.CD0, 5), round(b.f, 3)
    (0.00863, 0.688)

Units
-----
SI.  Drag areas ``f = CD * S_ref`` are in m^2 -- the unit a drag buildup is
actually additive in, which is why the tables below are built in it and
converted to coefficients only at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from . import airfoil as _airfoil
from . import atmos, geom, wing
from .fleet import Aircraft, Body, Planform
from .geom import Panel

__all__ = [
    "Row",
    "Buildup",
    "Polar",
    "flat_plate_cf",
    "form_factor_surface",
    "form_factor_body",
    "buildup",
    "strip_viscous_drag",
    "wave_drag",
    "drag_divergence_mach",
    "body_form_factor_valid",
    "sphere_cd",
    "body_cd_frontal",
    "crest_critical_mach",
    "oswald_efficiency",
    "span_efficiency_with_fuselage",
    "effective_diameter",
    "polar",
]

G0 = 9.80665

#: Typical **interference** markups, as a fraction of the component sum.  The
#: buildup treats each component in isolation; assembling them costs 3-8% more
#: (Raymer, quoted in the course text).  Almost never beneficial.
INTERFERENCE = {"typical": 0.05, "low": 0.03, "high": 0.08}

#: Typical **protuberance and roughness** markups, as a fraction of the
#: component sum.  Hinges, antennas, gaps, rivets, tape seams and control
#: horns.  The spread between classes is large and it is not a detail: an RC
#: aircraft's markup can exceed its entire tail drag.
PROTUBERANCE = {
    "jet_transport": 0.035,   # 2-5%
    "propeller": 0.075,       # 5-10%
    "rc_model": 0.10,         # at least 10%, and often much more
    "sailplane": 0.02,        # sealed, filled and polished
}

#: Drag coefficients on **frontal** area for bluff items lying across the flow.
#: Not in the course text, which handles such items through the protuberance
#: markup.  Provided because an aircraft with exposed gear legs and lift struts
#: carries a large, identifiable drag that a markup buries: see the note in
#: :func:`buildup`.
CROSSFLOW_CD = {
    "cylinder": 0.90,      # round tube, subcritical Reynolds number
    "streamline": 0.10,    # a proper streamline-section strut
    "faired": 0.25,        # a fairing over a round leg
}


# --- skin friction and form factors -----------------------------------------


def flat_plate_cf(Re, mach: float = 0.0, xtr: float = 0.0) -> np.ndarray:
    """Flat-plate skin-friction coefficient, with transition handled properly.

    Parameters
    ----------
    Re : float or array_like
        Reynolds number on the full length.
    mach : float
        Compressibility correction, ``cf / (1 + 0.144 M^2)^0.65``.
    xtr : float
        Transition location as a fraction of the length, 0 to 1.  ``0`` is
        fully turbulent -- the right default for anything behind a propeller or
        with a bug on the leading edge.  ``1`` is fully laminar.  In between,
        the two regions are combined by the momentum-thickness method in the
        course text's appendix rather than by blending coefficients, which is
        not the same thing and is optimistic.

    Returns
    -------
    ndarray

    Notes
    -----
    Fully laminar is Blasius, ``1.328/sqrt(Re)``; fully turbulent is
    Schlichting, ``0.074/Re^0.2``.

    A partially laminar plate is **not** the weighted average of the two.  The
    turbulent boundary layer downstream of transition does not start from
    nothing: it inherits the momentum thickness the laminar layer built up.
    The text's method finds the *effective* distance ``x_e`` upstream at which a
    turbulent layer would have reached that same momentum thickness, computes
    the turbulent friction over ``L - x_t + x_e``, and subtracts back the
    double-counted ``x_e``:

        ``Cf = Cf_1 (x_t/c) + Cf_3 (x_f/c) - Cf_2 (x_e/c)``

    Averaging the two branches instead **understates** the drag of a partly
    laminar surface, and the error grows with the laminar run -- exactly the
    case a sailplane cares about.

    The laminar value is also the floor referred to throughout the course: no
    surface's section drag coefficient can fall below the laminar flat plate at
    its own Reynolds number.  When one does, a lookup was extrapolated.
    """
    Re = np.asarray(Re, dtype=float)
    if np.any(Re <= 0):
        raise ValueError("Reynolds number must be positive")
    xtr = float(np.clip(xtr, 0.0, 1.0))

    if xtr <= 0.0:
        cf = 0.074 / Re**0.2
    elif xtr >= 1.0:
        cf = 1.328 / np.sqrt(Re)
    else:
        Re_xt = Re * xtr
        cf1 = 1.328 / np.sqrt(Re_xt)
        # laminar momentum thickness at transition, per unit length
        theta_l = 0.664 * xtr / np.sqrt(Re_xt)
        # distance a turbulent layer needs to reach that same thickness
        xe = (theta_l * Re**0.2 / 0.036) ** 1.25
        xf = 1.0 - xtr + xe
        cf3 = 0.074 / (Re * xf) ** 0.2
        cf2 = 0.074 / (Re * xe) ** 0.2
        cf = cf1 * xtr + cf3 * xf - cf2 * xe

    return cf / (1.0 + 0.144 * mach**2) ** 0.65


def form_factor_surface(
    thickness: float, mach: float = 0.0, sweep_c4_deg: float = 0.0
) -> float:
    """Form factor of a lifting surface -- Shevell's formula.

    ``k = 1 + Z (t/c) + 100 (t/c)^4``, with

    ``Z = (2 - M^2) cos(L) / sqrt(1 - M^2 cos^2(L))``

    which reduces to ``2 cos(L)`` in incompressible flow.

    Parameters
    ----------
    thickness : float
        ``t/c``.
    mach : float
        Freestream Mach number.
    sweep_c4_deg : float
        **Quarter-chord** sweep, degrees.

    Notes
    -----
    The form factor is the multiplier on flat-plate skin friction that accounts
    for the pressure drag a thick section carries.  A flat plate at zero lift
    has none; ``t/c`` of 0.15 costs about 35%.  The quartic term is negligible
    until the section is very thick, and then it is not.
    """
    tc = float(thickness)
    M = float(mach)
    cos_l = np.cos(np.radians(sweep_c4_deg))
    Z = (2.0 - M**2) * cos_l / np.sqrt(1.0 - M**2 * cos_l**2)
    return float(1.0 + Z * tc + 100.0 * tc**4)


def form_factor_body(fineness: float, strict: bool = False) -> float:
    """Form factor of a body of revolution -- the course text's fit.

    ``k = 1.675 - 0.09 fr + 0.003 fr^2`` for ``5 < fr < 15``, and ``1`` for
    ``fr >= 15``.  A quadratic fit to Shevell's figure, constrained to bottom
    out at 1 rather than dropping below it as a naive fit would.

    Parameters
    ----------
    fineness : float
        Length over maximum diameter.  For a non-circular section use the
        effective diameter :func:`effective_diameter` returns.
    strict : bool
        Raise a warning when ``fr < 5``, the text's stated lower limit.  Off by
        default, because a stubby body is not an error and the buildup reports
        it as data instead: :attr:`Row.extrapolated` marks the row and
        :meth:`Buildup.table` footnotes it.  Turn it on for a direct call where
        you want to be stopped.

    Notes
    -----
    The minimum sits near ``fr = 15``, but the *product* of form factor and
    wetted area has its own minimum much lower, because a long thin body has
    little pressure drag and a great deal of surface.  That tradeoff is why so
    many fuselages land near a fineness ratio of 6 to 8.
    """
    fr = float(fineness)
    if fr >= 15.0:
        return 1.0
    if strict and not body_form_factor_valid(fr):
        import warnings

        warnings.warn(
            f"form_factor_body called with fineness ratio {fr:.2f}.  The "
            "course text's fit is stated as valid only for fr > 5; below that "
            f"it is being extrapolated and returns "
            f"{1.675 - 0.09 * fr + 0.003 * fr**2:.3f}.  Treat a body this "
            "stubby as bluff instead (Body(drag_model='faired')).",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(1.675 - 0.09 * fr + 0.003 * fr**2)


def body_form_factor_valid(fineness: float) -> bool:
    """Whether :func:`form_factor_body` is inside its stated validity.

    The text gives the fit for ``fr > 5``.  Below that the quadratic is still
    well behaved -- it does not blow up the way some other correlations do --
    but it is extrapolated, and the aircraft this course actually builds sits
    there: RC-1's pod has a fineness ratio of 3.8, because a pod that has to
    hold a battery, an ESC, a receiver and three servos in 400 mm is stubby.
    That is worth reporting rather than hiding, and worth remembering when a
    transport-derived correlation is pointed at a model aeroplane.
    """
    return float(fineness) >= 5.0


def effective_diameter(max_cross_section: float) -> float:
    """``d_eff = sqrt(4 S_max / pi)`` for a non-circular body.

    The diameter of the circle with the same maximum cross-sectional area, so
    a rectangular pod and a round one of the same frontal size get the same
    fineness ratio.
    """
    return float(np.sqrt(4.0 * max_cross_section / np.pi))


def sphere_cd(Re) -> np.ndarray:
    """Drag coefficient of a sphere on **frontal** area, against diameter Reynolds number.

    The Clift-Gauvin correlation,

        ``CD = (24/Re)(1 + 0.15 Re^0.687) + 0.42 / (1 + 42500 Re^-1.16)``

    which spans Stokes flow to the subcritical plateau of about 0.47.

    .. warning::

       It does **not** capture the drag crisis.  Above a diameter Reynolds
       number of roughly 3e5 a smooth sphere's wake reattaches and its drag
       coefficient falls by a factor of four, to about 0.1; this correlation
       plateaus at 0.42 instead.  Nothing in this course's fleet has a body
       stubby enough to matter *and* fast enough to reach the crisis, but the
       limitation is real and it is why golf balls have dimples.
    """
    Re = np.asarray(Re, dtype=float)
    return (24.0 / Re) * (1.0 + 0.15 * Re**0.687) + 0.42 / (
        1.0 + 42500.0 * Re**-1.16
    )


#: Fineness ratio below which a body is treated as bluff rather than
#: streamlined.  See :func:`body_cd_frontal` for where the number comes from.
BLUFF_FINENESS = 4.0


def body_cd_frontal(fineness: float, Re_length: float, Re_diameter: float,
                    mach: float = 0.0, xtr: float = 0.0,
                    cone_fraction: float = 0.4,
                    bluff_below: float = BLUFF_FINENESS) -> float:
    """Drag coefficient of a body of revolution, on its **frontal** area.

    Two regimes, joined continuously.

    **Streamlined**, above ``bluff_below``: the course text's method, skin
    friction over the wetted area times a form factor, re-expressed on frontal
    area.  For a body with a gradually closing afterbody the flow stays
    attached and this is the right picture.

    **Bluff**, below ``bluff_below``: a short body cannot close gradually, the
    flow separates, and skin friction over a wetted area stops describing what
    is happening.  What is happening is closer to a sphere, so the coefficient
    is interpolated logarithmically between :func:`sphere_cd` at a fineness
    ratio of 1 -- where the body *is* a sphere -- and the streamlined value at
    the crossover.

    Why the crossover sits at 4 rather than at the text's stated limit of 5:
    the text's fit, extrapolated, gives 0.12 on frontal area at a fineness of
    4, and Hoerner's measured streamline bodies of revolution sit at 0.10-0.13
    there, so it is still describing reality.  At a fineness of 2 it gives
    0.068 against a measured 0.20, and at 1 it gives 0.036 against a sphere's
    0.47.  It fails below about 3, not below 5.

    Parameters
    ----------
    fineness : float
        Length over effective diameter.
    Re_length, Re_diameter : float
        Reynolds numbers on the length and on the effective diameter.
    mach, xtr, cone_fraction : float
    bluff_below : float
        Fineness ratio below which the bluff treatment takes over.

    Returns
    -------
    float
        ``CD`` referenced to the maximum cross-sectional area.
    """
    fr = float(fineness)
    shape = 1.0 - 0.25 * float(np.clip(cone_fraction, 0.0, 1.0))

    def streamlined(f):
        # Cf * k * (pi d L * shape) / (pi d^2 / 4) = Cf * k * 4 * shape * f
        cf = float(flat_plate_cf(Re_length, mach, xtr))
        return cf * form_factor_body(f) * 4.0 * shape * f

    if fr >= bluff_below:
        return streamlined(fr)

    cd_sphere = float(sphere_cd(Re_diameter))
    cd_match = streamlined(bluff_below)
    if fr <= 1.0:
        return cd_sphere
    t = np.log(fr) / np.log(bluff_below)
    return float(np.exp((1.0 - t) * np.log(cd_sphere) + t * np.log(cd_match)))


def oswald_efficiency(e_inv: float, CDp: float, aspect_ratio: float,
                      K: float = 0.38) -> float:
    """Oswald efficiency factor from the inviscid span efficiency.

    ``e = 1 / (1/e_inv + K CDp pi AR)``

    The book-keeping device that folds the viscous lift-dependent drag,
    ``K CDp CL^2``, back into an induced-drag-shaped term so the total polar
    keeps the form ``CD = CDp + CL^2/(pi AR e)``.

    ``e_inv`` is near 1 for a well designed wing; ``e`` typically lands between
    0.7 and 0.9, and the whole difference is viscous.  Quoting one when you
    mean the other is a common and expensive mistake.
    """
    return float(1.0 / (1.0 / e_inv + K * CDp * np.pi * aspect_ratio))


def span_efficiency_with_fuselage(fuselage_diameter: float, span: float,
                                  e_clean: float = 0.98) -> float:
    """Inviscid span efficiency reduced by the fuselage carry-through.

    ``e_inv = 0.98 [1 - 2 (d_f / b)^2]``

    The fuselage interrupts the lift distribution at the centreline, where an
    elliptical loading wants to be highest.  On a transport the penalty is a
    couple of per cent; on a fat short-span aircraft it is much more.
    """
    return float(e_clean * (1.0 - 2.0 * (fuselage_diameter / span) ** 2))


# --- the component buildup --------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One component's contribution to the parasitic drag.

    Attributes
    ----------
    name, kind : str
    S_wet : float
        Wetted area, m^2.
    length : float
        Reference length the Reynolds number is built on, m.
    Re : float
    cf : float
        Flat-plate skin friction.
    FF : float
        Form factor.
    f : float
        Drag area, ``cf * FF * S_wet``, m^2.  Additive across components, which
        coefficients are not until they share a reference area.
    cd_frontal : float
        For a body, the drag coefficient this row implies on the body's maximum
        cross-sectional area -- the number to compare against published
        body-drag data.  ``nan`` for a lifting surface.
    extrapolated : str
        Empty when the row is inside every correlation's validity; otherwise a
        short note saying which one was extrapolated and by how much.
    """

    name: str
    kind: str
    S_wet: float
    length: float
    Re: float
    cf: float
    FF: float
    f: float
    cd_frontal: float = float("nan")
    extrapolated: str = ""

    @property
    def CD0(self) -> float:
        """This row's drag area -- divide by ``S_ref`` for a coefficient."""
        return self.f


@dataclass(frozen=True)
class Buildup:
    """A component drag buildup at one flight condition.

    Attributes
    ----------
    rows : tuple of Row
    S_ref : float
        Reference area the coefficients use, m^2.
    interference : float
        Interference allowance, as a fraction of the component sum.  3-8%.
    protuberance : float
        Protuberance and roughness allowance, as a fraction of the component
        sum.  2-5% for a jet transport, 5-10% for a propeller aircraft, at
        least 10% for a small RC model.
    f_other : float
        A flat drag area, m^2, for anything the geometry does not resolve at
        all.  Zero unless you set it.
    V, altitude, mach : float
    skipped : tuple of str
        Bodies whose cross-section the sources do not publish, so no wetted
        area could be computed.  The DC-3's fuselage is one: the course model
        carries it, the nacelles and the gear together in ``f_other`` instead.
        A buildup with a non-empty ``skipped`` is not a whole-aircraft drag
        estimate unless ``f_other`` covers what is missing.
    """

    rows: Tuple[Row, ...]
    S_ref: float
    interference: float
    protuberance: float
    f_other: float
    V: float
    altitude: float
    mach: float
    skipped: Tuple[str, ...] = ()

    @property
    def f_components(self) -> float:
        """Summed component drag area before the markup, m^2."""
        return float(sum(r.f for r in self.rows))

    @property
    def extrapolated_rows(self) -> Tuple[Row, ...]:
        """Rows whose correlations were used outside their stated validity."""
        return tuple(r for r in self.rows if r.extrapolated)

    @property
    def markup(self) -> float:
        """Interference plus protuberance, as one fraction."""
        return self.interference + self.protuberance

    @property
    def f(self) -> float:
        """Total drag area including markups and ``f_other``, m^2."""
        return self.f_components * (1.0 + self.markup) + self.f_other

    @property
    def CD0(self) -> float:
        """Parasitic drag coefficient on ``S_ref``."""
        return self.f / self.S_ref

    @property
    def drag(self) -> float:
        """Parasitic drag force at the buildup's own condition, N."""
        return self.f * atmos.at(self.altitude).q(self.V)

    def fractions(self) -> Dict[str, float]:
        """Each component's share of the component total, before markup."""
        total = self.f_components
        return {r.name: r.f / total for r in self.rows}

    def table(self) -> str:
        """A printable buildup table -- the deliverable of a drag estimate."""
        w = max(14, max((len(r.name) for r in self.rows), default=14))
        lines = [
            f"{'component':<{w}} {'S_wet':>9} {'Re':>10} {'cf':>9} "
            f"{'FF':>6} {'CD_fr':>7} {'f (m2)':>9} {'share':>7}",
            "-" * (w + 63),
        ]
        total = self.f_components
        for r in self.rows:
            flag = " *" if r.extrapolated else ""
            ff = f"{r.FF:6.3f}" if np.isfinite(r.FF) else f"{'--':>6}"
            cdf = f"{r.cd_frontal:7.3f}" if np.isfinite(r.cd_frontal) else f"{'--':>7}"
            lines.append(
                f"{r.name:<{w}} {r.S_wet:9.3f} {r.Re:10.3e} {r.cf:9.5f} "
                f"{ff} {cdf} {r.f:9.5f} {100 * r.f / total:6.1f}%{flag}"
            )
        lines.append("-" * (w + 63))
        blank = f"{'':9} {'':10} {'':9} {'':6} {'':7}"
        lines.append(f"{'components':<{w}} {blank} {total:9.5f} {100.0:6.1f}%")
        if self.interference:
            label = f"interference ({100 * self.interference:.0f}%)"
            lines.append(f"{label:<{w}} {blank} {total * self.interference:9.5f}")
        if self.protuberance:
            label = f"protuberance ({100 * self.protuberance:.0f}%)"
            lines.append(f"{label:<{w}} {blank} {total * self.protuberance:9.5f}")
        if self.f_other:
            lines.append(f"{'f_other':<{w}} {blank} {self.f_other:9.5f}")
        lines.append(f"{'TOTAL':<{w}} {blank} {self.f:9.5f}")
        lines.append(f"\nCD0 = {self.CD0:.5f} on S_ref = {self.S_ref:.3f} m2")
        for r in self.extrapolated_rows:
            lines.append(f"  * {r.name}: {r.extrapolated}")
        if self.skipped:
            lines.append(
                "\nnot resolved (no published cross-section): "
                + ", ".join(self.skipped)
                + "\n  -> this is not a whole-aircraft buildup unless f_other covers them"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<drag.Buildup {len(self.rows)} components  "
            f"f={self.f:.4f} m2  CD0={self.CD0:.5f}>"
        )


def buildup(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    interference: float = 0.05,
    protuberance: float = 0.05,
    f_other: float = 0.0,
    cooling: float = 0.0,
    xtr: float = 0.0,
    xtr_wing: Optional[float] = None,
    carry_through: bool = True,
    bluff_below: float = BLUFF_FINENESS,
    dT: float = 0.0,
    include: Optional[Sequence[str]] = None,
) -> Buildup:
    """Component parasitic drag buildup, by the method in the course text.

    For each component, ``f = k C_f S_wet``, summed, then marked up for
    interference and protuberances.

    Parameters
    ----------
    aircraft : Aircraft
    V : float
        True airspeed, m/s.
    altitude : float
        Geometric altitude, m.
    interference : float
        Interference markup, 0.03 to 0.08.  See :data:`INTERFERENCE`.
    protuberance : float
        Protuberance and roughness markup.  See :data:`PROTUBERANCE`: about
        0.035 for a jet transport, 0.075 for a propeller aircraft, 0.10 or
        more for a small RC model.
    f_other : float
        Flat drag area for items the geometry does not resolve at all, m^2.
        The DC-3's simplified geometry has no fuselage, nacelles or gear, and
        the course carries 1.6 m^2 here to stand in for them.
    cooling : float
        Cooling drag area, m^2 -- momentum lost by air pushed through an
        engine's cooling fins and out again.  Zero for a jet, a sailplane or an
        electric model; 10 to 30 per cent of total drag on a piston single.
        Added to ``f_other``.  A first estimate is ``0.05 P / (rho V^3)``.
    xtr : float
        Laminar run fraction on bodies.
    xtr_wing : float, optional
        Laminar run on lifting surfaces; defaults to ``xtr``.
    bluff_below : float
        Fineness ratio below which a body is treated as bluff rather than
        streamlined -- see :func:`body_cd_frontal`.  Set it to 0 to force the
        streamlined method everywhere.
    carry_through : bool
        Subtract the wing and tail carry-through inside the widest body, so
        wetted area is exposed area.  The text's definition; turning it off
        recovers the cruder "use the reference area" treatment.
    dT : float
        Temperature offset, K.
    include : sequence of str, optional
        Restrict to these component kinds, e.g. ``["wing"]`` for a wing-only
        buildup to compare against a strip integration.

    Returns
    -------
    Buildup

    Notes
    -----
    **Bluff items.**  The text's method covers streamlined bodies of
    revolution, and rolls everything else into the protuberance markup.  That
    works until an aircraft carries something genuinely bluff in the airstream
    -- unfaired gear legs, lift struts, exposed wheels -- whose drag scales
    with frontal area and is an order of magnitude larger than a streamlined
    item of the same size.  A :class:`~flightlab.fleet.Body` may set
    ``drag_model="crossflow"`` (or ``"faired"``) to be treated that way
    instead, using :data:`CROSSFLOW_CD`.  The Cessna 172S uses it; without it
    a clean buildup puts its L/D near 17 against a published best glide of 9.
    """
    air = atmos.at(altitude, dT)
    M = float(air.mach(V))
    xtr_wing = xtr if xtr_wing is None else xtr_wing

    ref_panel = geom.resolve(aircraft.wing) if aircraft.wing else None
    S_ref = ref_panel.area if ref_panel else 1.0

    body_width = 0.0
    if carry_through and aircraft.bodies:
        widths = []
        for b in aircraft.bodies:
            if getattr(b, "drag_model", "streamlined") != "streamlined":
                continue
            try:
                widths.append(geom.body_diameter(b))
            except ValueError:
                pass
        body_width = max(widths) if widths else 0.0

    rows: List[Row] = []

    for kind, plan in (
        ("wing", aircraft.wing),
        ("htail", aircraft.htail),
        ("vtail", aircraft.vtail),
    ):
        if plan is None or (include is not None and kind not in include):
            continue
        p = geom.resolve(plan)
        width = body_width if kind == "wing" else 0.0
        S_wet = geom.wetted_area(p, width)
        Re = float(air.reynolds(V, p.mac))
        cf = float(flat_plate_cf(Re, M, xtr_wing))
        FF = form_factor_surface(p.thickness, M, p.sweep_c4_deg)
        rows.append(Row(kind, kind, S_wet, p.mac, Re, cf, FF,
                        cf * FF * S_wet, float('nan')))

    skipped: List[str] = []
    for body in aircraft.bodies:
        kind = _body_kind(body)
        if include is not None and kind not in include:
            continue
        model = getattr(body, "drag_model", "streamlined")
        if model in (
            "crossflow", "faired", "streamline_strut",
            "bluff_round_member", "faired_member", "streamlined_strut",
        ):
            try:
                frontal = body.frontal_area
            except ValueError:
                skipped.append(body.name)
                continue
            if model in {"faired", "faired_member"}:
                shape = "faired"
            elif model in {"streamline_strut", "streamlined_strut"}:
                shape = "streamline"
            elif model == "bluff_round_member":
                shape = "cylinder"
            else:
                shape = "cylinder" if body.diameter is not None else "streamline"
            cd_x = CROSSFLOW_CD[shape]
            rows.append(
                Row(body.name, "crossflow", frontal, body.length,
                    float(air.reynolds(V, body.length)), cd_x, float("nan"),
                    cd_x * frontal, cd_x)
            )
            continue
        try:
            S_wet = geom.body_wetted_area(body)
            d = geom.body_diameter(body)
            fr = body.fineness
        except ValueError:
            skipped.append(body.name)
            continue
        Re = float(air.reynolds(V, body.length))
        Re_d = float(air.reynolds(V, d))
        cf = float(flat_plate_cf(Re, M, xtr))
        cone = float(getattr(body, "cone_fraction", 0.4))
        cd = body_cd_frontal(fr, Re, Re_d, M, xtr, cone, bluff_below)
        S_max = 0.25 * np.pi * d**2 * body.count
        f_body = cd * S_max
        if fr >= bluff_below:
            FF = form_factor_body(fr)
            note = ""
        else:
            FF = float("nan")
            note = (
                f"fineness {fr:.2f} is below {bluff_below:.0f}, so this body is "
                "treated as bluff: CD on frontal area, blended toward a sphere"
            )
        rows.append(Row(body.name, kind, S_wet, body.length, Re, cf, FF,
                        f_body, cd, note))

    if not rows:
        raise ValueError(
            f"{aircraft.label}: no components matched include={include}"
        )

    return Buildup(
        rows=tuple(rows),
        skipped=tuple(skipped),
        S_ref=S_ref,
        interference=float(interference),
        protuberance=float(protuberance),
        f_other=float(f_other) + float(cooling),
        V=float(V),
        altitude=float(altitude),
        mach=M,
    )


def _body_kind(body: Body) -> str:
    name = body.name.lower()
    for key in ("nacelle", "boom", "pod", "strut"):
        if key in name:
            return key
    if "gear" in name or "leg" in name:
        return "gear"
    return "fuselage"


# --- strip integration ------------------------------------------------------


def strip_viscous_drag(
    solution: "wing.Solution",
    table: Optional["_airfoil.Table"] = None,
    section: Optional[str] = None,
    surface: Optional[str] = None,
    check_floor: bool = True,
) -> Dict[str, object]:
    """Wing viscous drag by integrating section data along the span.

    ``D_v = integral of cd * q * c dy``, with ``cd`` looked up at each
    station's **own** local ``cl`` and **own** local Reynolds number.  This is
    the wing's zero-lift and lift-dependent profile drag together, in one
    number -- the buildup's ``cf * FF * S_wet`` and its ``K CD_p CL^2`` term
    both at once, and not separable from each other afterwards.

    Parameters
    ----------
    solution : flightlab.wing.Solution
    table : flightlab.airfoil.Table, optional
        Built from the wing's section over the local Reynolds range if omitted.
    section : str, optional
        Section to build that table from.
    surface : str, optional
        Which surface's stations to integrate.  Defaults to the first.
    check_floor : bool
        Assert that no station's ``cd`` falls below the laminar flat plate at
        its own Reynolds number.  A violation means the lookup extrapolated.

    Returns
    -------
    dict
        ``drag`` (N), ``CD`` (on the solution's reference area), ``cd``
        (per station), ``y``, ``chord``, ``Re``, ``cl``, and ``clamped`` --
        the number of stations whose Reynolds number fell outside the table.
    """
    name = surface or solution.surfaces[0]
    s = solution.surface_slices[name]
    y, chord, cl, Re, ds = (
        solution.y[s], solution.chord[s], solution.cl[s],
        solution.Re[s], solution.ds[s],
    )

    if table is None:
        sec = section or "naca2412"
        table = _airfoil.table(sec, Re=(0.5 * Re.min(), 2.0 * Re.max()))

    cd = np.asarray(table.cd(cl, Re), dtype=float)
    clamped = int(np.sum(table.out_of_range(Re)))

    if check_floor:
        floor = flat_plate_cf(Re, xtr=1.0)
        bad = cd < floor
        if np.any(bad):
            i = int(np.argmax(bad))
            raise ValueError(
                f"station {i} at Re = {Re[i]:.3e} returned cd = {cd[i]:.6f}, "
                f"below the laminar flat plate {float(floor[i]):.6f}.  The "
                "section lookup is being extrapolated; widen the table's "
                "Reynolds range."
            )

    q = solution.q
    D = float(np.sum(cd * chord * ds)) * q
    return {
        "drag": D,
        "CD": D / (q * solution.reference["area"]),
        "cd": cd,
        "y": y,
        "chord": chord,
        "Re": Re,
        "cl": cl,
        "clamped": clamped,
    }


# --- compressibility --------------------------------------------------------


def drag_divergence_mach(
    thickness, CL, sweep_c4_deg=0.0
):
    """Drag-divergence Mach number, from the Korn equation with simple sweep.

    ``M_dd = 0.95/cos(L) - (t/c)/cos^2(L) - CL/(10 cos^3(L))``

    Parameters
    ----------
    thickness : float or array_like
        ``t/c``; must be positive and less than one.
    CL : float or array_like
    sweep_c4_deg : float or array_like
        Quarter-chord sweep, degrees.

    Notes
    -----
    An empirical correlation, and the reason a transonic wing can be reasoned
    about long before the physics is derivable.  Its three terms are the three
    levers a transonic wing designer has: sweep it, thin it, or fly it at lower
    lift coefficient.  Sweep appears in all three and dominates.
    """
    thickness = np.asarray(thickness, dtype=float)
    CL = np.asarray(CL, dtype=float)
    sweep = np.asarray(sweep_c4_deg, dtype=float)
    if np.any((thickness <= 0) | (thickness >= 1)):
        raise ValueError("thickness must be a t/c ratio between 0 and 1")
    if np.any(np.abs(sweep) >= 85.0):
        raise ValueError("the course correlation requires |quarter-chord sweep| < 85 deg")
    cos_l = np.cos(np.radians(sweep))
    result = 0.95 / cos_l - thickness / cos_l**2 - CL / (10.0 * cos_l**3)
    return float(result) if result.ndim == 0 else result


def crest_critical_mach(
    thickness, CL, sweep_c4_deg=0.0
):
    """``M_cc = M_dd - 0.11`` -- where the drag rise begins.

    Below it there are no shocks and no wave drag.  The offset follows from
    defining drag divergence as ``dCD/dM = 0.1`` and differentiating the
    quartic rise in :func:`wave_drag`.
    """
    return drag_divergence_mach(thickness, CL, sweep_c4_deg) - 0.11


def wave_drag(mach, thickness, CL, sweep_c4_deg=0.0):
    """Compressibility drag coefficient.

    ``CD_c = 20 (M - M_cc)^4`` for ``M_cc < M < 1``, and zero below.

    A quartic with a hard switch is not what a real drag rise looks like right
    at its onset, but it captures what matters: the penalty is negligible, then
    it is not, and the transition takes about 0.05 in Mach number.
    """
    mach = np.asarray(mach, dtype=float)
    if np.any(mach < 0):
        raise ValueError("mach must be nonnegative")
    m_cc = crest_critical_mach(thickness, CL, sweep_c4_deg)
    excess = np.clip(mach - m_cc, 0.0, None)
    result = 20.0 * excess**4
    return float(result) if result.ndim == 0 else result


# --- the complete polar -----------------------------------------------------


@dataclass(frozen=True)
class Polar:
    """A complete drag polar: total drag coefficient against lift coefficient.

    Attributes
    ----------
    CL : ndarray
    CD : ndarray
        The total.
    CD0, CD_i, CD_visc_lift, CD_wave : ndarray
        The four contributions.  For ``method="strip"`` the split is different:
        ``CD0`` is the wing's viscous drag at zero total lift plus the non-wing
        parasitic, and ``CD_visc_lift`` is the *change* in wing viscous drag
        from that baseline -- which can be negative, because a cambered section
        moves toward its minimum-drag condition as lift comes on.
    e : ndarray
        Inviscid span efficiency at each point.
    V, altitude, mach, S_ref : float
    method : str
        ``"buildup"`` or ``"strip"``.
    """

    CL: np.ndarray
    CD: np.ndarray
    CD0: np.ndarray
    CD_i: np.ndarray
    CD_visc_lift: np.ndarray
    CD_wave: np.ndarray
    e: np.ndarray
    V: float
    altitude: float
    mach: float
    S_ref: float
    method: str
    aspect_ratio: float = float("nan")

    @property
    def LD(self) -> np.ndarray:
        """Lift-to-drag ratio at each point."""
        return self.CL / self.CD

    @property
    def LD_max(self) -> float:
        """Maximum lift-to-drag ratio."""
        return float(np.max(self.LD))

    @property
    def CL_at_LD_max(self) -> float:
        """Lift coefficient at maximum ``L/D``."""
        return float(self.CL[int(np.argmax(self.LD))])

    @property
    def CD_at_LD_max(self) -> float:
        """Drag coefficient at maximum ``L/D``."""
        return float(self.CD[int(np.argmax(self.LD))])

    def speed_for(self, mass: float, n: float = 1.0) -> np.ndarray:
        """True airspeed at each point of the polar for a given mass, m/s."""
        air = atmos.at(self.altitude)
        return np.sqrt(
            2.0 * n * mass * G0 / (air.density * self.S_ref * np.maximum(self.CL, 1e-9))
        )

    def V_LD_max(self, mass: float) -> float:
        """Speed at maximum ``L/D`` for a given mass, m/s."""
        return float(self.speed_for(mass)[int(np.argmax(self.LD))])

    def CD_at(self, CL) -> np.ndarray:
        """Total drag coefficient interpolated at ``CL``."""
        return np.interp(np.asarray(CL, dtype=float), self.CL, self.CD)

    def drag(self, CL, mass: float, n: float = 1.0) -> np.ndarray:
        """Drag force at a given ``CL`` and mass, N."""
        air = atmos.at(self.altitude)
        CL = np.asarray(CL, dtype=float)
        V = np.sqrt(2.0 * n * mass * G0 / (air.density * self.S_ref * CL))
        return self.CD_at(CL) * air.q(V) * self.S_ref

    def split_at(self, CL: float) -> Dict[str, float]:
        """The drag breakdown at one lift coefficient, as fractions of the total."""
        i = int(np.argmin(np.abs(self.CL - CL)))
        total = self.CD[i]
        return {
            "CL": float(self.CL[i]),
            "CD": float(total),
            "parasitic": float(self.CD0[i] / total),
            "induced": float(self.CD_i[i] / total),
            "viscous_lift": float(self.CD_visc_lift[i] / total),
            "wave": float(self.CD_wave[i] / total),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<drag.Polar {self.method}  {len(self.CL)} points  "
            f"L/D_max={self.LD_max:.2f} at CL={self.CL_at_LD_max:.3f}>"
        )


def polar(
    aircraft: Aircraft,
    V: float,
    altitude: float = 0.0,
    CL=None,
    method: str = "buildup",
    e: Optional[float] = None,
    K: float = 0.38,
    interference: float = 0.05,
    protuberance: float = 0.05,
    f_other: float = 0.0,
    cooling: float = 0.0,
    xtr: float = 0.0,
    xtr_wing: Optional[float] = None,
    ns: int = 40,
    nc: int = 6,
    table: Optional["_airfoil.Table"] = None,
    section: Optional[str] = None,
    dT: float = 0.0,
    wave: bool = True,
) -> Polar:
    """Build a complete drag polar.

    Parameters
    ----------
    aircraft : Aircraft
    V : float
        True airspeed the polar is evaluated at, m/s.  Sets the Reynolds and
        Mach numbers.  A polar is only a curve in ``CL``; its ``CD0`` still
        depends on speed, and quoting one without its condition is incomplete.
    altitude : float
    CL : array_like, optional
        Lift coefficients.  Defaults to 0 to 1.4 in 29 steps.
    method : {"buildup", "strip"}
        ``"buildup"`` is the early-design method: component parasitic drag,
        an assumed span efficiency, and ``K CD_p CL^2`` for the lift-dependent
        viscous term.  ``"strip"`` integrates real section data along the span
        for the wing and takes induced drag from the Trefftz plane, using the
        buildup only for the non-wing components.
    e : float, optional
        Span efficiency for ``method="buildup"``.  Defaults to 0.98, the sort
        of number a designer uses before there is a planform.  Ignored by
        ``method="strip"``, which computes it.
    K : float
        Lift-dependent viscous coefficient in ``K CD_p CL^2``.  About 0.38 for
        a conventional aircraft.  Ignored by ``method="strip"``.
    interference, protuberance, f_other, cooling, xtr, xtr_wing : float
        Passed to :func:`buildup`.
    ns, nc : int
        Panelling for ``method="strip"``.
    table : flightlab.airfoil.Table, optional
    section : str, optional
    wave : bool
        Include the transonic drag rise.

    Returns
    -------
    Polar
    """
    if CL is None:
        CL = np.linspace(0.0, 1.4, 29)
    CL = np.atleast_1d(np.asarray(CL, dtype=float))

    air = atmos.at(altitude, dT)
    M = float(air.mach(V))
    p = geom.resolve(aircraft.wing)
    S_ref, AR = p.area, p.aspect_ratio

    CD_wave = (
        wave_drag(M, p.thickness, CL, p.sweep_c4_deg)
        if wave
        else np.zeros_like(CL)
    )

    if method == "buildup":
        b = buildup(
            aircraft, V, altitude, interference=interference,
            protuberance=protuberance, f_other=f_other, cooling=cooling,
            xtr=xtr, xtr_wing=xtr_wing, dT=dT,
        )
        CD0 = np.full_like(CL, b.CD0)
        e_val = 0.98 if e is None else e
        CD_i = CL**2 / (np.pi * AR * e_val)
        CD_vl = K * b.CD0 * CL**2
        e_arr = np.full_like(CL, e_val)

    elif method == "strip":
        non_wing = [
            k for k in ("htail", "vtail", "fuselage", "pod", "boom", "nacelle")
        ]
        try:
            b_other = buildup(
                aircraft, V, altitude, interference=interference,
                protuberance=protuberance, f_other=f_other, cooling=cooling,
                xtr=xtr, xtr_wing=xtr_wing, dT=dT, include=non_wing,
            )
            f_nonwing = b_other.f
        except ValueError:
            f_nonwing = f_other + cooling  # a wing-only aircraft
        CD_other = f_nonwing / S_ref

        sec = section or p.section
        if table is None:
            probe = wing.analyze(aircraft.wing, 0.0, V, altitude, ns=ns, nc=nc)
            table = _airfoil.table(
                sec, Re=(0.4 * probe.Re.min(), 2.5 * probe.Re.max())
            )

        visc, e_list = [], []
        for cl_target in CL:
            sol = wing.trim_to_CL(
                aircraft.wing, float(cl_target), V, altitude, ns=ns, nc=nc,
                bracket=(-15.0, 20.0),
            )
            r = strip_viscous_drag(sol, table=table, check_floor=False)
            visc.append(r["CD"])
            e_list.append(sol.e_inv)
        visc = np.array(visc)
        e_arr = np.array(e_list)

        # baseline: the wing's viscous drag at zero total lift
        sol0 = wing.trim_to_CL(
            aircraft.wing, 0.0, V, altitude, ns=ns, nc=nc, bracket=(-15.0, 20.0)
        )
        visc0 = strip_viscous_drag(sol0, table=table, check_floor=False)["CD"]

        CD0 = np.full_like(CL, visc0 + CD_other)
        CD_vl = visc - visc0
        CD_i = CL**2 / (np.pi * AR * e_arr)
        # the same empirical lift-dependent term for the non-wing components,
        # which the strip integration does not resolve
        CD_vl = CD_vl + K * CD_other * CL**2

    else:
        raise ValueError(f"method must be 'buildup' or 'strip', not {method!r}")

    return Polar(
        CL=CL,
        CD=CD0 + CD_i + CD_vl + CD_wave,
        CD0=CD0,
        CD_i=CD_i,
        CD_visc_lift=CD_vl,
        CD_wave=CD_wave,
        e=e_arr,
        V=float(V),
        altitude=float(altitude),
        mach=M,
        S_ref=S_ref,
        method=method,
        aspect_ratio=AR,
    )
