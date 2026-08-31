"""``flightlab.geom`` -- planform arithmetic, wetted areas, and vortex-lattice grids.

The bridge between :mod:`flightlab.fleet`, which holds aircraft as tables, and
:mod:`flightlab.vlm`, which wants panel grids.  It also holds the geometry every
other module keeps asking for: mean aerodynamic chord and where it sits, chord
and sweep as functions of span station, and wetted area.

Nothing here is a model.  It is all definitions and trigonometry, which is why
it is provided rather than assigned: there is no physics in it to learn, and a
sign error in a sweep angle looks exactly like an aerodynamic result.

    >>> from flightlab import geom
    >>> from flightlab.fleet import ASW27
    >>> p = geom.resolve(ASW27.wing)
    >>> round(p.mac, 3), round(p.root_chord, 3)
    (0.637, 0.857)

Conventions
-----------
Body axes: ``x`` aft, ``y`` right, ``z`` up, origin at the wing root leading
edge unless an aircraft says otherwise.  Angles are **degrees** in this
module's interfaces and converted to radians only where :mod:`flightlab.vlm`
requires it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import foil
from .fleet import Aircraft, Body, Planform
from .vlm import Cosine, Uniform, wing_to_grid

__all__ = [
    "Panel",
    "resolve",
    "chord_at",
    "sweep_at",
    "mac",
    "exposed_area",
    "wetted_area",
    "body_wetted_area",
    "body_diameter",
    "surface_grid",
    "aircraft_surfaces",
    "reference",
]


# --- resolved planform ------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """A planform with every derived quantity filled in.

    :class:`flightlab.fleet.Planform` stores what its sources publish, which for
    several aircraft is span, area and taper but not the chords.  This resolves
    the set to a complete, self-consistent description.

    Attributes
    ----------
    span : float
        Tip to tip, m.  For a fin, the height, with ``vertical`` set.
    area : float
        Reference area, m^2.  For a fin, the single-surface area.
    root_chord, tip_chord, taper, mac : float
        m, m, dimensionless, m.
    y_mac, x_mac : float
        Spanwise station of the mean aerodynamic chord and the ``x`` of its
        leading edge, m, relative to the root leading edge.
    sweep_le_deg, sweep_c4_deg, dihedral_deg, twist_deg, incidence_deg : float
    thickness : float
        ``t/c``.
    x_le, z : float
        Root leading edge position in body axes, m.
    section : str
        Airfoil designation to hand to :mod:`flightlab.foil`.
    vertical : bool
    """

    span: float
    area: float
    root_chord: float
    tip_chord: float
    taper: float
    mac: float
    y_mac: float
    x_mac: float
    sweep_le_deg: float
    sweep_c4_deg: float
    dihedral_deg: float
    twist_deg: float
    incidence_deg: float
    thickness: float
    x_le: float
    z: float
    section: str
    vertical: bool
    assumed: frozenset = frozenset()

    @property
    def aspect_ratio(self) -> float:
        """``b^2 / S``."""
        return self.span**2 / self.area

    @property
    def standard_mean_chord(self) -> float:
        """``S / b`` for a wing, ``S / height`` for a fin, m.

        Not the same thing as :attr:`mac`, and the difference is not small: the
        787's is 6.27 m against a 7.41 m MAC, an 18% gap, because the standard
        mean chord averages the chord by span and the aerodynamic one averages
        it by chord squared.  Sources publish both, sometimes without saying
        which.  Moments are non-dimensionalized by the MAC.
        """
        return self.area / (self.span if not self.vertical else self.semispan)

    @property
    def semispan(self) -> float:
        """Half span for a wing or tailplane; full height for a fin."""
        return self.span if self.vertical else 0.5 * self.span

    @property
    def x_c4_mac(self) -> float:
        """``x`` of the MAC quarter chord in body axes, m.

        The point moments are taken about by convention, and the one a static
        margin is quoted against.
        """
        return self.x_le + self.x_mac + 0.25 * self.mac

    def chord(self, eta) -> np.ndarray:
        """Chord at fractional semispan ``eta`` in [0, 1], m."""
        eta = np.asarray(eta, dtype=float)
        return self.root_chord * (1.0 - (1.0 - self.taper) * eta)

    def reynolds(self, V: float, state) -> float:
        """Reynolds number on the MAC at speed ``V`` in atmosphere ``state``."""
        return state.reynolds(V, self.mac)


def _first(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _area_factor(planform: Planform) -> float:
    """How the reference area relates to ``semispan * chord``.

    A wing's area counts both sides: ``S = semispan * cr * (1 + taper)``.  A
    fin's area counts one surface over its full height, so the same expression
    carries a factor of one half.  Getting this backwards halves or doubles
    every fin chord, and since the fin area is what tail volume uses, the error
    hides until a directional stability number comes out twice what it should.
    """
    return 0.5 if planform.vertical else 1.0


def resolve(
    planform: Planform,
    section: Optional[str] = None,
    default_taper: float = 1.0,
) -> Panel:
    """Fill in every derived quantity of a :class:`~flightlab.fleet.Planform`.

    Chords come from whichever pair of {area, span, taper, root chord, tip
    chord} the aircraft's sources publish.  Sweep angles convert between
    leading-edge and quarter-chord using the resolved taper, so an aircraft
    with only one of them gets the other.

    Parameters
    ----------
    planform : Planform
    section : str, optional
        Override the airfoil designation.
    default_taper : float
        Taper to assume when the sources publish neither chord nor taper --
        only the span and the area.  The ASG 29's wing and the 787's tail
        surfaces are in that position.  The default of 1.0 is rectangular,
        chosen because it is visibly an assumption rather than a plausible
        invention; pass a real value when you have one.

    Returns
    -------
    Panel
        With :attr:`Panel.assumed` naming anything that was assumed rather than
        derived, so a report can say so.

    Raises
    ------
    ValueError
        If the published set is not enough to determine the chords -- which is
        better than inventing a taper ratio and burying it.
    """
    b, S = planform.span, planform.area
    semispan = b if planform.vertical else 0.5 * b
    assumed = set()

    cr, ct, taper = planform.root_chord, planform.tip_chord, planform.taper
    if cr is not None and ct is not None:
        taper = ct / cr
    elif cr is not None and taper is not None:
        ct = taper * cr
    elif ct is not None and taper is not None:
        cr = ct / taper
    elif cr is not None:
        ct = 2.0 * S / semispan - cr if planform.vertical else 2.0 * S / b - cr
        taper = ct / cr
    else:
        if taper is None:
            taper = default_taper
            assumed.add("taper")
        cr = S / (_area_factor(planform) * semispan * (1.0 + taper))
        ct = taper * cr

    if cr <= 0 or ct <= 0:
        raise ValueError(
            f"resolved a non-positive chord (root {cr:.4g} m, tip {ct:.4g} m); "
            "the published span, area and taper are not mutually consistent"
        )

    # mean aerodynamic chord of a straight-tapered panel.  Deliberately *not*
    # planform.mean_chord: sources usually publish S/b there, which is the
    # standard mean chord and differs from the MAC on any tapered wing.
    mac_val = (2.0 / 3.0) * cr * (1.0 + taper + taper**2) / (1.0 + taper)
    y_mac = semispan * (1.0 + 2.0 * taper) / (3.0 * (1.0 + taper))

    sweep_le = planform.sweep_le_deg
    sweep_c4 = planform.sweep_c4_deg
    if sweep_le is None and sweep_c4 is None:
        sweep_le = sweep_c4 = 0.0
    elif sweep_le is None:
        sweep_le = math.degrees(
            math.atan(
                math.tan(math.radians(sweep_c4))
                + 0.25 * (cr - ct) / semispan
            )
        )
    elif sweep_c4 is None:
        sweep_c4 = math.degrees(
            math.atan(
                math.tan(math.radians(sweep_le))
                - 0.25 * (cr - ct) / semispan
            )
        )

    x_mac = y_mac * math.tan(math.radians(sweep_le))

    return Panel(
        span=b,
        area=S,
        root_chord=cr,
        tip_chord=ct,
        taper=taper,
        mac=mac_val,
        y_mac=y_mac,
        x_mac=x_mac,
        sweep_le_deg=sweep_le,
        sweep_c4_deg=sweep_c4,
        dihedral_deg=planform.dihedral_deg,
        twist_deg=planform.twist_deg,
        incidence_deg=planform.incidence_deg,
        thickness=_thickness(planform, assumed),
        x_le=planform.x_le if planform.x_le is not None else 0.0,
        z=planform.z,
        section=section or planform.section_file or planform.section,
        vertical=planform.vertical,
        assumed=frozenset(assumed),
    )


def _thickness(planform: Planform, assumed: set) -> float:
    """``t/c``, defaulting to 0.12 and saying so when the source is silent."""
    if planform.thickness is not None:
        return planform.thickness
    assumed.add("thickness")
    return 0.12


# --- planform functions -----------------------------------------------------


def chord_at(panel: Panel, y) -> np.ndarray:
    """Chord at spanwise station ``y`` (m from the centreline), m."""
    eta = np.abs(np.asarray(y, dtype=float)) / panel.semispan
    return panel.chord(eta)


def sweep_at(panel: Panel, x_over_c: float) -> float:
    """Sweep of the constant-``x/c`` line, degrees.

    ``sweep_at(panel, 0.5)`` is the mid-chord sweep the compressibility
    correlations want; ``sweep_at(panel, 0.25)`` returns the quarter-chord
    sweep it was resolved from.
    """
    tan_le = math.tan(math.radians(panel.sweep_le_deg))
    return math.degrees(
        math.atan(tan_le - x_over_c * (panel.root_chord - panel.tip_chord) / panel.semispan)
    )


def mac(panel: Panel) -> Tuple[float, float, float]:
    """``(mac, y_mac, x_mac)`` -- length, station, and leading-edge x, m."""
    return panel.mac, panel.y_mac, panel.x_mac


def exposed_area(panel: Panel, body_width: float = 0.0) -> float:
    """Planform area outside the fuselage, m^2.

    Parasitic drag acts on what would get wet if the aircraft were dipped in a
    fluid, and the part of the wing inside the fuselage would not.  Subtracts
    the carry-through: the chord at the side of the body, times the body width.

    With ``body_width = 0`` this returns the full reference area, which is the
    right answer for a wing mounted on top of a slim pod and the wrong one for
    a wing through a wide fuselage.
    """
    if body_width <= 0.0:
        return panel.area
    eta = min(0.5 * body_width / panel.semispan, 1.0)
    c_side = float(panel.chord(eta))
    carry = 0.5 * body_width * (panel.root_chord + c_side)
    sides = 1.0 if panel.vertical else 1.0
    return max(panel.area - sides * carry, 0.1 * panel.area)


def wetted_area(panel: Panel, body_width: float = 0.0) -> float:
    """Wetted area of a lifting surface, m^2.

    ``S_wet = 2 (1 + 0.2 t/c) S_exposed`` -- the course text's expression.  The
    factor of two is both sides of the exposed planform; the ``t/c`` term
    approximates the extra surface the section's curvature adds.

    Parameters
    ----------
    panel : Panel
    body_width : float
        Width of the body the surface passes through, m.  The carry-through is
        subtracted, because it does not get wet.  Zero leaves the full
        reference area.
    """
    return 2.0 * (1.0 + 0.2 * panel.thickness) * exposed_area(panel, body_width)


def body_wetted_area(body: Body) -> float:
    """Wetted area of a fuselage, pod, boom or nacelle, m^2.

    Built the way the course text builds it: the cylindrical portion
    contributes ``pi d l_cyl`` and each rounded end contributes
    ``0.75 pi d l_cone``.  With a fraction ``f`` of the length taken as
    nose and tail cones together, that is ``pi d L (1 - 0.25 f)``.

    Multiplied by ``body.count``, so a twin's two nacelles are one call.
    """
    d = body_diameter(body)
    f = float(np.clip(getattr(body, "cone_fraction", 0.4), 0.0, 1.0))
    return body.count * np.pi * d * body.length * (1.0 - 0.25 * f)


def body_diameter(body: Body) -> float:
    """Effective diameter of a body, m.

    ``d_eff = sqrt(4 S_max / pi)`` from the maximum cross-sectional area, so a
    rectangular pod and a round one of the same frontal size share a fineness
    ratio.  A round body returns its own diameter exactly.
    """
    if body.diameter is not None:
        return body.diameter
    if body.width is not None and body.height is not None:
        s_max = 0.25 * np.pi * body.width * body.height  # elliptical section
        return float(np.sqrt(4.0 * s_max / np.pi))
    if body.width is not None:
        return body.width
    raise ValueError(f"{body.name}: no cross-section dimension given")


# --- vortex lattice grids ---------------------------------------------------


def surface_grid(
    panel: Panel,
    ns: int = 40,
    nc: int = 6,
    mirror: bool = False,
    camber: bool = True,
    spacing_s=None,
    spacing_c=None,
):
    """Build a vortex-lattice grid for one straight-tapered surface.

    Parameters
    ----------
    panel : Panel
        A resolved planform.
    ns, nc : int
        Spanwise and chordwise panel counts on the **semispan**.  Cosine
        spanwise spacing is the default because it clusters panels at the tip,
        where the loading gradient is, and reaches a converged span efficiency
        at roughly a third the panel count of uniform spacing.
    mirror : bool
        Build both sides explicitly.  Needed for lateral work: with
        ``symmetric=True`` in the solver every roll and yaw derivative comes
        back exactly zero, which is correct and looks like a broken code.
        Ignored for a vertical surface, which has only one side.
    camber : bool
        Deflect the panels onto the section's camber line.  The camber line is
        all an inviscid vortex lattice sees of a section, and leaving it out
        moves the zero-lift angle to zero, which is wrong for every cambered
        wing in the fleet.
    spacing_s, spacing_c : vlm spacing objects, optional
        Default ``Cosine()`` spanwise and ``Uniform()`` chordwise.

    Returns
    -------
    grid, ratios
        Ready to hand to :func:`flightlab.vlm.steady_analysis`.

    Notes
    -----
    Twist is applied linearly from root to tip, negative being washout, and is
    added to the root incidence.  Angles go into :mod:`flightlab.vlm` in radians;
    this function does the conversion.
    """
    spacing_s = Cosine() if spacing_s is None else spacing_s
    spacing_c = Uniform() if spacing_c is None else spacing_c

    # A centreline fin already lies in the plane of symmetry, so mirroring it
    # lays a second copy exactly on top of the first.  The duplicated panels
    # make the influence matrix singular and the whole solve returns NaN, which
    # is a confusing way to find out.  Mirroring is for surfaces with two
    # sides; a fin has one.
    if panel.vertical:
        mirror = False

    semispan = panel.semispan
    tan_le = math.tan(math.radians(panel.sweep_le_deg))

    xle = [0.0, semispan * tan_le]
    if panel.vertical:
        # a fin lives in the x-z plane: its "span" runs along z
        yle = [0.0, 0.0]
        zle = [0.0, semispan]
    else:
        dihedral = math.radians(panel.dihedral_deg)
        yle = [0.0, semispan * math.cos(dihedral)]
        zle = [0.0, semispan * math.sin(dihedral)]

    chord = [panel.root_chord, panel.tip_chord]
    theta = [
        math.radians(panel.incidence_deg),
        math.radians(panel.incidence_deg + panel.twist_deg),
    ]

    fc = None
    if camber and panel.section:
        try:
            f = foil.load(panel.section).camber_function()
            fc = [f, f]
        except Exception:
            fc = None  # an unresolvable section is a flat plate, not a crash

    grid, ratios = wing_to_grid(
        xle, yle, zle, chord, theta, [0.0, 0.0], ns, nc,
        fc=fc, spacing_s=spacing_s, spacing_c=spacing_c, mirror=mirror,
    )
    return grid, ratios


def aircraft_surfaces(
    aircraft: Aircraft,
    ns: int = 40,
    nc: int = 6,
    tail: bool = True,
    fin: bool = False,
    mirror: bool = False,
    tail_incidence_deg: Optional[float] = None,
) -> Tuple[List, List, List[str]]:
    """Grids for an aircraft's lifting surfaces.

    Parameters
    ----------
    aircraft : Aircraft
    ns, nc : int
        Panel counts per surface semispan.
    tail, fin : bool
        Include the horizontal tail and the vertical fin.  The fin contributes
        nothing longitudinal, so it is off by default and on for lateral work.
    mirror : bool
        Mirror the surfaces explicitly rather than relying on solver symmetry.
    tail_incidence_deg : float, optional
        Override the horizontal tail's incidence -- the variable a longitudinal
        trim solve turns.

    Returns
    -------
    grids, ratios, names
    """
    from .vlm import translate

    grids, ratios, names = [], [], []

    for label, plan, include in (
        ("wing", aircraft.wing, True),
        ("htail", aircraft.htail, tail),
        ("vtail", aircraft.vtail, fin),
    ):
        if plan is None or not include:
            continue
        panel = resolve(plan)
        if label == "htail" and tail_incidence_deg is not None:
            panel = Panel(**{**panel.__dict__, "incidence_deg": tail_incidence_deg})
        grid, r = surface_grid(panel, ns=ns, nc=nc, mirror=mirror)
        if panel.x_le or panel.z:
            grid = translate(grid, (panel.x_le, 0.0, panel.z))
        grids.append(grid)
        ratios.append(r)
        names.append(label)

    if not grids:
        raise ValueError(f"{aircraft.label} has no lifting surfaces to build")
    return grids, ratios, names


def reference(aircraft: Aircraft, V: float, x_ref: Optional[float] = None):
    """A :class:`flightlab.vlm.Reference` for an aircraft.

    Areas and lengths come from the wing; the moment reference defaults to the
    wing MAC quarter chord, which is what a static margin is quoted against.
    """
    from .vlm import Reference

    panel = resolve(aircraft.wing)
    x_ref = panel.x_c4_mac if x_ref is None else x_ref
    return Reference(panel.area, panel.mac, panel.span, [x_ref, 0.0, 0.0], V)
