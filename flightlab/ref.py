"""``flightlab.ref`` -- reference values, so a verification assertion is one line.

Everything here carries its provenance, because a verification rung is only as
good as the number you are checking against.  Three kinds of entry, and the
difference matters:

``DEFINED``
    Values that *are* the definition (the 1976 standard atmosphere's base
    constants).  Assert against these to machine precision.

``ANALYTIC``
    Closed forms.  Exact within their own assumptions, so a disagreement is
    either your bug or a violated assumption -- and telling those apart is the
    skill.

``MEASURED`` / ``PUBLISHED``
    Somebody else's number.  Cite it, and expect real disagreement.  Where a
    value was read off a published chart it is given as a **range**, not a point,
    because chart-reading precision is not three digits.

``TOOL``
    Output of a tool, recorded so you can tell a code change from a physics
    change.  **Not truth.**  A tool agreeing with itself proves nothing.

``COURSE MODEL``
    A transparent teaching assumption supplied to make an exercise runnable.
    Its validity box and provenance are part of the data; it is not measured
    aircraft performance.

Examples
--------
::

    from flightlab import ref

    assert abs(rho_sl - ref.ATMOS_SL["density"]) < 1e-4
    assert cd >= ref.laminar_flat_plate_cf(Re)      # analytic lower bound
    assert abs(e_inv - ref.ELLIPTICAL_E_INV) < 5e-3 # known induced-drag limit
    lo, hi = ref.CDP_WIDEBODY_CRUISE                # course plausibility range
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ATMOS_SL",
    "ATMOS_CONSTANTS",
    "ATMOS_CALCULATOR",
    "PROVO_ALTITUDE",
    "ELLIPTICAL_E_INV",
    "elliptical_induced_drag",
    "elliptical_root_bending_moment",
    "lifting_line_CL_alpha",
    "laminar_flat_plate_cf",
    "ideal_propulsive_efficiency",
    "motor_peak_efficiency",
    "OPTIMAL_BATTERY_MASS_FRACTION",
    "BATTERY_SOC_REFERENCE",
    "phugoid_period_approx",
    "phugoid_damping_approx",
    "RC1_LONGITUDINAL_MATRICES",
    "B787_ENGINE_MODEL",
    "b787_thrust_available",
    "AVL_SIMPLE_WING",
    "NEURALFOIL_NACA2412",
    "PUBLISHED_NACA2412",
    "NEURALFOIL_VS_PUBLISHED_NACA2412",
    "DC3_COURSE_SOLUTION",
    "SATURN_V_S_IC_TARGETS",
    "CDP_WIDEBODY_CRUISE",
    "SWET_SREF_WIDEBODY",
    "MOTOR_PEAK_EFFICIENCY_BAND",
    "CHAIN_EFFICIENCY_BAND",
    "UIUC_LOW_SPEED_AIRFOIL_DATA",
]

# --- standard atmosphere ----------------------------------------------------

#: Sea-level base values of the 1976 U.S. Standard Atmosphere.  DEFINED.
#: These are the standard's defining constants, not measurements, so your
#: implementation should reproduce them exactly.
ATMOS_SL = {
    "temperature": 288.15,  # K
    "pressure": 101325.0,  # Pa
    "density": 1.225,  # kg/m^3
    "speed_of_sound": 340.294,  # m/s
    "viscosity": 1.7894e-5,  # Pa*s
}

#: Constants of the 1976 standard.  DEFINED.
ATMOS_CONSTANTS = {
    "R": 287.0528,  # J/(kg*K), specific gas constant for air
    "gamma": 1.4,
    "g0": 9.80665,  # m/s^2
    "lapse_rate_troposphere": -0.0065,  # K/m
    "tropopause_altitude": 11000.0,  # m (geopotential)
    "tropopause_temperature": 216.65,  # K
    # Sutherland's law for air
    "sutherland_mu0": 1.716e-5,  # Pa*s
    "sutherland_T0": 273.15,  # K
    "sutherland_S": 110.4,  # K
}

#: An independent atmosphere table for the published-data rung.  The point of
#: that rung is to check against something that is *not* your own code, so use
#: this rather than re-deriving the standard from its own formulas.
ATMOS_CALCULATOR = "https://www.digitaldutch.com/atmoscalc/"

#: Geometric altitude of the Provo flying field, m.  Used all semester.
PROVO_ALTITUDE = 1400.0


# --- analytic limits --------------------------------------------------------

#: Inviscid span efficiency of an elliptical planform.  ANALYTIC.
#: A vortex lattice approaches this from above as the spanwise panel count
#: rises; expect roughly 1.001 at 80 panels per side with cosine spacing.
#: Calibrate your resolution here *before* trusting a planform whose answer you
#: do not know.
ELLIPTICAL_E_INV = 1.0


def elliptical_induced_drag(L, q, b):
    """Induced drag of an elliptically loaded wing.  ANALYTIC.

    .. math:: D_i = \\frac{L^2}{q \\pi b^2}

    Parameters
    ----------
    L : float or array_like
        Lift, N.
    q : float or array_like
        Dynamic pressure, Pa.
    b : float or array_like
        Span, m.

    Returns
    -------
    float or ndarray
        Induced drag, N.
    """
    return np.asarray(L, dtype=float) ** 2 / (
        np.asarray(q, dtype=float) * np.pi * np.asarray(b, dtype=float) ** 2
    )


def elliptical_root_bending_moment(L, b):
    """Root bending moment of an elliptically loaded wing.  ANALYTIC.

    .. math:: M_{root} = \\frac{L b}{3 \\pi}

    Parameters
    ----------
    L : float or array_like
        Total lift on the whole wing, N.
    b : float or array_like
        Full span, m.

    Returns
    -------
    float or ndarray
        Bending moment at the root, N*m.

    Notes
    -----
    ``L`` is the lift of the **whole wing**, and the moment is carried by one
    side.  If your numerical integration comes out exactly a factor of two off
    this, you integrated the whole span instead of the half span -- which is the
    single most common error in the ``loads`` module.
    """
    return np.asarray(L, dtype=float) * np.asarray(b, dtype=float) / (3.0 * np.pi)


def lifting_line_CL_alpha(cl_alpha, AR, e=1.0):
    """Finite-wing lift curve slope from lifting-line theory.  ANALYTIC.

    .. math:: C_{L\\alpha} = \\frac{c_{l\\alpha}}{1 + c_{l\\alpha}/(\\pi AR\\, e)}

    Parameters
    ----------
    cl_alpha : float or array_like
        Section lift curve slope, **per radian**.
    AR : float or array_like
        Aspect ratio.
    e : float or array_like
        Span efficiency.

    Returns
    -------
    float or ndarray
        Per radian.

    Notes
    -----
    This is a useful independent check. It agrees with a vortex lattice at high
    aspect ratio and degrades as ``AR`` falls -- roughly 4% low at ``AR = 25``,
    10% at ``AR = 7.5`` and 17% at ``AR = 3.2`` for a flat plate.  Knowing
    *which* assumption fails is the graded part.
    """
    cla = np.asarray(cl_alpha, dtype=float)
    return cla / (1.0 + cla / (np.pi * np.asarray(AR, dtype=float) * e))


def laminar_flat_plate_cf(Re):
    """Blasius laminar flat-plate skin friction coefficient.  ANALYTIC.

    .. math:: c_f = \\frac{1.328}{\\sqrt{Re}}

    Parameters
    ----------
    Re : float or array_like
        Reynolds number based on the reference length.

    Returns
    -------
    float or ndarray

    Notes
    -----
    A hard **lower bound**.  A drag model returning a ``c_f`` or a section
    ``c_d`` below this value is not optimistic, it is wrong, and it is a common
    failure of a hastily written correlation. It is a useful assertion inside
    any strip-integration loop.
    """
    return 1.328 / np.sqrt(np.asarray(Re, dtype=float))


def ideal_propulsive_efficiency(T, rho, V, A):
    """Actuator-disk upper bound on propulsive efficiency.  ANALYTIC.

    .. math::

        \\eta_{ideal} = \\frac{2}{1 + \\sqrt{1 + T/(\\tfrac{1}{2}\\rho V^2 A)}}

    Parameters
    ----------
    T : float or array_like
        Thrust, N.
    rho : float or array_like
        Density, kg/m^3.
    V : float or array_like
        Flight speed, m/s.
    A : float or array_like
        Disk area, m^2.

    Returns
    -------
    float or ndarray

    Notes
    -----
    A consistently defined propeller efficiency should generally sit below
    this ideal reference.  A small apparent violation is first a reason to
    audit disk area, coefficient conventions, and interpolation.
    """
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    V = np.asarray(V, dtype=float)
    A = np.asarray(A, dtype=float)
    return 2.0 / (1.0 + np.sqrt(1.0 + T / (0.5 * rho * V**2 * A)))


def motor_peak_efficiency(I0, R, V):
    """Maximum efficiency of the three-parameter motor model.  ANALYTIC.

    .. math:: \\eta_{max} = \\left(1 - \\sqrt{I_0 R / V}\\right)^2

    Parameters
    ----------
    I0 : float or array_like
        No-load current, A.
    R : float or array_like
        Winding resistance, ohms.
    V : float or array_like
        Terminal voltage, V.

    Returns
    -------
    float or ndarray

    Notes
    -----
    Note what is absent: ``Kv``.  Peak efficiency ranks with ``I0*R/V`` and not
    with the speed constant, an important motor-selection result.
    """
    return (
        1.0
        - np.sqrt(
            np.asarray(I0, dtype=float)
            * np.asarray(R, dtype=float)
            / np.asarray(V, dtype=float)
        )
    ) ** 2


#: Battery mass fraction that maximizes endurance for a constant-weight
#: electric aircraft with the structure frozen.  ANALYTIC.
#: Endurance goes as ``m_batt/(m_fixed + m_batt)**1.5``, whose maximum is at
#: ``m_batt = 2*m_fixed`` -- two thirds of the airplane in battery.  The same
#: point is also where the corresponding closed sizing loop stops converging.
OPTIMAL_BATTERY_MASS_FRACTION = 2.0 / 3.0


def phugoid_period_approx(V, g=9.80665):
    """Classical phugoid period approximation.  ANALYTIC.

    .. math:: T \\approx \\pi \\sqrt{2}\\, V / g

    Parameters
    ----------
    V : float or array_like
        Flight speed, m/s.
    g : float

    Returns
    -------
    float or ndarray
        Period, s.

    Notes
    -----
    Because this scales linearly with ``V``, halving the flight speed must halve
    the period.  If your state matrix does not do that, you have a normalization
    error in a rate derivative, and this test finds it faster than reading the
    matrix will.
    """
    return np.pi * np.sqrt(2.0) * np.asarray(V, dtype=float) / g


def phugoid_damping_approx(LD):
    """Classical phugoid damping ratio approximation.  ANALYTIC.

    .. math:: \\zeta \\approx \\frac{1}{\\sqrt{2}\\, L/D}

    Parameters
    ----------
    LD : float or array_like
        Lift-to-drag ratio at the flight condition.

    Returns
    -------
    float or ndarray

    Notes
    -----
    Your drag polar predicts a *dynamic* property, which is worth a moment's
    thought.
    """
    return 1.0 / (np.sqrt(2.0) * np.asarray(LD, dtype=float))


# --- supplied teaching models ----------------------------------------------

#: Reduced-order dimensional longitudinal matrices for modal demonstrations.
#: COURSE MODEL.
#:
#: The state is ``[u, gamma, alpha, q]``: forward-speed perturbation (m/s),
#: flight-path angle (rad), angle of attack (rad), and pitch rate (rad/s).
#: The model uses a phugoid block in ``[u, gamma]`` and a short-period block in
#: ``[alpha, q]``, with one-way kinematic coupling chosen so
#: ``q = alpha_dot + gamma_dot``. This lets students learn modal interpretation
#: without first assembling dimensional derivatives. The
#: phugoid block is calibrated to ``T = pi*sqrt(2)*V/g`` and
#: ``zeta = 1/(sqrt(2)*L/D)`` using ``L/D = 9.6``.  The short-period natural
#: frequency is a course estimate (12 rad/s at cruise, proportional to speed)
#: with damping ratio 0.45.  These are not flight-identified RC-1 dynamics and
#: are not an independent validation of a student-built dynamics model.
RC1_LONGITUDINAL_MATRICES = {
    "state_order": ("u", "gamma", "alpha", "q"),
    "state_units": ("m/s", "rad", "rad", "rad/s"),
    "input": None,
    "assumptions": (
        "small perturbations about steady level flight",
        "reduced-order phugoid and short-period blocks with kinematic coupling",
        "phugoid calibrated to V and L/D = 9.6",
        "short-period frequency and damping are course estimates",
        "no controls, lateral motion, propulsion dynamics, or nonlinear stall",
    ),
    "cruise": {
        "speed": 9.1,
        "A": np.array([
            [-0.22512274, -9.80665, 0.0, 0.0],
            [0.23753642, 0.0, 0.0, 0.0],
            [-0.23753642, 0.0, 0.0, 1.0],
            [0.0, 0.0, -144.0, -10.8],
        ]),
        "reference_eigenvalues": (
            -0.11256137 + 1.52209279j,
            -0.11256137 - 1.52209279j,
            -5.4 + 10.71634266j,
            -5.4 - 10.71634266j,
        ),
    },
    "cruise_1p25": {
        "speed": 11.375,
        "A": np.array([
            [-0.18009818, -9.80665, 0.0, 0.0],
            [0.15202331, 0.0, 0.0, 0.0],
            [-0.15202331, 0.0, 0.0, 1.0],
            [0.0, 0.0, -225.0, -13.5],
        ]),
        "reference_eigenvalues": (
            -0.09004909 + 1.21767424j,
            -0.09004909 - 1.21767424j,
            -6.75 + 13.39542832j,
            -6.75 - 13.39542832j,
        ),
    },
    "provenance": "ME 415 reduced-order teaching model; not measured aircraft data",
}


#: Simple two-engine thrust-lapse and TSFC model for the 787 exercises.
#: COURSE MODEL, not a Boeing or GE engine deck. The sea-level
#: thrust is tied to the published pair of 285 kN takeoff ratings; the lapse
#: exponents, Mach factor, and TSFC are conceptual-design estimates.
B787_ENGINE_MODEL = {
    "sea_level_total_thrust": 570_000.0,  # N, two engines
    "sea_level_density": 1.225,  # kg/m^3
    "density_exponent": 0.7,
    "mach_slope": 0.25,
    "minimum_mach_factor": 0.65,
    "weight_tsfc_per_hour": 0.53,
    "weight_tsfc_per_second": 0.53 / 3600.0,
    "mass_tsfc_kg_per_N_hour": 0.53 / 9.80665,
    "thrust_valid_mach": (0.20, 0.90),
    "thrust_valid_geometric_altitude": (0.0, 13_000.0),
    "tsfc_valid_mach": (0.70, 0.90),
    "tsfc_valid_geometric_altitude": (8_000.0, 13_000.0),
    "provenance": (
        "ME 415 conceptual-design estimate; sea-level rating from the GEnx-1B "
        "published takeoff thrust, lapse and TSFC not manufacturer data"
    ),
}


#: Synthetic LiPo cell curves for battery circuit exercises. COURSE MODEL.
#: Values are deliberately low resolution and should be linearly interpolated;
#: they are not a characterization of a particular cell chemistry or product.
#: ``resistance_multiplier`` scales each catalog battery's mid-SOC
#: ``cell_resistance`` value.
BATTERY_SOC_REFERENCE = {
    "soc": np.array([0.0, 0.10, 0.20, 0.50, 0.80, 1.0]),
    "open_circuit_voltage_per_cell": np.array([3.00, 3.40, 3.60, 3.75, 3.90, 4.20]),
    "resistance_multiplier": np.array([2.50, 1.80, 1.30, 1.00, 1.00, 1.10]),
    "usable_soc": (0.20, 0.90),
    "interpolation": "linear; raise outside 0 <= SOC <= 1",
    "provenance": "ME 415 synthetic teaching curve; not measured cell data",
}


def b787_thrust_available(rho, mach, altitude=None, *, validate=True):
    """Total thrust available from both engines in the course 787 model.

    Parameters
    ----------
    rho : float or array_like
        Atmospheric density, kg/m^3.
    mach : float or array_like
        Flight Mach number.
    altitude : float or array_like, optional
        Geometric altitude, m.  Supply it when ``validate=True`` so both stated
        thrust-model validity limits can be checked.
    validate : bool
        If true, raise ``ValueError`` outside Mach 0.20--0.90 or geometric
        altitude 0--13 km. If altitude is omitted, only Mach is checked.

    Returns
    -------
    float or ndarray
        Total thrust available, N.

    Notes
    -----
    This smooth lapse law is for a performance-model exercise. It is not a
    proprietary engine deck or a certification-level takeoff model, and it
    should not be extrapolated outside the stated subsonic box.
    """
    model = B787_ENGINE_MODEL
    rho_arr = np.asarray(rho, dtype=float)
    mach_arr = np.asarray(mach, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError("density must be positive")
    if validate:
        lo_m, hi_m = model["thrust_valid_mach"]
        if np.any((mach_arr < lo_m) | (mach_arr > hi_m)):
            raise ValueError(f"course 787 model is valid only for {lo_m} <= Mach <= {hi_m}")
        if altitude is not None:
            altitude_arr = np.asarray(altitude, dtype=float)
            lo_h, hi_h = model["thrust_valid_geometric_altitude"]
            if np.any((altitude_arr < lo_h) | (altitude_arr > hi_h)):
                raise ValueError(
                    f"course 787 model is valid only from {lo_h:g} to {hi_h:g} m"
                )
    mach_factor = np.maximum(
        model["minimum_mach_factor"], 1.0 - model["mach_slope"] * mach_arr
    )
    thrust = (
        model["sea_level_total_thrust"]
        * (rho_arr / model["sea_level_density"]) ** model["density_exponent"]
        * mach_factor
    )
    return float(thrust) if thrust.ndim == 0 else thrust


# --- solver verification ----------------------------------------------------

#: AVL results for the simple wing used throughout the VLM test suite.
#: PUBLISHED (AVL output, via the VortexLattice.jl test suite).
#:
#: Geometry: two sections, ``xle = [0, 0.4]``, ``yle = [0, 7.5]``,
#: ``zle = [0, 0]``, ``chord = [2.2, 1.8]`` divided by ``cos(theta)``,
#: ``theta = 2 deg``, no dihedral.  ``Sref = 30``, ``cref = 2``, ``bref = 15``,
#: moment reference at ``x = 0.5``, ``Vinf = 1``.  12 spanwise panels.
AVL_SIMPLE_WING = {
    "geometry": {
        "xle": (0.0, 0.4),
        "yle": (0.0, 7.5),
        "zle": (0.0, 0.0),
        "chord": (2.2, 1.8),
        "theta_deg": (2.0, 2.0),
        "Sref": 30.0,
        "cref": 2.0,
        "bref": 15.0,
        "rref": (0.5, 0.0, 0.0),
        "Vinf": 1.0,
        "ns": 12,
        "nc": 1,
        "note": "divide chord by cos(theta): AVL measures chord along x",
    },
    # uniform spanwise spacing, alpha = 1 deg
    "uniform_alpha1": {
        "CL": 0.24324, "CD": 0.00243, "CDi_trefftz": 0.00245, "Cm": -0.02252
    },
    # cosine spanwise spacing, alpha = 1 deg
    "cosine_alpha1": {
        "CL": 0.23744, "CD": 0.00254, "CDi_trefftz": 0.00243, "Cm": -0.02165
    },
    # uniform, alpha = 8 deg
    "uniform_alpha8": {
        "CL": 0.80348, "CD": 0.02651, "CDi_trefftz": 0.02696, "Cm": -0.07399
    },
    # uniform, alpha = 1 deg, beta = 15 deg
    "sideslip15": {
        "CL": 0.22695, "CD": 0.00227, "CDi_trefftz": 0.0022852,
        "Cm": -0.02101, "Cl": -0.00644, "Cn": 0.00012,
    },
    # stability derivatives, uniform, alpha = 1 deg (chord NOT divided by cos)
    "derivatives_alpha1": {
        "CLa": 4.638088, "Cma": -0.429247, "Clb": -0.025749, "Cnb": 0.000466,
        "Clp": -0.518725, "Cmq": -0.517094, "Clr": 0.064243,
        "Cnp": -0.019846, "Cnr": -0.000898,
    },
    "source": (
        "AVL, via the VortexLattice.jl test suite "
        "(https://github.com/byuflowlab/VortexLattice.jl)"
    ),
}


# --- airfoils ---------------------------------------------------------------

#: NeuralFoil output for a NACA 2412, recorded so a change in this package can
#: be told from a change in physics.  **TOOL** output, not truth.
#:
#: Conditions: UIUC ``naca2412`` coordinates, ``model_size="xlarge"``, lift
#: slope fitted by least squares over the stated range in degrees, reported per
#: radian.
#:
#: The headline result is the first entry: **6.28 per radian at Re = 3e6**,
#: which is ``2*pi`` to three digits, and it is robust -- unchanged across
#: NeuralFoil's medium, large, xlarge and xxlarge models and across coordinate
#: refinements.
#:
#: The three low-Reynolds entries illustrate an important ambiguity. They are three fits to
#: **the same curve** and they disagree by roughly 40%, because the curve has a
#: steep segment near zero lift and is simply not straight there.  Which one is
#: "the lift curve slope" is a question about the airfoil, not about the
#: software.  Note also that ``analysis_confidence`` stays above 0.95 through
#: that whole region: the tool is confident exactly where the three fits
#: disagree.
NEURALFOIL_NACA2412 = {
    "cl_alpha_per_rad": {
        ("Re3e6", -2.0, 8.0): 6.28,
        ("Re1e5", -2.0, 4.0): 7.87,
        ("Re1e5", 1.0, 4.0): 5.77,
        ("Re1e5", -2.0, 8.0): 6.35,
    },
    "confidence_Re1e5": {14.0: 0.70, 16.0: 0.32},
    "model_size": "xlarge",
    "tolerance": 0.15,
    "source": "computed with this package; NOT an independent reference",
}

#: NACA 2412 section characteristics near ``Re = 3e6``, read from the published
#: charts.  **PUBLISHED**, and given as **ranges** because reading a printed
#: chart is not a three-digit operation.  Check against the source before you
#: cite any of it.
#:
#: These ranges show which outputs deserve more trust. The lift slope and the
#: zero-lift angle are easier; ``cl_max`` is
#: the hard one, because it depends on a stall the surrogate smooths.
PUBLISHED_NACA2412 = {
    "alpha_L0_deg": (-2.10, -1.90),
    "cl_alpha_per_deg": (0.100, 0.110),
    "cd_min": (0.0058, 0.0068),
    "cl_max": (1.55, 1.70),
    "source": (
        "Abbott and von Doenhoff, Theory of Wing Sections, "
        "NACA 2412 section data. Values read from the charts; verify against "
        "the source before citing."
    ),
    "note": "ranges, not point values -- chart-reading precision",
}

#: How NeuralFoil actually compares against the ranges above, at ``Re = 3e6``.
#: **Recorded, not corrected.**  Two of the four land outside the published box,
#: which identifies the quantities for which the tool should be treated most
#: cautiously:
#:
#: * ``cl_alpha`` -- inside the box.  Trust it.
#: * ``alpha_L0`` -- about 0.15 deg more negative than published.  Small, and in
#:   a quantity you can measure to about that precision anyway.
#: * ``cd_min`` -- about 12% **below** the published minimum.  The surrogate is
#:   optimistic about minimum drag, so a drag buildup resting on it is
#:   optimistic too.
#: * ``cl_max`` -- roughly 6% **above** the published maximum, and worse than
#:   that suggests: a surrogate that never stalls sharply hands you a maximum
#:   that depends on how far you swept.  Define your criterion explicitly.
NEURALFOIL_VS_PUBLISHED_NACA2412 = {
    "alpha_L0_deg": -2.22,
    "cl_alpha_per_deg": 0.1077,
    "cd_min": 0.0052,
    "cl_max_swept_to_20deg": 1.78,
    "verdict": {
        "cl_alpha": "inside the published range",
        "alpha_L0": "0.15 deg low, acceptable",
        "cd_min": "12% below published -- optimistic",
        "cl_max": "6% above published, and sweep-dependent -- do not trust",
    },
    "conditions": "UIUC naca2412 coordinates, model_size='xlarge', Re = 3e6",
}

#: Where the measured low-Reynolds-number airfoil polars live. A comparison
#: against one of these matters because agreement at high ``Re`` tells you
#: nothing about the regime a model airplane actually flies in.
#:
#: The data is published as PDF volumes rather than machine-readable files, so
#: it is not bundled -- read the value off the plot and say that you did.
#: ``sd7037`` is bundled as coordinates and appears in these volumes, which
#: makes it the natural choice.
UIUC_LOW_SPEED_AIRFOIL_DATA = {
    "index": "https://m-selig.ae.illinois.edu/uiuc_lsat.html",
    "volumes": (
        "https://m-selig.ae.illinois.edu/uiuc_lsat/Low-Speed-Airfoil-Data-V1.pdf",
        "https://m-selig.ae.illinois.edu/uiuc_lsat/Low-Speed-Airfoil-Data-V2.pdf",
        "https://m-selig.ae.illinois.edu/uiuc_lsat/Low-Speed-Airfoil-Data-V3.pdf",
        "https://m-selig.ae.illinois.edu/uiuc_lsat/Low-Speed-Airfoil-Data-V5.pdf",
    ),
    "suggested_section": "sd7037",
    "source": "Selig et al., Summary of Low-Speed Airfoil Data, UIUC",
}


# --- whole-aircraft reference numbers --------------------------------------

#: The existing course solution for the simplified DC-3 wing at 93 m/s and
#: 11,000 kg, on the 79.75 m^2 trapezoidal planform.  PUBLISHED (course).
#:
#: These viscous-drag entries are WING ONLY. A whole-aircraft comparison adds the same nominal
#: 1.6 m^2 non-wing drag area and non-wing interaction model to both methods.
#: The gap between the handbook wing drag and the strip integration -- 2191 N
#: against 1685 N -- is the assignment.  Note that the strip integration lands
#: close to XFLR5's independent 1713 N, which is a second-method rung the
#: handbook method does not get.
DC3_COURSE_SOLUTION = {
    "induced_drag": 1150.0,  # N, inviscid
    "viscous_drag_handbook": 2191.0,  # N, Method A
    "viscous_drag_strip": 1685.0,  # N, Method B
    "viscous_drag_xflr5": 1713.0,  # N, independent check on Method B
    "e_inv": 0.977,  # inviscid span efficiency of the simplified planform
    "nonwing_drag_area_nominal": 1.6,  # m^2, COURSE MODEL
    "nonwing_drag_area_range": (1.2, 2.0),  # m^2, COURSE MODEL
    "conditions": {
        "mass": 11000.0,
        "speed": 93.0,
        "altitude": 10000.0 * 0.3048,
        "Sref": 79.75,
    },
    "source": "existing ME 415 course solution",
}

#: Saturn V S-IC values that R 1's sizing loop should land near.  PUBLISHED.
#: Treat as the answer key, not as an input.
SATURN_V_S_IC_TARGETS = {
    "height": 42.1,  # m
    "diameter": 10.06,  # m
    "propellant_mass": 2_077_000.0,  # kg
    "dry_mass": 130_000.0,  # kg
    "source": "NASA, Saturn V News Reference",
}


# --- bounds worth carrying in your head ------------------------------------

#: Cruise parasitic drag coefficient plausibility box for the course widebody
#: buildup.  COURSE MODEL, not a direct published measurement of the 787.
#: Use it to judge whether a buildup is plausible and, if not, which geometry
#: or omitted-drag assumption may be responsible.
CDP_WIDEBODY_CRUISE = (0.017, 0.021)

#: Wetted-area ratio plausibility box for a modern widebody.  COURSE MODEL.
SWET_SREF_WIDEBODY = (6.0, 7.0)

#: Peak efficiency these small outrunner motors actually achieve.  PUBLISHED
#: range.  A model returning 95% has a sign error or a missing loss term.
MOTOR_PEAK_EFFICIENCY_BAND = (0.70, 0.85)

#: Total electrical-to-propulsive chain efficiency at a good operating point.
#: PUBLISHED range.  If you get 0.85 you have double-counted something.
CHAIN_EFFICIENCY_BAND = (0.45, 0.60)
