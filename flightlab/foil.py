"""``flightlab.foil`` -- section aerodynamics from NeuralFoil.

NeuralFoil is a neural network trained on a very large number of XFOIL runs.
It is fast, smooth, vectorized, and **it always returns an answer**.  That last
property is what makes this course's sweeps affordable and it is also the single
most dangerous thing about it: a solver that fails to converge is telling you
something, and a surrogate that has left its training data is not.

So every call returns ``confidence`` alongside the coefficients.  Keep it.  Do
not average it away, do not drop it.  And note what it does and does not mean:
it tells you whether the network has seen cases like this one.  It does not tell
you whether the number you extracted from its output means what you think it
means.  Those are different failures and only one of them has a warning
attached.

Units and angles
----------------
``alpha`` is in **degrees**.  ``Re`` is the chord Reynolds number.  Lengths are
normalized by chord.  Coefficients come back non-dimensional, with ``cm`` about
the quarter chord.

Compressibility
---------------
NeuralFoil is incompressible -- it has no Mach input at all.  ``mach`` here
applies a **Prandtl-Glauert** correction to ``cl`` and ``cm``
(divide by ``sqrt(1 - M^2)``) and leaves ``cd`` untouched.  That is a linear
subsonic correction with **no drag rise in it**: it will not tell you anything
about the 787 at Mach 0.85 beyond the lift-curve steepening.  Transonic drag
rise belongs in the separate empirical compressibility model, not this module.
``mach=0`` (the default) applies no correction at all, which is the right setting
for every low-speed case in the
course.

Examples
--------
::

    from flightlab import foil

    s = foil.load("naca2412")
    r = foil.aero(s, alpha=[0, 2, 4, 6], Re=3e6)
    r["cl"], r["cd"], r["cm"], r["confidence"]

    # trip the boundary layer forward on the upper surface
    dirty = foil.aero(s, alpha=4.0, Re=1e5, xtr_upper=0.05)
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "Section",
    "load",
    "naca4",
    "from_coordinates",
    "from_dat_text",
    "available",
    "aero",
    "polar",
    "DATA_DIR",
]

DATA_DIR = Path(__file__).parent / "data" / "airfoils"

#: NeuralFoil model size.  ``"xlarge"`` is NeuralFoil's own default and costs
#: essentially nothing here (>100k points/s), so it is the default.  The
#: course's headline verification result -- a NACA 2412 lift slope of 2*pi at
#: Re = 3e6 -- is unchanged across "medium", "large", "xlarge" and "xxlarge".
DEFAULT_MODEL_SIZE = "xlarge"


# --- sections ---------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """An airfoil section: a name and a closed coordinate loop.

    Attributes
    ----------
    name : str
        Human-readable name.
    coordinates : ndarray, shape (n, 2)
        ``(x/c, z/c)`` ordered the standard way: trailing edge, forward over
        the upper surface to the leading edge, then aft along the lower surface
        back to the trailing edge.
    source : str
        Where the coordinates came from, for citing in a report.
    """

    name: str
    coordinates: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        xy = np.asarray(self.coordinates, dtype=float).reshape(-1, 2)
        object.__setattr__(self, "coordinates", xy)

    @property
    def x(self) -> np.ndarray:
        """Chordwise coordinates, ``x/c``."""
        return self.coordinates[:, 0]

    @property
    def z(self) -> np.ndarray:
        """Vertical coordinates, ``z/c``."""
        return self.coordinates[:, 1]

    @property
    def thickness(self) -> float:
        """Maximum thickness, ``t/c``."""
        xu, zu, zl = self._upper_lower()
        return float(np.max(zu - zl))

    @property
    def camber(self) -> float:
        """Maximum camber, ``z/c`` of the mean line."""
        xu, zu, zl = self._upper_lower()
        return float(np.max(np.abs(0.5 * (zu + zl))))

    def _upper_lower(self, n=201):
        """Interpolate upper and lower surfaces onto a common x grid."""
        x, z = self.x, self.z
        ile = int(np.argmin(x))
        xu, zu = x[: ile + 1][::-1], z[: ile + 1][::-1]  # LE -> TE
        xl, zl = x[ile:], z[ile:]
        xq = np.linspace(max(xu[0], xl[0]), min(xu[-1], xl[-1]), n)
        return xq, np.interp(xq, xu, zu), np.interp(xq, xl, zl)

    def camber_line(self, x=None):
        """Mean camber line ``z/c`` at chordwise stations ``x``.

        This is the only thing an inviscid vortex lattice sees of an airfoil,
        so it is what you pass to ``wing_to_grid``'s ``fc`` argument.

        Parameters
        ----------
        x : array_like, optional
            Stations ``x/c`` in ``[0, 1]``.  Defaults to a 201-point grid.

        Returns
        -------
        x, z : ndarray
        """
        xq, zu, zl = self._upper_lower()
        zc = 0.5 * (zu + zl)
        if x is None:
            return xq, zc
        return np.asarray(x, dtype=float), np.interp(x, xq, zc)

    def camber_function(self):
        """Return ``f(x/c) -> z/c``, ready for ``wing_to_grid(fc=[...])``."""
        xq, zc = self.camber_line()

        def f(xx):
            return np.interp(xx, xq, zc)

        return f

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Section({self.name!r}, {len(self.coordinates)} points, "
            f"t/c={self.thickness:.4f}, camber={self.camber:.4f})"
        )


def _parse_dat(text: str, name_hint: str = ""):
    """Parse a UIUC ``.dat`` file in either Selig or Lednicer format.

    Selig files run the coordinates as one loop from the trailing edge over the
    upper surface and back.  Lednicer files give point counts, then the upper
    surface leading-edge-first, then the lower surface.  Telling them apart is
    the whole job: the Lednicer count line has both values greater than one,
    which no normalized coordinate ever is.
    """
    lines = text.splitlines()
    # the first line is always the name, never coordinates -- several UIUC
    # files carry a thickness in the title ("E212  (10.55%)") that would
    # otherwise parse as a point
    name = lines[0].strip() if lines else name_hint
    rows: List[Optional[np.ndarray]] = []

    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            rows.append(None)  # blank line: a section break in Lednicer files
            continue
        nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 2:
            try:
                rows.append(np.array([float(nums[0]), float(nums[1])]))
                continue
            except ValueError:
                pass
        rows.append(None)

    pts = [r for r in rows if r is not None]
    if len(pts) < 5:
        raise ValueError(f"could not parse airfoil coordinates for {name_hint!r}")

    first = pts[0]
    if first[0] > 1.5 and first[1] > 1.5:
        # Lednicer: first numeric row is the point counts
        n_upper = int(round(first[0]))
        n_lower = int(round(first[1]))
        body = pts[1:]
        if len(body) < n_upper + n_lower:
            raise ValueError(
                f"{name_hint!r}: Lednicer header claims "
                f"{n_upper}+{n_lower} points but only {len(body)} were found"
            )
        upper = np.array(body[:n_upper])
        lower = np.array(body[n_upper : n_upper + n_lower])
        # upper runs LE->TE; reverse it and drop the duplicated leading edge
        coords = np.vstack([upper[::-1], lower[1:]])
    else:
        coords = np.array(pts)

    return name.strip(), coords


@functools.lru_cache(maxsize=None)
def _load_file(stem: str) -> Section:
    path = DATA_DIR / f"{stem}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"no bundled coordinates for {stem!r}. Available: "
            f"{', '.join(available())}"
        )
    name, coords = _parse_dat(path.read_text(errors="replace"), stem)
    return Section(
        name=name,
        coordinates=coords,
        source=(
            "UIUC Airfoil Coordinates Database, "
            "https://m-selig.ae.illinois.edu/ads/coord_database.html"
        ),
    )


def available() -> List[str]:
    """Names of the airfoil coordinate files bundled with the package."""
    return sorted(p.stem for p in DATA_DIR.glob("*.dat"))


def naca4(designation: str, n=121) -> Section:
    """Generate a NACA 4-digit section from its designation.

    Parameters
    ----------
    designation : str
        Four digits, optionally prefixed with ``naca``, e.g. ``"2412"`` or
        ``"naca0009"``.
    n : int
        Number of points per surface.  Cosine-spaced toward the leading edge.

    Returns
    -------
    Section
    """
    digits = re.sub(r"[^0-9]", "", designation)
    if len(digits) != 4:
        raise ValueError(f"{designation!r} is not a NACA 4-digit designation")
    m = int(digits[0]) / 100.0  # max camber
    p = int(digits[1]) / 10.0  # location of max camber
    t = int(digits[2:]) / 100.0  # thickness

    beta = np.linspace(0.0, np.pi, n)
    x = (1.0 - np.cos(beta)) / 2.0

    yt = (
        5.0
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4  # open trailing edge, the classic definition
        )
    )

    if m > 0.0 and 0.0 < p < 1.0:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        dyc = np.where(
            x < p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p) ** 2 * (p - x),
        )
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)

    theta = np.arctan(dyc)
    xu, zu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, zl = x + yt * np.sin(theta), yc - yt * np.cos(theta)

    coords = np.vstack(
        [
            np.column_stack([xu[::-1], zu[::-1]]),  # TE -> LE over the top
            np.column_stack([xl[1:], zl[1:]]),  # LE -> TE along the bottom
        ]
    )
    return Section(
        name=f"NACA {digits}",
        coordinates=coords,
        source="generated from the NACA 4-digit equations",
    )


def from_coordinates(coordinates, name="custom") -> Section:
    """Wrap a raw coordinate array as a :class:`Section`.

    Parameters
    ----------
    coordinates : array_like, shape (n, 2)
        ``(x/c, z/c)``, trailing edge over the upper surface to the leading
        edge and back along the lower surface.
    name : str
    """
    return Section(name=name, coordinates=np.asarray(coordinates, dtype=float),
                   source="user supplied")


def from_dat_text(text: str, name_hint: str = "custom") -> Section:
    """Parse an airfoil ``.dat`` file that has already been read as text.

    Both the common Selig and Lednicer coordinate layouts are accepted.  This
    is the browser-upload counterpart of the bundled-file loader: callers do
    not need to write an uploaded file to disk before analyzing it.
    """
    name, coordinates = _parse_dat(text, name_hint)
    return Section(name=name or name_hint, coordinates=coordinates, source="user supplied .dat file")


def load(name: str) -> Section:
    """Load a section by name.

    Resolution order: a bundled UIUC coordinate file, then a NACA 4-digit
    designation.

    Parameters
    ----------
    name : str
        A bundled file stem such as ``"sd7037"``, ``"clarky"`` or
        ``"fx62k131"``, or a 4-digit NACA designation such as ``"naca0009"``.

    Returns
    -------
    Section

    Examples
    --------
    >>> load("naca0012").camber < 1e-12
    True
    """
    stem = name.strip().lower().replace(" ", "").replace("-", "")
    if (DATA_DIR / f"{stem}.dat").exists():
        return _load_file(stem)
    digits = re.sub(r"[^0-9]", "", stem)
    if stem.startswith("naca") and len(digits) == 4:
        return naca4(digits)
    if len(digits) == 4 and digits == stem:
        return naca4(digits)
    raise FileNotFoundError(
        f"unknown section {name!r}. Bundled sections: {', '.join(available())}; "
        "or give a NACA 4-digit designation, or use from_coordinates()."
    )


def _as_section(section) -> Section:
    if isinstance(section, Section):
        return section
    if isinstance(section, str):
        return load(section)
    arr = np.asarray(section, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return from_coordinates(arr)
    raise TypeError(
        "section must be a Section, a name, or an (n, 2) coordinate array"
    )


# --- aerodynamics -----------------------------------------------------------


def aero(
    section,
    alpha,
    Re,
    mach=0.0,
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    model_size=DEFAULT_MODEL_SIZE,
) -> Dict[str, np.ndarray]:
    """Section coefficients from NeuralFoil.

    Parameters
    ----------
    section : Section, str, or ndarray
        A :class:`Section`, a name understood by :func:`load`, or an
        ``(n, 2)`` coordinate array.
    alpha : float or array_like
        Angle of attack, **degrees**.
    Re : float or array_like
        Chord Reynolds number.  Broadcasts against ``alpha``.
    mach : float or array_like
        Freestream Mach number.  Applies a Prandtl-Glauert correction to ``cl``
        and ``cm`` only; see the module docstring.  Must be below 1.
    n_crit : float or array_like
        Transition criterion.  9 is a clean wind tunnel; 5-7 is a rougher
        surface or a more turbulent stream.
    xtr_upper, xtr_lower : float or array_like
        Forced transition location, ``x/c``.  ``1.0`` means "let the model
        decide"; smaller values trip the boundary layer there, which is what a
        contaminated leading edge does to a laminar section.
    model_size : str
        NeuralFoil network size: ``"xxsmall"`` .. ``"xxlarge"``.

    Returns
    -------
    dict
        ``cl``, ``cd``, ``cm`` (about the quarter chord), ``confidence``
        (NeuralFoil's ``analysis_confidence``), and ``top_xtr``/``bot_xtr``
        (the transition locations the model actually found).  Every entry has
        the broadcast shape of the inputs.

    Raises
    ------
    ImportError
        If NeuralFoil is not installed.
    ValueError
        If any Mach number is at or above 1, where the correction is invalid.
    """
    try:
        import neuralfoil as nf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "flightlab.foil needs NeuralFoil. Install it with "
            "`pip install neuralfoil`."
        ) from exc

    sec = _as_section(section)

    alpha_b, Re_b, mach_b, nc_b, xu_b, xl_b = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in
          (alpha, Re, mach, n_crit, xtr_upper, xtr_lower))
    )
    shape = alpha_b.shape

    if np.any(mach_b >= 1.0) or np.any(mach_b < 0.0):
        raise ValueError(
            "mach must be in [0, 1); the Prandtl-Glauert correction in this "
            "module is subsonic only and has no drag rise in it"
        )
    if np.any(Re_b <= 0.0):
        raise ValueError("Re must be positive")

    raw = nf.get_aero_from_coordinates(
        coordinates=sec.coordinates,
        alpha=np.atleast_1d(alpha_b).ravel(),
        Re=np.atleast_1d(Re_b).ravel(),
        n_crit=np.atleast_1d(nc_b).ravel(),
        xtr_upper=np.atleast_1d(xu_b).ravel(),
        xtr_lower=np.atleast_1d(xl_b).ravel(),
        model_size=model_size,
    )

    def get(key):
        return np.asarray(raw[key], dtype=float).reshape(shape)

    beta = np.sqrt(1.0 - mach_b**2)
    return {
        "cl": get("CL") / beta,
        "cd": get("CD"),
        "cm": get("CM") / beta,
        "confidence": get("analysis_confidence"),
        "top_xtr": get("Top_Xtr"),
        "bot_xtr": get("Bot_Xtr"),
    }


def polar(
    section,
    alpha=None,
    Re=1e6,
    mach=0.0,
    n_crit=9.0,
    xtr_upper=1.0,
    xtr_lower=1.0,
    model_size=DEFAULT_MODEL_SIZE,
) -> Dict[str, np.ndarray]:
    """Sweep angle of attack at one condition.

    A thin convenience wrapper over :func:`aero` with a default alpha range.
    For the higher-level polar summaries and ``cd(cl, Re)`` interpolator used
    in homework, see :mod:`flightlab.airfoil`.

    Parameters
    ----------
    section : Section, str, or ndarray
    alpha : array_like, optional
        Angles of attack in degrees.  Defaults to ``-8`` to ``16`` in
        0.25-degree steps.
    Re : float
    mach, n_crit, xtr_upper, xtr_lower, model_size
        As in :func:`aero`.

    Returns
    -------
    dict
        As :func:`aero`, plus the ``alpha`` array that was used.
    """
    if alpha is None:
        alpha = np.arange(-8.0, 16.0 + 1e-9, 0.25)
    alpha = np.asarray(alpha, dtype=float)
    out = aero(
        section, alpha, Re, mach, n_crit, xtr_upper, xtr_lower, model_size
    )
    out["alpha"] = alpha
    return out
