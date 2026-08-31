"""``flightlab.props`` -- UIUC propeller data files and a reader.

This module bundles measured propeller data and parses it. It deliberately does
not hide the individual test runs. The interpolated coefficient model and the
motor-propeller torque match are in :mod:`flightlab.propulsion`.

Coefficient definitions
-----------------------
The UIUC data uses the standard propeller definitions, in which ``n`` is in
**revolutions per second**, not rad/s:

.. math::

    J = \\frac{V}{n D}, \\quad
    C_T = \\frac{T}{\\rho n^2 D^4}, \\quad
    C_P = \\frac{P}{\\rho n^3 D^5}, \\quad
    \\eta = \\frac{J\\, C_T}{C_P}

Your motor model works in rad/s.  A factor of ``2*pi`` between the two is the
most common unit error in propeller analysis, and it produces answers that are wrong by a
plausible-looking amount rather than obviously broken ones.  ``Run.omega``
gives you rad/s explicitly so you do not have to remember which way the
conversion goes.

What is measured, and where it is not
------------------------------------
Each propeller has one **static** run (``J = 0``, a sweep over RPM) and several
**advance-ratio** runs, each at a roughly fixed RPM.  Notice what that means
for the edges: static thrust lives at ``J = 0`` and high-speed flight lives near
the largest measured ``J``, and both are near the ends of the data.  Use
:attr:`Propeller.J_range` to find out what fraction of your operating envelope
is extrapolation and must be reported.

Examples
--------
::

    from flightlab import props

    props.available()                  # -> ['apce_10x5', 'apce_10x7', ...]
    p = props.load("apce_10x7")        # RC-1's propeller
    p.diameter                         # 0.254 m
    for run in p.runs:
        run.rpm, run.J, run.CT, run.CP, run.eta
    p.static.rpm, p.static.CT, p.static.CP
    p.J_range                          # (min, max) over all runs
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "Run",
    "StaticRun",
    "Geometry",
    "Propeller",
    "available",
    "load",
    "DATA_DIR",
    "SOURCE",
]

DATA_DIR = Path(__file__).parent / "data" / "props"

SOURCE = (
    "UIUC Propeller Data Site, Selig et al., "
    "https://m-selig.ae.illinois.edu/props/propDB.html"
)

INCH = 0.0254

#: Manufacturer prefixes appearing in UIUC filenames.
FAMILIES = {
    "apce": "APC Electric",
    "apcsf": "APC Sport",
    "apcsp": "APC Sport (pusher)",
    "apccf": "APC Competition",
    "apcff": "APC Free Flight",
    "apc29ff": "APC Free Flight (2.9)",
    "ance": "Aeronaut Electric",
    "ancf": "Aeronaut CAM Folding",
    "da4002": "Dan Allen",
    "da4022": "Dan Allen",
    "da4052": "Dan Allen",
    "grcp": "Graupner CAM Prop",
    "grcsp": "Graupner CAM Slim Prop",
    "grsn": "Graupner Super Nylon",
    "gwsdd": "GWS Direct Drive",
    "gwssf": "GWS Slow Fly",
    "kavfk": "Kavan",
    "kyosho": "Kyosho",
    "ma": "Master Airscrew",
    "mae": "Master Airscrew Electric",
    "mas": "Master Airscrew Scimitar",
    "magf": "Master Airscrew G/F",
    "zin": "Zinger",
}


@dataclass(frozen=True)
class Run:
    """One measured advance-ratio sweep at a roughly constant RPM.

    Attributes
    ----------
    rpm : float
        Nominal rotational speed of the run, rev/min, from the filename.
    J : ndarray
        Advance ratio ``V/(n D)``.
    CT, CP : ndarray
        Thrust and power coefficients (see the module docstring).
    eta : ndarray
        Efficiency as tabulated, ``J*CT/CP``.
    file : str
        Source filename, for citation.
    """

    rpm: float
    J: np.ndarray
    CT: np.ndarray
    CP: np.ndarray
    eta: np.ndarray
    file: str = ""

    @property
    def n(self) -> float:
        """Rotational speed in **revolutions per second**."""
        return self.rpm / 60.0

    @property
    def omega(self) -> float:
        """Rotational speed in **rad/s**.

        Provided so the ``2*pi`` conversion is written down once, here, rather
        than remembered in the middle of a torque balance.
        """
        return self.rpm * 2.0 * np.pi / 60.0

    @property
    def eta_check(self) -> np.ndarray:
        """``J*CT/CP`` recomputed from the columns.

        Comparing this against :attr:`eta` is a useful one-line verification of
        the propeller data.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.CP != 0.0, self.J * self.CT / self.CP, 0.0)

    def __len__(self) -> int:
        return len(self.J)


@dataclass(frozen=True)
class StaticRun:
    """The static (``J = 0``) sweep over RPM.

    Attributes
    ----------
    rpm : ndarray
        Rotational speeds, rev/min.
    CT, CP : ndarray
        Thrust and power coefficients at zero advance ratio.
    file : str
    """

    rpm: np.ndarray
    CT: np.ndarray
    CP: np.ndarray
    file: str = ""

    @property
    def n(self) -> np.ndarray:
        """Rotational speeds in revolutions per second."""
        return self.rpm / 60.0

    @property
    def omega(self) -> np.ndarray:
        """Rotational speeds in rad/s."""
        return self.rpm * 2.0 * np.pi / 60.0

    def __len__(self) -> int:
        return len(self.rpm)


@dataclass(frozen=True)
class Geometry:
    """Measured blade geometry.

    Attributes
    ----------
    r_R : ndarray
        Radial station, ``r/R``.
    c_R : ndarray
        Local chord over radius, ``c/R``.
    beta_deg : ndarray
        Local blade angle, **degrees**.
    file : str
    """

    r_R: np.ndarray
    c_R: np.ndarray
    beta_deg: np.ndarray
    file: str = ""


@dataclass(frozen=True)
class Propeller:
    """A propeller with its measured performance data.

    Attributes
    ----------
    name : str
        The UIUC key, e.g. ``"apce_10x7"``.
    family : str
        Manufacturer prefix, e.g. ``"apce"``.
    manufacturer : str
        Expanded family name.
    diameter_in, pitch_in : float
        Nominal diameter and pitch, **inches**, from the name.
    blades : int
    runs : tuple of Run
        Advance-ratio sweeps, ordered by RPM.
    static : StaticRun, optional
    geometry : Geometry, optional
    source : str
    """

    name: str
    family: str
    diameter_in: float
    pitch_in: float
    blades: int = 2
    runs: Tuple[Run, ...] = ()
    static: Optional[StaticRun] = None
    geometry: Optional[Geometry] = None
    source: str = SOURCE

    @property
    def manufacturer(self) -> str:
        """Expanded manufacturer name."""
        return FAMILIES.get(self.family, self.family)

    @property
    def diameter(self) -> float:
        """Diameter in **metres**."""
        return self.diameter_in * INCH

    @property
    def pitch(self) -> float:
        """Nominal pitch in **metres**."""
        return self.pitch_in * INCH

    @property
    def radius(self) -> float:
        """Radius in metres."""
        return 0.5 * self.diameter

    @property
    def disk_area(self) -> float:
        """Swept disk area, m^2.  Needed for the momentum-theory bound."""
        return np.pi * self.radius**2

    @property
    def pitch_diameter_ratio(self) -> float:
        """``p/D``.  Sets roughly where peak efficiency sits in ``J``."""
        return self.pitch_in / self.diameter_in

    @property
    def rpm_values(self) -> np.ndarray:
        """Nominal RPM of each advance-ratio run."""
        return np.array([r.rpm for r in self.runs])

    @property
    def J_range(self) -> Tuple[float, float]:
        """Smallest and largest advance ratio anywhere in the data.

        Anything outside this is extrapolation. Compare the operating envelope
        against this range and report any uncovered portion.
        """
        if not self.runs:
            raise ValueError(f"{self.name} has no advance-ratio runs")
        lo = min(float(r.J.min()) for r in self.runs)
        hi = max(float(r.J.max()) for r in self.runs)
        return lo, hi

    def run(self, rpm) -> Run:
        """The advance-ratio run whose nominal RPM is closest to ``rpm``."""
        if not self.runs:
            raise ValueError(f"{self.name} has no advance-ratio runs")
        i = int(np.argmin(np.abs(self.rpm_values - float(rpm))))
        return self.runs[i]

    def all_points(self):
        """Every measured advance-ratio point, pooled across runs.

        Returns
        -------
        rpm, J, CT, CP, eta : ndarray
            Flat arrays of equal length.  Pooling the runs is how you build a
            ``CT(J)`` fit that uses all the data; whether pooling is legitimate
            is a question about Reynolds number, and worth a sentence.
        """
        rpm = np.concatenate([np.full(len(r), r.rpm) for r in self.runs])
        J = np.concatenate([r.J for r in self.runs])
        CT = np.concatenate([r.CT for r in self.runs])
        CP = np.concatenate([r.CP for r in self.runs])
        eta = np.concatenate([r.eta for r in self.runs])
        return rpm, J, CT, CP, eta

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<Propeller {self.name}: {self.manufacturer} "
            f"{self.diameter_in:g}x{self.pitch_in:g}, "
            f"{len(self.runs)} runs"
            f"{', static' if self.static is not None else ''}"
            f"{', geometry' if self.geometry is not None else ''}>"
        )


# --- parsing ----------------------------------------------------------------

_SIZE = re.compile(r"^(?P<family>[a-z0-9]+?)_(?P<dia>\d+(?:\.\d+)?)x(?P<pitch>\d+(?:\.\d+)?)")


def _read_table(path: Path, ncols: int) -> np.ndarray:
    """Read a whitespace table, skipping the header and any short rows."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < ncols:
            continue
        try:
            rows.append([float(v) for v in parts[:ncols]])
        except ValueError:
            continue  # the header line
    if not rows:
        raise ValueError(f"no numeric rows found in {path.name}")
    return np.array(rows)


def _decode_size(stem: str):
    """Pull the family and the nominal diameter and pitch out of a filename."""
    m = _SIZE.match(stem)
    if not m:
        raise ValueError(f"cannot decode propeller size from {stem!r}")
    dia = float(m.group("dia"))
    pitch_raw = m.group("pitch")
    pitch = float(pitch_raw)
    # UIUC encodes some fractional pitches without the point: "13x65" is 13x6.5
    if "." not in pitch_raw and pitch > dia:
        pitch = pitch / 10.0
    return m.group("family"), dia, pitch


_KEY = re.compile(r"^([a-z0-9]+_\d+(?:\.\d+)?x\d+(?:\.\d+)?)_")


def available() -> List[str]:
    """Names of the propellers bundled with the package."""
    names = set()
    for path in DATA_DIR.glob("*.txt"):
        m = _KEY.match(path.stem)
        if m:
            names.add(m.group(1))
    return sorted(names)


@functools.lru_cache(maxsize=None)
def load(name: str) -> Propeller:
    """Load a propeller and all of its data files.

    Parameters
    ----------
    name : str
        UIUC key such as ``"apce_10x7"``.  Case and surrounding whitespace are
        ignored.

    Returns
    -------
    Propeller

    Raises
    ------
    FileNotFoundError
        If no bundled files match.
    """
    key = name.strip().lower()
    files = sorted(DATA_DIR.glob(f"{key}_*.txt"))
    if not files:
        raise FileNotFoundError(
            f"no bundled data for propeller {name!r}. "
            f"Available: {', '.join(available())}"
        )

    family, dia, pitch = _decode_size(key)
    runs: List[Run] = []
    static = None
    geometry = None

    for path in files:
        tail = path.stem[len(key) + 1 :]
        if tail == "geom":
            t = _read_table(path, 3)
            geometry = Geometry(
                r_R=t[:, 0], c_R=t[:, 1], beta_deg=t[:, 2], file=path.name
            )
        elif tail.startswith("static"):
            t = _read_table(path, 3)
            order = np.argsort(t[:, 0])
            static = StaticRun(
                rpm=t[order, 0], CT=t[order, 1], CP=t[order, 2], file=path.name
            )
        else:
            m = re.search(r"_(\d+)$", tail)
            if m is None:
                continue
            t = _read_table(path, 4)
            order = np.argsort(t[:, 0])
            runs.append(
                Run(
                    rpm=float(m.group(1)),
                    J=t[order, 0],
                    CT=t[order, 1],
                    CP=t[order, 2],
                    eta=t[order, 3],
                    file=path.name,
                )
            )

    runs.sort(key=lambda r: r.rpm)
    return Propeller(
        name=key,
        family=family,
        diameter_in=dia,
        pitch_in=pitch,
        runs=tuple(runs),
        static=static,
        geometry=geometry,
    )
