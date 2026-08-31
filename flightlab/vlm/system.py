"""System state container and per-panel properties.

Ported from VortexLattice.jl (``src/system.jl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .freestream import Freestream
from .panel import Surface
from .reference import Reference

__all__ = ["PanelProperties", "System", "get_surface_properties"]

#: The solver works at unit density; forces are non-dimensionalized by the
#: reference dynamic pressure before they are returned, so density cancels.
RHO = 1.0


@dataclass
class PanelProperties:
    """Per-panel results for one surface.

    All quantities are non-dimensional.  Arrays are shaped ``(nc, ns)`` for
    scalars and ``(nc, ns, 3)`` for vectors.

    Attributes
    ----------
    gamma : ndarray, shape (nc, ns)
        Panel circulation normalized by the reference velocity, ``Gamma/Vref``.
    velocity : ndarray, shape (nc, ns, 3)
        Local velocity at the bound vortex centre, normalized by ``Vref``.
        Includes the freestream, body rotation, and all induced contributions.
    cfb, cfl, cfr : ndarray, shape (nc, ns, 3)
        Force coefficient on the panel's top (bound), left, and right vortex
        filaments, non-dimensionalized by ``q*Sref``.  These are *panel* forces
        already divided by the whole reference area, so summing them over every
        panel gives the body force coefficient directly.
    """

    gamma: np.ndarray
    velocity: np.ndarray
    cfb: np.ndarray
    cfl: np.ndarray
    cfr: np.ndarray

    @property
    def shape(self) -> tuple:
        """``(nc, ns)``."""
        return self.gamma.shape

    @property
    def cf_total(self) -> np.ndarray:
        """Total force coefficient per panel, ``cfb + cfl + cfr``."""
        return self.cfb + self.cfl + self.cfr


@dataclass
class System:
    """Holds the geometry, the flow condition, and everything the solve produced.

    You normally get one back from :func:`flightlab.vlm.steady_analysis` rather
    than building it yourself.

    Attributes
    ----------
    surfaces : list of Surface
        The panelled lifting surfaces.
    grids : list of ndarray
        The corner grids the surfaces were built from, shape ``(3, nc+1, ns+1)``.
        Needed by :func:`lifting_line_geometry`.
    ratios : list of ndarray
        Panel placement ratios, shape ``(2, nc, ns)``.
    reference : Reference
        Reference quantities used for non-dimensionalization.
    freestream : Freestream
        Flow condition (angles in radians).
    symmetric : list of bool
        Per surface: whether a mirror image was used.
    surface_id : list of int
        Per surface: finite-core grouping.
    trailing_vortices : list of bool
    xhat : ndarray, shape (3,)
        Trailing vortex direction.
    AIC : ndarray, shape (N, N)
        Influence coefficient matrix.
    w : ndarray, shape (N,)
        Normal-velocity right-hand side.
    gamma : ndarray, shape (N,)
        Panel circulations, dimensional (m^2/s).
    dgamma : dict
        Derivatives of ``gamma`` with respect to ``alpha``, ``beta`` (per rad)
        and ``p``, ``q``, ``r`` (per rad/s), if derivatives were requested.
    properties : list of PanelProperties
    dproperties : dict of list of PanelProperties
    """

    surfaces: List[Surface]
    grids: List[np.ndarray] = field(default_factory=list)
    ratios: List[np.ndarray] = field(default_factory=list)
    reference: Optional[Reference] = None
    freestream: Optional[Freestream] = None
    symmetric: Sequence[bool] = ()
    surface_id: Sequence[int] = ()
    trailing_vortices: Sequence[bool] = ()
    xhat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    AIC: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    dgamma: Optional[dict] = None
    properties: Optional[List[PanelProperties]] = None
    dproperties: Optional[dict] = None

    @property
    def nsurf(self) -> int:
        """Number of surfaces."""
        return len(self.surfaces)

    @property
    def npanels(self) -> int:
        """Total number of panels across all surfaces."""
        return int(sum(s.size for s in self.surfaces))

    @property
    def offsets(self) -> np.ndarray:
        """Start index of each surface in the flattened panel ordering."""
        return np.concatenate(([0], np.cumsum([s.size for s in self.surfaces])))

    @property
    def has_derivatives(self) -> bool:
        """Whether freestream derivatives were computed."""
        return self.dproperties is not None

    def surface_gamma(self, isurf: int) -> np.ndarray:
        """Circulation on surface ``isurf``, shaped ``(nc, ns)``."""
        offs = self.offsets
        return self.gamma[offs[isurf] : offs[isurf + 1]].reshape(
            self.surfaces[isurf].shape
        )


def get_surface_properties(system: System) -> List[PanelProperties]:
    """Return the per-panel properties for each surface.

    Parameters
    ----------
    system : System

    Returns
    -------
    list of PanelProperties
        One entry per surface, each holding ``(nc, ns)``-shaped arrays.
    """
    if system.properties is None:
        raise RuntimeError(
            "no panel properties available -- run steady_analysis with "
            "near_field_analysis=True"
        )
    return system.properties
