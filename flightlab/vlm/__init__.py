"""``flightlab.vlm`` -- a steady vortex lattice method.

A Python port of the steady solver in `VortexLattice.jl
<https://github.com/byuflowlab/VortexLattice.jl>`_ (BYU FLOW Lab).  Validated
against the same AVL reference cases as the original; see
``tests/test_vlm_avl.py``.

Everything here is **inviscid** and knows nothing about airfoils beyond their
camber line.  That is not a limitation to work around -- it is the seam the
course is built on.  Section drag comes from :mod:`flightlab.foil`, and putting the
two together is the assignment.

What is ported
--------------
``steady_analysis``, ``body_forces``, ``far_field_drag``,
``stability_derivatives``, ``body_derivatives``,
``lifting_line_coefficients``, ``get_surface_properties``,
``wing_to_grid``, ``grid_to_surface_panels``, ``lifting_line_geometry``, and
the ``Uniform``/``Sine``/``Cosine`` spacing schemes.

What is not
-----------
The unsteady solver, free wakes, the nonlinear viscous-coupled analysis, rotor
geometry helpers, OpenVSP import and the VTK writers.

Units and angles
----------------
SI throughout: metres, m/s, m^2.  ``Freestream`` stores ``alpha`` and ``beta``
in **radians**, matching the Julia package; use
:meth:`Freestream.from_degrees` at the boundary.  ``wing_to_grid`` takes twist
and dihedral in **radians** for the same reason.  Rotation rates are rad/s in
the standard flight-dynamics sign convention.

Two traps worth knowing before you start
----------------------------------------
**``symmetric=True`` zeroes the lateral results.**  It halves the panel count by
using a mirror image, and the mirror image cancels ``CY``, ``Cl`` and ``Cn``
exactly -- so every roll and yaw derivative comes back as ``0.0``.  Use it for
longitudinal work; for anything lateral, build a mirrored geometry with
``wing_to_grid(..., mirror=True)`` and pass ``symmetric=False``.

**Induced drag from ``far_field_drag``, not from ``body_forces``.**  The
Trefftz-plane value is far less sensitive to how the wing was panelled than the
near-field pressure drag is. It is still not insensitive, so check panel
convergence on unfamiliar geometry.

Quick start
-----------
::

    import numpy as np
    from flightlab.vlm import (Reference, Freestream, Stability, Cosine,
                           wing_to_grid, steady_analysis, body_forces,
                           far_field_drag)

    b, S = 15.0, 30.0
    grid, ratios = wing_to_grid(
        xle=[0.0, 0.4], yle=[0.0, b / 2], zle=[0.0, 0.0],
        chord=[2.2, 1.8], theta=np.radians([2.0, 2.0]), phi=[0.0, 0.0],
        ns=24, nc=1, spacing_s=Cosine())

    ref = Reference(S=S, c=2.0, b=b, r=[0.5, 0.0, 0.0], V=1.0)
    fs = Freestream.from_degrees(Vinf=1.0, alpha=1.0)

    system = steady_analysis([grid], ref, fs, symmetric=True, ratios=[ratios])
    CF, CM = body_forces(system, frame=Stability())
    CD, CY, CL = CF
    CDi = far_field_drag(system)
    e_inv = CL**2 / (np.pi * b**2 / S * CDi)
"""

from .analyses import steady_analysis
from .farfield import far_field_drag, trefftz_panels
from .freestream import Freestream
from .geometry import (
    AbstractSpacing,
    Cosine,
    Sine,
    Uniform,
    grid_to_surface_panels,
    lifting_line_geometry,
    rotate,
    translate,
    wing_to_grid,
)
from .induced import influence_coefficients, induced_velocity
from .nearfield import (
    body_forces,
    body_forces_derivatives,
    lifting_line_coefficients,
    near_field_forces,
)
from .panel import Surface
from .reference import Body, Reference, Stability, Wind
from .stability import body_derivatives, stability_derivatives
from .system import PanelProperties, System, get_surface_properties

__all__ = [
    # geometry
    "wing_to_grid",
    "grid_to_surface_panels",
    "lifting_line_geometry",
    "translate",
    "rotate",
    "Surface",
    "AbstractSpacing",
    "Uniform",
    "Sine",
    "Cosine",
    # conditions
    "Reference",
    "Freestream",
    "Body",
    "Stability",
    "Wind",
    # solve
    "steady_analysis",
    "System",
    "PanelProperties",
    "get_surface_properties",
    # results
    "body_forces",
    "body_forces_derivatives",
    "far_field_drag",
    "lifting_line_coefficients",
    "stability_derivatives",
    "body_derivatives",
    # lower level
    "near_field_forces",
    "influence_coefficients",
    "induced_velocity",
    "trefftz_panels",
]
