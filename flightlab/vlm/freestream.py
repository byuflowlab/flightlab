"""Freestream definition and the rotation matrices between reference frames.

Ported from VortexLattice.jl (``src/freestream.jl``).

Angle convention
----------------
``alpha`` and ``beta`` are stored in **radians** inside this class, because the
rotation matrices need radians and mixing units inside a solver is how sign
errors hide.  Every function a human calls at the top level of ``flightlab.vlm``
takes degrees; use :meth:`Freestream.from_degrees` to build one from degrees.

Rotation-rate convention
------------------------
``Omega = (p, q, r)`` uses the standard flight-dynamics convention: positive
``p`` rolls the right wing down, positive ``q`` pitches the nose up, positive
``r`` yaws the nose right.  Internally the solver applies the rotation vector
``(-p, q, -r)`` because the geometric axes have z up while the dynamics
convention has z down.  You do not need to apply that flip yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Freestream"]


@dataclass(frozen=True)
class Freestream:
    """Freestream velocity and body rotation rates.

    Parameters
    ----------
    Vinf : float
        Freestream velocity magnitude, m/s.
    alpha : float
        Angle of attack, **radians**.
    beta : float
        Sideslip angle, **radians**.
    Omega : array_like, shape (3,)
        Body rotation rates ``(p, q, r)`` in rad/s about the reference location
        ``Reference.r``.  Standard dynamics convention (see module docstring).
    """

    Vinf: float = 1.0
    alpha: float = 0.0
    beta: float = 0.0
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "Vinf", float(self.Vinf))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "beta", float(self.beta))
        Omega = np.asarray(self.Omega, dtype=float).reshape(3)
        object.__setattr__(self, "Omega", Omega)

    @classmethod
    def from_degrees(cls, Vinf=1.0, alpha=0.0, beta=0.0, Omega=(0.0, 0.0, 0.0)):
        """Build a :class:`Freestream` from ``alpha`` and ``beta`` in degrees.

        ``Omega`` stays in rad/s -- it is a rate, not an angle a human sketches.
        """
        return cls(Vinf, np.radians(alpha), np.radians(beta), Omega)

    @property
    def alpha_deg(self) -> float:
        """Angle of attack in degrees."""
        return float(np.degrees(self.alpha))

    @property
    def beta_deg(self) -> float:
        """Sideslip angle in degrees."""
        return float(np.degrees(self.beta))

    def replace(self, **kwargs) -> "Freestream":
        """Return a copy with the named fields replaced (radians for angles)."""
        data = {
            "Vinf": self.Vinf,
            "alpha": self.alpha,
            "beta": self.beta,
            "Omega": self.Omega,
        }
        data.update(kwargs)
        return Freestream(**data)


# --- rotation matrices ------------------------------------------------------


def body_to_stability(fs: Freestream) -> np.ndarray:
    """Rotation matrix from the body frame to the stability frame."""
    sa, ca = np.sin(fs.alpha), np.cos(fs.alpha)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def body_to_stability_alpha(fs: Freestream):
    """Body-to-stability rotation matrix and its derivative wrt ``alpha``."""
    sa, ca = np.sin(fs.alpha), np.cos(fs.alpha)
    R = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    R_a = np.array([[-sa, 0.0, ca], [0.0, 0.0, 0.0], [-ca, 0.0, -sa]])
    return R, R_a


def stability_to_wind(fs: Freestream) -> np.ndarray:
    """Rotation matrix from the stability frame to the wind frame."""
    sb, cb = np.sin(fs.beta), np.cos(fs.beta)
    return np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])


def body_to_wind(fs: Freestream) -> np.ndarray:
    """Rotation matrix from the body frame to the wind frame."""
    return stability_to_wind(fs) @ body_to_stability(fs)


# --- velocities -------------------------------------------------------------


def freestream_velocity(fs: Freestream) -> np.ndarray:
    """Freestream velocity vector in the body frame, m/s."""
    sa, ca = np.sin(fs.alpha), np.cos(fs.alpha)
    sb, cb = np.sin(fs.beta), np.cos(fs.beta)
    return fs.Vinf * np.array([ca * cb, -sb, sa * cb])


def freestream_velocity_derivatives(fs: Freestream):
    """Freestream velocity and its derivatives wrt ``alpha`` and ``beta``."""
    sa, ca = np.sin(fs.alpha), np.cos(fs.alpha)
    sb, cb = np.sin(fs.beta), np.cos(fs.beta)
    V = fs.Vinf * np.array([ca * cb, -sb, sa * cb])
    V_a = fs.Vinf * np.array([-sa * cb, 0.0, ca * cb])
    V_b = fs.Vinf * np.array([-ca * sb, -cb, -sa * sb])
    return V, (V_a, V_b)


def rotational_velocity(r, fs: Freestream, ref) -> np.ndarray:
    """Velocity at ``r`` due to body rotation about ``ref.r``.

    Parameters
    ----------
    r : array_like, shape (..., 3)
        Point(s) at which to evaluate the velocity, m.

    Returns
    -------
    ndarray, shape (..., 3)
    """
    r = np.asarray(r, dtype=float)
    dr = r - ref.r
    p, q, rr = fs.Omega
    # flip p and r: geometric axes have z up, the dynamics convention has z down
    Omega = np.array([-p, q, -rr])
    return np.cross(dr, Omega)


def rotational_velocity_derivatives(r, fs: Freestream, ref):
    """Rotational velocity and its derivatives wrt ``(p, q, r)``.

    Returns ``(Vrot, (dV_dp, dV_dq, dV_dr))`` with each array shaped like ``r``.
    """
    r = np.asarray(r, dtype=float)
    dr = r - ref.r
    Vrot = rotational_velocity(r, fs, ref)

    dx, dy, dz = dr[..., 0], dr[..., 1], dr[..., 2]
    zero = np.zeros_like(dx)
    # d/dp of cross(dr, (-p, q, -r)) = cross(dr, (-1, 0, 0))
    Vrot_p = -np.stack([zero, dz, -dy], axis=-1)
    # d/dq = cross(dr, (0, 1, 0))
    Vrot_q = np.stack([-dz, zero, dx], axis=-1)
    # d/dr = cross(dr, (0, 0, -1))
    Vrot_r = -np.stack([dy, -dx, zero], axis=-1)
    return Vrot, (Vrot_p, Vrot_q, Vrot_r)
