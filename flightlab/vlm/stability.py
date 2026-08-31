"""Stability and body-axis derivatives.

Ported from VortexLattice.jl (``src/stability.jl``).

Normalization -- read this before assembling a state matrix
-----------------------------------------------------------
:func:`stability_derivatives` returns derivatives **in the stability frame**,
with respect to:

* ``alpha``, ``beta`` -- per **radian**;
* ``p``, ``q``, ``r`` -- per **non-dimensional rate**, where
  ``phat = p*b/(2V)``, ``qhat = q*c/(2V)``, ``rhat = r*b/(2V)``.

Those are the AVL conventions and the ones most textbooks tabulate, but the
factor of two and the choice of ``b`` versus ``c`` are exactly where
dimensionalizing goes wrong.  To get a dimensional derivative, multiply by
``2V/b`` (for ``p`` and ``r``) or ``2V/c`` (for ``q``).

Moment coefficients are normalized by ``(b, c, b)`` for ``(Cl, Cm, Cn)`` and
carry the flight-dynamics sign convention: positive ``Cl`` rolls right wing
down, positive ``Cn`` yaws nose right.

.. warning::

   A surface solved with ``symmetric=True`` contributes **exactly zero** to
   ``CY``, ``Cl`` and ``Cn`` and to all of their derivatives, because the mirror
   image cancels them.  So ``Cl_beta``, ``Cn_beta``, ``Cl_p``, ``Cn_r`` and
   friends all come back as ``0.0`` -- not small, exactly zero.  For any lateral
   or roll/yaw work (including dihedral/fin sweeps and Dutch-roll or spiral
   modes), build a **mirrored** geometry with ``wing_to_grid(mirror=True)`` and
   pass ``symmetric=False``.  Longitudinal derivatives are identical either way.
"""

from __future__ import annotations

import numpy as np

from .freestream import body_to_stability_alpha, freestream_velocity
from .nearfield import DERIVS, body_forces_derivatives
from .system import System

__all__ = ["stability_derivatives", "body_derivatives"]


def stability_derivatives(system: System):
    """Force and moment derivatives in the stability frame.

    Parameters
    ----------
    system : System
        Must have been solved with ``derivatives=True``.

    Returns
    -------
    dCF : dict
        Keys ``'alpha'``, ``'beta'``, ``'p'``, ``'q'``, ``'r'``; each value is a
        3-vector ``(dCD, dCY, dCL)`` in the stability frame.
    dCM : dict
        Same keys; each value is ``(dCl, dCm, dCn)``.

    Examples
    --------
    Neutral point, in mean chords aft of the moment reference point::

        dCF, dCM = stability_derivatives(system)
        x_np = ref.r[0] - dCM["alpha"][1] / dCF["alpha"][2] * ref.c
    """
    CFb, CMb, dCFb, dCMb = body_forces_derivatives(system)
    fs, ref = system.freestream, system.reference
    lengths = ref.lengths

    # restore the reference lengths so the moment vector rotates as a vector
    CMb = CMb * lengths
    dCMb = {k: v * lengths for k, v in dCMb.items()}

    R, R_a = body_to_stability_alpha(fs)

    CFs = R @ CFb
    CMs = R @ CMb
    dCFs = {k: R @ v for k, v in dCFb.items()}
    dCMs = {k: R @ v for k, v in dCMb.items()}
    # rotating into the stability frame makes the alpha derivative pick up the
    # rotation matrix's own alpha dependence
    dCFs["alpha"] = dCFs["alpha"] + R_a @ CFb
    dCMs["alpha"] = dCMs["alpha"] + R_a @ CMb

    # convert the body-frame rate derivatives to stability-frame rates:
    # Omega_body = R.T @ Omega_stability, so d(Omega_body)/d(Omega_stab) = R.T
    Omega_s = R @ fs.Omega
    rate_body = ("p", "q", "r")

    def to_stability_rate(d, row):
        return sum(d[rate_body[m]] * R[row, m] for m in range(3))

    dCFs_rate = [to_stability_rate(dCFs, row) for row in range(3)]
    dCMs_rate = [to_stability_rate(dCMs, row) for row in range(3)]

    # a body rotating at fixed stability-frame rate changes its body rates as
    # alpha changes, which feeds back into the alpha derivative
    dOmega_b_dalpha = R_a.T @ Omega_s
    extra_F = sum(dCFs[rate_body[m]] * dOmega_b_dalpha[m] for m in range(3))
    extra_M = sum(dCMs[rate_body[m]] * dOmega_b_dalpha[m] for m in range(3))

    # non-dimensional rate scalings
    scale = (2.0 * ref.V / ref.b, 2.0 * ref.V / ref.c, 2.0 * ref.V / ref.b)

    dCF = {
        "alpha": dCFs["alpha"] + extra_F,
        "beta": dCFs["beta"],
        "p": dCFs_rate[0] * scale[0],
        "q": dCFs_rate[1] * scale[1],
        "r": dCFs_rate[2] * scale[2],
    }
    dCM = {
        "alpha": (dCMs["alpha"] + extra_M) / lengths,
        "beta": dCMs["beta"] / lengths,
        "p": dCMs_rate[0] * scale[0] / lengths,
        "q": dCMs_rate[1] * scale[1] / lengths,
        "r": dCMs_rate[2] * scale[2] / lengths,
    }
    return dCF, dCM


def body_derivatives(system: System):
    """Force and moment derivatives in the body frame.

    Derivatives are with respect to the body velocity components ``u``, ``v``,
    ``w`` (per m/s) and the body rates ``p``, ``q``, ``r`` (per rad/s), which is
    the form a linearized equation of motion wants directly.

    Parameters
    ----------
    system : System
        Must have been solved with ``derivatives=True``.

    Returns
    -------
    dCF, dCM : dict
        Keys ``'u'``, ``'v'``, ``'w'``, ``'p'``, ``'q'``, ``'r'``.
    """
    CF, CM, dCF, dCM = body_forces_derivatives(system)
    u, v, w = freestream_velocity(system.freestream)

    u2, v2, w2 = u * u, v * v, w * w
    a_u = -w / (u2 + w2)
    a_w = u / (u2 + w2)
    b_u = -v / (u2 + v2)
    b_v = u / (u2 + v2)

    out_F = {
        "u": dCF["alpha"] * a_u + dCF["beta"] * b_u,
        "v": dCF["beta"] * b_v,
        "w": dCF["alpha"] * a_w,
        "p": dCF["p"],
        "q": dCF["q"],
        "r": dCF["r"],
    }
    out_M = {
        "u": dCM["alpha"] * a_u + dCM["beta"] * b_u,
        "v": dCM["beta"] * b_v,
        "w": dCM["alpha"] * a_w,
        "p": dCM["p"],
        "q": dCM["q"],
        "r": dCM["r"],
    }
    return out_F, out_M
