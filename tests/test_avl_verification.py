"""Verify the project's lattice, trim, and dynamic modes against AVL 3.52.

``tests/data/avl_reference.json`` holds AVL's answers for the cases in
``avl_cases.py``; ``record_avl_reference.py`` regenerates it from a local AVL.
The comparisons run without AVL.  When ``FLIGHTLAB_AVL`` points at an
executable, one extra test re-runs AVL and checks the recorded file is current.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import avl_cases  # noqa: E402
from flightlab import avl  # noqa: E402

reference = avl_cases.load_reference()


def _rel(ours, theirs, tol, label):
    assert abs(ours - theirs) <= tol * abs(theirs), f"{label}: ours {ours:+.5g}, AVL {theirs:+.5g}"


def test_recorded_reference_has_the_expected_shape():
    assert reference["settings"]["trailing_leg_forces"] is True
    assert set(reference["cases"]) == set(avl_cases.CASES)
    modes = reference["cases"]["verification_aircraft"]["modes"]
    assert len(modes["longitudinal"]) == 3 and len(modes["lateral"]) == 3


def test_simple_wing_derivatives_match_avl_with_trailing_leg_forces():
    """The VortexLattice.jl reference wing.  AVL turned trailing-leg forces off
    by default in 3.51; with them on, as in every earlier AVL and in this
    package, the sideslip and rate derivatives agree closely."""
    ours = avl_cases.ours("simple_wing")["derivatives"]
    theirs = reference["cases"]["simple_wing"]["derivatives"]
    for name, tol in (("CLa", 0.01), ("Cma", 0.02), ("Clp", 0.01), ("Clr", 0.02), ("Cnp", 0.02),
                      ("Clb", 0.03), ("CYp", 0.10), ("Cmq", 0.02), ("CLq", 0.02)):
        _rel(ours[name], theirs[name], tol, name)


def test_verification_aircraft_derivatives_match_avl():
    ours = avl_cases.ours("verification_aircraft")
    theirs = reference["cases"]["verification_aircraft"]
    d, a = ours["derivatives"], theirs["derivatives"]
    # Longitudinal: the camber slope at the control point and the tail out of
    # the wake put both codes on the same problem.
    for name, tol in (("CLa", 0.01), ("Cma", 0.03), ("CLq", 0.01), ("Cmq", 0.01), ("CDa", 0.02)):
        _rel(d[name], a[name], tol, name)
    # Lateral: the fin and dihedral terms agree to a few per cent; the two
    # small roll-rate couplings differ more between codes and are checked on
    # an absolute basis.
    for name, tol in (("CYb", 0.08), ("Cnb", 0.05), ("Clb", 0.15), ("Clp", 0.01), ("Cnr", 0.05),
                      ("Clr", 0.06), ("CYr", 0.06)):
        _rel(d[name], a[name], tol, name)
    for name, tol in (("Cnp", 0.005), ("CYp", 0.04)):
        assert abs(d[name] - a[name]) < tol, (name, d[name], a[name])


def test_verification_aircraft_trims_where_avl_does():
    ours = avl_cases.ours("verification_aircraft")["trim"]
    theirs = reference["cases"]["verification_aircraft"]["trim"]
    assert ours["CL"] == pytest.approx(theirs["CL"], rel=1e-3)
    # The residual camber difference (AVL splines the airfoil file, the
    # project differentiates its camber function) is worth about 0.1 degree.
    assert abs(ours["alpha"] - theirs["alpha"]) < 0.2
    assert abs(ours["deflection"] - theirs["deflection"]) < 0.2
    assert abs(ours["x_np"] - theirs["x_np"]) < 0.003


def test_verification_aircraft_modes_match_avl():
    """Eigenvalues with AVL's own model assumptions: constant profile drag, no
    alpha-dot terms, apparent mass of the air included."""
    ours = avl_cases.split_modes(avl_cases.ours("verification_aircraft")["modes"])
    theirs = avl_cases.split_modes(reference["cases"]["verification_aircraft"]["modes"])
    phugoid, short, dutch, roll, spiral = ours
    phugoid_a, short_a, dutch_a, roll_a, spiral_a = theirs
    _rel(phugoid.imag, phugoid_a.imag, 0.05, "phugoid frequency")
    _rel(phugoid.real, phugoid_a.real, 0.10, "phugoid damping")
    _rel(dutch.imag, dutch_a.imag, 0.05, "dutch roll frequency")
    _rel(dutch.real, dutch_a.real, 0.05, "dutch roll damping")
    _rel(roll, roll_a, 0.05, "roll subsidence")
    assert abs(spiral - spiral_a) < 0.05, ("spiral", spiral, spiral_a)
    # The short period is heavily damped and near the real/complex boundary,
    # where the individual roots move a lot for a small change in the
    # coefficients; its trace and determinant are the robust comparison.
    def trace_det(roots):
        roots = roots if len(roots) == 2 else [roots[0], roots[0].conjugate()]
        return sum(roots).real, (roots[0] * roots[1]).real
    trace, det = trace_det(short)
    trace_a, det_a = trace_det(short_a)
    _rel(trace, trace_a, 0.05, "short-period trace")
    _rel(det, det_a, 0.10, "short-period determinant")


def test_apparent_mass_matches_avl_printout():
    """AVL prints its apparent inertia for the same lump; recorded from its log."""
    added = avl_cases.ours("verification_aircraft")["apparent"]
    assert added["Ixx"] == pytest.approx(0.01794, rel=0.01)
    assert added["Iyy"] == pytest.approx(0.005147, rel=0.01)
    assert added["Izz"] == pytest.approx(0.002512, rel=0.01)


@pytest.mark.skipif(avl.find_avl() is None, reason="no AVL executable (set FLIGHTLAB_AVL)")
def test_recorded_reference_is_current_against_local_avl():
    for name in avl_cases.CASES:
        fresh = avl_cases.theirs(name)
        stored = reference["cases"][name]
        for key, value in fresh["derivatives"].items():
            assert value == pytest.approx(stored["derivatives"][key], rel=1e-3, abs=1e-5), (name, key)
        if "modes" in fresh:
            for family in ("longitudinal", "lateral"):
                flat = [value for pair in fresh["modes"][family] for value in pair]
                flat_stored = [value for pair in stored["modes"][family] for value in pair]
                assert flat == pytest.approx(flat_stored, rel=1e-3, abs=1e-4), family
