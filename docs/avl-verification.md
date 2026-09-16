# Verification against AVL

FlightLab's vortex lattice, trim, stability derivatives, and dynamic modes are
checked against AVL 3.52 (Drela and Youngren's Athena Vortex Lattice). The
checks live in `tests/test_avl_verification.py` and run on every test run
without AVL installed: `tests/data/avl_reference.json` holds AVL's answers,
recorded by `tests/record_avl_reference.py`.

## Running AVL yourself

Build AVL from its source distribution (gfortran plus an X11 library; on a Mac
`brew install gcc libx11` supplies both) and point FlightLab at the executable:

```bash
export FLIGHTLAB_AVL=/path/to/avl
uv run python tests/record_avl_reference.py   # refresh the recorded answers
uv run pytest tests/test_avl_verification.py  # also checks the record is current
```

`flightlab.avl` writes any project as an AVL geometry and mass file and runs
AVL in batch, so a student can cross-check a design too:

```python
from flightlab import avl
from flightlab.project import AircraftProject

project = AircraftProject.load("mydesign.flightlab.json")
result = avl.run(project, alpha=3.0, deflection=0.0, ns=28, nc=4)
print(result.totals["CLtot"], result.derivatives["Cma"], result.derivatives["Cnb"])
```

## What is compared

Two cases, both without bodies because the two codes model fuselages differently.

**The AVL reference wing.** The tapered, twisted wing that VortexLattice.jl and
FlightLab's own `tests/test_vlm_avl.py` use. All stability-axis derivatives
with respect to alpha, sideslip, and the three rates agree to one or two per
cent.

**The verification aircraft.** The blank project's wing, tail, and fin with
three changes that put both codes on the same problem: the tailplane raised
clear of the wing's wake, a 15 mm gap between fin and tailplane roots, and the
geometry origin at the centre of gravity. For it the tests compare:

| quantity | agreement |
|---|---|
| lift, pitch-moment and drag slopes; pitch damping | 1 to 3 % |
| trimmed angle of attack and tail deflection | within 0.2 degree |
| neutral point | within 3 mm |
| side force, directional stiffness, roll damping, yaw damping | 1 to 8 % |
| dihedral effect | 15 % (see below) |
| phugoid and dutch roll frequency and damping; roll subsidence | 2 to 5 % |
| short period trace and determinant | 5 and 10 % |
| apparent mass and inertia of the air | 1 % |

## Choices the comparison depends on

**Trailing-leg forces.** AVL can put Kutta-Joukowski forces on the chordwise
vortex legs lying on each surface. That was its default until version 3.51,
and it is what FlightLab's lattice does. It mainly changes the rolling moment
due to sideslip of a wing at angle of attack (by a factor of eight on the
reference wing). The comparison switches it on; with AVL's new default the
dihedral-effect and roll-rate side-force derivatives would not match.

**Camber.** Both codes see an airfoil only through its camber line. AVL
applies the camber slope at each control point; FlightLab does the same since
September 2026 (before that the camber was built into the panel geometry,
which under-predicted the lift of a NACA 2412 camber line by 13 % at four
chordwise panels). A residual 2 % difference in the camber lift remains, which
is the 0.1 degree difference in trimmed angle of attack.

**Apparent mass.** For a one-kilogram model the air a surface accelerates with
it is a large fraction of the pitch and roll inertia. AVL includes it in its
eigenmode analysis; FlightLab's `analyze_dynamic_stability` includes the same
strip formula. AVL takes the apparent inertia about its geometry origin rather
than the centre of gravity, so the verification aircraft puts the two at the
same point.

**Wake proximity.** With the blank project's tail in its original place, 3 to
4 cm above the wing's wake sheet, the two codes disagree on pitch stiffness by
25 % even though they agree within 2 % once the tail is raised. FlightLab
rotates a twisted section's geometry, AVL only its boundary condition, so the
sheets leave the wing about a centimetre apart, and the downwash at a tail
that close to the sheet is sensitive to that. Neither code is trustworthy for
a tail inside the wake; a real wake rolls up and follows the flow.

**Surface junctions.** A fin whose root lies exactly on the tailplane's root
line makes every lateral derivative depend on the paneling in both codes.
FlightLab uses a finite vortex core of five per cent of local chord between
different surfaces, which converges such junctions; AVL has no core, so the
verification geometry keeps a gap.

## What is not compared

Body (fuselage) increments, the alpha-dot estimate, the profile-drag model,
and the propulsion derivatives have no AVL counterpart. Their plausibility is
covered by the ordinary unit tests.
