# Using a Workbench project from Python

The workbench is the easiest place to enter and check an aircraft. Python is the
better place to repeat an analysis, make a constrained change, compare many
candidates, or use a FlightLab tool that is not exposed in the workbench.

The normal workflow is:

1. build and visually check the baseline aircraft in the workbench;
2. save its `.flightlab.json` file beside the script;
3. load that file, make a fresh copy for each candidate, and change the copy;
4. validate the candidate and run an analysis;
5. collect named outputs and plot or tabulate them.

FlightLab uses SI units. Human-facing angles are in degrees. The low-level
`flightlab.vlm` module is the exception and uses radians.

## Load and inspect a project

```python
from flightlab.project import AircraftProject

project = AircraftProject.load("my-aircraft.flightlab.json")

print([surface.name for surface in project.surfaces])
print([body.name for body in project.bodies])
print([item.name for item in project.masses])
print([case.name for case in project.cases])
print([item.name for item in project.propulsion.propulsors])
print(project.validate())
```

The project is the editable input model. Its principal collections are:

| Input | Location | Important fields |
|---|---|---|
| lifting surfaces | `project.surfaces` | `stations`, `orientation`, `purpose`, `symmetric`, trim-control fields |
| surface stations | `surface.stations` | `x_le`, `y`, `z`, `chord`, `twist_deg`, `airfoil` |
| bodies | `project.bodies` | `length`, `width`, `height`, `diameter`, `x_nose`, `drag_model` |
| user mass rows | `project.masses` | `mass`, `x`, `y`, `z`, distribution and attachment fields |
| flight cases | `project.cases` | `speed`, `altitude`, `load_factor`, drag and transition assumptions |
| reference geometry | `project.reference` | reference mode, surface selection, or manual quantities |
| propulsion installation | `project.propulsion` | battery, state of charge, positions, and `propulsors` |
| structural definition | `project.structure` | surface, spar geometry, allowable, modulus, ultimate factor |

Use the named accessors when a script needs a particular item:

```python
wing = project.surface_named("Main wing")
fuselage = project.body_named("fuselage")
payload = project.mass_named("Payload")
motor = project.propulsor_named("Propulsor 1")
cruise = project.case("Cruise")       # raises KeyError if it is absent

if wing is None or fuselage is None:
    raise KeyError("This script expects 'Main wing' and 'fuselage'")
```

The `*_named` geometry and component accessors return `None` when no match is
found. `project.case(name)` raises `KeyError`, because an analysis cannot run
without the requested case.

## Change geometry or an operating condition

Project input dataclasses are intentionally editable:

```python
wing = project.surface_named("Main wing")
fuselage = project.body_named("fuselage")
if wing is None or fuselage is None:
    raise KeyError("expected geometry is missing")

# Absolute body axes: x aft, y right, z up. Lengths are metres.
wing.stations[-1].chord = 0.120
wing.stations[-1].twist_deg = -2.0
fuselage.length = 0.950
fuselage.width = 0.090
fuselage.height = 0.120

project.case("Cruise").speed = 14.0
project.require_valid()                # raises one message listing input errors
project.save("modified-aircraft.flightlab.json")
```

`require_valid()` is a preflight check. It raises `ValueError` containing all
validation errors; warnings do not make it fail. Project-level aerodynamic
analyses validate internally too, but an explicit check immediately after the
edits identifies a bad candidate before an expensive loop. `project.validate()`
returns the error and warning records when a script wants to inspect them.
`project.save()` writes the current inputs without validating them.

Changing a station changes the full piecewise-linear lifting-surface geometry
used by VLM. A body is not a VLM panel surface; its dimensions feed the
documented empirical body-drag correlations, mass attachments, and approximate
stability corrections.

## Run the project-level analyses

Use the project-level functions for a saved workbench aircraft:

| Question | Function | Main return type |
|---|---|---|
| trimmed aerodynamics, mass/CG, and drag at one case | `run_design_point` | `DesignPoint` |
| whole-aircraft alpha sweep at fixed control deflection | `aircraft_polar` | `AircraftPolar` |
| span load and preliminary two-cap spar sizing | `analyze_structure` | `ProjectStructuralAnalysis` |
| motor/propeller/battery match over speed | `analyze_propulsion` | `PropulsionAnalysis` |
| linear longitudinal and lateral modes | `analyze_dynamic_stability` | `DynamicStability` |

```python
import numpy as np

from flightlab.project_analysis import (
    aircraft_polar,
    analyze_dynamic_stability,
    analyze_propulsion,
    analyze_structure,
    run_design_point,
)

case = project.case("Cruise")
point = run_design_point(project, case)
polar = aircraft_polar(
    project,
    case,
    alpha=np.linspace(-5.0, 15.0, 21),
    trim_deflection=point.trim.trim_deflection,
)
structure = analyze_structure(project, case, load_factor=3.0, speed=18.0)
power = analyze_propulsion(project, case, speed=np.linspace(5.0, 30.0, 26))
modes = analyze_dynamic_stability(project, case)
```

The optional `ns` and `nc` arguments control spanwise and chordwise VLM panel
resolution. Keep them fixed when comparing candidates. Increase them for a
mesh-convergence check, not merely to produce more digits.

## Find and use the outputs

Analysis results are immutable dataclasses with named fields. Use those names,
not tuple positions. For a `DesignPoint` called `point`, the most commonly used
outputs are:

| Quantity | Expression | Units |
|---|---|---|
| trimmed angle of attack | `point.trim.alpha` | deg |
| pitch-control deflection | `point.trim.trim_deflection` | deg |
| lift coefficient | `point.trim.solution.CL` | dimensionless |
| induced drag coefficient | `point.trim.solution.CD_i` | dimensionless |
| total drag coefficient | `point.CD_total` | dimensionless |
| drag force | `point.drag` | N |
| lift-to-drag ratio | `point.lift_to_drag` | dimensionless |
| static margin | `point.trim.static_margin` | chord fraction |
| mass and CG | `point.mass_properties.mass`, `.x_cg`, `.y_cg`, `.z_cg` | kg, m |
| component drag buildup | `point.buildup.rows` or `.table()` | rows or printable text |
| model warnings | `point.warnings` | strings |

```python
print(point.mass_properties.table())
print(point.buildup.table())
print(f"alpha = {point.trim.alpha:.3f} deg")
print(f"CL = {point.trim.solution.CL:.4f}")
print(f"CD = {point.CD_total:.5f}")
print(f"D = {point.drag:.3f} N")
print(f"L/D = {point.lift_to_drag:.2f}")

for warning in point.warnings:
    print("WARNING:", warning)
```

The sweep results carry NumPy arrays ready to plot:

```python
import matplotlib.pyplot as plt

plt.plot(polar.alpha, polar.CL, label="CL")
plt.plot(polar.alpha, polar.Cm, label="Cm")
plt.xlabel("angle of attack [deg]")
plt.grid()
plt.legend()
plt.show()
```

Useful nested results include:

- `polar.alpha`, `CL`, `CD`, `CD_profile`, `CD_i`, `Cm`, and `LD`;
- `structure.span_load.y`, `lift`, `shear`, `moment`, and `root_moment`;
- `structure.sizing`, `cap_thickness`, `deflection`, and `EI`;
- `power.speed`, `thrust_available`, `drag_required`, `current`, `rpm`,
  efficiency and power arrays, plus `power.extrapolated`;
- `modes.longitudinal.table()`, `modes.lateral.table()`,
  `modes.derivatives.table()`, and `modes.warnings`.

Python can list every field on any result without relying on this page:

```python
from dataclasses import fields

print([field.name for field in fields(point)])
print([field.name for field in fields(point.trim)])
help(type(point))
```

## A safe parameter-sweep pattern

Keep one baseline unchanged and deep-copy it for every candidate. This avoids a
later iteration inheriting a change that was not intended to be cumulative.

```python
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from flightlab.project import AircraftProject
from flightlab.project_analysis import run_design_point

baseline = AircraftProject.load("my-aircraft.flightlab.json")
values = np.linspace(0.10, 0.20, 11)
drag = []

for tip_chord in values:
    candidate = deepcopy(baseline)
    wing = candidate.surface_named("Main wing")
    if wing is None:
        raise KeyError("Main wing not found")
    wing.stations[-1].chord = float(tip_chord)
    candidate.require_valid()

    result = run_design_point(candidate, candidate.case("Cruise"), ns=28, nc=4)
    drag.append(result.drag)

plt.plot(values, drag, "o-")
plt.xlabel("wing tip chord [m]")
plt.ylabel("total drag [N]")
plt.grid()
plt.show()
```

Store all outputs needed for the comparison during the loop. Do not rerun a
candidate later merely to recover a value that could have been recorded.

For a two-parameter study, use nested loops and append one dictionary per
candidate. A list of dictionaries can be passed directly to
`pandas.DataFrame` when pandas is available.

## Worked constrained-body example

This example varies pod length while holding both volume and width-to-height
ratio fixed. It illustrates how to encode a design constraint; it is not the
only kind of study the API supports.

```python
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from flightlab.project import AircraftProject
from flightlab.project_analysis import run_design_point

baseline = AircraftProject.load("rc1-hw1.flightlab.json")
lengths = np.linspace(0.250, 0.695, 31)
volume = 0.400 * 0.110 * 0.100       # m^3
width_to_height = 1.10
drag = []

for length in lengths:
    candidate = deepcopy(baseline)
    pod = candidate.body_named("Fuselage pod")
    if pod is None:
        raise KeyError("Fuselage pod not found")

    # V = L*w*h and w/h = r imply h = sqrt(V/(L*r)), w = r*h.
    height = np.sqrt(volume / (length * width_to_height))
    width = width_to_height * height
    pod.length = float(length)
    pod.width = float(width)
    pod.height = float(height)
    candidate.require_valid()

    result = run_design_point(candidate, candidate.case("Cruise"), ns=28, nc=4)
    drag.append(result.drag)

plt.plot(lengths, drag, "o-")
plt.xlabel("pod length [m]")
plt.ylabel("total aircraft drag [N]")
plt.grid()
plt.show()
```

## Analyze conditions that are not saved cases

Use `dataclasses.replace` to make a temporary case without changing the
project's saved case:

```python
from dataclasses import replace

cruise = project.case("Cruise")
for speed in (10.0, 12.0, 14.0, 16.0):
    trial = replace(cruise, speed=speed)
    result = run_design_point(project, trial)
    print(speed, result.drag, result.trim.alpha)
```

Use a copied project instead when the candidate changes physical geometry,
mass, installed hardware, or saved structural properties.

## Go beyond the workbench-level analyses

FlightLab also exposes topic modules for atmosphere, airfoils, wing analysis,
drag, performance, propulsion, loads, stability, and plotting:

```python
import flightlab

flightlab.show_tools()                 # list topics
flightlab.show_tools("performance")    # functions, inputs, outputs, limits
print(flightlab.example("performance"))

from flightlab import atmos
air = atmos.at(1400.0)
print(air.density, air.q(12.0), air.mach(12.0), air.reynolds(12.0, 0.160))

help(atmos.at)
```

From a terminal, the same discovery system is available as:

```bash
python -m flightlab
python -m flightlab performance
python -m flightlab performance --example
```

Prefer the project-level functions when starting from a workbench JSON file.
Use the topic modules when the assignment asks a narrower analytical question
or needs a model that is not part of the integrated design point.

## Common mistakes

- Editing the baseline inside a loop when changes are not meant to accumulate.
- Comparing candidates with different `ns` or `nc` resolution.
- Using `alpha_deg` from a `FlightCase` as the trimmed angle. It is only the
  solver's initial guess; use `result.trim.alpha`.
- Plotting `CD_total` when the requested quantity is force. Use `result.drag`
  in newtons, or compute force with an explicitly stated reference area and
  dynamic pressure.
- Changing width but forgetting height when volume or aspect ratio is a design
  constraint.
- Treating bodies as VLM surfaces. Their aerodynamics use empirical models.
- Ignoring `result.warnings` or propulsion extrapolation flags.
- Calling `flightlab.vlm` directly for a routine project study. It is a
  lower-level radian interface; the project API handles the course conventions.
