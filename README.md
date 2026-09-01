# `flightlab`

`flightlab` is the student analysis package for BYU ME 415, Flight Vehicle Design. It
contains the numerical tools, reference aircraft, and measured data used by the aircraft
homework. The graded work is choosing a model and flight condition, checking the result,
interpreting the physics, and making a design decision—not rebuilding these solvers.

The package includes section aerodynamics, a vortex-lattice method, drag buildup, stability
and trim, electric-propulsion matching, aircraft performance, flight loads, simple spar
sizing, plotting helpers, and a small aircraft/component catalog. Rocket solvers are not
included because those are developed in class.

## Install once

FlightLab is a normal Python package. Use whichever environment manager you already have;
the project does not require Pixi. Run the commands from the cloned repository directory
that contains `pyproject.toml`.

```bash
git clone https://github.com/byuflowlab/flightlab.git
cd flightlab
```

### Pixi

Pixi installs the package, notebook interface, and test tools from `pyproject.toml`:

```bash
pixi install
pixi run test
pixi run workbench
pixi run notebook
```

`pixi run workbench` opens the general aircraft-design application. `pixi run notebook`
opens the first guided notebook, and `pixi run reference` opens the complete toolbox catalog.

### pip with a virtual environment

On macOS or Linux:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,notebook,workbench]"
.venv/bin/python -m pytest tests
.venv/bin/python -m flightlab workbench
.venv/bin/python -m jupyter lab notebooks/01_flight_condition_and_drag.ipynb
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,notebook,workbench]"
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m flightlab workbench
.venv\Scripts\python -m jupyter lab notebooks/01_flight_condition_and_drag.ipynb
```

### conda

Create and activate an environment, then install the local package with pip inside it:

```bash
conda create -n flightlab python=3.12 pip
conda activate flightlab
python -m pip install -e ".[dev,notebook,workbench]"
python -m pytest tests
python -m flightlab workbench
python -m jupyter lab notebooks/01_flight_condition_and_drag.ipynb
```

The editable install (`-e`) makes local package corrections available without reinstalling.
Check the installation with the Python launcher for your environment—for example, `pixi run
python`, `.venv/bin/python`, or the `python` in an activated conda environment:

```bash
pixi run python -c "import flightlab; print(flightlab.__version__)"
```

It should print `0.9.0`.

## Design an aircraft in the workbench

The workbench is the general interface for homework and the open-ended aircraft project:

```bash
pixi run workbench
```

or, from an activated pip or conda environment:

```bash
flightlab workbench
```

It opens locally in a browser. No FlightLab server or account is involved. A project contains:

- any number of piecewise-linear lifting surfaces, defined by editable station tables;
- independent surface orientation, descriptive purpose, and pitch-trim incidence control;
- coefficient reference quantities derived from one surface, a selected set of surfaces
  (such as both wings of a biplane), or manually entered values;
- a different airfoil at every station, including imported `.dat` coordinates;
- fuselages, nacelles, booms, struts, gear, and other drag components;
- point, span-distributed, and geometry-attached mass components;
- several named flight conditions and drag assumptions;
- project-owned editable motor, propeller, battery, and ESC definitions; one shared battery
  may feed any number of independently positioned propulsors.

The **Analysis** tab runs mass and CG, full-station VLM, longitudinal trim, lifting-surface
neutral point, and component profile/body drag as one design point. It also plots the aircraft
lift curve, drag polar, lift-to-drag ratio, pitching moment, and span loading. The VLM retains
every surface station. Lifting-surface profile drag integrates each strip's local chord,
Reynolds number, lift coefficient, and interpolated station airfoils; bodies use documented
empirical correlations. Separate **Propulsion** and **Dynamic stability** tabs expose the
electric-chain match, efficiency and power curves, and longitudinal/lateral eigenmodes.
The aircraft view marks CG independently of optional numbered mass-component markers, shows
3D/planform/side/front views, draws each propeller at its specified disk diameter, and can
overlay the lifting-surface panel mesh used by the current discretization controls. Table
headers and the principal analysis controls provide hover help. Four-digit NACA sections can
be generated directly in the Airfoils tab; NACA 0009 is included in the initial selectors.
The Lifting surfaces tab has a mass-free, selected-surface panel preview (including vertical
tails), while the Mass tab has dedicated planform and side views keyed to the numbered mass
rows and CG marker.
Airfoil curves, the Analysis polar and spanwise solution, the Propulsion speed sweep, and the
Loads span-load/deflection solution can be downloaded as unit-labeled CSV files for
calculations and plots beyond the workbench. Propulsion sweep limits and point count are
editable, so a characteristic speed or thrust/drag crossing need not depend on the selected
flight case's default plotting range.
The **Loads & structures** tab defaults to a direct RC design case specified by flight speed
and load factor. A maneuver/gust V–n envelope remains an optional mode. Both modes use the
current project mass and geometry for a selected surface's VLM span load, shear and bending
moment, preliminary two-cap spar sizing, and a cap-only beam-deflection estimate.
Dynamic lifting-surface derivatives currently use the equivalent-surface adapter, augmented
with empirical slender-body/crossflow derivatives and fixed-throttle propulsion derivatives;
those limitations are shown next to the results.

Mass rows can attach a known total mass to a lifting surface or body so that CG and inertia
follow the geometry. A density can instead determine mass from solid volume, or from wetted
surface area when a skin thickness is also supplied. These are explicit mass-distribution
models, not automatic structural-weight predictions.

The **Propulsion** tab starts with copies of the course catalog inside each project. Students
can replace provisional motor and battery values with measurements, add components, and import
propeller coefficient CSV files with `rpm`, `J`, `CT`, and `CP` columns. The editable component
library and measured propeller points are saved in the project JSON; no package source or global
configuration file needs to be changed. Battery and propulsor positions also place their catalog
masses in the aircraft mass model when that option is enabled; multiple motor/ESC/propeller rows
share the selected battery and its total-current voltage sag.

Use **Save project** to download a human-readable `.flightlab.json` file. The **Python** tab
shows how to open that same file and reproduce the analysis in a script or notebook:

```python
from flightlab.project import AircraftProject
from flightlab.project_analysis import aircraft_polar, run_design_point

project = AircraftProject.load("my_aircraft.flightlab.json")
result = run_design_point(project, project.case("Cruise"))
polar = aircraft_polar(project, project.case("Cruise"))
```

The current workbench is a preliminary-design tool, not a body-panel clone of XFLR5. Bodies
enter through handbook drag, mass properties, and documented stability approximations; the
VLM resolves lifting surfaces.

## Learn the toolbox in stages

The `notebooks/` directory contains a sequence of guided lessons. They introduce analysis
ideas one topic at a time; they are not the aircraft-project data model or the only available
interface. Students can open the same saved project from ordinary Python as the course builds
toward integrated design work.

| Stage | Notebook | Main question |
|---:|---|---|
| 1 | `01_flight_condition_and_drag.ipynb` | What flight regime are we in, and where does parasite drag come from? |
| 2 | `02_airfoil_analysis.ipynb` | How does a section change with Reynolds number and transition? |
| 3 | `03_wing_design.ipynb` | How do span, taper, and twist affect loading and induced drag? |
| 4 | `04_stability_and_trim.ipynb` | How do CG position and tail size affect trim and static margin? |
| 5 | `05_complete_aircraft_drag.ipynb` | How do the drag contributions combine into a complete polar? |
| 6 | `06_propulsion_matching.ipynb` | Where does a battery–motor–propeller combination actually operate? |
| 7 | `07_aircraft_performance.ipynb` | How does the polar determine characteristic speeds and glide behavior? |
| 8 | `08_loads_and_structures.ipynb` | How does the flight envelope become a wing bending load? |

Start with the notebook that matches the current course topic; later notebooks assume ideas
introduced earlier. `notebooks/reference_toolbox.ipynb` remains available as a searchable
catalog, but it is not intended to be the first student experience.

You can also discover capabilities without Jupyter. From a terminal:

```bash
python -m flightlab
python -m flightlab wings
python -m flightlab wings --example
```

Or from Python:

```python
import flightlab

flightlab.show_tools()
flightlab.show_tools("propulsion")
code = flightlab.example("propulsion")
```

## Start an analysis

Import modules by topic and aircraft by name:

```python
import numpy as np
import matplotlib.pyplot as plt

from flightlab import atmos, drag, geom
from flightlab.fleet import C172

air = atmos.at(2438.4)
wing = geom.resolve(C172.wing)
V = 63.8

q = air.q(V)
Re = air.reynolds(V, wing.mac)
buildup = drag.buildup(C172, V, altitude=2438.4)

print(f"density = {air.density:.3f} kg/m^3")
print(f"Re = {Re:.3e}")
print(buildup.table())
```

You can also import the package first and access modules lazily:

```python
import flightlab

air = flightlab.atmos.at(1400.0)
```

Use `help` when you are unsure about inputs, units, or returned fields:

```python
help(atmos.at)
help(drag.buildup)
help(buildup)
```

## Conventions

- Use SI units: metres, kilograms, seconds, newtons, pascals, watts, and kelvin.
- Speeds are true airspeed unless a name explicitly says EAS.
- Altitudes are geometric altitude in metres.
- Student-facing angles are degrees.
- Aircraft-project coefficients use the reference area, span, and chord selected in the
  project. Other APIs state their reference convention explicitly.
- `flightlab.vlm` is the one low-level exception: it uses radians to remain consistent with
  the reference solver. Most homework should use `wing`, `stability`, or `loads` instead.

Functions that naturally represent a sweep accept NumPy arrays. Atmosphere calculations,
the wave-drag model, gust loads, and the turbofan model can therefore be evaluated directly
over a grid. Iterative solvers such as `wing.trim_to_weight` and
`propulsion.operating_point` analyze one condition per call; loop over them or use the
provided sweep helper.

## Find the right module

| Module | Use it for |
|---|---|
| `atmos` | Standard atmosphere, dynamic pressure, Mach, Reynolds number, TAS/EAS |
| `fleet` | Aircraft geometry, mass data, operating points, and documented placeholders |
| `geom` | Resolving planforms, local chord, wetted area, and solver grids |
| `airfoil` | Student-facing NeuralFoil polars and section lookup tables |
| `wing` | Finite-wing loading, induced drag, span efficiency, and first-station stall |
| `drag` | Component buildup, complete polars, strip drag, and transonic correlations |
| `stability` | Mass properties, trim, neutral point, derivatives, and dynamic modes |
| `catalog` | Motors, propellers, batteries, and controllers used in the course |
| `propulsion` | Motor–propeller matching, battery voltage, rotors, and turbofan lapse |
| `performance` | Characteristic speeds, climb, glide, ceiling, range, and endurance |
| `loads` | V–n diagrams, gusts, span loads, inertial relief, spar sizing, deflection |
| `plot` | Course plotting helpers; each accepts an optional Matplotlib `ax` |
| `explorers` | Optional notebook mini GUIs and their generated Python |
| `project`, `project_analysis` | Saved project aircraft and integrated design-point analyses |
| `workbench` | Optional local browser interface for editing and analyzing a project |
| `case`, `live` | Optional cached and interactive workflows for design exploration |
| `foil`, `props`, `vlm` | Lower-level interfaces behind `airfoil`, `propulsion`, and `wing` |
| `ref` | Analytic checks, published ranges, and clearly labeled course models |

The modules follow the homework sequence:

```text
atmos ──> airfoil ──┐
                    ├──> wing ──> drag ──> performance
geom  ──────────────┘       └──> loads
fleet ──> stability
catalog ──> propulsion
```

## Working with aircraft data

Fleet aircraft are immutable dataclasses. Import them instead of retyping their data, and
use `dataclasses.replace` to make a design candidate:

```python
from dataclasses import replace

from flightlab.fleet import ASW27

longer_wing = replace(
    ASW27.wing,
    span=18.0,
    root_chord=None,
    tip_chord=None,
    mean_chord=None,
)
candidate = replace(ASW27, wing=longer_wing)
```

Setting dependent chord values to `None` tells `geom.resolve` to recompute them from span,
area, and taper. The original `ASW27` object remains unchanged.

Most analysis functions return either a documented dataclass or a dictionary with named
fields. Prefer names over tuple positions:

```python
solution = flightlab.wing.trim_to_weight(candidate, mass=500.0, V=29.0)
print(solution.CL, solution.CD_i, solution.e_inv)

envelope = flightlab.loads.vn_diagram(candidate, mass=500.0, CL_max=1.4)
print(envelope["V_stall"], envelope["V_A"])
```

## Model limits that must appear in your interpretation

- NeuralFoil is a reduced-order section model. Its confidence value indicates model
  coverage, not experimental uncertainty, and it is not a transonic wave-drag model.
- The VLM is steady and inviscid. Section drag and stall limits enter through separate
  models.
- The 787 wave-drag and engine models are conceptual-design correlations, not proprietary
  aerodynamic or engine decks.
- The ASW-27B uses an FX 62-K-131 stand-in because its actual DU-series coordinates are not
  public.
- Several fleet geometries are simplified single trapezoids. A calculated value can be
  internally consistent without representing every detail of the real aircraft.
- UIUC propeller data are measured, but the catalog's motor winding resistance, motor
  no-load current, and battery internal resistance are provisional until course
  thrust-stand measurements replace them.
- Those electrical values are component inputs, not solver constants. Pass `Motor` and
  `Battery` objects to `propulsion.operating_point`; use `motor.with_measurements(...)` and
  `battery.with_measurements(...)` to substitute measured values without editing the package.
- A propeller operating point outside measured advance ratio is an extrapolation even if
  the nonlinear solver converges. Check `op.extrapolated`, `op.well_covered`, and
  `op.extrapolated_reason`.
- Limit and ultimate structural loads are not interchangeable. `loads.spar_sizing` expects
  a limit moment and applies the supplied limit-to-ultimate factor.

## Common mistakes

- Passing RPM where a function requires rad/s. Use `op.omega`, `op.rpm`, or the conversion
  constants in `propulsion`; do not guess the conversion.
- Mixing a component coefficient's frontal or wetted reference area with the aircraft wing
  reference area. Drag area (`CD * S_ref`) is often the safer comparison.
- Mutating or retyping fleet data instead of building a candidate with `replace`.
- Silently accepting interpolation or extrapolation outside an airfoil or propeller model's
  coverage.
- Calling `flightlab.vlm` directly and supplying degrees. Use the higher-level course modules
  unless the assignment explicitly asks for the low-level solver.
- Reporting a precise-looking number without the simplification, stand-in, or provisional
  input that controls its credibility.

The package raises `ValueError` for common out-of-domain inputs and includes the expected
units or range in the message. Read that message before changing a solver tolerance or
expanding a search bracket.

## Package verification

The test suite checks analytic limits, AVL reference cases, fleet and catalog consistency,
NeuralFoil regressions, plotting entry points, full analysis workflows, and every interface
used directly in draft 5. Run it after changing the package:

```bash
.venv/bin/python -m pytest tests
```

NeuralFoil is intentionally capped below version 0.4 because its network weights affect
course regression values. Update that dependency only together with the airfoil regression
data and a documented review of changed results.
