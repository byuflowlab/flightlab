# FlightLab notebook sequence

These notebooks introduce FlightLab one course topic at a time. They are guided lessons, not
separate restricted versions of the aircraft-design environment. Use `pixi run workbench` to
create, save, and integrate a project aircraft; use these notebooks to develop the analysis
ideas that the workbench brings together.

| Stage | Open this notebook | Explore |
|---:|---|---|
| 1 | [Flight condition and parasite drag](01_flight_condition_and_drag.ipynb) | Atmosphere, Mach, Reynolds number, required lift, and component drag |
| 2 | [Airfoil analysis](02_airfoil_analysis.ipynb) | Section polars, Reynolds number, transition, and model confidence |
| 3 | [Wing design](03_wing_design.ipynb) | Span, taper, twist, span loading, and induced drag |
| 4 | [Stability and trim](04_stability_and_trim.ipynb) | CG, horizontal-tail size, trim, and static margin |
| 5 | [Complete-aircraft drag](05_complete_aircraft_drag.ipynb) | Parasite, induced, viscous, and wave-drag contributions |
| 6 | [Propulsion matching](06_propulsion_matching.ipynb) | Battery, motor, propeller, torque balance, and data coverage |
| 7 | [Aircraft performance](07_aircraft_performance.ipynb) | Stall, best glide, minimum power, and minimum sink |
| 8 | [Loads and structures](08_loads_and_structures.ipynb) | Maneuver/gust envelopes, span loads, and root bending moment |

Start at the stage that matches the current homework. Change the inputs, select **Run
analysis**, and use the **Results** and **Python** tabs together. The plots support exploration;
the generated Python is the starting point for a reproducible homework analysis.

From the repository root, launch the first notebook with one of:

```bash
pixi run notebook
```

```bash
python -m jupyter lab notebooks/01_flight_condition_and_drag.ipynb
```

The second command works in pip or conda environments after installing the `notebook` extra
described in the main [FlightLab README](../README.md).

The [reference toolbox](reference_toolbox.ipynb) is a searchable overview for later in the
course. It deliberately contains more modules than a new student needs at once.
