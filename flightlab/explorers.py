"""Task-focused notebook explorers for FlightLab.

Each explorer is a small, optional ``ipywidgets`` interface around the same
public functions used in ordinary analysis code.  The Results tab is for
building intuition; the Python tab exposes a reproducible starting point for
the student's own notebook or script.

The module is importable without notebook dependencies.  Calling an explorer
without the ``notebook`` extra installed raises a short installation message.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from typing import Callable, Dict

import numpy as np

from . import airfoil, atmos, catalog, drag, geom, loads, performance, propulsion, stability, wing
from .fleet import ASW27, C172, RC1


def _notebook_dependencies():
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import HTML, clear_output, display
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "FlightLab notebook explorers require JupyterLab and ipywidgets. "
            "Install the 'notebook' extra with your environment manager."
        ) from exc
    return widgets, plt, HTML, clear_output, display


def _metrics(HTML, display, values):
    cells = "".join(
        "<div style='min-width:125px;padding:8px 12px;border-right:1px solid #d8dee8'>"
        f"<div style='font-size:11px;color:#667085'>{escape(str(label))}</div>"
        f"<div style='font-size:17px;font-weight:650'>{escape(str(value))}</div></div>"
        for label, value in values
    )
    display(HTML(f"<div style='display:flex;flex-wrap:wrap;border:1px solid #d8dee8'>{cells}</div>"))


def _code_html(code: str) -> str:
    return (
        "<div style='font-size:12px;color:#667085;margin:0 0 7px'>"
        "Copy this into a new cell, then change or extend it.</div>"
        "<pre style='margin:0;padding:14px;overflow:auto;background:#172033;color:#e6edf7;"
        "border-radius:6px;font-size:12px;line-height:1.5;white-space:pre-wrap'>"
        f"{escape(code)}</pre>"
    )


def _explorer(
    title: str,
    introduction: str,
    controls: Dict[str, object],
    analyze: Callable[[Dict[str, object], tuple], str],
    limitations,
):
    widgets, plt, HTML, clear_output, display = _notebook_dependencies()

    run_button = widgets.Button(
        description="Run analysis",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width="100%"),
    )
    form = widgets.VBox(
        [*controls.values(), run_button],
        layout=widgets.Layout(width="285px", min_width="285px", margin="0 16px 0 0"),
    )
    result = widgets.Output(layout=widgets.Layout(min_height="390px"))
    code = widgets.HTML(value=_code_html("Run the analysis to generate Python."))
    tabs = widgets.Tab(
        children=(result, code),
        layout=widgets.Layout(width="100%", min_width="0"),
    )
    tabs.set_title(0, "Results")
    tabs.set_title(1, "Python")

    limit_items = "".join(f"<li>{escape(str(item))}</li>" for item in limitations)
    header = widgets.HTML(
        "<div style='padding:4px 0 12px'>"
        f"<div style='font-size:22px;font-weight:700'>{escape(title)}</div>"
        f"<div style='font-size:13px;color:#58657a;max-width:820px'>{escape(introduction)}</div>"
        "</div>"
    )
    limits_box = widgets.HTML(
        "<div style='margin-top:12px;padding:10px 12px;border-left:4px solid #d49b16;"
        "background:#fff8df;font-size:12px'><b>Model limits</b>"
        f"<ul style='margin:5px 0 0 18px;padding:0'>{limit_items}</ul></div>"
    )
    body = widgets.HBox(
        [form, tabs],
        layout=widgets.Layout(width="100%", align_items="flex-start"),
    )
    app = widgets.VBox([header, body, limits_box], layout=widgets.Layout(width="100%"))

    def rerun(_=None):
        values = {name: control.value for name, control in controls.items()}
        with result:
            clear_output(wait=True)
            try:
                generated = analyze(values, (plt, HTML, display))
            except Exception as exc:
                display(
                    HTML(
                        "<div style='padding:14px;border:1px solid #d92d20;color:#912018'>"
                        f"<b>{escape(type(exc).__name__)}</b>: {escape(str(exc))}</div>"
                    )
                )
                generated = "# This input combination did not produce a valid result.\n" + str(exc)
        code.value = _code_html(generated)

    run_button.on_click(rerun)
    rerun()
    return app


def flight_condition_explorer():
    """Atmosphere, nondimensional conditions, and parasite-drag sources."""
    widgets, *_ = _notebook_dependencies()
    aircraft = {"Cessna 172S": C172, "RC-1": RC1, "ASW-27B": ASW27}
    controls = {
        "aircraft": widgets.Dropdown(options=aircraft, value=C172, description="Aircraft"),
        "speed": widgets.FloatSlider(value=50.0, min=5.0, max=120.0, step=1.0, description="TAS [m/s]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=0.0, min=0.0, max=6000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "interference": widgets.FloatSlider(value=0.05, min=0.0, max=0.20, step=0.01, description="Interference", readout_format=".2f", continuous_update=False),
        "protuberance": widgets.FloatSlider(value=0.05, min=0.0, max=0.20, step=0.01, description="Protuberance", readout_format=".2f", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        ac, V, h = v["aircraft"], v["speed"], v["altitude"]
        air = atmos.at(h)
        panel = geom.resolve(ac.wing)
        buildup = drag.buildup(
            ac, V, altitude=h,
            interference=v["interference"], protuberance=v["protuberance"],
        )
        mass = ac.mass.get("gross") or ac.mass.get("mtow")
        cl_required = mass * 9.80665 / (air.q(V) * panel.area) if mass else float("nan")
        _metrics(HTML, display, (
            ("Density", f"{air.density:.4f} kg/m³"),
            ("Dynamic pressure", f"{air.q(V):.1f} Pa"),
            ("Mach", f"{air.mach(V):.3f}"),
            ("Re on MAC", f"{air.reynolds(V, panel.mac):.3e}"),
            ("Required CL", f"{cl_required:.3f}"),
            ("Parasite CD", f"{buildup.CD0:.4f}"),
        ))
        names = [row.name for row in buildup.rows]
        areas = [row.f for row in buildup.rows]
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        ax.barh(names, areas, color="#3d6da8")
        ax.set(xlabel="unmarked drag area [m²]", title=f"{ac.label}: component parasite drag")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''from flightlab import atmos, drag, geom
from flightlab.fleet import {ac.label}

V = {V:.6g}
h = {h:.6g}
air = atmos.at(h)
panel = geom.resolve({ac.label}.wing)
buildup = drag.buildup(
    {ac.label}, V, altitude=h,
    interference={v["interference"]:.6g},
    protuberance={v["protuberance"]:.6g},
)
print(air.density, air.q(V), air.mach(V))
print(air.reynolds(V, panel.mac))
print(buildup.table())'''

    return _explorer(
        "Flight condition and parasite drag",
        "Change the aircraft and flight condition, then connect density and speed to Mach, Reynolds number, required lift, and the sources of parasite drag.",
        controls, analyze,
        ("The atmosphere is the 1976 standard model.", "Fleet geometries are early-design representations.", "Cooling and installed-system drag require explicit allowances."),
    )


def airfoil_explorer():
    """Low-speed section polar and model-coverage explorer."""
    widgets, *_ = _notebook_dependencies()
    sections = ("naca2412", "s1223", "sd7037", "e212", "clarky", "fx62k131")
    controls = {
        "section": widgets.Dropdown(options=sections, value="naca2412", description="Section"),
        "reynolds": widgets.FloatLogSlider(value=5e5, base=10, min=4.7, max=6.7, step=0.05, description="Reynolds", continuous_update=False),
        "alpha": widgets.IntRangeSlider(value=(-4, 14), min=-8, max=20, step=1, description="α range [°]", continuous_update=False),
        "transition": widgets.FloatSlider(value=1.0, min=0.05, max=1.0, step=0.05, description="xtr/c", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        alpha = np.linspace(v["alpha"][0], v["alpha"][1], 73)
        polar = airfoil.polar(
            v["section"], Re=v["reynolds"], alpha=alpha,
            xtr_upper=v["transition"], xtr_lower=v["transition"],
        )
        _metrics(HTML, display, (
            ("Section", polar.section),
            ("cl max in sweep", f"{polar.cl_max:.3f}"),
            ("α at cl max", f"{polar.alpha_stall:.2f}°"),
            ("minimum cd", f"{polar.cd_min:.5f}"),
            ("minimum confidence", f"{np.min(polar.confidence):.3f}"),
        ))
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
        axes[0].plot(polar.alpha, polar.cl, color="#245b9e")
        axes[0].set(xlabel="angle of attack [deg]", ylabel="$c_l$")
        axes[1].plot(polar.cd, polar.cl, color="#a34d36")
        axes[1].set(xlabel="$c_d$", ylabel="$c_l$")
        axes[2].plot(polar.alpha, polar.confidence, color="#397a55")
        axes[2].set(xlabel="angle of attack [deg]", ylabel="confidence")
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''import numpy as np
from flightlab import airfoil

alpha = np.linspace({v["alpha"][0]}, {v["alpha"][1]}, 73)
polar = airfoil.polar(
    {v["section"]!r}, Re={v["reynolds"]:.6g}, alpha=alpha,
    xtr_upper={v["transition"]:.6g},
    xtr_lower={v["transition"]:.6g},
)
print(polar.cl_max, polar.cd_min)
print(polar.cl, polar.cd, polar.cm, polar.confidence)'''

    return _explorer(
        "Airfoil analysis",
        "Explore how section lift, drag, moment, and model coverage change with Reynolds number, angle of attack, and forced transition.",
        controls, analyze,
        ("NeuralFoil is a reduced-order, subsonic section model.", "Confidence measures model coverage, not experimental uncertainty.", "An interior maximum in the swept range is needed before calling a value cl max."),
    )


def wing_design_explorer():
    """Planform, twist, span loading, and induced-drag explorer."""
    widgets, *_ = _notebook_dependencies()
    controls = {
        "span": widgets.FloatSlider(value=15.0, min=12.0, max=18.0, step=0.25, description="Span [m]", continuous_update=False),
        "taper": widgets.FloatSlider(value=0.35, min=0.20, max=1.0, step=0.05, description="Taper", continuous_update=False),
        "twist": widgets.FloatSlider(value=-2.0, min=-6.0, max=0.0, step=0.5, description="Tip twist [°]", continuous_update=False),
        "speed": widgets.FloatSlider(value=29.0, min=20.0, max=40.0, step=0.5, description="TAS [m/s]", continuous_update=False),
        "mass": widgets.FloatSlider(value=500.0, min=350.0, max=650.0, step=10.0, description="Mass [kg]", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        new_wing = replace(
            ASW27.wing, span=v["span"], taper=v["taper"], twist_deg=v["twist"],
            root_chord=None, tip_chord=None, mean_chord=None,
        )
        candidate = replace(ASW27, wing=new_wing)
        sol = wing.trim_to_weight(candidate, mass=v["mass"], V=v["speed"], ns=36, nc=5)
        _metrics(HTML, display, (
            ("Trim α", f"{sol.alpha:.2f}°"),
            ("CL", f"{sol.CL:.3f}"),
            ("Induced CD", f"{sol.CD_i:.5f}"),
            ("Induced drag", f"{sol.induced_drag:.1f} N"),
            ("Span efficiency", f"{sol.e_inv:.3f}"),
            ("Aspect ratio", f"{new_wing.aspect_ratio:.1f}"),
        ))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
        axes[0].plot(sol.y, sol.ccl, label="computed")
        axes[0].plot(sol.y, sol.elliptical(), "k--", label="elliptical, same lift")
        axes[0].set(xlabel="span station y [m]", ylabel="$c c_l$ [m]", title="Span loading")
        axes[0].legend(fontsize=8)
        axes[1].plot(sol.y, sol.cl, color="#a34d36")
        axes[1].set(xlabel="span station y [m]", ylabel="local $c_l$", title="Local section lift")
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''from dataclasses import replace
from flightlab import wing
from flightlab.fleet import ASW27

new_wing = replace(
    ASW27.wing,
    span={v["span"]:.6g}, taper={v["taper"]:.6g},
    twist_deg={v["twist"]:.6g},
    root_chord=None, tip_chord=None, mean_chord=None,
)
candidate = replace(ASW27, wing=new_wing)
solution = wing.trim_to_weight(
    candidate, mass={v["mass"]:.6g}, V={v["speed"]:.6g},
    ns=60, nc=6,
)
print(solution.alpha, solution.CL, solution.CD_i, solution.e_inv)'''

    return _explorer(
        "Wing design",
        "Change span, taper, and washout while holding area fixed. Compare the distribution that controls induced drag with the local section lift that controls stall onset.",
        controls, analyze,
        ("The live view uses coarse VLM panelling for responsiveness; rerun reporting values at converged resolution.", "The VLM is inviscid and does not predict separated stall.", "Changing span without changing structural mass is a hypothetical aerodynamic comparison."),
    )


def stability_explorer():
    """Longitudinal trim, CG, tail-size, and static-margin explorer."""
    widgets, *_ = _notebook_dependencies()
    controls = {
        "speed": widgets.FloatSlider(value=11.5, min=8.0, max=18.0, step=0.25, description="TAS [m/s]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=1400.0, min=0.0, max=3000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "mass": widgets.FloatSlider(value=0.75, min=0.55, max=1.0, step=0.01, description="Mass [kg]", continuous_update=False),
        "cg_fraction": widgets.FloatSlider(value=0.30, min=0.10, max=0.55, step=0.01, description="CG / MAC", readout_format=".2f", continuous_update=False),
        "tail_scale": widgets.FloatSlider(value=1.0, min=0.65, max=1.5, step=0.05, description="Tail area ×", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        new_tail = replace(
            RC1.htail, area=RC1.htail.area * v["tail_scale"],
            root_chord=None, tip_chord=None, mean_chord=None,
        )
        candidate = replace(RC1, htail=new_tail)
        panel = geom.resolve(candidate.wing)
        x_mac_le = panel.x_le + panel.x_mac
        x_cg = x_mac_le + v["cg_fraction"] * panel.mac
        tr = stability.trim(
            candidate, V=v["speed"], altitude=v["altitude"],
            mass=v["mass"], x_cg=x_cg, ns=28, nc=4,
        )
        tail_lift = tr.solution.surface("htail").strip_lift
        np_fraction = (tr.x_np - x_mac_le) / panel.mac
        _metrics(HTML, display, (
            ("Trim α", f"{tr.alpha:.2f}°"),
            ("Tail incidence", f"{tr.tail_incidence:.2f}°"),
            ("Tail lift", f"{tail_lift:.2f} N"),
            ("Neutral point", f"{np_fraction:.1%} MAC"),
            ("Static margin", f"{tr.static_margin:.1%}"),
            ("Trim converged", "yes" if tr.converged else "no"),
        ))
        fig, ax = plt.subplots(figsize=(7.5, 2.2))
        ax.hlines(0, 0, 1, color="0.35", lw=3)
        ax.plot(v["cg_fraction"], 0, "o", ms=10, label="CG")
        ax.plot(np_fraction, 0, "D", ms=9, label="neutral point")
        ax.axvspan(v["cg_fraction"], np_fraction, alpha=0.18, color="#397a55")
        ax.set(xlim=(0, 0.7), ylim=(-0.2, 0.2), xlabel="fraction of wing MAC", yticks=[], title="CG and neutral point")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()
        return f'''from dataclasses import replace
from flightlab import geom, stability
from flightlab.fleet import RC1

new_tail = replace(
    RC1.htail, area=RC1.htail.area * {v["tail_scale"]:.6g},
    root_chord=None, tip_chord=None, mean_chord=None,
)
candidate = replace(RC1, htail=new_tail)
panel = geom.resolve(candidate.wing)
x_cg = panel.x_le + panel.x_mac + {v["cg_fraction"]:.6g} * panel.mac
trim = stability.trim(
    candidate, V={v["speed"]:.6g}, altitude={v["altitude"]:.6g},
    mass={v["mass"]:.6g}, x_cg=x_cg, ns=40, nc=6,
)
print(trim.alpha, trim.tail_incidence, trim.static_margin)'''

    return _explorer(
        "Stability and trim",
        "Move the center of gravity and resize the horizontal tail. See the difference between finding a trim condition and retaining a positive stability margin.",
        controls, analyze,
        ("The analysis is linearized about a conventional, small-disturbance flight condition.", "The slider changes tail area without estimating its structural mass or packaging cost.", "A trimmable point is not automatically controllable across the whole CG envelope."),
    )


def drag_explorer():
    """Complete polar and drag-component explorer."""
    widgets, *_ = _notebook_dependencies()
    aircraft = {"Cessna 172S": C172, "ASW-27B": ASW27, "RC-1": RC1}
    controls = {
        "aircraft": widgets.Dropdown(options=aircraft, value=C172, description="Aircraft"),
        "speed": widgets.FloatSlider(value=50.0, min=8.0, max=120.0, step=1.0, description="TAS [m/s]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=0.0, min=0.0, max=6000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "e": widgets.FloatSlider(value=0.82, min=0.55, max=1.0, step=0.01, description="Span eff. e", continuous_update=False),
        "cooling": widgets.FloatSlider(value=0.0, min=0.0, max=0.015, step=0.001, description="Cooling ΔCD", readout_format=".3f", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        ac = v["aircraft"]
        cl = np.linspace(0.05, 1.5, 60)
        polar = drag.polar(
            ac, V=v["speed"], altitude=v["altitude"], CL=cl,
            method="buildup", e=v["e"], cooling=v["cooling"],
        )
        _metrics(HTML, display, (
            ("L/D max", f"{polar.LD_max:.2f}"),
            ("CL at L/D max", f"{polar.CL_at_LD_max:.3f}"),
            ("CD at L/D max", f"{polar.CD_at_LD_max:.4f}"),
            ("Mach", f"{polar.mach:.3f}"),
            ("Method", polar.method),
        ))
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))
        axes[0].plot(polar.CD, polar.CL, color="#172f55", lw=2, label="total")
        axes[0].plot(polar.CD0, polar.CL, "--", label="parasite")
        axes[0].plot(polar.CD_i, polar.CL, "--", label="induced")
        axes[0].set(xlabel="$C_D$", ylabel="$C_L$", title="Complete polar")
        axes[0].legend(fontsize=8)
        axes[1].plot(polar.CL, polar.CD0, label="parasite")
        axes[1].plot(polar.CL, polar.CD_i, label="induced")
        axes[1].plot(polar.CL, polar.CD_visc_lift, label="viscous lift")
        axes[1].plot(polar.CL, polar.CD_wave, label="wave")
        axes[1].set(xlabel="$C_L$", ylabel="drag contribution", title="Drag breakdown")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''import numpy as np
from flightlab import drag
from flightlab.fleet import {ac.label}

CL = np.linspace(0.05, 1.5, 60)
polar = drag.polar(
    {ac.label}, V={v["speed"]:.6g}, altitude={v["altitude"]:.6g},
    CL=CL, method="buildup", e={v["e"]:.6g},
    cooling={v["cooling"]:.6g},
)
print(polar.LD_max, polar.CL_at_LD_max)
print(polar.CD0, polar.CD_i, polar.CD_visc_lift, polar.CD_wave)'''

    return _explorer(
        "Complete-aircraft drag",
        "Build a complete polar and separate parasite, induced, viscous lift-dependent, and wave-drag contributions before interpreting maximum L/D.",
        controls, analyze,
        ("The buildup method uses early-design component correlations and an assumed span efficiency.", "The supplied wave-drag relation is conceptual, not a transonic solver.", "A polar belongs to the speed and altitude at which its Reynolds and Mach numbers were evaluated."),
    )


def propulsion_explorer():
    """Battery–motor–propeller operating-point explorer."""
    widgets, *_ = _notebook_dependencies()
    controls = {
        "motor": widgets.Dropdown(options=sorted(catalog.MOTORS), value="M1000", description="Motor"),
        "propeller": widgets.Dropdown(options=sorted(catalog.PROPELLERS), value="P10x7", description="Propeller"),
        "battery": widgets.Dropdown(options=sorted(catalog.BATTERIES), value="B3S1300", description="Battery"),
        "speed": widgets.FloatSlider(value=9.1, min=0.0, max=20.0, step=0.5, description="TAS [m/s]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=1400.0, min=0.0, max=3000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "soc": widgets.FloatSlider(value=0.90, min=0.20, max=1.0, step=0.05, description="State of charge", readout_format=".2f", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        motor = catalog.MOTORS[v["motor"]]
        prop_entry = catalog.PROPELLERS[v["propeller"]]
        battery = catalog.BATTERIES[v["battery"]]
        op = propulsion.operating_point(
            motor, prop_entry.data, battery, V=v["speed"],
            altitude=v["altitude"], soc=v["soc"], esc="ESC30",
        )
        speeds = np.linspace(0.0, 20.0, 21)
        sweep = propulsion.sweep_speed(
            motor, prop_entry.data, battery, speeds,
            altitude=v["altitude"], soc=v["soc"], esc="ESC30",
        )
        _metrics(HTML, display, (
            ("Thrust", f"{op.thrust:.2f} N"),
            ("Current", f"{op.current:.2f} A"),
            ("Shaft speed", f"{op.rpm:.0f} rpm"),
            ("Advance ratio", f"{op.J:.3f}"),
            ("Total efficiency", f"{op.efficiency_total:.1%}"),
            ("Data coverage", "extrapolated" if op.extrapolated else "measured range"),
        ))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
        axes[0].plot(speeds, [point.thrust for point in sweep], label="thrust")
        axes[0].axvline(v["speed"], color="0.4", ls="--")
        axes[0].set(xlabel="airspeed [m/s]", ylabel="thrust [N]")
        axes[1].plot(speeds, [point.current for point in sweep], label="current")
        axes[1].plot(speeds, [100 * point.efficiency_total for point in sweep], label="efficiency [%]")
        axes[1].axvline(v["speed"], color="0.4", ls="--")
        axes[1].set(xlabel="airspeed [m/s]", ylabel="current [A] or efficiency [%]")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        if op.extrapolated_reason:
            display(HTML(f"<p><b>Coverage note:</b> {escape(op.extrapolated_reason)}</p>"))
        return f'''from flightlab import catalog, propulsion

motor = catalog.MOTORS[{v["motor"]!r}]
propeller = catalog.PROPELLERS[{v["propeller"]!r}]
battery = catalog.BATTERIES[{v["battery"]!r}]
op = propulsion.operating_point(
    motor, propeller.data, battery,
    V={v["speed"]:.6g}, altitude={v["altitude"]:.6g},
    soc={v["soc"]:.6g}, esc="ESC30",
)
print(op.thrust, op.current, op.rpm, op.J)
print(op.extrapolated, op.extrapolated_reason)'''

    return _explorer(
        "Propulsion-system matching",
        "Select a battery, motor, and measured propeller dataset. See the coupled torque balance, electrical load, efficiency, and whether the operating point lies inside the data.",
        controls, analyze,
        ("Catalog electrical properties are provisional until replaced by measured course values.", "A converged torque balance can still lie outside the measured propeller range.", "The speed sweep holds state of charge and throttle fixed."),
    )


def performance_explorer():
    """Characteristic-speed, drag-required, and power-required explorer."""
    widgets, *_ = _notebook_dependencies()
    controls = {
        "mass": widgets.FloatSlider(value=1111.0, min=750.0, max=1250.0, step=10.0, description="Mass [kg]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=0.0, min=0.0, max=5000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "cl_max": widgets.FloatSlider(value=1.5, min=1.0, max=2.2, step=0.05, description="CL max", continuous_update=False),
        "polar_speed": widgets.FloatSlider(value=50.0, min=35.0, max=75.0, step=1.0, description="Polar TAS [m/s]", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        polar = drag.polar(C172, V=v["polar_speed"], altitude=v["altitude"], CL=np.linspace(0.05, v["cl_max"], 70), e=0.82)
        speed_data = performance.speeds(polar, mass=v["mass"], CL_max=v["cl_max"])
        glide = performance.glide(polar, mass=v["mass"])
        speeds = np.linspace(max(speed_data["V_stall"] * 1.02, 20.0), 80.0, 100)
        curve = performance.drag_curve(polar, v["mass"], speeds)
        _metrics(HTML, display, (
            ("Stall speed", f"{speed_data['V_stall']:.1f} m/s"),
            ("Best-glide speed", f"{speed_data['V_LD_max']:.1f} m/s"),
            ("Best L/D", f"{speed_data['LD_max']:.1f}"),
            ("Minimum-power speed", f"{speed_data['V_min_power']:.1f} m/s"),
            ("Minimum sink", f"{glide['min_sink']:.2f} m/s"),
        ))
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
        axes[0].plot(curve["V"], curve["drag"], color="#245b9e")
        axes[0].axvline(speed_data["V_LD_max"], color="0.4", ls="--", label="best glide")
        axes[0].set(xlabel="true airspeed [m/s]", ylabel="drag required [N]")
        axes[0].legend(fontsize=8)
        axes[1].plot(curve["V"], curve["power"] / 1000.0, color="#a34d36")
        axes[1].axvline(speed_data["V_min_power"], color="0.4", ls="--", label="minimum power")
        axes[1].set(xlabel="true airspeed [m/s]", ylabel="power required [kW]")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''import numpy as np
from flightlab import drag, performance
from flightlab.fleet import C172

polar = drag.polar(
    C172, V={v["polar_speed"]:.6g}, altitude={v["altitude"]:.6g},
    CL=np.linspace(0.05, {v["cl_max"]:.6g}, 70), e=0.82,
)
speeds = performance.speeds(
    polar, mass={v["mass"]:.6g}, CL_max={v["cl_max"]:.6g},
)
glide = performance.glide(polar, mass={v["mass"]:.6g})
print(speeds)
print(glide)'''

    return _explorer(
        "Aircraft performance",
        "Turn a complete drag polar into stall, best-glide, minimum-power, and minimum-sink conditions. Compare the shapes of drag-required and power-required curves.",
        controls, analyze,
        ("The result inherits the assumptions of the selected drag polar.", "The polar is evaluated at one reference speed even though it is used over a speed range.", "Available thrust or power must be added before drawing climb or ceiling conclusions."),
    )


def loads_explorer():
    """Maneuver/gust envelope and span-load explorer."""
    widgets, *_ = _notebook_dependencies()
    controls = {
        "mass": widgets.FloatSlider(value=500.0, min=350.0, max=650.0, step=10.0, description="Mass [kg]", continuous_update=False),
        "altitude": widgets.FloatSlider(value=0.0, min=0.0, max=5000.0, step=100.0, description="Altitude [m]", continuous_update=False),
        "cl_max": widgets.FloatSlider(value=1.4, min=0.9, max=1.8, step=0.05, description="CL max", continuous_update=False),
        "n_pos": widgets.FloatSlider(value=5.3, min=2.5, max=7.0, step=0.1, description="Positive limit", continuous_update=False),
        "gust": widgets.FloatSlider(value=15.24, min=5.0, max=20.0, step=0.5, description="Gust [m/s]", continuous_update=False),
        "load_factor": widgets.FloatSlider(value=3.0, min=1.0, max=5.3, step=0.1, description="Span-load n", continuous_update=False),
    }

    def analyze(v, deps):
        plt, HTML, display = deps
        n_neg = -0.4 * v["n_pos"]
        vn = loads.vn_diagram(
            ASW27, mass=v["mass"], CL_max=v["cl_max"],
            altitude=v["altitude"], n_pos=v["n_pos"], n_neg=n_neg,
        )
        Vg = np.linspace(0.0, vn["V_max"], 100)
        gust_positive = loads.gust_load_factor(
            ASW27, V=Vg, gust=v["gust"], mass=v["mass"],
            CL_alpha=5.2, altitude=v["altitude"],
        )
        applied_n = min(v["load_factor"], v["n_pos"])
        span = loads.span_load(
            ASW27, mass=v["mass"], n=applied_n,
            V=vn["V_A"], altitude=v["altitude"], ns=36,
        )
        _metrics(HTML, display, (
            ("Stall speed", f"{vn['V_stall']:.1f} m/s"),
            ("Corner speed", f"{vn['V_A']:.1f} m/s"),
            ("Positive limit", f"{vn['n_pos']:.1f} g"),
            ("Positive ultimate", f"{vn['n_ultimate_pos']:.1f} g"),
            ("Root moment", f"{span.root_moment / 1000:.1f} kN·m"),
        ))
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
        axes[0].plot(vn["V_upper"], vn["n_upper"], color="#245b9e")
        axes[0].plot(vn["V_lower"], vn["n_lower"], color="#245b9e")
        axes[0].plot(Vg, gust_positive, "--", label="positive gust")
        axes[0].plot(Vg, 2.0 - gust_positive, "--", label="negative gust")
        axes[0].set(xlabel="true airspeed [m/s]", ylabel="load factor n", title="Maneuver and gust envelope")
        axes[0].legend(fontsize=8)
        axes[1].plot(span.y, span.lift / 1000.0, label="lift [kN/m]")
        axes[1].plot(span.y, span.moment / 1000.0, label="moment [kN m]")
        axes[1].set(xlabel="semispan station [m]", title=f"Span loads at n={applied_n:.1f}")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        return f'''import numpy as np
from flightlab import loads
from flightlab.fleet import ASW27

vn = loads.vn_diagram(
    ASW27, mass={v["mass"]:.6g}, CL_max={v["cl_max"]:.6g},
    altitude={v["altitude"]:.6g}, n_pos={v["n_pos"]:.6g},
    n_neg={n_neg:.6g},
)
span = loads.span_load(
    ASW27, mass={v["mass"]:.6g}, n={applied_n:.6g},
    V=vn["V_A"], altitude={v["altitude"]:.6g}, ns=60,
)
print(vn["V_stall"], vn["V_A"], span.root_moment)'''

    return _explorer(
        "Loads and structures",
        "Build maneuver and gust boundaries, identify the corner speed, and connect a selected load factor to spanwise lift and root bending moment.",
        controls, analyze,
        ("Limit and ultimate loads are different; the interface reports both.", "The gust relation is a reduced-order certification-style model.", "The span-load solve omits structural inertial relief unless it is supplied explicitly."),
    )


__all__ = [
    "flight_condition_explorer", "airfoil_explorer", "wing_design_explorer",
    "stability_explorer", "drag_explorer", "propulsion_explorer",
    "performance_explorer", "loads_explorer",
]
