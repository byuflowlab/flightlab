"""Plain-text capability browser for ``python -m flightlab``."""

import argparse
import sys

from .capabilities import example, format_tools


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "workbench":
        launch_workbench(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        prog="python -m flightlab",
        description="Browse FlightLab analyses, inputs, outputs, and model limits.",
    )
    parser.add_argument("topic", nargs="?", help="topic such as atmosphere, wings, or propulsion")
    parser.add_argument("--example", action="store_true", help="print only runnable starter code")
    args = parser.parse_args()

    if args.example:
        if not args.topic:
            parser.error("--example requires a topic")
        print(example(args.topic))
    else:
        print(format_tools(args.topic))


def _session_factory(workbench_class, first):
    """Hand a pre-built workbench to the first browser tab, then build fresh ones.

    Building the first :class:`Workbench` also warms matplotlib and NeuralFoil,
    which is most of the wait.  Doing it before the browser opens means the
    first page appears almost as soon as the tab connects instead of sitting
    blank while the server builds it; every later tab still gets its own state.
    """
    pending = [first]

    def create():
        workbench = pending.pop() if pending else workbench_class()
        return workbench.view()

    return create


def launch_workbench(argv=None) -> None:
    """Launch the optional local browser workbench."""
    parser = argparse.ArgumentParser(
        prog="flightlab workbench",
        description="Open the FlightLab aircraft-design workbench.",
    )
    parser.add_argument("--port", type=int, default=0, help="local port; 0 chooses an available port")
    parser.add_argument("--no-open", action="store_true", help="start the server without opening a browser")
    args = parser.parse_args(argv)
    # Importing Panel, matplotlib, and the numerical workbench can take a
    # while on student hardware, so narrate each otherwise-silent phase in the
    # launcher window.
    print("Preparing FlightLab (loading scientific libraries)...", flush=True)
    try:
        import panel as pn
    except ImportError as exc:
        raise SystemExit(
            "The workbench needs Panel. Install FlightLab with the 'workbench' extra."
        ) from exc
    from .workbench import Workbench

    print("Building the workbench page (the first one is the slow one)...", flush=True)
    first = Workbench()
    print("Starting the local server. Keep this window open while FlightLab is running.", flush=True)
    if not args.no_open:
        print("Your web browser will open in a moment.", flush=True)

    pn.serve(
        _session_factory(Workbench, first),
        title="FlightLab Workbench",
        show=not args.no_open,
        port=args.port,
    )


if __name__ == "__main__":
    main()
