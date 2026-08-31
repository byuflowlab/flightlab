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


def launch_workbench(argv=None) -> None:
    """Launch the optional local browser workbench."""
    parser = argparse.ArgumentParser(
        prog="flightlab workbench",
        description="Open the FlightLab aircraft-design workbench.",
    )
    parser.add_argument("--port", type=int, default=0, help="local port; 0 chooses an available port")
    parser.add_argument("--no-open", action="store_true", help="start the server without opening a browser")
    args = parser.parse_args(argv)
    try:
        import panel as pn
    except ImportError as exc:
        raise SystemExit(
            "The workbench needs Panel. Install FlightLab with the 'workbench' extra."
        ) from exc
    from .workbench import create_workbench

    pn.serve(
        create_workbench,
        title="FlightLab Workbench",
        show=not args.no_open,
        port=args.port,
    )


if __name__ == "__main__":
    main()
