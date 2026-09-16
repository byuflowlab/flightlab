"""Regenerate tests/data/avl_reference.json from a local AVL.

    FLIGHTLAB_AVL=/path/to/avl python tests/record_avl_reference.py

Run it when the AVL export changes or a new AVL release is adopted; commit the
result.  The verification test compares the project against this file, so it
needs no AVL to run.
"""

from pathlib import Path
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
import avl_cases  # noqa: E402
from flightlab import avl  # noqa: E402


def main() -> None:
    path = avl.find_avl()
    if path is None:
        raise SystemExit("no AVL found: set FLIGHTLAB_AVL or put avl on the PATH")
    banner = subprocess.run([path], input="quit\n", capture_output=True, text=True).stdout
    version = next((line.split("Version")[-1].strip() for line in banner.splitlines() if "Version" in line), "unknown")
    record = {
        "avl_version": version,
        "settings": {"trailing_leg_forces": True, "ns": avl_cases.NS, "nc": avl_cases.NC,
                     "cd_profile": avl_cases.CD_PROFILE},
        "cases": {name: avl_cases.theirs(name, path) for name in avl_cases.CASES},
    }
    avl_cases.REFERENCE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {avl_cases.REFERENCE} from AVL {version}")


if __name__ == "__main__":
    main()
