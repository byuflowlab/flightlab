# FlightLab student release checklist

## One-time launch

1. Commit and push `student_setup/`, the HW1 notebook, the root README changes,
   the setup tests, and the GitHub Actions workflow.
2. Open the repository's **Actions** tab and confirm that **Student launcher
   smoke tests** passes on macOS, Windows, and Linux.
3. Post these three files on the course site:
   - `student_setup/downloads/FlightLab-macOS.zip`
   - `student_setup/downloads/FlightLab-Windows.zip`
   - `student_setup/downloads/FlightLab-Linux.zip`
4. Link students to the Colab notebook:
   `https://colab.research.google.com/github/byuflowlab/flightlab/blob/main/notebooks/hw1_starter.ipynb`
5. Ask students to do the first launch before class because it downloads roughly
   150--200 MB.

## Weekly code update

1. Finish and test the FlightLab changes. If saved project files can no longer
   be read, bump `FORMAT_VERSION` in `flightlab/project.py` in the same commit
   (see the [file-format note](README.md#changing-the-saved-project-file-format)).
2. Commit and push them to `main`.
3. Run `git rev-parse HEAD` and copy the full 40-character commit hash.
4. Replace the hash in `student_setup/release.txt` and push that one-file change.
5. Announce that students should close and reopen FlightLab. Their existing
   launcher checks the release file and installs the promoted build into a
   fresh environment (a minute or two; already-downloaded libraries are reused).

Do not rebuild or repost the ZIP files for an ordinary code update. Rebuild them
only when a launcher or its `START HERE.txt` file changes.

## Rollback

Put the last known-good commit hash back in `student_setup/release.txt` and push
the change. Students return to that build the next time they launch FlightLab.
Saved project files remain in the students' download folders and are not removed
by either an update or rollback.
