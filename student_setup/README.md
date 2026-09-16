# FlightLab student launchers

These launchers are the no-Git, no-Python-install route for ME 415 students.
Give each student the ZIP file for their operating system. The first launch
downloads a private copy of Python and FlightLab; later launches start that
copy directly.

## Student directions

1. Download the launcher for your computer.
2. Open it by double-clicking it (on Linux, run it from a terminal; see below).
3. Keep the small launcher window open while using FlightLab in your browser.
4. Close the launcher window when you are finished.

The first launch needs an internet connection and can take several minutes.
It downloads roughly 150--200 MB; students should do this before class instead
of having the entire class start at once. Later launches take a few seconds:
the launcher only checks a small release file and then starts the installed
copy. This setup does not change the computer's normal Python installation
and does not require Git.

### macOS

Open **Start FlightLab.command**. If macOS blocks it because it was downloaded,
Control-click the file, choose **Open**, and then choose **Open** again. This is
normally needed only once.

### Windows

Open **Start FlightLab.cmd**. If Windows SmartScreen appears, choose **More
info**, verify that the file came from the ME 415 course, and choose **Run
anyway**.

### Linux

Open a terminal in the unzipped folder and run `bash "Start FlightLab.sh"`.
Most desktop file managers can also run it directly (right-click, **Run as a
Program**), or make it executable once with `chmod +x "Start FlightLab.sh"`.
The script needs `curl` or `wget`, which nearly every distribution includes.
It is written for any recent distribution, not only Ubuntu.

## Updating FlightLab

Keep using the same launcher. Each time it starts, it checks the small
`student_setup/release.txt` file in the FlightLab repository. If the instructor
has promoted a new course build, the launcher installs it into a fresh
environment and leaves saved `.flightlab.json` project files alone. If that
install fails part-way (for example, the connection drops), the previously
installed build starts instead. If the update check itself fails, the
already-installed build starts without any network access.

## Library versions are frozen for the semester

The `workbench` extra in `pyproject.toml` lists every third-party package at
one exact version, resolved once for Windows, macOS, and Linux on Python 3.12.
Because the launchers install `flightlab[workbench]` from the promoted commit,
every student receives the same libraries no matter when they first install or
update. Only FlightLab itself changes from one promoted build to the next.

This exists because of a real failure. In September 2026 Tornado 6.5.9 was
published on a Monday and broke every static file in Bokeh-based apps. Any
student whose first install happened that day got a workbench with no styling
and a wall of `AttributeError` messages, while students who had installed a
day earlier were fine. Exact pins make "which day did you install" irrelevant.

The pins are generated, not hand-edited. The loose inputs are the core
`dependencies` in `pyproject.toml` and `student_setup/workbench.in`. About once
a year, before the course starts, refresh them with `uv` on the PATH:

```bash
python student_setup/update_pins.py
pixi run test            # or pytest in any environment with the new pins
python -m flightlab workbench
```

Then commit and promote as usual. Do not edit the block between `BEGIN PINS`
and `END PINS` by hand; rerun the script instead. Editing `workbench.in`
(for example to move to a new Panel major version) also requires rerunning it.

## Resetting a student's installation

A student whose installed environment is broken, for whatever reason, can force
a clean reinstall of the current course build. Re-downloading the launcher ZIP
alone does **not** do this: the environment lives outside the ZIP and the
launcher will keep starting it. Deleting `installed.txt` makes the next launch
rebuild the environment; deleting the whole folder also re-downloads `uv` and
Python. Both take the several-minute first-install time again.

**Windows.** Press the Windows key and R together, paste
`%LOCALAPPDATA%\FlightLab`, and press Enter. File Explorer opens the FlightLab
folder (it is inside the hidden `AppData` folder, so browsing to it does not
work). Delete `installed.txt`, or go up one level and delete the whole
`FlightLab` folder. Then open **Start FlightLab.cmd** again.

**macOS.** Click the desktop so Finder is active, press Shift, Command, and G
together, paste `~/.local/share/flightlab`, and press Return. Drag
`installed.txt` to the Trash, or press Command and the up arrow and drag the
whole `flightlab` folder to the Trash. Then open **Start FlightLab.command**
again. From Terminal the equivalent is
`rm ~/.local/share/flightlab/installed.txt`.

**Linux.** Same folder as macOS: `rm ~/.local/share/flightlab/installed.txt`
(or `rm -rf ~/.local/share/flightlab`), then run the launcher again. If
`XDG_DATA_HOME` is set, the folder is under that directory instead.

## Changing the saved project file format

Saved `.flightlab.json` files carry a `format_version` (currently 3, set by
`FORMAT_VERSION` in `flightlab/project.py`). Old formats are not migrated. When
a change makes previously saved files unreadable, because a required field was
added, removed, or renamed, bump `FORMAT_VERSION` in the same commit. The
loader then refuses every older file with a message that names the version the
file has and the version this build reads, and the workbench shows that same
message when a student opens the file. Students re-create the design in the
current workbench. Adding an optional field with a default, or changing how the
solver uses an existing field, does not need a bump; those files keep working.

Releasing such a change is the ordinary double push described below: push the
code, then put its commit hash in `release.txt` and push again.

## Homework notebooks and Colab

The Colab link opens the public notebook directly from this GitHub repository;
it does not upload a second copy maintained inside Colab. Google requires a
Google-account sign-in to execute the notebook. Students should choose **File →
Save a copy in Drive** before editing, or download the completed `.ipynb` file
before closing the session. The HW1 notebook covers only Problem 1b.

## What the launchers do

They install `uv` into a FlightLab-only folder in the student's user account
(`~/.local/share/flightlab` on macOS and Linux, `%LOCALAPPDATA%\FlightLab` on
Windows). The first time a course build is seen, `uv` downloads Python 3.12,
creates a virtual environment in that folder, and installs
`flightlab[workbench]` from the promoted Git commit. The commit hash is
recorded in `installed.txt` next to the environment. On every later launch the
launcher compares that file with `release.txt` and, when they match, starts the
environment's own Python directly. Nothing requires administrator access.

Earlier launchers ran `uv tool run` on every start. That re-resolved the
dependency set against PyPI each time, so a routine matplotlib or pandas
release could trigger a surprise multi-minute reinstall in class, and two
students could end up with different library versions of the same course
build. Installing once per promoted commit removes both problems, and the exact
pins in the `workbench` extra (see above) mean the dependency versions a
student receives do not depend on when that install happened.

The initial fallback release is Git commit
`0ee06b60ba2d657cb7dbe324faef81d2c8be8e5a` while the package is prepared for
PyPI.

For a weekly release, first push and test the new code. Then put that tested
40-character commit hash in `student_setup/release.txt` and push that one-file
change. Students receive the new build the next time they launch FlightLab.
Commits that have not been placed in `release.txt` are never sent to students.

The **Student launcher smoke tests** GitHub Actions workflow runs the real
launcher twice (install, then reuse) on macOS, Windows, and Ubuntu. It runs
automatically when launcher files change and can also be started manually from
the repository's **Actions** tab. Changing only `release.txt` does not rerun the
large first-install tests.

After editing any launcher, rebuild the student ZIP files:

```bash
./student_setup/build_downloads.sh
```

The files to post on the course site are then in `student_setup/downloads/`.
The short [release checklist](RELEASE_CHECKLIST.md) covers the initial course-site
launch, weekly promotion, and rollback.

Once FlightLab is on PyPI, the release lookup inside the launchers can be
simplified to use normal numbered releases. The student workflow does not
otherwise change.
