# FlightLab student launchers

These launchers are the no-Git, no-Python-install route for ME 415 students.
Give each student the ZIP file for their operating system. The first launch
downloads a private copy of Python and FlightLab; later launches reuse it.

## Student directions

1. Download the launcher for your computer.
2. Open it by double-clicking it.
3. Keep the small launcher window open while using FlightLab in your browser.
4. Close the launcher window when you are finished.

The first launch needs an internet connection and can take several minutes.
It downloads roughly 150--200 MB; students should do this before class instead
of having the entire class start at once. Later launches are normally much
faster. This setup does not change the computer's normal Python installation
and does not require Git.

### macOS

Open **Start FlightLab.command**. If macOS blocks it because it was downloaded,
Control-click the file, choose **Open**, and then choose **Open** again. This is
normally needed only once.

### Windows

Open **Start FlightLab.cmd**. If Windows SmartScreen appears, choose **More
info**, verify that the file came from the ME 415 course, and choose **Run
anyway**.

## Updating FlightLab

Keep using the same launcher. Each time it starts, it checks the small
`student_setup/release.txt` file in the FlightLab repository. If the instructor
has promoted a new course build, the launcher downloads it automatically and
leaves saved `.flightlab.json` project files alone. If the update check fails,
an already-downloaded build can still start offline.

## Homework notebooks and Colab

The Colab link opens the public notebook directly from this GitHub repository;
it does not upload a second copy maintained inside Colab. Google requires a
Google-account sign-in to execute the notebook. Students should choose **File →
Save a copy in Drive** before editing, or download the completed `.ipynb` file
before closing the session. The HW1 notebook covers only Problem 1b.

## What the launchers do

They install `uv` into a FlightLab-only folder in the student's user account.
`uv` then downloads Python 3.12 and runs the workbench in an isolated cached
environment. Nothing requires administrator access. The initial fallback
release is Git commit `0ee06b60ba2d657cb7dbe324faef81d2c8be8e5a` while the
package is prepared for PyPI.

For a weekly release, first push and test the new code. Then put that tested
40-character commit hash in `student_setup/release.txt` and push that one-file
change. Students receive the new build the next time they launch FlightLab.
Commits that have not been placed in `release.txt` are never sent to students.

The **Student launcher smoke tests** GitHub Actions workflow runs the real
launcher on both macOS and Windows. It runs automatically when launcher files
change and can also be started manually from the repository's **Actions** tab.
Changing only `release.txt` does not rerun the large first-install tests.

After editing either launcher, rebuild the student ZIP files:

```bash
./student_setup/build_downloads.sh
```

The files to post on the course site are then in `student_setup/downloads/`.
The short [release checklist](RELEASE_CHECKLIST.md) covers the initial course-site
launch, weekly promotion, and rollback.

Once FlightLab is on PyPI, the release lookup inside the launchers can be
simplified to use normal numbered releases. The student workflow does not
otherwise change.
