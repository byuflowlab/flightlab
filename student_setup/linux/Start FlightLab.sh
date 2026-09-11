#!/usr/bin/env bash

# Terminal launcher for the ME 415 FlightLab workbench on Linux.
# FlightLab is pinned so every student uses the same course version.
#
# The first launch downloads uv, Python 3.12, and the course build into a
# FlightLab-only folder under ~/.local/share/flightlab.  Later launches start
# that installed copy directly, so nothing is downloaded or re-resolved again
# until the instructor promotes a new course build.
#
# Run it with:  bash "Start FlightLab.sh"

set -u
set -o pipefail

FLIGHTLAB_DEFAULT_COMMIT="0ee06b60ba2d657cb7dbe324faef81d2c8be8e5a"
FLIGHTLAB_RELEASE_URL="https://raw.githubusercontent.com/byuflowlab/flightlab/main/student_setup/release.txt"
FLIGHTLAB_DATA_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/flightlab"
FLIGHTLAB_UV_DIR="${FLIGHTLAB_DATA_ROOT}/uv"
FLIGHTLAB_UV="${FLIGHTLAB_UV_DIR}/uv"
FLIGHTLAB_RELEASE_FILE="${FLIGHTLAB_DATA_ROOT}/release.txt"
FLIGHTLAB_RELEASE_TEMP="${FLIGHTLAB_DATA_ROOT}/release-download.txt"
FLIGHTLAB_ENV_DIR="${FLIGHTLAB_DATA_ROOT}/env"
FLIGHTLAB_ENV_TEMP="${FLIGHTLAB_DATA_ROOT}/env-install"
FLIGHTLAB_INSTALLED_FILE="${FLIGHTLAB_DATA_ROOT}/installed.txt"
FLIGHTLAB_PYTHON="${FLIGHTLAB_ENV_DIR}/bin/python"
FLIGHTLAB_TEST_ONLY="${FLIGHTLAB_TEST_ONLY:-0}"

wait_for_key() {
    echo "Press any key to close this window."
    if [[ "${FLIGHTLAB_TEST_ONLY}" != "1" ]]; then
        read -r -s -n 1
    fi
}

# Download a URL to a file with whichever of curl or wget the machine has.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf --connect-timeout 5 --max-time 15 "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --timeout=15 --tries=1 -O "$2" "$1"
    else
        return 1
    fi
}

echo "FlightLab Workbench"
echo "=============================================="
echo

mkdir -p "${FLIGHTLAB_DATA_ROOT}"

if [[ ! -x "${FLIGHTLAB_UV}" ]]; then
    echo "First-time setup: downloading the FlightLab launcher..."
    echo "This does not need administrator access."
    echo
    mkdir -p "${FLIGHTLAB_UV_DIR}"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${FLIGHTLAB_UV_DIR}" sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${FLIGHTLAB_UV_DIR}" sh
    else
        echo "This computer has neither curl nor wget. Install one of them (for example"
        echo "'sudo apt install curl' on Ubuntu) and run this launcher again."
        wait_for_key
        exit 1
    fi
    if [[ ! -x "${FLIGHTLAB_UV}" ]]; then
        echo
        echo "Setup could not be downloaded. Check the internet connection and try again."
        echo "If the problem continues, take a screenshot of this window and send it to your TA."
        wait_for_key
        exit 1
    fi
fi

echo "Checking for a course update..."
if fetch "${FLIGHTLAB_RELEASE_URL}" "${FLIGHTLAB_RELEASE_TEMP}" \
    && grep -Eq '^[0-9a-f]{40}$' "${FLIGHTLAB_RELEASE_TEMP}"; then
    mv "${FLIGHTLAB_RELEASE_TEMP}" "${FLIGHTLAB_RELEASE_FILE}"
elif [[ -f "${FLIGHTLAB_RELEASE_FILE}" ]] \
    && grep -Eq '^[0-9a-f]{40}$' "${FLIGHTLAB_RELEASE_FILE}"; then
    echo "The update check was unavailable; using the most recent downloaded version."
else
    echo "The update check was unavailable; using the version included with this launcher."
    printf '%s\n' "${FLIGHTLAB_DEFAULT_COMMIT}" > "${FLIGHTLAB_RELEASE_FILE}"
fi

read -r FLIGHTLAB_COMMIT < "${FLIGHTLAB_RELEASE_FILE}"
FLIGHTLAB_INSTALLED=""
if [[ -f "${FLIGHTLAB_INSTALLED_FILE}" ]]; then
    read -r FLIGHTLAB_INSTALLED < "${FLIGHTLAB_INSTALLED_FILE}"
fi

if [[ ! -x "${FLIGHTLAB_PYTHON}" || "${FLIGHTLAB_INSTALLED}" != "${FLIGHTLAB_COMMIT}" ]]; then
    FLIGHTLAB_REQUIREMENT="flightlab[workbench] @ https://github.com/byuflowlab/flightlab/archive/${FLIGHTLAB_COMMIT}.zip"
    echo "Installing course build ${FLIGHTLAB_COMMIT:0:8}..."
    echo "The first install downloads Python and FlightLab's libraries (150-200 MB)"
    echo "and can take several minutes. Later updates reuse what is already downloaded."
    echo
    rm -rf "${FLIGHTLAB_ENV_TEMP}"
    if "${FLIGHTLAB_UV}" python install 3.12 \
        && "${FLIGHTLAB_UV}" venv --quiet --python 3.12 "${FLIGHTLAB_ENV_TEMP}" \
        && "${FLIGHTLAB_UV}" pip install --python "${FLIGHTLAB_ENV_TEMP}/bin/python" "${FLIGHTLAB_REQUIREMENT}"; then
        rm -rf "${FLIGHTLAB_ENV_DIR}"
        mv "${FLIGHTLAB_ENV_TEMP}" "${FLIGHTLAB_ENV_DIR}"
        printf '%s\n' "${FLIGHTLAB_COMMIT}" > "${FLIGHTLAB_INSTALLED_FILE}"
        echo
    else
        rm -rf "${FLIGHTLAB_ENV_TEMP}"
        echo
        if [[ -x "${FLIGHTLAB_PYTHON}" && -n "${FLIGHTLAB_INSTALLED}" ]]; then
            echo "The update could not be installed; starting the previously installed build instead."
            FLIGHTLAB_COMMIT="${FLIGHTLAB_INSTALLED}"
        else
            echo "FlightLab could not be installed. Check the internet connection and try again."
            echo "If the problem continues, take a screenshot of this window and send it to your TA."
            wait_for_key
            exit 1
        fi
    fi
fi

echo "Course build: ${FLIGHTLAB_COMMIT:0:8}"
echo "Starting FlightLab. Your web browser will open when it is ready."
echo "If no browser opens, copy the http://localhost address printed below into one."
echo
echo "Keep this window open while using FlightLab."
echo "Close this window, or press Control-C, when you are finished."
echo

if [[ "${FLIGHTLAB_TEST_ONLY}" == "1" ]]; then
    "${FLIGHTLAB_PYTHON}" -m flightlab
else
    "${FLIGHTLAB_PYTHON}" -m flightlab workbench
fi

FLIGHTLAB_STATUS=$?
if [[ ${FLIGHTLAB_STATUS} -ne 0 && ${FLIGHTLAB_STATUS} -ne 130 ]]; then
    echo
    echo "FlightLab stopped because of an error."
    echo "Take a screenshot of this window and send it to your TA."
    wait_for_key
fi

exit ${FLIGHTLAB_STATUS}
