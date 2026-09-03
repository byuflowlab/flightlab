#!/bin/zsh

# Double-click launcher for the ME 415 FlightLab workbench.
# FlightLab is pinned so every student uses the same course version.

set -u
set -o pipefail

FLIGHTLAB_DEFAULT_COMMIT="0ee06b60ba2d657cb7dbe324faef81d2c8be8e5a"
FLIGHTLAB_RELEASE_URL="https://raw.githubusercontent.com/byuflowlab/flightlab/main/student_setup/release.txt"
FLIGHTLAB_DATA_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/flightlab"
FLIGHTLAB_UV_DIR="${FLIGHTLAB_DATA_ROOT}/uv"
FLIGHTLAB_UV="${FLIGHTLAB_UV_DIR}/uv"
FLIGHTLAB_RELEASE_FILE="${FLIGHTLAB_DATA_ROOT}/release.txt"
FLIGHTLAB_RELEASE_TEMP="${FLIGHTLAB_DATA_ROOT}/release-download.txt"
FLIGHTLAB_NETWORK_OPTIONS=()
FLIGHTLAB_TEST_ONLY="${FLIGHTLAB_TEST_ONLY:-0}"

echo "FlightLab Workbench"
echo "=============================================="
echo

if [[ ! -x "${FLIGHTLAB_UV}" ]]; then
    echo "First-time setup: downloading the FlightLab launcher..."
    echo "This does not need administrator access."
    echo
    mkdir -p "${FLIGHTLAB_UV_DIR}"
    if ! curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${FLIGHTLAB_UV_DIR}" sh; then
        echo
        echo "Setup could not be downloaded. Check the internet connection and try again."
        echo "Press any key to close this window."
        if [[ "${FLIGHTLAB_TEST_ONLY}" != "1" ]]; then
            read -k 1
        fi
        exit 1
    fi
    if [[ ! -x "${FLIGHTLAB_UV}" ]]; then
        echo
        echo "Setup finished without creating the launcher."
        echo "Take a screenshot of this window and send it to your TA."
        echo "Press any key to close this window."
        if [[ "${FLIGHTLAB_TEST_ONLY}" != "1" ]]; then
            read -k 1
        fi
        exit 1
    fi
fi

echo "Checking for a course update..."
if curl -LsSf --connect-timeout 5 --max-time 15 \
    "${FLIGHTLAB_RELEASE_URL}" -o "${FLIGHTLAB_RELEASE_TEMP}" \
    && grep -Eq '^[0-9a-f]{40}$' "${FLIGHTLAB_RELEASE_TEMP}"; then
    mv "${FLIGHTLAB_RELEASE_TEMP}" "${FLIGHTLAB_RELEASE_FILE}"
elif [[ -f "${FLIGHTLAB_RELEASE_FILE}" ]] \
    && grep -Eq '^[0-9a-f]{40}$' "${FLIGHTLAB_RELEASE_FILE}"; then
    echo "The update check was unavailable; using the most recent downloaded version."
    FLIGHTLAB_NETWORK_OPTIONS=(--offline)
else
    echo "The update check was unavailable; using the version included with this launcher."
    printf '%s\n' "${FLIGHTLAB_DEFAULT_COMMIT}" > "${FLIGHTLAB_RELEASE_FILE}"
fi

read -r FLIGHTLAB_COMMIT < "${FLIGHTLAB_RELEASE_FILE}"
FLIGHTLAB_REQUIREMENT="flightlab[workbench] @ https://github.com/byuflowlab/flightlab/archive/${FLIGHTLAB_COMMIT}.zip"

echo "Course build: ${FLIGHTLAB_COMMIT[1,8]}"
echo "Starting FlightLab. The first launch can take several minutes."
echo "Your web browser will open when it is ready."
echo
echo "Keep this window open while using FlightLab."
echo "Close this window, or press Control-C, when you are finished."
echo

if [[ "${FLIGHTLAB_TEST_ONLY}" == "1" ]]; then
    "${FLIGHTLAB_UV}" tool run \
        "${FLIGHTLAB_NETWORK_OPTIONS[@]}" \
        --python 3.12 \
        --from "${FLIGHTLAB_REQUIREMENT}" \
        flightlab
else
    "${FLIGHTLAB_UV}" tool run \
        "${FLIGHTLAB_NETWORK_OPTIONS[@]}" \
        --python 3.12 \
        --from "${FLIGHTLAB_REQUIREMENT}" \
        flightlab workbench
fi

FLIGHTLAB_STATUS=$?
if [[ ${FLIGHTLAB_STATUS} -ne 0 && ${FLIGHTLAB_STATUS} -ne 130 ]]; then
    echo
    echo "FlightLab stopped because of an error."
    echo "Take a screenshot of this window and send it to your TA."
    echo "Press any key to close this window."
    if [[ "${FLIGHTLAB_TEST_ONLY}" != "1" ]]; then
        read -k 1
    fi
fi

exit ${FLIGHTLAB_STATUS}
