#!/bin/sh

# Rebuild the two ZIP files posted for students. Run from any directory.

set -eu

FLIGHTLAB_SETUP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FLIGHTLAB_DOWNLOAD_DIR="${FLIGHTLAB_SETUP_DIR}/downloads"

mkdir -p "${FLIGHTLAB_DOWNLOAD_DIR}"

cd "${FLIGHTLAB_SETUP_DIR}/macos"
zip -q -r -FS "${FLIGHTLAB_DOWNLOAD_DIR}/FlightLab-macOS.zip" .

cd "${FLIGHTLAB_SETUP_DIR}/windows"
zip -q -r -FS "${FLIGHTLAB_DOWNLOAD_DIR}/FlightLab-Windows.zip" .

echo "Student downloads are ready in:"
echo "${FLIGHTLAB_DOWNLOAD_DIR}"
