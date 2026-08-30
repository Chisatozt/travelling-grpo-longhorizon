#!/usr/bin/env bash

# Install the only environment used by this project: TravelGym.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -e "$SCRIPT_DIR/gyms/TravelGym"
echo "TravelGym installed successfully."
