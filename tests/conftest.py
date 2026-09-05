"""Test bootstrap for the repository's src-layout project package."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
TRAVELGYM_SOURCE_ROOT = ROOT / "environments" / "TravelGym"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(TRAVELGYM_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAVELGYM_SOURCE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
