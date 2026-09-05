"""Stable repository paths for the source-layout package."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def _looks_like_repository(path: Path) -> bool:
    return (path / "verl").is_dir() and (path / "data").is_dir()


def _default_repository_root() -> Path:
    # In a checkout, the current directory is the most reliable anchor.  In
    # editable installs the package location also points back into the
    # checkout; in a wheel install it does not, so callers should run from the
    # repository or set TRAVEL_GRPO_REPOSITORY_ROOT explicitly.
    candidates = [Path.cwd(), *PACKAGE_ROOT.parents]
    for candidate in candidates:
        if _looks_like_repository(candidate):
            return candidate.resolve()
    return PACKAGE_ROOT.parents[1].resolve()


DEFAULT_REPOSITORY_ROOT = _default_repository_root()
REPOSITORY_ROOT = Path(
    os.environ.get("TRAVEL_GRPO_REPOSITORY_ROOT", DEFAULT_REPOSITORY_ROOT)
).expanduser().resolve()

__all__ = ["DEFAULT_REPOSITORY_ROOT", "PACKAGE_ROOT", "REPOSITORY_ROOT"]
