"""CLI entry point for :mod:`sft.task_pools`."""

from __future__ import annotations

try:
    from .task_pools import main
except ImportError:  # direct ``python sft/build_task_pools.py``
    from task_pools import main


if __name__ == "__main__":  # pragma: no cover
    main()
