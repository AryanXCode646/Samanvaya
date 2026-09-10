"""Samanvaya package namespace for repository-side real-pair workflows."""

from __future__ import annotations

__all__ = ["__version__", "main"]
__version__ = "0.3.0"


def main(argv: list[str] | None = None) -> int:
    """Expose the package-level CLI entrypoint for python -m samanvaya and install scripts."""
    from .__main__ import main as _main

    return _main(argv)
