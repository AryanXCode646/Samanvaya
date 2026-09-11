"""Reproducibility metadata helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def git_commit_sha(root: Path | None = None) -> str:
    """Return the exact repository commit or an explicit unavailable marker."""
    cwd = root or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "GIT_COMMIT_UNAVAILABLE"
    value = result.stdout.strip()
    return value or "GIT_COMMIT_UNAVAILABLE"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def provenance_record(**values: Any) -> dict[str, Any]:
    return {
        "repository_commit": git_commit_sha(),
        **values,
    }
