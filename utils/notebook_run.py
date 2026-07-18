"""Helpers shared by the FER training notebooks.

The training entry point owns creation of the immutable run directory.  A
notebook can still render data previews before training by writing them to a
run-specific staging directory and publishing them only after provenance.json
exists in the final run directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def timestamped_run_id(
    prefix: str,
    seed: Optional[int] = None,
    now: Optional[datetime] = None,
) -> str:
    """Return a filesystem-safe local-time run id with microsecond precision."""

    prefix = str(prefix).strip().lower().replace(" ", "-")
    if not prefix or any(character in prefix for character in "/\\"):
        raise ValueError("prefix must be a non-empty safe path component")
    started_at = now or datetime.now().astimezone()
    if started_at.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    timestamp = started_at.strftime("%Y%m%dT%H%M%S.%f%z")
    suffix = f"-seed{int(seed)}" if seed is not None else ""
    return f"{prefix}-{timestamp}{suffix}"


def prepare_notebook_staging(repo_root: str | Path, run_id: str) -> Path:
    """Create a run-specific staging directory without reserving OUTPUT_DIR."""

    run_id = str(run_id).strip()
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single safe path component")
    staging = Path(repo_root).expanduser().resolve() / "outputs" / ".notebook_staging" / run_id
    (staging / "visuals").mkdir(parents=True, exist_ok=True)
    return staging


def publish_notebook_artifacts(
    output_dir: str | Path,
    staging_dir: str | Path,
    *,
    console_text: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Publish staged notebook visuals and console output into one explicit run."""

    output = Path(output_dir).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    if not (output / "provenance.json").is_file():
        raise FileNotFoundError(
            f"Refusing to publish notebook artifacts outside a completed run initialization: {output}"
        )

    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    staged_visuals = staging / "visuals"
    if staged_visuals.is_dir():
        for source in sorted(staged_visuals.iterdir()):
            if not source.is_file():
                continue
            target = visual_dir / source.name
            shutil.copy2(source, target)
            copied.append(str(target))

    console_path = output / "notebook_console.log"
    console_path.write_text(console_text, encoding="utf-8")
    payload = {
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "staging_dir": str(staging),
        "console_log": str(console_path),
        "visuals": copied,
        **dict(metadata or {}),
    }
    metadata_path = output / "notebook_run.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
