"""Download-once and cache-aware landmark preprocessing for notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from datasets.anatomy import ANATOMY_ARTIFACT_SCHEMA_VERSION


DEFAULT_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_once(url: str, destination: str | Path) -> tuple[Path, bool]:
    """Download a model atomically; return ``(path, downloaded_now)``."""

    destination = Path(destination).expanduser().resolve()
    if destination.is_file() and destination.stat().st_size > 0:
        return destination, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded model is empty: {url}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, True


def _cache_metadata_path(output_manifest: Path) -> Path:
    return output_manifest.with_suffix(output_manifest.suffix + ".landmark_cache.json")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nonempty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def validate_landmark_manifest_layout(
    manifest: str | Path,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Verify that a landmark manifest and every referenced artifact are colocated.

    The training loader resolves relative ``landmark_path`` values against the
    manifest directory. Requiring the derived manifest to live directly in the
    dataset directory keeps the artifact bundle portable between the local
    checkout and JupyterHub.
    """

    manifest = Path(manifest).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Landmark manifest does not exist: {manifest}")
    if manifest.parent != data_dir:
        raise ValueError(
            f"Landmark manifest must be stored directly in the dataset directory: "
            f"manifest={manifest}, data_dir={data_dir}"
        )

    record_count = 0
    relative_reference_count = 0
    missing_references = []
    outside_data_dir = []
    missing_artifacts = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_count += 1
            record = json.loads(line)
            reference = str(record.get("landmark_path") or "").strip()
            if not reference:
                missing_references.append(line_number)
                continue
            reference_path = Path(reference)
            relative_reference_count += int(not reference_path.is_absolute())
            artifact = (
                reference_path.resolve()
                if reference_path.is_absolute()
                else (manifest.parent / reference_path).resolve()
            )
            try:
                artifact.relative_to(data_dir)
            except ValueError:
                outside_data_dir.append(str(artifact))
                continue
            if not artifact.is_file():
                missing_artifacts.append(str(artifact))

    if record_count == 0:
        raise ValueError(f"Landmark manifest is empty: {manifest}")
    if missing_references:
        raise ValueError(
            f"Landmark manifest has {len(missing_references)} record(s) without landmark_path; "
            f"first line(s): {missing_references[:5]}"
        )
    if outside_data_dir:
        raise ValueError(
            f"Landmark manifest references {len(outside_data_dir)} artifact(s) outside {data_dir}; "
            f"first: {outside_data_dir[0]}"
        )
    if missing_artifacts:
        raise FileNotFoundError(
            f"Landmark manifest references {len(missing_artifacts)} missing artifact(s); "
            f"first: {missing_artifacts[0]}"
        )
    if relative_reference_count != record_count:
        raise ValueError(
            "All landmark_path values must be relative so the dataset remains portable "
            f"between machines; relative={relative_reference_count}, records={record_count}"
        )
    return {
        "manifest": str(manifest),
        "data_dir": str(data_dir),
        "record_count": record_count,
        "artifact_count": record_count,
        "all_references_relative": True,
    }


def prepare_cached_landmarks(
    *,
    repo_root: str | Path,
    manifest: str | Path,
    output_manifest: str | Path,
    artifact_root: str | Path,
    model_path: str | Path,
    root_dir: str | Path,
    jitter_repeats: int = 4,
    jitter_pixels: int = 2,
    seed: int = 42,
    force_rebuild: bool = False,
    python_executable: Optional[str] = None,
    log_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Build landmark artifacts once and fast-path identical later notebook runs."""

    repo_root = Path(repo_root).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve()
    output_manifest = Path(output_manifest).expanduser().resolve()
    artifact_root = Path(artifact_root).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    root_dir = Path(root_dir).expanduser().resolve()
    builder = repo_root / "tools" / "build_face_landmark_artifacts.py"
    for required in (manifest, model_path, builder):
        if not required.is_file():
            raise FileNotFoundError(required)

    record_count = _nonempty_line_count(manifest)
    signature = {
        "artifact_schema_version": ANATOMY_ARTIFACT_SCHEMA_VERSION,
        "manifest_sha256": _sha256(manifest),
        "model_sha256": _sha256(model_path),
        "root_dir": str(root_dir),
        "jitter_repeats": int(jitter_repeats),
        "jitter_pixels": int(jitter_pixels),
        "seed": int(seed),
        "record_count": record_count,
    }
    metadata_path = _cache_metadata_path(output_manifest)
    cached = _read_json(metadata_path)
    artifact_count = sum(1 for path in artifact_root.glob("*.json") if path.is_file()) if artifact_root.exists() else 0
    cache_hit = bool(
        not force_rebuild
        and output_manifest.is_file()
        and cached.get("signature") == signature
        and cached.get("output_manifest_sha256") == _sha256(output_manifest)
        and artifact_count >= record_count
    )
    if cache_hit:
        if log_path is not None:
            log_path = Path(log_path).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(
                    {
                        "event": "landmark_cache_hit",
                        "output_manifest": str(output_manifest),
                        "artifact_root": str(artifact_root),
                        "record_count": record_count,
                        "artifact_count": artifact_count,
                        "metadata_path": str(metadata_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "cache_hit": True,
            "output_manifest": str(output_manifest),
            "artifact_root": str(artifact_root),
            "record_count": record_count,
            "artifact_count": artifact_count,
            "metadata_path": str(metadata_path),
        }

    command = [
        python_executable or sys.executable,
        str(builder),
        "--manifest",
        str(manifest),
        "--output-manifest",
        str(output_manifest),
        "--artifact-root",
        str(artifact_root),
        "--model-path",
        str(model_path),
        "--root-dir",
        str(root_dir),
        "--jitter-repeats",
        str(int(jitter_repeats)),
        "--jitter-pixels",
        str(int(jitter_pixels)),
        "--seed",
        str(int(seed)),
    ]
    if force_rebuild:
        command.append("--overwrite")
    if log_path is None:
        subprocess.run(command, cwd=repo_root, check=True)
    else:
        log_path = Path(log_path).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        log_path.write_text(
            (completed.stdout or "") + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    artifact_count = sum(1 for path in artifact_root.glob("*.json") if path.is_file())
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "output_manifest_sha256": _sha256(output_manifest),
        "artifact_count": artifact_count,
        "command": command,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "cache_hit": False,
        "output_manifest": str(output_manifest),
        "artifact_root": str(artifact_root),
        "record_count": record_count,
        "artifact_count": artifact_count,
        "metadata_path": str(metadata_path),
    }
