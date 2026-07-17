"""Immutable experiment directories and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import yaml


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(args: Iterable[str]) -> bytes:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return b""


def _manifest_hashes(path: str) -> Dict[str, Any]:
    if not path:
        return {"path": "", "sha256": None, "split_sha256": {}, "split_counts": {}}
    manifest = Path(path).expanduser().resolve()
    raw = manifest.read_bytes()
    split_rows = defaultdict(list)
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        split_rows[str(record.get("split", "train")).lower()].append(
            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
    return {
        "path": str(manifest),
        "sha256": _sha256_bytes(raw),
        "split_sha256": {
            split: _sha256_bytes(("\n".join(sorted(rows)) + "\n").encode("utf-8"))
            for split, rows in sorted(split_rows.items())
        },
        "split_counts": {split: len(rows) for split, rows in sorted(split_rows.items())},
    }


def _dependency_versions() -> Dict[str, str | None]:
    names = ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scikit-learn", "tqdm")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> Dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append({"index": index, "name": props.name, "total_memory": props.total_memory})
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": cuda_devices,
    }


def initialize_immutable_run(cfg: Dict[str, Any], run_id: str = "") -> Path:
    """Create OUTPUT_DIR/run_id and persist all inputs needed to identify the run."""
    root = Path(cfg["OUTPUT_DIR"]).expanduser().resolve()
    requested = run_id or str(cfg.get("TRAIN", {}).get("RUN_ID") or "").strip()
    if not requested:
        requested = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if requested in {".", ".."} or Path(requested).name != requested:
        raise ValueError("run_id must be a single safe path component")
    run_dir = root / requested
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Run {requested!r} already exists and is immutable: {run_dir}") from exc

    cfg.setdefault("TRAIN", {})["RUN_ID"] = requested
    cfg["TRAIN"]["OUTPUT_ROOT"] = str(root)
    cfg["OUTPUT_DIR"] = str(run_dir)
    diff = _git(["diff", "--binary", "HEAD"])
    status = _git(["status", "--porcelain=v1"])
    provenance = {
        "run_id": requested,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": _git(["rev-parse", "HEAD"]).decode().strip() or None,
            "dirty": bool(status.strip()),
            "diff_sha256": _sha256_bytes(diff),
            "status_sha256": _sha256_bytes(status),
        },
        "manifest": _manifest_hashes(cfg.get("DATASETS", {}).get("MANIFEST", "")),
        "dependencies": _dependency_versions(),
        "seed": cfg.get("SOLVER", {}).get("SEED"),
        "hardware": _hardware(),
    }
    (run_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def artifact_dir(output_root: str | os.PathLike[str], run_id: str) -> Path:
    """Resolve artifacts by explicit run_id; never guess or select the latest run."""
    if not run_id:
        raise ValueError("run_id is required; notebooks must not infer the latest run")
    path = Path(output_root).expanduser().resolve() / run_id
    if not (path / "provenance.json").is_file():
        raise FileNotFoundError(f"Unknown or incomplete run_id {run_id!r}: {path}")
    return path
