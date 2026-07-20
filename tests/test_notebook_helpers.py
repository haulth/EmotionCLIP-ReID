import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from utils.notebook_run import (
    prepare_notebook_staging,
    publish_notebook_artifacts,
    timestamped_run_id,
)


def test_timestamped_run_id_is_local_time_safe_and_precise():
    now = datetime(2026, 7, 18, 23, 4, 5, 123456, tzinfo=timezone(timedelta(hours=7)))
    assert timestamped_run_id("FER2013", seed=1234, now=now) == (
        "fer2013-20260718T230405.123456+0700-seed1234"
    )


def test_publish_notebook_artifacts_requires_initialized_run(tmp_path: Path):
    staging = prepare_notebook_staging(tmp_path, "run-1")
    with pytest.raises(FileNotFoundError):
        publish_notebook_artifacts(tmp_path / "outputs" / "run-1", staging)


def test_publish_notebook_artifacts_copies_visuals_and_log(tmp_path: Path):
    staging = prepare_notebook_staging(tmp_path, "run-1")
    (staging / "visuals" / "distribution.png").write_bytes(b"png")
    output = tmp_path / "experiment" / "run-1"
    output.mkdir(parents=True)
    (output / "provenance.json").write_text("{}", encoding="utf-8")

    payload = publish_notebook_artifacts(
        output,
        staging,
        console_text="epoch=1\n",
        metadata={"run_id": "run-1"},
    )

    assert (output / "visuals" / "distribution.png").read_bytes() == b"png"
    assert (output / "notebook_console.log").read_text(encoding="utf-8") == "epoch=1\n"
    saved = json.loads((output / "notebook_run.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "run-1"
    assert payload["visuals"] == [str(output.resolve() / "visuals" / "distribution.png")]


@pytest.mark.parametrize(
    "notebook_name",
    [
        "emotionclip_reid_jupyterhub_fer2013.ipynb",
        "emotionclip_reid_jupyterhub_rafdb.ipynb",
    ],
)
def test_training_notebooks_forward_resolved_runtime_controls(notebook_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo_root / "notebooks" / notebook_name).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    required_controls = (
        "GPU_IDS = [0, 1]",
        "SOLVER.STAGE1.MAX_EPOCHS",
        "SOLVER.STAGE1.BASE_EPOCHS",
        "SOLVER.STAGE1.GEOMETRY_EPOCHS",
        "SOLVER.STAGE2.MAX_EPOCHS",
        "SOLVER.STAGE1.IMS_PER_BATCH",
        "SOLVER.STAGE2.IMS_PER_BATCH",
        "DATALOADER.NUM_WORKERS",
        "DATALOADER.PIN_MEMORY",
        "RESOLVED_TRAIN_CFG = load_emotion_cfg",
        "*TRAIN_OVERRIDES",
        "--gpus",
    )
    for control in required_controls:
        assert control in source
    assert "GPU_ID =" not in source

    training_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and any(token in "".join(cell.get("source", [])) for token in ("TRAIN_OVERRIDES =", "train_cmd ="))
    ]
    assert training_cells
    assert all(cell.get("execution_count") is None and not cell.get("outputs") for cell in training_cells)
