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
