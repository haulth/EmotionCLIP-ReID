import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import utils.notebook_progress as notebook_progress
from utils.notebook_run import (
    prepare_notebook_staging,
    publish_notebook_artifacts,
    timestamped_run_id,
)
from utils.notebook_landmarks import validate_landmark_manifest_layout
from config.emotion_defaults import get_default_emotion_cfg, load_emotion_cfg
from utils.notebook_progress import _is_progress_update, _progress_html


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
    "line",
    [
        "Stage1-base 1/200:  25%|██▌       | 5/20 [00:01<00:03, loss=1.23]",
        "Stage1-geometry 2/50:  50%|█████     | 10/20 [00:02<00:02, loss=0.95]",
        "Stage2 3/100:  75%|███████▌  | 15/20 [00:03<00:01, loss=0.71]",
    ],
)
def test_notebook_progress_recognizes_all_training_phase_labels(line: str):
    assert _is_progress_update(line)


def test_notebook_progress_html_keeps_stage1_phase_in_single_display(monkeypatch):
    class FakeHTML:
        def __init__(self, data):
            self.data = data

    monkeypatch.setattr(notebook_progress, "HTML", FakeHTML)
    rendered = _progress_html("Stage1-base 1/200:  25%|██▌       | 5/20 [00:01<00:03]")
    rendered_text = getattr(rendered, "data", rendered)

    assert "Stage1-base 1/200:" in rendered_text
    assert "5/20 (25.0%)" in rendered_text


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

def test_validate_landmark_manifest_layout_accepts_colocated_relative_artifacts(tmp_path: Path):
    data_dir = tmp_path / "RAF-DB"
    artifact_dir = data_dir / "anatomy_v3"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "sample.json").write_text("{}", encoding="utf-8")
    manifest = data_dir / "manifest_anatomy.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "image.jpg", "landmark_path": "anatomy_v3/sample.json"}) + "\n",
        encoding="utf-8",
    )

    report = validate_landmark_manifest_layout(manifest, data_dir)
    assert report["record_count"] == 1
    assert report["all_references_relative"] is True


def test_validate_landmark_manifest_layout_rejects_missing_artifact(tmp_path: Path):
    data_dir = tmp_path / "RAF-DB"
    data_dir.mkdir()
    manifest = data_dir / "manifest_anatomy.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "image.jpg", "landmark_path": "anatomy_v3/missing.json"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing artifact"):
        validate_landmark_manifest_layout(manifest, data_dir)


def test_rafdb_notebooks_share_explicit_colocated_landmark_paths():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_sources = {}
    for notebook_name in (
        "emotionclip_reid_jupyterhub_build_landmarks.ipynb",
        "emotionclip_reid_jupyterhub_rafdb.ipynb",
    ):
        notebook = json.loads((repo_root / "notebooks" / notebook_name).read_text(encoding="utf-8"))
        notebook_sources[notebook_name] = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    build_source = notebook_sources["emotionclip_reid_jupyterhub_build_landmarks.ipynb"]
    train_source = notebook_sources["emotionclip_reid_jupyterhub_rafdb.ipynb"]
    for source in (build_source, train_source):
        assert "RAFDB_DATA_DIR" in source
        assert "RAFDB_ANATOMY_MANIFEST" in source
        assert "RAFDB_ANATOMY_DIR" in source
        assert "validate_landmark_manifest_layout" in source
    assert "DATASETS.REQUIRE_ANATOMY" in train_source
    assert "MODEL.ROUTING.MODE" in train_source
    assert "MODEL.UNCERTAINTY.USE_ANATOMY_QUALITY" in train_source


def test_anatomy_is_required_by_default_and_quick_presets():
    default_cfg = get_default_emotion_cfg()
    assert default_cfg["MODEL"]["ROUTING"]["MODE"] == "hybrid"
    assert default_cfg["MODEL"]["GEOMETRY"]["ENABLED"] is True
    assert default_cfg["MODEL"]["UNCERTAINTY"]["USE_ANATOMY_QUALITY"] is True
    assert default_cfg["DATASETS"]["REQUIRE_ANATOMY"] is True
    assert default_cfg["DATASETS"]["ALLOW_ANATOMY_FALLBACK"] is False

    repo_root = Path(__file__).resolve().parents[1]
    for config_name in (
        "vit_b16_emotionclip_hf_fer2013_quick.yml",
        "vit_b16_emotionclip_fer2013_quick.yml",
        "vit_b16_emotionclip_rafdb_quick.yml",
    ):
        cfg = load_emotion_cfg(str(repo_root / "configs" / "emotion" / config_name))
        assert cfg["MODEL"]["ROUTING"]["MODE"] == "hybrid"
        assert cfg["MODEL"]["GEOMETRY"]["ENABLED"] is True
        assert cfg["MODEL"]["UNCERTAINTY"]["USE_ANATOMY_QUALITY"] is True
        assert cfg["DATASETS"]["REQUIRE_ANATOMY"] is True
        assert cfg["DATASETS"]["MANIFEST"].endswith("manifest_anatomy.jsonl")


def test_hf_fer2013_notebook_and_presets_enable_stage1b_geometry_prompt():
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "emotionclip_reid_jupyterhub_fer2013.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "ANATOMY_PROMPT_MODE = 'quality'" in source
    assert "STAGE1_MODE = 'both'" in source
    assert "STAGE1_GEOMETRY_EPOCHS = 10" in source
    assert "MODEL.ANATOMY_PROMPT.MODE" in source
    assert "GEOMETRY_EPOCHS'] > 0" in source

    for config_name in (
        "vit_b16_emotionclip_hf_fer2013_quick.yml",
        "vit_b16_emotionclip_fer2013_quick.yml",
    ):
        cfg = load_emotion_cfg(str(repo_root / "configs" / "emotion" / config_name))
        assert cfg["MODEL"]["ANATOMY_PROMPT"]["MODE"] == "quality"
        assert cfg["SOLVER"]["STAGE1"]["MODE"] == "both"
        assert cfg["SOLVER"]["STAGE1"]["BASE_EPOCHS"] == 190
        assert cfg["SOLVER"]["STAGE1"]["GEOMETRY_EPOCHS"] == 10
