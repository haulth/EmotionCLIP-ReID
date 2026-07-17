import json

import pytest

from utils.run_artifacts import artifact_dir, initialize_immutable_run


def test_run_directory_is_immutable_and_records_provenance(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '\n'.join([
            '{"image_path":"a.jpg","emotion":"neutral","split":"train"}',
            '{"image_path":"b.jpg","emotion":"neutral","split":"val"}',
            '{"image_path":"c.jpg","emotion":"neutral","split":"test"}',
        ]) + '\n', encoding="utf-8"
    )
    cfg = {
        "OUTPUT_DIR": str(tmp_path / "runs"),
        "TRAIN": {},
        "SOLVER": {"SEED": 7},
        "DATASETS": {"MANIFEST": str(manifest)},
    }
    run_dir = initialize_immutable_run(cfg, "run-a")
    metadata = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))

    assert cfg["OUTPUT_DIR"] == str(run_dir)
    assert metadata["seed"] == 7
    assert metadata["manifest"]["split_counts"] == {"test": 1, "train": 1, "val": 1}
    assert "sha" in metadata["git"] and "diff_sha256" in metadata["git"]
    assert (run_dir / "resolved_config.yml").is_file()
    assert artifact_dir(tmp_path / "runs", "run-a") == run_dir
    with pytest.raises(FileExistsError):
        initialize_immutable_run({**cfg, "OUTPUT_DIR": str(tmp_path / "runs")}, "run-a")


def test_artifact_lookup_requires_explicit_run_id(tmp_path):
    with pytest.raises(ValueError):
        artifact_dir(tmp_path, "")
