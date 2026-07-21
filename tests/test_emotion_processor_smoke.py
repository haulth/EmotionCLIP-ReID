import json
import os

import pytest
import torch

from processor.processor_emotionclip import (
    EMOTION_CHECKPOINT_SCHEMA_VERSION,
    corrupt_anatomy_for_reliability,
    do_train_emotion_stage1,
    do_train_emotion_stage2,
    evaluate_sealed_test,
    load_emotion_checkpoint,
    save_checkpoint,
    _format_log_value,
)
from datasets.anatomy import ANATOMY_DESCRIPTOR_VERSION, empty_anatomy_inputs


class TinyEmotionModel(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.class_names = tuple(f"c{i}" for i in range(num_classes))
        self.num_classes = num_classes
        self.backbone_name = "Tiny"
        self.prompt = torch.nn.Parameter(torch.randn(num_classes, 4) * 0.01)
        self.encoder = torch.nn.Linear(3, 4)
        self.classifier = torch.nn.Linear(4, num_classes)
        self.reliability_head = torch.nn.Linear(4, 1)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def set_train_stage(self, stage):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == 1:
            self.prompt.requires_grad_(True)
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in self.classifier.parameters():
                parameter.requires_grad_(True)
            for parameter in self.reliability_head.parameters():
                parameter.requires_grad_(True)
            self.logit_scale.requires_grad_(True)

    def get_text_features(self):
        return torch.nn.functional.normalize(self.prompt, dim=-1)

    def forward(self, images=None, labels=None, get_text=False, get_image=False, text_features=None):
        if get_text:
            return self.get_text_features()
        pooled = images.mean(dim=(2, 3))
        features = torch.nn.functional.normalize(self.encoder(pooled), dim=-1)
        if get_image:
            return features
        text_features = self.get_text_features() if text_features is None else text_features
        alignment_logits = self.logit_scale.exp() * features @ text_features.t()
        logits = self.classifier(features) + alignment_logits
        probabilities = torch.softmax(logits, dim=-1)
        raw_strength = self.reliability_head(features).squeeze(-1)
        strength = self.num_classes + torch.nn.functional.softplus(raw_strength)
        alpha = 1 + probabilities.detach() * (strength.unsqueeze(-1) - self.num_classes)
        uncertainty = self.num_classes / strength
        return {
            "logits": logits,
            "alignment_logits": alignment_logits,
            "probabilities": probabilities,
            "dirichlet_mean": alpha / strength.unsqueeze(-1),
            "alpha": alpha,
            "strength": strength,
            "raw_strength": raw_strength,
            "uncertainty": uncertainty,
            "class_ambiguity": -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1),
            "region_disagreement": torch.zeros(images.shape[0]),
            "region_disagreement_valid": torch.ones(images.shape[0], dtype=torch.bool),
            "region_quality": torch.ones(images.shape[0], 3),
        }


def _loader():
    batches = []
    for _ in range(2):
        batches.append(
            {
                "images": torch.rand(4, 3, 8, 8),
                "labels": torch.tensor([0, 1, 2, 3]),
                "image_paths": ["a", "b", "c", "d"],
                "metadata": [{} for _ in range(4)],
            }
        )
    return batches


def test_processor_stage1_stage2_cpu_smoke(tmp_path):
    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE1": {"MAX_EPOCHS": 1, "IMS_PER_BATCH": 4, "LOG_PERIOD": 100, "CHECKPOINT_PERIOD": 1},
            "STAGE2": {
                "MAX_EPOCHS": 1,
                "LOG_PERIOD": 100,
                "CHECKPOINT_PERIOD": 1,
                "EVAL_PERIOD": 1,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "EDL_ANNEALING_EPOCHS": 1,
            },
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(1)
    optimizer = torch.optim.SGD([model.prompt], lr=0.01)
    do_train_emotion_stage1(cfg, model, _loader(), optimizer)

    model.set_train_stage(2)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    metrics = do_train_emotion_stage2(cfg, model, _loader(), _loader(), optimizer)
    assert metrics["num_samples"] == 8
    assert os.path.exists(tmp_path / "best_emotionclip.pth")
    assert os.path.exists(tmp_path / "trainable_parameters.json")


def test_stage2_gradient_accumulation_controls_optimizer_steps(tmp_path):
    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE2": {
                "MAX_EPOCHS": 1,
                "LOG_PERIOD": 100,
                "EVAL_PERIOD": 1,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "GRADIENT_ACCUMULATION_STEPS": 2,
                "AMP_ENABLED": False,
                "MAX_GRAD_NORM": 1.0,
            }
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(2)
    optimizer = CountingSGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    do_train_emotion_stage2(cfg, model, _loader(), None, optimizer)

    assert optimizer.step_calls == 1


def test_stage2_nonfinite_loss_fails_before_optimizer_step(tmp_path):
    class NonFiniteEmotionModel(TinyEmotionModel):
        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            if isinstance(outputs, dict):
                outputs["logits"] = outputs["logits"] * torch.tensor(float("nan"))
            return outputs

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE2": {
                "MAX_EPOCHS": 1,
                "LOG_PERIOD": 100,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "AMP_ENABLED": False,
                "MAX_GRAD_NORM": 1.0,
                "FAIL_ON_NONFINITE": True,
            }
        },
    }
    model = NonFiniteEmotionModel()
    model.set_train_stage(2)
    optimizer = CountingSGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    with pytest.raises(FloatingPointError, match="Non-finite Stage 2 loss"):
        do_train_emotion_stage2(cfg, model, _loader(), None, optimizer)

    assert optimizer.step_calls == 0
    diagnostic = json.loads((tmp_path / "training_failure.json").read_text(encoding="utf-8"))
    assert diagnostic["reason"] == "non-finite loss"
    assert diagnostic["losses"]["loss"]["finite"] is False


def test_stage2_nonfinite_parameter_after_optimizer_step_fails_closed(tmp_path):
    class PoisonSGD(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
            return result

    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE2": {
                "MAX_EPOCHS": 1,
                "LOG_PERIOD": 100,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "AMP_ENABLED": False,
                "MAX_GRAD_NORM": 1.0,
                "FAIL_ON_NONFINITE": True,
            }
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(2)
    optimizer = PoisonSGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    with pytest.raises(FloatingPointError, match="parameters after optimizer step"):
        do_train_emotion_stage2(cfg, model, _loader(), None, optimizer)

    diagnostic = json.loads((tmp_path / "training_failure.json").read_text(encoding="utf-8"))
    assert diagnostic["reason"] == "non-finite parameters after optimizer step"
    assert diagnostic["nonfinite_parameters"]


def test_stage2_early_stopping_uses_validation_patience(tmp_path):
    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE2": {
                "MAX_EPOCHS": 5,
                "LOG_PERIOD": 100,
                "EVAL_PERIOD": 1,
                "EARLY_STOPPING_PATIENCE": 2,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "AMP_ENABLED": False,
                "MAX_GRAD_NORM": 1.0,
            }
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(2)
    # lr=0 keeps validation predictions unchanged across epochs.
    optimizer = CountingSGD([p for p in model.parameters() if p.requires_grad], lr=0.0)

    do_train_emotion_stage2(cfg, model, _loader(), _loader(), optimizer)

    assert optimizer.step_calls == 6  # 2 batches x (best epoch + 2 stale epochs)


def test_small_learning_rates_use_scientific_log_format():
    assert _format_log_value(5e-6) == "5.0000e-06"
    assert _format_log_value(0.5) == "0.5000"


def test_stage1_validation_writes_and_selects_prompt_checkpoint(tmp_path):
    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE1": {
                "MAX_EPOCHS": 1,
                "IMS_PER_BATCH": 4,
                "LOG_PERIOD": 100,
                "EVAL_PERIOD": 1,
                "SELECTION_METRIC": "macro_f1",
            }
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(1)
    optimizer = torch.optim.SGD([model.prompt], lr=0.01)
    metrics = do_train_emotion_stage1(cfg, model, _loader(), optimizer, val_loader=_loader())
    assert metrics["selection_split"] == "val"
    assert os.path.exists(tmp_path / "best_emotionclip_stage1.pth")
    assert os.path.exists(tmp_path / "stage1_metrics_epoch_1.json")
    assert os.path.exists(tmp_path / "stage1_validation_metrics.csv")
    assert os.path.exists(tmp_path / "stage1_selection.json")


def test_stage1b_fails_closed_when_geometry_statistics_are_missing(tmp_path):
    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "DATASETS": {"ALLOW_ANATOMY_FALLBACK": False},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE1": {
                "MODE": "both",
                "BASE_EPOCHS": 1,
                "GEOMETRY_EPOCHS": 1,
                "MAX_EPOCHS": 2,
                "IMS_PER_BATCH": 4,
                "LOG_PERIOD": 100,
            }
        },
    }
    model = TinyEmotionModel()
    model.set_train_stage(1)
    optimizer = torch.optim.SGD([model.prompt], lr=0.01)

    with pytest.raises(RuntimeError, match="no reliable anatomy statistics"):
        do_train_emotion_stage1(cfg, model, _loader(), optimizer)


def test_fixed_epoch_training_skips_model_selection_and_test_is_explicit(tmp_path):
    cfg = {
        "MODEL": {"DEVICE": "cpu"},
        "OUTPUT_DIR": str(tmp_path),
        "SOLVER": {
            "STAGE2": {
                "MAX_EPOCHS": 1,
                "LOG_PERIOD": 100,
                "EVAL_PERIOD": 1,
                "BETA_ALIGN": 0.1,
                "LAMBDA_UNC": 0.01,
                "EDL_ANNEALING_EPOCHS": 1,
            }
        },
        "TEST": {"OUTPUT_FILE": "test_metrics.json"},
    }
    model = TinyEmotionModel()
    model.set_train_stage(2)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    metrics = do_train_emotion_stage2(cfg, model, _loader(), None, optimizer)

    assert metrics is None
    assert not os.path.exists(tmp_path / "best_emotionclip.pth")
    assert os.path.exists(tmp_path / "last_emotionclip.pth")

    test_metrics = evaluate_sealed_test(
        cfg,
        model,
        _loader(),
        checkpoint_path=str(tmp_path / "last_emotionclip.pth"),
        selection_split="fixed_epoch_no_validation",
    )
    assert test_metrics["evaluation_split"] == "test"
    assert test_metrics["selection_split"] == "fixed_epoch_no_validation"
    assert len(test_metrics["analysis_outputs"]) == 8
    assert set(test_metrics["analysis_outputs"][0]) >= {
        "class_ambiguity",
        "region_disagreement",
        "extrinsic_unreliability",
    }
    assert os.path.exists(tmp_path / "test_metrics.json")


class _CheckpointPrompt(torch.nn.Module):
    def __init__(self, token_value):
        super().__init__()
        self.n_ctx = 4
        self.prompt_prefix = "A photo of a face with"
        self.prompt_suffix_template = "showing a {emotion} expression."
        self.ctx = torch.nn.Parameter(torch.ones(1, 4, 2))
        self.register_buffer("token_prefix", torch.full((1, 1, 2), float(token_value)))
        self.register_buffer("token_suffix", torch.full((1, 1, 2), float(token_value)))
        self.register_buffer("tokenized_prompts", torch.full((1, 2), int(token_value), dtype=torch.long))


class _CheckpointModel(torch.nn.Module):
    def __init__(self, token_value=0):
        super().__init__()
        self.class_names = ("a", "b")
        self.backbone_name = "Tiny"
        self.routing_mode = "hybrid"
        self.reliability_use_anatomy_quality = True
        self.prompt_learner = _CheckpointPrompt(token_value)
        self.anatomy_fusion = torch.nn.Linear(2, 2)
        self.anatomy_fusion.geometry_enabled = True
        self.anatomy_fusion.fusion_mode = "gated_residual"
        self.classifier = torch.nn.Linear(2, 2)
        self.fusion = torch.nn.Linear(2, 2)
        self.reliability_head = torch.nn.Linear(2, 1)


def test_checkpoint_schema_preserves_current_derived_prompt_buffers(tmp_path):
    source = _CheckpointModel(token_value=9)
    path = save_checkpoint(source, str(tmp_path), "stage2.pth", epoch=3, stage=2)
    payload = torch.load(path, map_location="cpu")
    assert payload["schema_version"] == EMOTION_CHECKPOINT_SCHEMA_VERSION
    assert payload["model_signature"]["anatomy_descriptor_version"] == ANATOMY_DESCRIPTOR_VERSION
    assert payload["model_signature"]["routing_mode"] == "hybrid"

    target = _CheckpointModel(token_value=2)
    target.prompt_learner.ctx.data.zero_()
    load_emotion_checkpoint(target, path, strict=False)

    torch.testing.assert_close(target.prompt_learner.ctx, source.prompt_learner.ctx)
    torch.testing.assert_close(
        target.prompt_learner.token_prefix,
        torch.full_like(target.prompt_learner.token_prefix, 2.0),
    )
    torch.testing.assert_close(
        target.prompt_learner.tokenized_prompts,
        torch.full_like(target.prompt_learner.tokenized_prompts, 2),
    )


def test_checkpoint_rejects_stage1_or_incomplete_anatomy_for_inference(tmp_path):
    source = _CheckpointModel()
    stage1_path = save_checkpoint(source, str(tmp_path), "stage1.pth", epoch=1, stage=1)
    with pytest.raises(RuntimeError, match="Stage 1 only"):
        load_emotion_checkpoint(_CheckpointModel(), stage1_path, strict=False)

    complete_path = save_checkpoint(source, str(tmp_path), "complete.pth", epoch=1, stage=2)
    payload = torch.load(complete_path, map_location="cpu")
    payload["model"] = {
        key: value
        for key, value in payload["model"].items()
        if not key.startswith("anatomy_fusion.")
    }
    incomplete_path = tmp_path / "incomplete.pth"
    torch.save(payload, incomplete_path)
    with pytest.raises(RuntimeError, match="complete trained anatomy_fusion"):
        load_emotion_checkpoint(_CheckpointModel(), str(incomplete_path), strict=False)

    payload = torch.load(complete_path, map_location="cpu")
    payload["model"] = {
        key: value
        for key, value in payload["model"].items()
        if not key.startswith("reliability_head.")
    }
    incomplete_runtime_path = tmp_path / "incomplete_runtime.pth"
    torch.save(payload, incomplete_runtime_path)
    with pytest.raises(RuntimeError, match="missing trained Stage 2 runtime modules"):
        load_emotion_checkpoint(
            _CheckpointModel(),
            str(incomplete_runtime_path),
            strict=False,
        )


def test_corruption_invalidates_occluded_anatomy_evidence():
    single = empty_anatomy_inputs()
    anatomy = {
        key: torch.stack((value, value), dim=0)
        for key, value in single.items()
    }
    anatomy["region_landmarks"][:, :, 0] = 0.5
    anatomy["region_landmark_mask"][:, :, 0] = True
    anatomy["region_landmark_weights"][:, :, 0] = 1.0
    anatomy["region_quality"].fill_(1.0)
    anatomy["geometry_validity"].fill_(True)
    occlusion_mask = torch.ones(2, 8, 8, dtype=torch.bool)

    corrupted = corrupt_anatomy_for_reliability(anatomy, occlusion_mask)

    assert not corrupted["region_landmark_mask"].any()
    torch.testing.assert_close(corrupted["region_quality"], torch.zeros(2, 3))
    assert not corrupted["geometry_validity"].any()
    assert not corrupted["anatomy_available"].any()
