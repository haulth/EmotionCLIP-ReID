import os

import torch

from processor.processor_emotionclip import (
    do_train_emotion_stage1,
    do_train_emotion_stage2,
    evaluate_sealed_test,
)


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
        return {
            "logits": logits,
            "alignment_logits": alignment_logits,
            "probabilities": probabilities,
            "dirichlet_mean": alpha / strength.unsqueeze(-1),
            "alpha": alpha,
            "strength": strength,
            "raw_strength": raw_strength,
            "uncertainty": self.num_classes / strength,
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
    assert os.path.exists(tmp_path / "test_metrics.json")
