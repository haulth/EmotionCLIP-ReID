import numpy as np
import torch

from datasets.emotion_manifest import CANONICAL_EMOTIONS
from loss.emotion_losses import (
    dirichlet_statistics,
    evidential_ce_loss,
    emotion_stage2_loss,
    reliability_correctness_loss,
)
from model.emotionclip_model import decoupled_dirichlet
from utils.fer_metrics import compute_fer_metrics, ood_detection_metrics


def test_decoupled_dirichlet_is_offset_invariant_and_consistent():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.1, 0.0]])
    strength = torch.tensor([4.0, 8.0])
    result = decoupled_dirichlet(logits, strength)
    shifted = decoupled_dirichlet(logits + 100.0, strength)
    assert result["probabilities"].shape == logits.shape
    assert result["uncertainty"].shape == (2,)
    torch.testing.assert_close(result["probabilities"].sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(result["probabilities"], shifted["probabilities"])
    torch.testing.assert_close(result["alpha"], shifted["alpha"])
    torch.testing.assert_close(result["uncertainty"], shifted["uncertainty"])
    torch.testing.assert_close(result["alpha"].sum(dim=-1), strength)
    assert torch.all(result["alpha"] >= 1.0)
    assert torch.all((result["uncertainty"] > 0) & (result["uncertainty"] <= 1))

    statistics = dirichlet_statistics(result["alpha"])
    torch.testing.assert_close(statistics["dirichlet_mean"], result["dirichlet_mean"])


def test_decoupled_dirichlet_strength_is_monotonic_and_probabilities_are_softmax():
    logits = torch.randn(4, 7)
    low = decoupled_dirichlet(logits, torch.full((4,), 8.0))
    high = decoupled_dirichlet(logits, torch.full((4,), 20.0))
    torch.testing.assert_close(low["probabilities"], logits.softmax(dim=-1))
    torch.testing.assert_close(low["probabilities"], high["probabilities"])
    assert torch.all(high["uncertainty"] < low["uncertainty"])


def test_emotion_stage2_loss_has_gradients():
    logits = torch.randn(4, 7, requires_grad=True)
    raw_strength = torch.randn(4, requires_grad=True)
    strength = 7.0 + torch.nn.functional.softplus(raw_strength)
    probabilities = logits.softmax(dim=-1)
    alpha = 1.0 + probabilities.detach() * (strength.unsqueeze(-1) - 7.0)
    outputs = {
        "logits": logits,
        "alignment_logits": logits * 0.5,
        "raw_strength": raw_strength,
        "alpha": alpha,
    }
    targets = torch.tensor([0, 1, 2, 3])
    losses = emotion_stage2_loss(outputs, targets)
    losses["loss"].backward()
    assert logits.grad is not None
    assert raw_strength.grad is not None
    assert float(evidential_ce_loss(alpha.detach(), targets)) > 0


def test_reliability_loss_does_not_backpropagate_to_classifier_logits():
    logits = torch.randn(4, 7, requires_grad=True)
    raw_strength = torch.randn(4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = reliability_correctness_loss(raw_strength, logits, targets)
    loss.backward()
    assert raw_strength.grad is not None
    assert logits.grad is None


def test_fer_metrics_basic_values():
    labels = [0, 1, 1, 2]
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05, 0, 0, 0, 0],
            [0.1, 0.8, 0.1, 0, 0, 0, 0],
            [0.7, 0.2, 0.1, 0, 0, 0, 0],
            [0.1, 0.1, 0.8, 0, 0, 0, 0],
        ]
    )
    metrics = compute_fer_metrics(labels, probabilities, [0.1, 0.2, 0.9, 0.3], CANONICAL_EMOTIONS)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["confusion_matrix"][1][0] == 1
    assert metrics["error_auroc"] == 1.0
    assert metrics["error_aupr"] == 1.0
    assert metrics["nll"] > 0.0
    assert metrics["brier"] > 0.0


def test_ood_metrics_reward_separated_uncertainty_scores():
    metrics = ood_detection_metrics([0.05, 0.1, 0.2], [0.8, 0.9, 0.95])
    assert metrics == {"ood_auroc": 1.0, "ood_aupr": 1.0, "ood_fpr95": 0.0}
