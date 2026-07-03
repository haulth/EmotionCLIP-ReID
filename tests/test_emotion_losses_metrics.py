import numpy as np
import torch

from datasets.emotion_manifest import CANONICAL_EMOTIONS
from loss.emotion_losses import dirichlet_probabilities, evidential_ce_loss, emotion_stage2_loss
from utils.fer_metrics import compute_fer_metrics


def test_evidential_probabilities_sum_to_one_and_uncertainty_shape():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.1, 0.0]])
    result = dirichlet_probabilities(logits)
    assert result["probabilities"].shape == logits.shape
    assert result["uncertainty"].shape == (2,)
    torch.testing.assert_close(result["probabilities"].sum(dim=-1), torch.ones(2))


def test_emotion_stage2_loss_has_gradients():
    logits = torch.randn(4, 7, requires_grad=True)
    outputs = {"logits": logits, "alignment_logits": logits * 0.5}
    targets = torch.tensor([0, 1, 2, 3])
    losses = emotion_stage2_loss(outputs, targets)
    losses["loss"].backward()
    assert logits.grad is not None
    assert float(evidential_ce_loss(logits.detach(), targets)) > 0


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
