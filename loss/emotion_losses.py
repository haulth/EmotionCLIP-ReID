from typing import Dict

import torch
import torch.nn.functional as F


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.shape[-1]
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    first = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True) - torch.lgamma(
        torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype)
    )
    second = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
    return (first + second).squeeze(-1)


def evidential_ce_loss(alpha: torch.Tensor, targets: torch.Tensor, annealing_coef: float = 1.0) -> torch.Tensor:
    """Evidential CE for an already constructed Dirichlet parameter tensor."""
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape (batch, num_classes)")
    if torch.any(alpha < 1.0):
        raise ValueError("all Dirichlet alpha values must be at least 1")
    num_classes = alpha.shape[-1]
    labels = F.one_hot(targets, num_classes=num_classes).to(dtype=alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    nll = (labels * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
    alpha_tilde = labels + (1.0 - labels) * alpha
    kl = dirichlet_kl_to_uniform(alpha_tilde)
    return (nll + float(annealing_coef) * kl).mean()


def dirichlet_statistics(alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return Dirichlet-only statistics; these are not classifier probabilities."""
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape (batch, num_classes)")
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    dirichlet_mean = alpha / strength
    uncertainty = alpha.shape[-1] / strength.squeeze(-1)
    return {
        "alpha": alpha,
        "strength": strength.squeeze(-1),
        "dirichlet_mean": dirichlet_mean,
        "uncertainty": uncertainty,
    }


def reliability_correctness_loss(
    raw_strength: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Train reliability as probability that the detached class decision is correct."""
    correct = (logits.detach().argmax(dim=-1) == targets).to(dtype=raw_strength.dtype)
    return F.binary_cross_entropy_with_logits(raw_strength, correct)


def reliability_ranking_loss(
    clean_strength: torch.Tensor,
    shifted_strength: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Encourage clean examples to have greater strength than shifted/OOD examples."""
    return F.relu(float(margin) + shifted_strength - clean_strength).mean()


def emotion_stage2_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    beta_align: float = 0.5,
    lambda_unc: float = 0.05,
    edl_annealing: float = 1.0,
    reliability_target: str = "correctness",
    lambda_gate: float = 0.0,
    lambda_temperature: float = 0.0,
) -> Dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(outputs["logits"], targets)
    align_loss = F.cross_entropy(outputs["alignment_logits"], targets)
    if reliability_target == "correctness":
        unc_loss = reliability_correctness_loss(outputs["raw_strength"], outputs["logits"], targets)
        reliability_weight = float(lambda_unc) * float(edl_annealing)
    elif reliability_target == "edl":
        unc_loss = evidential_ce_loss(outputs["alpha"], targets, annealing_coef=edl_annealing)
        reliability_weight = float(lambda_unc)
    else:
        raise ValueError(f"Unknown reliability_target {reliability_target!r}; expected 'correctness' or 'edl'")
    gate_loss = outputs.get("gate_regularization", cls_loss.new_zeros(()))
    temperature_loss = outputs.get("temperature_regularization", cls_loss.new_zeros(()))
    total = (
        cls_loss
        + float(beta_align) * align_loss
        + reliability_weight * unc_loss
        + float(lambda_gate) * gate_loss
        + float(lambda_temperature) * temperature_loss
    )
    return {
        "loss": total,
        "classification": cls_loss.detach(),
        "alignment": align_loss.detach(),
        "uncertainty": unc_loss.detach(),
        "reliability": unc_loss.detach(),
        "gate": gate_loss.detach(),
        "temperature": temperature_loss.detach(),
    }
