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


def evidential_ce_loss(logits: torch.Tensor, targets: torch.Tensor, annealing_coef: float = 1.0) -> torch.Tensor:
    num_classes = logits.shape[-1]
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    labels = F.one_hot(targets, num_classes=num_classes).to(dtype=logits.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    nll = (labels * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
    alpha_tilde = labels + (1.0 - labels) * alpha
    kl = dirichlet_kl_to_uniform(alpha_tilde)
    return (nll + float(annealing_coef) * kl).mean()


def dirichlet_probabilities(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probabilities = alpha / strength
    uncertainty = logits.shape[-1] / strength.squeeze(-1)
    return {"evidence": evidence, "alpha": alpha, "probabilities": probabilities, "uncertainty": uncertainty}


def emotion_stage2_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    beta_align: float = 0.5,
    lambda_unc: float = 0.05,
    edl_annealing: float = 1.0,
) -> Dict[str, torch.Tensor]:
    cls_loss = F.cross_entropy(outputs["logits"], targets)
    align_loss = F.cross_entropy(outputs["alignment_logits"], targets)
    unc_loss = evidential_ce_loss(outputs["logits"], targets, annealing_coef=edl_annealing)
    total = cls_loss + float(beta_align) * align_loss + float(lambda_unc) * unc_loss
    return {
        "loss": total,
        "classification": cls_loss.detach(),
        "alignment": align_loss.detach(),
        "uncertainty": unc_loss.detach(),
    }
