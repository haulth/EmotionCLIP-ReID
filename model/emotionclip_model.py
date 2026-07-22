import os
import math
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.anatomy import (
    ANATOMY_REGIONS,
    MAX_GEOMETRY_FEATURES,
    MAX_REGION_LANDMARKS,
    NUM_ANATOMY_REGIONS,
    geometry_feature_definition_mask,
)
from datasets.emotion_manifest import CANONICAL_EMOTIONS
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()
ANATOMY_PROMPT_CONDITIONING_VERSION = 2


def decoupled_dirichlet(
    logits: torch.Tensor,
    strength: torch.Tensor,
    detach_class_prob: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a Dirichlet distribution without deriving evidence from logit scale/offset."""
    if logits.ndim < 2:
        raise ValueError("logits must have shape (..., num_classes)")
    if strength.shape != logits.shape[:-1]:
        raise ValueError(
            f"strength shape {tuple(strength.shape)} must match logits batch shape {tuple(logits.shape[:-1])}"
        )

    num_classes = logits.shape[-1]
    class_probabilities = F.softmax(logits, dim=-1)
    evidence_probabilities = class_probabilities.detach() if detach_class_prob else class_probabilities
    strength = strength.to(dtype=logits.dtype)
    if torch.any(strength < float(num_classes)):
        raise ValueError("Dirichlet strength must be at least the number of classes")

    alpha = 1.0 + evidence_probabilities * (strength.unsqueeze(-1) - float(num_classes))
    dirichlet_mean = alpha / strength.unsqueeze(-1).clamp_min(1e-12)
    uncertainty = float(num_classes) / strength.clamp_min(1e-12)
    return {
        "probabilities": class_probabilities,
        "dirichlet_mean": dirichlet_mean,
        "alpha": alpha,
        "strength": strength,
        "uncertainty": uncertainty,
    }


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ExpressionAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("down", nn.Linear(dim, bottleneck_dim)),
                    ("act", QuickGELU()),
                    ("drop", nn.Dropout(dropout)),
                    ("up", nn.Linear(bottleneck_dim, dim)),
                ]
            )
        )
        nn.init.zeros_(self.net.up.weight)
        nn.init.zeros_(self.net.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(prompts.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AnatomyPromptResidual(nn.Module):
    """Class-level median/MAD residual for upper/middle/lower context tokens."""

    VALID_MODES = {"legacy", "disabled", "role_only", "median", "median_mad", "quality", "random", "shuffled"}

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        geometry_dim: int = MAX_GEOMETRY_FEATURES,
        hidden_dim: int = 32,
        gate_init: float = -4.0,
        mode: str = "quality",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.geometry_dim = int(geometry_dim)
        self.mode = str(mode).lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unknown anatomy prompt mode {mode!r}; expected one of {sorted(self.VALID_MODES)}")
        self.projectors = nn.ModuleList()
        for _ in ANATOMY_REGIONS:
            projector = nn.Sequential(
                nn.Linear(4 * self.geometry_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(embedding_dim)),
            )
            nn.init.zeros_(projector[-1].weight)
            nn.init.zeros_(projector[-1].bias)
            self.projectors.append(projector)
        self.gate_logits = nn.Parameter(torch.full((NUM_ANATOMY_REGIONS,), float(gate_init)))
        shape = (self.num_classes, NUM_ANATOMY_REGIONS, self.geometry_dim)
        self.register_buffer("class_median", torch.zeros(shape))
        self.register_buffer("class_scale", torch.zeros(shape))
        self.register_buffer("class_quality", torch.zeros(self.num_classes, NUM_ANATOMY_REGIONS))
        self.register_buffer("class_valid_rate", torch.zeros(shape))
        self.register_buffer("class_uncertainty", torch.ones(shape))
        self.register_buffer("statistics_ready", torch.tensor(False))
        generator = torch.Generator().manual_seed(1729)
        self.register_buffer("random_median", torch.randn(shape, generator=generator) * 0.1)
        self.register_buffer("random_scale", torch.rand(shape, generator=generator) * 0.1)
        random_validity = geometry_feature_definition_mask().unsqueeze(0).expand(
            self.num_classes,
            -1,
            -1,
        )
        self.register_buffer("random_validity", random_validity.to(dtype=torch.float32))

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown anatomy prompt mode {mode!r}; expected one of {sorted(self.VALID_MODES)}")
        self.mode = mode

    def set_statistics(self, statistics: Dict[str, torch.Tensor]) -> None:
        expected = self.class_median.shape
        for key, target in (
            ("median", self.class_median),
            ("scale", self.class_scale),
            ("valid_rate", self.class_valid_rate),
            ("uncertainty", self.class_uncertainty),
        ):
            value = statistics[key].detach().to(device=target.device, dtype=target.dtype)
            if value.shape != expected:
                raise ValueError(f"{key} shape {tuple(value.shape)} must equal {tuple(expected)}")
            target.copy_(value)
        quality = statistics["quality"].detach().to(self.class_quality)
        if quality.shape != self.class_quality.shape:
            raise ValueError(
                f"quality shape {tuple(quality.shape)} must equal {tuple(self.class_quality.shape)}"
            )
        self.class_quality.copy_(quality)
        self.statistics_ready.fill_(True)

    def _condition(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode in {"legacy", "disabled", "role_only"}:
            return (
                self.class_median,
                self.class_scale,
                self.class_valid_rate,
                self.class_uncertainty,
                torch.zeros_like(self.class_quality),
            )
        if self.mode == "random":
            return (
                self.random_median,
                self.random_scale,
                self.random_validity,
                torch.zeros_like(self.random_median),
                torch.ones_like(self.class_quality),
            )
        if not bool(self.statistics_ready):
            return (
                self.class_median,
                self.class_scale,
                self.class_valid_rate,
                self.class_uncertainty,
                torch.zeros_like(self.class_quality),
            )
        median = self.class_median
        scale = self.class_scale
        valid_rate = self.class_valid_rate
        uncertainty = self.class_uncertainty
        quality = self.class_quality
        if self.mode == "shuffled":
            permutation = torch.arange(self.num_classes, device=median.device).roll(1)
            return (
                median[permutation],
                scale[permutation],
                valid_rate[permutation],
                uncertainty[permutation],
                quality[permutation],
            )
        if self.mode == "median":
            return (
                median,
                torch.zeros_like(scale),
                valid_rate,
                torch.zeros_like(uncertainty),
                torch.ones_like(quality),
            )
        if self.mode == "median_mad":
            return median, scale, valid_rate, torch.zeros_like(uncertainty), torch.ones_like(quality)
        return median, scale, valid_rate, uncertainty, quality

    def forward(self, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        median, scale, valid_rate, uncertainty, quality = self._condition()
        if labels is not None:
            labels = labels.to(median.device)
            median = median[labels]
            scale = scale[labels]
            valid_rate = valid_rate[labels]
            uncertainty = uncertainty[labels]
            quality = quality[labels]
        valid_rate = valid_rate.clamp(0.0, 1.0)
        condition = torch.cat(
            (
                median * valid_rate,
                torch.log1p(scale.clamp_min(0.0)) * valid_rate,
                valid_rate,
                uncertainty.clamp_min(0.0) * valid_rate,
            ),
            dim=-1,
        )
        residuals = torch.stack(
            [projector(condition[:, index]) for index, projector in enumerate(self.projectors)],
            dim=1,
        )
        gates = torch.sigmoid(self.gate_logits).view(1, NUM_ANATOMY_REGIONS, 1)
        return gates * quality.unsqueeze(-1) * torch.tanh(residuals)

    def regularization(self, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self(labels).square().mean()


class EmotionPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        dtype: torch.dtype,
        token_embedding: nn.Embedding,
        n_ctx: int = 4,
        prompt_prefix: str = "A photo of a face with",
        prompt_suffix_template: str = "showing a {emotion} expression.",
        class_specific_context: bool = True,
        geometry_dim: int = MAX_GEOMETRY_FEATURES,
        geometry_hidden_dim: int = 32,
        geometry_gate_init: float = -4.0,
        geometry_mode: str = "quality",
    ):
        super().__init__()
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.n_ctx = int(n_ctx)
        if self.n_ctx < 4:
            raise ValueError("Anatomy role prompting requires at least four context tokens")
        if str(geometry_mode).lower() == "legacy":
            prompt_prefix = "A photo of a face showing"
            prompt_suffix_template = "expression of {emotion}."
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix_template = prompt_suffix_template

        placeholder = " ".join(["X"] * self.n_ctx)
        prompts = [
            f"{self.prompt_prefix} {placeholder} {self.prompt_suffix_template.format(emotion=name)}"
            for name in self.class_names
        ]
        tokenized_prompts = clip.tokenize(prompts)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        prefix_len = len(_tokenizer.encode(self.prompt_prefix)) + 1
        if class_specific_context:
            ctx_vectors = torch.empty(self.num_classes, self.n_ctx, embedding.shape[-1], dtype=dtype)
        else:
            ctx_vectors = torch.empty(1, self.n_ctx, embedding.shape[-1], dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        self.geometry_residual = AnatomyPromptResidual(
            num_classes=self.num_classes,
            embedding_dim=embedding.shape[-1],
            geometry_dim=geometry_dim,
            hidden_dim=geometry_hidden_dim,
            gate_init=geometry_gate_init,
            mode=geometry_mode,
        )
        self.register_buffer("token_prefix", embedding[:, :prefix_len, :])
        self.register_buffer("token_suffix", embedding[:, prefix_len + self.n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def set_geometry_statistics(self, statistics: Dict[str, torch.Tensor]) -> None:
        self.geometry_residual.set_statistics(statistics)

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        use_geometry: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            prefix = self.token_prefix
            suffix = self.token_suffix
            tokenized = self.tokenized_prompts
            if self.ctx.shape[0] == 1:
                ctx = self.ctx.expand(self.num_classes, -1, -1)
            else:
                ctx = self.ctx
        else:
            labels = labels.to(self.token_prefix.device)
            prefix = self.token_prefix[labels]
            suffix = self.token_suffix[labels]
            tokenized = self.tokenized_prompts[labels]
            if self.ctx.shape[0] == 1:
                ctx = self.ctx.expand(labels.shape[0], -1, -1)
            else:
                ctx = self.ctx[labels]

        if use_geometry:
            geometry = self.geometry_residual(labels).to(dtype=ctx.dtype)
            ctx = ctx.clone()
            ctx[:, 1:4] = ctx[:, 1:4] + geometry

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts, tokenized


def load_clip_to_cpu(backbone_name: str, h_resolution: int, w_resolution: int, vision_stride_size: int):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)


def install_expression_adapters(visual_encoder: nn.Module, bottleneck_dim: int, dropout: float) -> int:
    if not hasattr(visual_encoder, "transformer"):
        raise ValueError("Expression adapters are only implemented for CLIP ViT visual encoders")
    count = 0
    for block in visual_encoder.transformer.resblocks:
        width = block.ln_2.normalized_shape[0]
        block.emotion_adapter = ExpressionAdapter(width, bottleneck_dim=bottleneck_dim, dropout=dropout)
        count += 1
    return count


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def normalized_fusion_prior(weights: Sequence[float]) -> torch.Tensor:
    prior = torch.as_tensor(weights, dtype=torch.float32)
    if prior.shape != (3,):
        raise ValueError("fusion weights must contain classifier, global, and local values")
    if not torch.all(torch.isfinite(prior)) or torch.any(prior <= 0):
        raise ValueError("fusion weights must be finite and strictly positive")
    return prior / prior.sum()


def validate_train_last_blocks(train_last_blocks: int, num_blocks: int) -> int:
    value = int(train_last_blocks)
    if not 0 <= value <= int(num_blocks):
        raise ValueError(f"train_last_blocks must be between 0 and {num_blocks}, got {value}")
    return value


class BranchFusion(nn.Module):
    """Scale-control and simplex fusion shared by fixed and adaptive ablations."""

    def __init__(
        self,
        feature_dim: int,
        prior_weights: Sequence[float],
        gate_mode: str = "fixed",
        scale_mode: str = "temperature",
        gate_hidden_dim: int = 128,
        gate_dropout: float = 0.1,
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
        initial_temperatures: Sequence[float] = (1.0, 1.0, 1.0),
        learn_temperatures: bool = True,
    ):
        super().__init__()
        self.gate_mode = str(gate_mode).lower()
        self.scale_mode = str(scale_mode).lower()
        if self.gate_mode not in {"fixed", "simplex", "sample_dependent"}:
            raise ValueError("gate_mode must be 'fixed', 'simplex', or 'sample_dependent'")
        if self.scale_mode not in {"none", "temperature", "rms"}:
            raise ValueError("scale_mode must be 'none', 'temperature', or 'rms'")

        prior = normalized_fusion_prior(prior_weights)
        self.register_buffer("prior", prior)
        if self.gate_mode == "simplex":
            self.fusion_gate_logits = nn.Parameter(prior.log())
        elif self.gate_mode == "sample_dependent":
            self.fusion_gate = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, int(gate_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(gate_dropout)),
                nn.Linear(int(gate_hidden_dim), 3),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            with torch.no_grad():
                self.fusion_gate[-1].bias.copy_(prior.log())

        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        if not 0 < self.min_temperature < self.max_temperature:
            raise ValueError("temperature bounds must satisfy 0 < min < max")
        initial = torch.as_tensor(initial_temperatures, dtype=torch.float32)
        if initial.shape != (3,) or torch.any(initial <= self.min_temperature) or torch.any(
            initial >= self.max_temperature
        ):
            raise ValueError("initial_temperatures must contain three values strictly inside the configured bounds")
        self.register_buffer("initial_temperatures", initial)
        ratio = (initial - self.min_temperature) / (self.max_temperature - self.min_temperature)
        raw_temperatures = torch.logit(ratio)
        if self.scale_mode == "temperature" and learn_temperatures:
            self.branch_raw_temperatures = nn.Parameter(raw_temperatures)
        else:
            self.register_buffer("branch_raw_temperatures", raw_temperatures)

    def temperatures(self) -> torch.Tensor:
        if self.scale_mode != "temperature":
            return torch.ones(3, device=self.prior.device, dtype=self.prior.dtype)
        return self.min_temperature + (self.max_temperature - self.min_temperature) * torch.sigmoid(
            self.branch_raw_temperatures
        )

    def gate(self, features: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "fixed":
            return self.prior.unsqueeze(0).expand(features.shape[0], -1)
        if self.gate_mode == "simplex":
            return F.softmax(self.fusion_gate_logits, dim=0).unsqueeze(0).expand(features.shape[0], -1)
        return F.softmax(self.fusion_gate(features), dim=-1)

    def _scale(self, branch_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temperatures = self.temperatures().to(dtype=branch_logits.dtype)
        if self.scale_mode == "temperature":
            return branch_logits / temperatures.view(1, 3, 1), temperatures
        if self.scale_mode == "rms":
            centered = branch_logits - branch_logits.mean(dim=-1, keepdim=True)
            rms = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            return centered / rms, temperatures
        return branch_logits, temperatures

    def forward(
        self,
        classifier_logits: torch.Tensor,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        branch_logits = torch.stack((classifier_logits, global_logits, local_logits), dim=1)
        scaled_branch_logits, temperatures = self._scale(branch_logits)
        gate = self.gate(features)
        final_logits = (gate.unsqueeze(-1) * scaled_branch_logits).sum(dim=1)
        alignment_logits = 0.5 * (scaled_branch_logits[:, 1] + scaled_branch_logits[:, 2])

        mean_gate = gate.mean(dim=0)
        gate_kl = (
            mean_gate * (mean_gate.clamp_min(1e-8).log() - self.prior.clamp_min(1e-8).log())
        ).sum()
        if self.scale_mode == "temperature":
            temperature_regularization = (
                temperatures.log() - self.initial_temperatures.to(temperatures).log()
            ).square().mean()
        else:
            temperature_regularization = final_logits.new_zeros(())
        gate_entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "logits": final_logits,
            "alignment_logits": alignment_logits,
            "branch_logits": branch_logits,
            "scaled_branch_logits": scaled_branch_logits,
            "fusion_gate": gate,
            "branch_temperatures": temperatures,
            "gate_entropy": gate_entropy,
            "gate_regularization": gate_kl,
            "temperature_regularization": temperature_regularization,
        }


class RegionPatchRouter(nn.Module):
    """Reliability-gated mixture of landmark Gaussian masks and learned attention."""

    def __init__(
        self,
        feature_dim: int,
        grid_size: Tuple[int, int],
        mode: str = "hybrid",
        sigma: float = 0.08,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in {"free", "anatomy", "hybrid"}:
            raise ValueError("routing mode must be 'free', 'anatomy', or 'hybrid'")
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("routing sigma must be positive")
        self.key_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.region_queries = nn.Parameter(torch.empty(NUM_ANATOMY_REGIONS, feature_dim))
        nn.init.normal_(self.region_queries, std=feature_dim**-0.5)
        grid_h, grid_w = map(int, grid_size)
        ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / float(grid_h)
        xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / float(grid_w)
        mesh_y, mesh_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("patch_centers", torch.stack((mesh_x, mesh_y), dim=-1).reshape(-1, 2))

    def _anatomical_attention(
        self,
        anatomy: Dict[str, torch.Tensor],
        patch_count: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        landmarks = anatomy["region_landmarks"].to(dtype=dtype)
        weights = anatomy["region_landmark_weights"].to(dtype=dtype)
        mask = anatomy["region_landmark_mask"].to(dtype=dtype)
        uncertainty = anatomy["region_landmark_uncertainty"].to(dtype=dtype)
        centers = self.patch_centers[:patch_count].to(device=landmarks.device, dtype=dtype)
        distance = landmarks.unsqueeze(-2) - centers.view(1, 1, 1, patch_count, 2)
        distance_squared = distance.square().sum(dim=-1)
        sigma = self.sigma * (1.0 + uncertainty.clamp_min(0.0))
        gaussian = torch.exp(-distance_squared / (2.0 * sigma.unsqueeze(-1).square().clamp_min(1e-8)))
        weighted = gaussian * weights.unsqueeze(-1) * mask.unsqueeze(-1)
        scores = weighted.sum(dim=2)
        normalizer = scores.sum(dim=-1, keepdim=True)
        attention = torch.where(normalizer > 0, scores / normalizer.clamp_min(1e-8), torch.zeros_like(scores))
        return attention, normalizer.squeeze(-1) > 0

    def forward(
        self,
        patch_features: torch.Tensor,
        anatomy: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        keys = self.key_projection(patch_features)
        free_scores = torch.einsum("rd,bpd->brp", self.region_queries.to(keys), keys) / math.sqrt(keys.shape[-1])
        free_attention = F.softmax(free_scores, dim=-1)
        anatomical_attention, anatomy_valid = self._anatomical_attention(
            anatomy,
            patch_features.shape[1],
            patch_features.dtype,
        )
        quality = anatomy["region_quality"].to(device=patch_features.device, dtype=patch_features.dtype).clamp(0.0, 1.0)
        quality = quality * anatomy_valid.to(dtype=quality.dtype)
        if self.mode == "free":
            route = free_attention
        elif self.mode == "anatomy":
            route = torch.where(anatomy_valid.unsqueeze(-1), anatomical_attention, free_attention)
        else:
            route = quality.unsqueeze(-1) * anatomical_attention + (1.0 - quality.unsqueeze(-1)) * free_attention
        region_features = torch.einsum("brp,bpd->brd", route, patch_features)
        if self.mode == "hybrid":
            # Avoid an extreme 1/x gradient when the learned and anatomical
            # routes barely overlap; global gradient clipping is the second guard.
            overlap = (free_attention * anatomical_attention.detach()).sum(dim=-1).clamp_min(1e-4)
            routing_loss = (quality * -overlap.log()).sum() / quality.sum().clamp_min(1.0)
        else:
            # S1 (free) and S2 (anatomy-only) are pure routing ablations.
            # Anatomy supervision of free attention is part of S3+, not S1.
            routing_loss = free_attention.new_zeros(())
        return {
            "region_visual_features": region_features,
            "routing_attention": route,
            "free_attention": free_attention,
            "anatomical_attention": anatomical_attention,
            "region_quality": quality,
            "routing_loss": routing_loss,
        }


class AnatomyRegionFusion(nn.Module):
    """Three visual streams with zero-init geometry residuals and region logits."""

    def __init__(
        self,
        feature_dim: int,
        grid_size: Tuple[int, int],
        routing_mode: str = "hybrid",
        routing_sigma: float = 0.08,
        geometry_dim: int = MAX_GEOMETRY_FEATURES,
        geometry_hidden_dim: int = 64,
        importance_hidden_dim: int = 128,
        geometry_gate_init: float = -4.0,
        geometry_enabled: bool = True,
        fusion_mode: str = "gated_residual",
        disagreement_quality_threshold: float = 0.5,
        disagreement_min_regions: int = 2,
    ):
        super().__init__()
        self.router = RegionPatchRouter(feature_dim, grid_size, mode=routing_mode, sigma=routing_sigma)
        self.geometry_encoders = nn.ModuleList()
        self.region_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in ANATOMY_REGIONS])
        for _ in ANATOMY_REGIONS:
            encoder = nn.Sequential(
                nn.Linear(3 * int(geometry_dim), int(geometry_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(geometry_hidden_dim), feature_dim),
            )
            nn.init.zeros_(encoder[-1].weight)
            nn.init.zeros_(encoder[-1].bias)
            self.geometry_encoders.append(encoder)
        self.geometry_gate_logits = nn.Parameter(
            torch.full((NUM_ANATOMY_REGIONS,), float(geometry_gate_init))
        )
        self.geometry_enabled = bool(geometry_enabled)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"gated_residual", "cross_attention"}:
            raise ValueError("geometry fusion_mode must be 'gated_residual' or 'cross_attention'")
        if self.fusion_mode == "cross_attention" and not self.geometry_enabled:
            raise ValueError("cross_attention geometry fusion requires geometry_enabled=True")
        self.landmark_projections = nn.ModuleList()
        self.landmark_cross_attentions = nn.ModuleList()
        if self.fusion_mode == "cross_attention":
            attention_heads = 4 if feature_dim % 4 == 0 else 1
            self.landmark_projections.extend(
                nn.Linear(4, feature_dim) for _ in ANATOMY_REGIONS
            )
            self.landmark_cross_attentions.extend(
                nn.MultiheadAttention(feature_dim, attention_heads, batch_first=True)
                for _ in ANATOMY_REGIONS
            )
        self.region_importance = nn.Sequential(
            nn.LayerNorm(feature_dim + 1),
            nn.Linear(feature_dim + 1, int(importance_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(importance_hidden_dim), 1),
        )
        nn.init.zeros_(self.region_importance[-1].weight)
        nn.init.zeros_(self.region_importance[-1].bias)
        self.disagreement_quality_threshold = float(disagreement_quality_threshold)
        self.disagreement_min_regions = int(disagreement_min_regions)

    @staticmethod
    def _weighted_jsd(
        region_logits: torch.Tensor,
        quality: torch.Tensor,
        quality_threshold: float,
        minimum_regions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = F.softmax(region_logits, dim=-1)
        usable = quality > 0
        valid = (quality.sum(dim=-1) >= float(quality_threshold)) & (
            usable.sum(dim=-1) >= int(minimum_regions)
        )
        weights = quality / quality.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mixture = (weights.unsqueeze(-1) * probabilities).sum(dim=1)
        mixture_entropy = -(mixture * mixture.clamp_min(1e-8).log()).sum(dim=-1)
        region_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        disagreement = mixture_entropy - (weights * region_entropy).sum(dim=-1)
        return torch.where(valid, disagreement.clamp_min(0.0), torch.zeros_like(disagreement)), valid

    def forward(
        self,
        patch_features: torch.Tensor,
        text_features: torch.Tensor,
        anatomy: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        routed = self.router(patch_features, anatomy)
        visual = routed["region_visual_features"]
        features = anatomy["geometry_features"].to(device=patch_features.device, dtype=patch_features.dtype)
        validity = anatomy["geometry_validity"].to(device=patch_features.device, dtype=patch_features.dtype)
        uncertainty = anatomy["geometry_uncertainty"].to(device=patch_features.device, dtype=patch_features.dtype)
        encoder_input = torch.cat((features * validity, validity, uncertainty), dim=-1)
        if self.geometry_enabled:
            geometry = torch.stack(
                [encoder(encoder_input[:, index]) for index, encoder in enumerate(self.geometry_encoders)],
                dim=1,
            )
        else:
            geometry = visual.new_zeros(visual.shape)
        quality = routed["region_quality"]
        gates = torch.sigmoid(self.geometry_gate_logits).view(1, NUM_ANATOMY_REGIONS, 1)
        if self.fusion_mode == "cross_attention":
            landmark_coords = anatomy["region_landmarks"].to(device=visual.device, dtype=visual.dtype)
            landmark_weights = anatomy["region_landmark_weights"].to(device=visual.device, dtype=visual.dtype)
            landmark_uncertainty = anatomy["region_landmark_uncertainty"].to(device=visual.device, dtype=visual.dtype)
            landmark_mask = anatomy["region_landmark_mask"].to(device=visual.device).bool()
            cross_outputs = []
            for index, (projection, attention) in enumerate(
                zip(self.landmark_projections, self.landmark_cross_attentions)
            ):
                token_input = torch.cat(
                    (
                        landmark_coords[:, index],
                        landmark_weights[:, index].unsqueeze(-1),
                        landmark_uncertainty[:, index].unsqueeze(-1),
                    ),
                    dim=-1,
                )
                usable = landmark_mask[:, index]
                has_landmarks = usable.any(dim=-1)
                safe_usable = usable.clone()
                safe_usable[~has_landmarks, 0] = True
                landmark_tokens = projection(token_input)
                attended, _ = attention(
                    visual[:, index : index + 1],
                    landmark_tokens,
                    landmark_tokens,
                    key_padding_mask=~safe_usable,
                    need_weights=False,
                )
                cross_outputs.append(attended.squeeze(1) * has_landmarks.unsqueeze(-1))
            geometry = torch.stack(cross_outputs, dim=1)
            fused_regions = torch.stack(
                [
                    norm(visual[:, index] + torch.tanh(geometry[:, index]))
                    for index, norm in enumerate(self.region_norms)
                ],
                dim=1,
            )
            geometry_gates = visual.new_ones(NUM_ANATOMY_REGIONS)
        else:
            fused_regions = torch.stack(
                [
                    norm(
                        visual[:, index]
                        + gates[:, index] * quality[:, index : index + 1] * torch.tanh(geometry[:, index])
                    )
                    for index, norm in enumerate(self.region_norms)
                ],
                dim=1,
            )
            geometry_gates = torch.sigmoid(self.geometry_gate_logits)
        normalized_regions = F.normalize(fused_regions, dim=-1)
        region_logits = torch.einsum("brd,cd->brc", normalized_regions, text_features)
        importance_input = torch.cat((fused_regions, quality.unsqueeze(-1)), dim=-1)
        importance = F.softmax(self.region_importance(importance_input).squeeze(-1), dim=-1)
        local_logits = (importance.unsqueeze(-1) * region_logits).sum(dim=1)
        disagreement, disagreement_valid = self._weighted_jsd(
            region_logits,
            quality,
            self.disagreement_quality_threshold,
            self.disagreement_min_regions,
        )
        routed.update(
            {
                "region_geometry_features": geometry,
                "region_features": fused_regions,
                "region_logits": region_logits,
                "region_importance": importance,
                "local_logits": local_logits,
                "region_disagreement": disagreement,
                "region_disagreement_valid": disagreement_valid,
                "geometry_gates": geometry_gates,
                "geometry_fusion_mode": self.fusion_mode,
            }
        )
        return routed


class EmotionCLIPModel(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str] = CANONICAL_EMOTIONS,
        backbone_name: str = "ViT-B-16",
        image_size: Sequence[int] = (224, 224),
        stride_size: Sequence[int] = (16, 16),
        n_ctx: int = 4,
        prompt_geometry_mode: str = "quality",
        prompt_geometry_hidden_dim: int = 32,
        prompt_geometry_gate_init: float = -4.0,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.0,
        topk_patches: int = 5,
        global_weight: float = 1.0,
        local_weight: float = 0.5,
        classifier_weight: float = 1.0,
        train_last_blocks: int = 0,
        fusion_gate_mode: str = "fixed",
        fusion_scale_mode: str = "temperature",
        fusion_gate_hidden_dim: int = 128,
        fusion_gate_dropout: float = 0.1,
        min_branch_temperature: float = 0.05,
        max_branch_temperature: float = 20.0,
        initial_branch_temperatures: Sequence[float] = (1.0, 1.0, 1.0),
        learn_branch_temperatures: bool = True,
        routing_mode: str = "hybrid",
        routing_sigma: float = 0.08,
        geometry_hidden_dim: int = 64,
        region_importance_hidden_dim: int = 128,
        geometry_gate_init: float = -4.0,
        geometry_enabled: bool = True,
        geometry_fusion_mode: str = "gated_residual",
        disagreement_quality_threshold: float = 0.5,
        disagreement_min_regions: int = 2,
        reliability_hidden_dim: int = 128,
        reliability_dropout: float = 0.1,
        detach_class_prob: bool = True,
        max_strength: Optional[float] = 100.0,
        max_abs_raw_strength: float = 20.0,
        reliability_use_anatomy_quality: bool = True,
        reliability_detach_visual_feature: bool = True,
    ):
        super().__init__()
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.backbone_name = backbone_name
        self.topk_patches = int(topk_patches)
        self.routing_mode = str(routing_mode).lower()
        if self.routing_mode not in {"topk", "free", "anatomy", "hybrid"}:
            raise ValueError("routing_mode must be 'topk', 'free', 'anatomy', or 'hybrid'")
        self.train_last_blocks = int(train_last_blocks)
        self.detach_class_prob = bool(detach_class_prob)
        self.reliability_use_anatomy_quality = bool(reliability_use_anatomy_quality)
        self.reliability_detach_visual_feature = bool(reliability_detach_visual_feature)
        self.max_strength = None if max_strength is None else float(max_strength)
        if self.max_strength is not None and self.max_strength <= self.num_classes:
            raise ValueError("max_strength must be greater than the number of classes")
        self.max_abs_raw_strength = float(max_abs_raw_strength)
        if self.max_abs_raw_strength <= 0:
            raise ValueError("max_abs_raw_strength must be positive")

        height, width = int(image_size[0]), int(image_size[1])
        stride = int(stride_size[0])
        h_resolution = int((height - 16) // stride + 1)
        w_resolution = int((width - 16) // int(stride_size[1]) + 1)
        clip_model = load_clip_to_cpu(backbone_name, h_resolution, w_resolution, stride)

        self.image_encoder = clip_model.visual
        self.prompt_learner = EmotionPromptLearner(
            self.class_names,
            clip_model.dtype,
            clip_model.token_embedding,
            n_ctx=n_ctx,
            class_specific_context=True,
            geometry_hidden_dim=prompt_geometry_hidden_dim,
            geometry_gate_init=prompt_geometry_gate_init,
            geometry_mode=prompt_geometry_mode,
        )
        self.text_encoder = EmotionTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        if backbone_name != "ViT-B-16":
            raise ValueError("EmotionCLIPModel v1 expects a CLIP ViT visual encoder")
        self.adapter_count = install_expression_adapters(self.image_encoder, adapter_dim, adapter_dropout)
        self.train_last_blocks = validate_train_last_blocks(
            self.train_last_blocks,
            len(self.image_encoder.transformer.resblocks),
        )

        feature_dim = clip_model.text_projection.shape[1]
        self.classifier = nn.Linear(feature_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

        reliability_hidden_dim = int(reliability_hidden_dim)
        if reliability_hidden_dim <= 0:
            raise ValueError("reliability_hidden_dim must be positive")
        self.anatomy_fusion = AnatomyRegionFusion(
            feature_dim=feature_dim,
            grid_size=(h_resolution, w_resolution),
            routing_mode="free" if self.routing_mode == "topk" else self.routing_mode,
            routing_sigma=routing_sigma,
            geometry_hidden_dim=geometry_hidden_dim,
            importance_hidden_dim=region_importance_hidden_dim,
            geometry_gate_init=geometry_gate_init,
            geometry_enabled=geometry_enabled,
            fusion_mode=geometry_fusion_mode,
            disagreement_quality_threshold=disagreement_quality_threshold,
            disagreement_min_regions=disagreement_min_regions,
        )

        reliability_context_dim = feature_dim + NUM_ANATOMY_REGIONS + 5
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(reliability_context_dim),
            nn.Linear(reliability_context_dim, reliability_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(reliability_dropout)),
            nn.Linear(reliability_hidden_dim, 1),
        )
        # Stage 1 does not train this head.  A neutral final layer guarantees
        # that a fresh Stage 2 starts with finite, calibrated reliability logits
        # instead of inheriting an arbitrary random confidence scale.
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)

        self.fusion = BranchFusion(
            feature_dim=feature_dim,
            prior_weights=(classifier_weight, global_weight, local_weight),
            gate_mode=fusion_gate_mode,
            scale_mode=fusion_scale_mode,
            gate_hidden_dim=fusion_gate_hidden_dim,
            gate_dropout=fusion_gate_dropout,
            min_temperature=min_branch_temperature,
            max_temperature=max_branch_temperature,
            initial_temperatures=initial_branch_temperatures,
            learn_temperatures=learn_branch_temperatures,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_train_stage(self, stage: int) -> None:
        _set_requires_grad(self, False)
        if stage == 1:
            _set_requires_grad(self.prompt_learner, True)
            return
        if stage != 2:
            raise ValueError(f"Unknown train stage {stage}; expected 1 or 2")

        _set_requires_grad(self.classifier, True)
        if hasattr(self, "anatomy_fusion"):
            _set_requires_grad(self.anatomy_fusion, True)
            if not self.anatomy_fusion.geometry_enabled:
                _set_requires_grad(self.anatomy_fusion.geometry_encoders, False)
                self.anatomy_fusion.geometry_gate_logits.requires_grad_(False)
            elif self.anatomy_fusion.fusion_mode == "cross_attention":
                _set_requires_grad(self.anatomy_fusion.geometry_encoders, False)
                self.anatomy_fusion.geometry_gate_logits.requires_grad_(False)
            else:
                _set_requires_grad(self.anatomy_fusion.landmark_projections, False)
                _set_requires_grad(self.anatomy_fusion.landmark_cross_attentions, False)
        _set_requires_grad(self.reliability_head, True)
        _set_requires_grad(self.fusion, True)

        resblocks = self.image_encoder.transformer.resblocks
        last_start = max(0, len(resblocks) - self.train_last_blocks)
        for idx, block in enumerate(resblocks):
            _set_requires_grad(block.emotion_adapter, True)
            if idx >= last_start:
                _set_requires_grad(block, True)
                _set_requires_grad(block.emotion_adapter, True)

    def set_stage1_phase(self, phase: str) -> None:
        """Freeze the base prompt during Stage 1B geometry-residual fitting."""
        phase = str(phase).lower()
        if phase not in {"base", "geometry", "both"}:
            raise ValueError("Stage 1 phase must be 'base', 'geometry', or 'both'")
        _set_requires_grad(self.prompt_learner, False)
        if phase in {"base", "both"}:
            self.prompt_learner.ctx.requires_grad_(True)
        if phase in {"geometry", "both"}:
            _set_requires_grad(self.prompt_learner.geometry_residual, True)

    def set_class_geometry_statistics(self, statistics: Dict[str, torch.Tensor]) -> None:
        self.prompt_learner.set_geometry_statistics(statistics)

    def get_text_features(
        self,
        labels: Optional[torch.Tensor] = None,
        normalize: bool = True,
        use_geometry: bool = True,
    ) -> torch.Tensor:
        prompts, tokenized_prompts = self.prompt_learner(labels, use_geometry=use_geometry)
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.float()
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor]:
        _, image_tokens, image_tokens_proj = self.image_encoder(images.type(self.image_encoder.conv1.weight.dtype))
        image_tokens_proj = image_tokens_proj.float()
        global_feature = image_tokens_proj[:, 0]
        patch_features = image_tokens_proj[:, 1:]
        if normalize:
            global_feature = F.normalize(global_feature, dim=-1)
            patch_features = F.normalize(patch_features, dim=-1)
        return {
            "tokens": image_tokens.float(),
            "global_feature": global_feature,
            "patch_features": patch_features,
        }

    def local_alignment_logits(self, patch_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        patch_logits = torch.einsum("bpd,cd->bpc", patch_features, text_features)
        k = min(max(1, self.topk_patches), patch_logits.shape[1])
        return patch_logits.topk(k, dim=1).values.mean(dim=1)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        get_text: bool = False,
        get_image: bool = False,
        text_features: Optional[torch.Tensor] = None,
        anatomy: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if get_text:
            return self.get_text_features(labels)
        if images is None:
            raise ValueError("images must be provided unless get_text=True")

        image_outputs = self.encode_image(images)
        if get_image:
            return image_outputs["global_feature"]

        if text_features is None:
            text_features = self.get_text_features()
        else:
            text_features = F.normalize(text_features.float(), dim=-1)

        # Stage 2 uses cosine-scale logits plus one constrained temperature per branch.
        # CLIP's shared logit_scale stays frozen to avoid a redundant scale degree of freedom.
        global_logits = image_outputs["global_feature"] @ text_features.t()
        if anatomy is None:
            batch_size = images.shape[0]
            anatomy = {
                "region_landmarks": images.new_zeros(
                    batch_size, NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS, 2
                ),
                "region_landmark_weights": images.new_zeros(
                    batch_size, NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS
                ),
                "region_landmark_uncertainty": images.new_ones(
                    batch_size, NUM_ANATOMY_REGIONS, MAX_REGION_LANDMARKS
                ),
                "region_landmark_mask": torch.zeros(
                    batch_size,
                    NUM_ANATOMY_REGIONS,
                    MAX_REGION_LANDMARKS,
                    device=images.device,
                    dtype=torch.bool,
                ),
                "geometry_features": images.new_zeros(
                    batch_size, NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES
                ),
                "geometry_validity": torch.zeros(
                    batch_size,
                    NUM_ANATOMY_REGIONS,
                    MAX_GEOMETRY_FEATURES,
                    device=images.device,
                    dtype=torch.bool,
                ),
                "geometry_uncertainty": images.new_ones(
                    batch_size, NUM_ANATOMY_REGIONS, MAX_GEOMETRY_FEATURES
                ),
                "region_quality": images.new_zeros(batch_size, NUM_ANATOMY_REGIONS),
                "pose_quality": images.new_zeros(batch_size),
                "crop_quality": images.new_zeros(batch_size),
            }
        if self.routing_mode == "topk":
            local_logits = self.local_alignment_logits(image_outputs["patch_features"], text_features)
            region_quality = anatomy["region_quality"].to(local_logits).clamp(0.0, 1.0)
            anatomy_outputs = {
                "local_logits": local_logits,
                "region_quality": region_quality,
                "routing_loss": local_logits.new_zeros(()),
                "region_disagreement": local_logits.new_zeros(local_logits.shape[0]),
                "region_disagreement_valid": torch.zeros(
                    local_logits.shape[0], device=local_logits.device, dtype=torch.bool
                ),
            }
        else:
            anatomy_outputs = self.anatomy_fusion(
                image_outputs["patch_features"],
                text_features,
                anatomy,
            )
            local_logits = anatomy_outputs["local_logits"]
        classifier_logits = self.classifier(image_outputs["global_feature"])

        # Temperature division, softmax and evidential statistics are sensitive
        # to FP16 range. Keep the large CLIP encoder under AMP, but force these
        # small heads to FP32 so AMP still provides the intended GPU memory win.
        with torch.autocast(device_type=images.device.type, enabled=False):
            fusion_outputs = self.fusion(
                classifier_logits.float(),
                global_logits.float(),
                local_logits.float(),
                image_outputs["global_feature"].float(),
            )
        final_logits = fusion_outputs["logits"]
        alignment_logits = fusion_outputs["alignment_logits"]
        region_quality = anatomy_outputs["region_quality"].to(image_outputs["global_feature"])
        reliability_quality = region_quality
        pose_quality = anatomy["pose_quality"].to(region_quality).reshape(-1, 1)
        crop_quality = anatomy["crop_quality"].to(region_quality).reshape(-1, 1)
        if not self.reliability_use_anatomy_quality:
            reliability_quality = torch.zeros_like(region_quality)
            pose_quality = torch.zeros_like(pose_quality)
            crop_quality = torch.zeros_like(crop_quality)
        minimum_quality = reliability_quality.min(dim=-1, keepdim=True).values
        mean_quality = reliability_quality.mean(dim=-1, keepdim=True)
        valid_region_ratio = (reliability_quality > 0).to(region_quality.dtype).mean(dim=-1, keepdim=True)
        reliability_visual_feature = image_outputs["global_feature"]
        if self.reliability_detach_visual_feature:
            reliability_visual_feature = reliability_visual_feature.detach()
        reliability_context = torch.cat(
            (
                reliability_visual_feature,
                reliability_quality,
                minimum_quality,
                mean_quality,
                valid_region_ratio,
                pose_quality,
                crop_quality,
            ),
            dim=-1,
        )
        with torch.autocast(device_type=images.device.type, enabled=False):
            reliability_context = reliability_context.float()
            raw_strength_unbounded = self.reliability_head(reliability_context).squeeze(-1)
            raw_strength = self.max_abs_raw_strength * torch.tanh(
                raw_strength_unbounded / self.max_abs_raw_strength
            )
            strength = float(self.num_classes) + F.softplus(raw_strength)
            if self.max_strength is not None:
                strength = strength.clamp_max(self.max_strength)
            evidential = decoupled_dirichlet(
                final_logits.float(),
                strength,
                detach_class_prob=self.detach_class_prob,
            )

            probabilities = evidential["probabilities"]
            class_ambiguity = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        outputs = {
            "logits": final_logits,
            "classifier_logits": classifier_logits,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "alignment_logits": alignment_logits,
            "scaled_branch_logits": fusion_outputs["scaled_branch_logits"],
            "fusion_gate": fusion_outputs["fusion_gate"],
            "branch_temperatures": fusion_outputs["branch_temperatures"],
            "gate_entropy": fusion_outputs["gate_entropy"],
            "gate_regularization": fusion_outputs["gate_regularization"],
            "temperature_regularization": fusion_outputs["temperature_regularization"],
            "probabilities": evidential["probabilities"],
            "dirichlet_mean": evidential["dirichlet_mean"],
            "alpha": evidential["alpha"],
            "strength": evidential["strength"],
            "raw_strength": raw_strength,
            "raw_strength_unbounded": raw_strength_unbounded,
            "uncertainty": evidential["uncertainty"],
            "extrinsic_unreliability": evidential["uncertainty"],
            "class_ambiguity": class_ambiguity,
            "region_disagreement": anatomy_outputs["region_disagreement"],
            "region_disagreement_valid": anatomy_outputs["region_disagreement_valid"],
            "region_quality": region_quality,
            "routing_loss": anatomy_outputs["routing_loss"],
            "global_feature": image_outputs["global_feature"],
            "patch_features": image_outputs["patch_features"],
            "text_features": text_features,
        }
        for key in (
            "region_logits",
            "region_importance",
            "routing_attention",
            "free_attention",
            "anatomical_attention",
            "region_features",
            "region_geometry_features",
            "geometry_gates",
        ):
            if key in anatomy_outputs:
                outputs[key] = anatomy_outputs[key]
        return outputs

    def save_stage1_descriptors(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with torch.no_grad():
            descriptors = self.get_text_features().detach().cpu()
        torch.save({"class_names": self.class_names, "text_features": descriptors}, output_path)
