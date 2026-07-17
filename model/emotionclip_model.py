import os
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.emotion_manifest import CANONICAL_EMOTIONS
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


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


class EmotionPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str],
        dtype: torch.dtype,
        token_embedding: nn.Embedding,
        n_ctx: int = 4,
        prompt_prefix: str = "A photo of a face showing",
        prompt_suffix_template: str = "expression of {emotion}.",
        class_specific_context: bool = True,
    ):
        super().__init__()
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.n_ctx = int(n_ctx)
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
        self.register_buffer("token_prefix", embedding[:, :prefix_len, :])
        self.register_buffer("token_suffix", embedding[:, prefix_len + self.n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        initial_temperatures: Sequence[float] = (0.1, 1.0, 1.0),
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


class EmotionCLIPModel(nn.Module):
    def __init__(
        self,
        class_names: Sequence[str] = CANONICAL_EMOTIONS,
        backbone_name: str = "ViT-B-16",
        image_size: Sequence[int] = (224, 224),
        stride_size: Sequence[int] = (16, 16),
        n_ctx: int = 4,
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
        initial_branch_temperatures: Sequence[float] = (0.1, 1.0, 1.0),
        learn_branch_temperatures: bool = True,
        reliability_hidden_dim: int = 128,
        reliability_dropout: float = 0.1,
        detach_class_prob: bool = True,
        max_strength: Optional[float] = 100.0,
    ):
        super().__init__()
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.backbone_name = backbone_name
        self.topk_patches = int(topk_patches)
        self.train_last_blocks = int(train_last_blocks)
        self.detach_class_prob = bool(detach_class_prob)
        self.max_strength = None if max_strength is None else float(max_strength)
        if self.max_strength is not None and self.max_strength <= self.num_classes:
            raise ValueError("max_strength must be greater than the number of classes")

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
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, reliability_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(reliability_dropout)),
            nn.Linear(reliability_hidden_dim, 1),
        )

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
        _set_requires_grad(self.reliability_head, True)
        _set_requires_grad(self.fusion, True)

        resblocks = self.image_encoder.transformer.resblocks
        last_start = max(0, len(resblocks) - self.train_last_blocks)
        for idx, block in enumerate(resblocks):
            _set_requires_grad(block.emotion_adapter, True)
            if idx >= last_start:
                _set_requires_grad(block, True)
                _set_requires_grad(block.emotion_adapter, True)

    def get_text_features(
        self,
        labels: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        prompts, tokenized_prompts = self.prompt_learner(labels)
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
        local_logits = self.local_alignment_logits(image_outputs["patch_features"], text_features)
        classifier_logits = self.classifier(image_outputs["global_feature"])

        fusion_outputs = self.fusion(
            classifier_logits,
            global_logits,
            local_logits,
            image_outputs["global_feature"],
        )
        final_logits = fusion_outputs["logits"]
        alignment_logits = fusion_outputs["alignment_logits"]
        raw_strength = self.reliability_head(image_outputs["global_feature"]).squeeze(-1)
        strength = float(self.num_classes) + F.softplus(raw_strength)
        if self.max_strength is not None:
            strength = strength.clamp_max(self.max_strength)
        evidential = decoupled_dirichlet(
            final_logits,
            strength,
            detach_class_prob=self.detach_class_prob,
        )

        return {
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
            "uncertainty": evidential["uncertainty"],
            "global_feature": image_outputs["global_feature"],
            "patch_features": image_outputs["patch_features"],
            "text_features": text_features,
        }

    def save_stage1_descriptors(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with torch.no_grad():
            descriptors = self.get_text_features().detach().cpu()
        torch.save({"class_names": self.class_names, "text_features": descriptors}, output_path)
