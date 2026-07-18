import torch

import model.emotionclip_model as emotionclip_model_module
from datasets.anatomy import (
    empty_anatomy_inputs,
    fit_class_geometry_statistics,
    geometry_feature_definition_mask,
)
from datasets.emotion_manifest import CANONICAL_EMOTIONS
from model.emotionclip_model import (
    AnatomyPromptResidual,
    AnatomyRegionFusion,
    BranchFusion,
    EmotionCLIPModel,
    EmotionPromptLearner,
    ExpressionAdapter,
    RegionPatchRouter,
    validate_train_last_blocks,
)


def test_emotion_prompt_learner_shapes():
    token_embedding = torch.nn.Embedding(50000, 512)
    learner = EmotionPromptLearner(
        CANONICAL_EMOTIONS,
        dtype=torch.float32,
        token_embedding=token_embedding,
        n_ctx=4,
    )
    prompts, tokenized = learner()
    assert prompts.shape == (len(CANONICAL_EMOTIONS), 77, 512)
    assert tokenized.shape == (len(CANONICAL_EMOTIONS), 77)

    labels = torch.tensor([0, 3])
    prompts, tokenized = learner(labels)
    assert prompts.shape == (2, 77, 512)
    assert tokenized.shape == (2, 77)


def test_expression_adapter_zero_initialized_residual():
    adapter = ExpressionAdapter(dim=8, bottleneck_dim=4)
    x = torch.randn(3, 2, 8)
    y = adapter(x)
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_fixed_fusion_has_simplex_gate_bounded_temperatures_and_gradients():
    fusion = BranchFusion(
        feature_dim=8,
        prior_weights=(1.0, 1.0, 0.5),
        gate_mode="fixed",
        scale_mode="temperature",
    )
    features = torch.randn(4, 8)
    branches = [torch.randn(4, 7, requires_grad=True) for _ in range(3)]
    outputs = fusion(*branches, features)

    expected_gate = torch.tensor([0.4, 0.4, 0.2]).expand(4, -1)
    torch.testing.assert_close(outputs["fusion_gate"], expected_gate)
    torch.testing.assert_close(outputs["fusion_gate"].sum(dim=-1), torch.ones(4))
    assert torch.all(outputs["fusion_gate"] >= 0)
    assert outputs["logits"].shape == (4, 7)
    assert torch.all(torch.isfinite(outputs["branch_temperatures"]))
    assert torch.all(outputs["branch_temperatures"] >= 0.05)
    assert torch.all(outputs["branch_temperatures"] <= 20.0)

    outputs["logits"].sum().backward()
    assert fusion.branch_raw_temperatures.grad is not None
    assert not hasattr(fusion, "fusion_gate_logits")


def test_simplex_and_sample_dependent_fusion_are_valid_ablation_modes():
    features = torch.randn(5, 8)
    branches = [torch.randn(5, 7) for _ in range(3)]
    for mode in ("simplex", "sample_dependent"):
        fusion = BranchFusion(
            feature_dim=8,
            prior_weights=(1.0, 1.0, 0.5),
            gate_mode=mode,
            scale_mode="rms",
        )
        outputs = fusion(*branches, features)
        assert outputs["fusion_gate"].shape == (5, 3)
        torch.testing.assert_close(outputs["fusion_gate"].sum(dim=-1), torch.ones(5))
        torch.testing.assert_close(outputs["fusion_gate"][0], torch.tensor([0.4, 0.4, 0.2]))
        outputs["logits"].sum().backward()
        if mode == "simplex":
            assert fusion.fusion_gate_logits.grad is not None
        else:
            assert fusion.fusion_gate[-1].weight.grad is not None


def test_train_last_blocks_validation():
    assert validate_train_last_blocks(0, 12) == 0
    assert validate_train_last_blocks(12, 12) == 12
    for invalid in (-1, 13):
        try:
            validate_train_last_blocks(invalid, 12)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected train_last_blocks={invalid} to fail")


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.original = torch.nn.Linear(4, 4)
        self.emotion_adapter = ExpressionAdapter(4, bottleneck_dim=2)


def _stage_test_model(last_blocks):
    model = EmotionCLIPModel.__new__(EmotionCLIPModel)
    torch.nn.Module.__init__(model)
    model.train_last_blocks = last_blocks
    model.prompt_learner = torch.nn.Linear(4, 4)
    model.classifier = torch.nn.Linear(4, 3)
    model.reliability_head = torch.nn.Linear(4, 1)
    model.fusion = BranchFusion(4, (1.0, 1.0, 0.5))
    model.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
    transformer = torch.nn.Module()
    transformer.resblocks = torch.nn.ModuleList([_DummyBlock(), _DummyBlock()])
    model.image_encoder = torch.nn.Module()
    model.image_encoder.transformer = transformer
    return model


def test_adapter_only_stage2_trainable_parameter_whitelist():
    model = _stage_test_model(last_blocks=0)
    model.set_train_stage(2)
    for name, parameter in model.named_parameters():
        if name.startswith("image_encoder"):
            if ".emotion_adapter." in name:
                assert parameter.requires_grad, name
            else:
                assert not parameter.requires_grad, name
    assert not model.logit_scale.requires_grad
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.reliability_head.parameters())
    assert model.fusion.branch_raw_temperatures.requires_grad


def _router_anatomy(quality):
    return {
        "region_landmarks": torch.tensor([[[[0.25, 0.25]], [[0.75, 0.25]], [[0.5, 0.75]]]]),
        "region_landmark_weights": torch.ones(1, 3, 1),
        "region_landmark_uncertainty": torch.zeros(1, 3, 1),
        "region_landmark_mask": torch.ones(1, 3, 1, dtype=torch.bool),
        "region_quality": torch.full((1, 3), float(quality)),
        "geometry_features": torch.zeros(1, 3, 12),
        "geometry_validity": torch.ones(1, 3, 12, dtype=torch.bool),
        "geometry_uncertainty": torch.zeros(1, 3, 12),
    }


def test_hybrid_router_is_exact_free_attention_fallback_at_zero_quality():
    router = RegionPatchRouter(feature_dim=8, grid_size=(2, 2), mode="hybrid", sigma=0.1)
    outputs = router(torch.randn(1, 4, 8), _router_anatomy(quality=0.0))
    torch.testing.assert_close(outputs["routing_attention"], outputs["free_attention"])
    torch.testing.assert_close(outputs["routing_attention"].sum(dim=-1), torch.ones(1, 3))


def test_routing_supervision_is_only_active_for_hybrid_ablation():
    patches = torch.randn(1, 4, 8)
    anatomy = _router_anatomy(quality=1.0)
    for mode in ("free", "anatomy"):
        outputs = RegionPatchRouter(8, (2, 2), mode=mode, sigma=0.1)(patches, anatomy)
        torch.testing.assert_close(outputs["routing_loss"], torch.tensor(0.0))
    hybrid_outputs = RegionPatchRouter(8, (2, 2), mode="hybrid", sigma=0.1)(
        patches,
        anatomy,
    )
    assert float(hybrid_outputs["routing_loss"].detach()) > 0.0


def test_geometry_fusion_zero_init_does_not_change_visual_regions():
    fusion = AnatomyRegionFusion(feature_dim=8, grid_size=(2, 2), routing_mode="free")
    patches = torch.randn(1, 4, 8)
    text = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    first = _router_anatomy(quality=1.0)
    second = {key: value.clone() for key, value in first.items()}
    second["geometry_features"].fill_(100.0)
    first_outputs = fusion(patches, text, first)
    second_outputs = fusion(patches, text, second)
    torch.testing.assert_close(first_outputs["region_features"], second_outputs["region_features"])


def test_unconditional_cross_attention_has_missing_region_visual_fallback():
    fusion = AnatomyRegionFusion(
        feature_dim=8,
        grid_size=(2, 2),
        routing_mode="free",
        fusion_mode="cross_attention",
    )
    patches = torch.randn(1, 4, 8)
    text = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    anatomy = _router_anatomy(quality=0.0)
    anatomy["region_landmark_mask"].zero_()
    routed = fusion.router(patches, anatomy)["region_visual_features"]
    expected = torch.stack(
        [norm(routed[:, index]) for index, norm in enumerate(fusion.region_norms)],
        dim=1,
    )
    outputs = fusion(patches, text, anatomy)
    assert torch.isfinite(outputs["region_features"]).all()
    torch.testing.assert_close(outputs["region_features"], expected)


def test_reliable_region_disagreement_requires_two_quality_regions():
    logits = torch.zeros(2, 3, 7)
    disagreement, valid = AnatomyRegionFusion._weighted_jsd(
        logits,
        torch.tensor([[0.4, 0.4, 0.0], [1.0, 0.0, 0.0]]),
        quality_threshold=0.5,
        minimum_regions=2,
    )
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(disagreement, torch.zeros(2))


def test_prompt_geometry_and_class_statistics_are_zero_init_and_robust():
    features = torch.zeros(5, 3, 12)
    features[:, :, 0] = torch.tensor([1.0, 2.0, 100.0, 4.0, 5.0]).view(-1, 1)
    validity = torch.ones_like(features, dtype=torch.bool)
    uncertainty = torch.zeros_like(features)
    quality = torch.ones(5, 3)
    labels = torch.tensor([0, 0, 0, 1, 1])
    statistics = fit_class_geometry_statistics(
        features,
        validity,
        uncertainty,
        quality,
        labels,
        num_classes=2,
        minimum_samples=1,
    )
    torch.testing.assert_close(statistics["median"][0, :, 0], torch.full((3,), 2.0))
    torch.testing.assert_close(statistics["scale"][0, :, 0], torch.full((3,), 1.4826))

    residual = AnatomyPromptResidual(num_classes=2, embedding_dim=8, mode="quality")
    residual.set_statistics(statistics)
    torch.testing.assert_close(residual(), torch.zeros(2, 3, 8))


def test_class_quality_ignores_reserved_padding_slots():
    features = torch.zeros(2, 3, 12)
    definition = geometry_feature_definition_mask()
    validity = definition.unsqueeze(0).expand(2, -1, -1).clone()
    statistics = fit_class_geometry_statistics(
        features,
        validity,
        torch.zeros_like(features),
        torch.ones(2, 3),
        torch.zeros(2, dtype=torch.long),
        num_classes=1,
        minimum_samples=1,
    )
    torch.testing.assert_close(statistics["quality"][0], torch.ones(3))


class _TinyResidualBlock(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.ln_2 = torch.nn.LayerNorm(width)


class _TinyVisual(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=16, stride=16, bias=False)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_TinyResidualBlock(width)])

    def forward(self, images):
        patches = self.conv1(images).flatten(2).transpose(1, 2)
        global_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat((global_token, patches), dim=1)
        return tokens, tokens, tokens


class _TinyClip(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.dtype = torch.float32
        self.visual = _TinyVisual(width)
        self.token_embedding = torch.nn.Embedding(50000, width)
        self.transformer = torch.nn.Identity()
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, width))
        self.ln_final = torch.nn.LayerNorm(width)
        self.text_projection = torch.nn.Parameter(torch.eye(width))
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))


def test_full_emotionclip_constructor_and_anatomy_forward_integration(monkeypatch):
    monkeypatch.setattr(
        emotionclip_model_module,
        "load_clip_to_cpu",
        lambda *_args, **_kwargs: _TinyClip(),
    )
    model = EmotionCLIPModel(
        backbone_name="ViT-B-16",
        image_size=(32, 32),
        stride_size=(16, 16),
        adapter_dim=4,
        routing_mode="hybrid",
        geometry_hidden_dim=4,
        region_importance_hidden_dim=4,
        reliability_hidden_dim=4,
    )
    single = empty_anatomy_inputs()
    anatomy = {
        key: torch.stack((value, value), dim=0)
        for key, value in single.items()
    }
    outputs = model(images=torch.randn(2, 3, 32, 32), anatomy=anatomy)
    assert outputs["logits"].shape == (2, 7)
    assert outputs["region_logits"].shape == (2, 3, 7)
    assert outputs["routing_attention"].shape == (2, 3, 4)
    assert torch.isfinite(outputs["uncertainty"]).all()
    outputs["raw_strength"].sum().backward()
    assert model.reliability_head[-1].weight.grad is not None
    assert model.image_encoder.conv1.weight.grad is None
