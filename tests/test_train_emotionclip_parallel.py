import pytest
import torch

from processor.processor_emotionclip import (
    EmotionDataParallel,
    _model_output_error,
    get_shared_text_features,
    unwrap_model,
)
from train_emotionclip import parse_gpu_ids, validate_gpu_topology


def test_parse_gpu_ids_accepts_two_cuda_indices():
    assert parse_gpu_ids("0,1") == [0, 1]
    assert parse_gpu_ids("") == []


def test_production_training_rejects_single_process_multi_gpu():
    validate_gpu_topology([0])
    with pytest.raises(ValueError, match="DistributedDataParallel"):
        validate_gpu_topology([0, 1])


@pytest.mark.parametrize("value", ["0,0", "0,-1", "0,x"])
def test_parse_gpu_ids_rejects_invalid_lists(value):
    with pytest.raises(ValueError):
        parse_gpu_ids(value)


def test_unwrap_model_preserves_plain_model_and_unwraps_data_parallel():
    model = torch.nn.Linear(2, 1)
    assert unwrap_model(model) is model
    assert unwrap_model(torch.nn.DataParallel(model)) is model


def test_emotion_data_parallel_gathers_batch_and_shared_outputs_correctly():
    parallel = EmotionDataParallel(torch.nn.Identity(), device_ids=[])
    outputs = [
        {
            "logits": torch.zeros(2, 7),
            "branch_temperatures": torch.tensor([1.0, 2.0, 3.0]),
            "gate_regularization": torch.tensor(1.0),
            "text_features": torch.ones(7, 4),
        },
        {
            "logits": torch.ones(3, 7),
            "branch_temperatures": torch.tensor([1.0, 2.0, 3.0]),
            "gate_regularization": torch.tensor(3.0),
            "text_features": torch.ones(7, 4),
        },
    ]
    gathered = parallel.gather(outputs, output_device=0)
    assert gathered["logits"].shape == (5, 7)
    assert gathered["branch_temperatures"].shape == (3,)
    assert torch.equal(gathered["branch_temperatures"], torch.tensor([1.0, 2.0, 3.0]))
    assert gathered["gate_regularization"].ndim == 0
    assert float(gathered["gate_regularization"]) == pytest.approx(2.2)
    assert gathered["text_features"].shape == (7, 4)


def test_emotion_data_parallel_rejects_divergent_shared_temperatures():
    parallel = EmotionDataParallel(torch.nn.Identity(), device_ids=[])
    outputs = [
        {"logits": torch.zeros(3, 7), "branch_temperatures": torch.ones(3)},
        {"logits": torch.zeros(3, 7), "branch_temperatures": torch.tensor([0.5, 1.0, 1.0])},
    ]

    with pytest.raises(RuntimeError, match="inconsistent shared output 'branch_temperatures'"):
        parallel.gather(outputs, output_device=0)


def test_emotion_data_parallel_rejects_multi_gpu_construction():
    with pytest.raises(RuntimeError, match="multi-GPU execution is disabled"):
        EmotionDataParallel(torch.nn.Identity(), device_ids=[0, 1])


def test_model_output_invariant_rejects_impossible_confidence():
    probabilities = torch.full((2, 7), 1.0 / 7.0)
    probabilities[0, 0] = 2.0
    outputs = {
        "logits": torch.zeros(2, 7),
        "alignment_logits": torch.zeros(2, 7),
        "probabilities": probabilities,
        "raw_strength": torch.zeros(2),
        "uncertainty": torch.ones(2),
    }

    assert "outside [0, 1]" in _model_output_error(outputs)


def test_shared_text_features_are_built_from_unwrapped_model():
    class TextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.descriptors = torch.nn.Parameter(torch.randn(7, 4))

        def get_text_features(self):
            return self.descriptors

    core_model = TextModel()
    parallel = EmotionDataParallel(core_model, device_ids=[])

    descriptors = get_shared_text_features(parallel)

    assert descriptors.shape == (7, 4)
    assert descriptors is core_model.descriptors
