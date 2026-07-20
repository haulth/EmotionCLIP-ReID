import pytest
import torch

from processor.processor_emotionclip import EmotionDataParallel, unwrap_model
from train_emotionclip import parse_gpu_ids


def test_parse_gpu_ids_accepts_two_cuda_indices():
    assert parse_gpu_ids("0,1") == [0, 1]
    assert parse_gpu_ids("") == []


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
            "branch_temperatures": torch.tensor([3.0, 4.0, 5.0]),
            "gate_regularization": torch.tensor(3.0),
            "text_features": torch.ones(7, 4) * 2,
        },
    ]
    gathered = parallel.gather(outputs, output_device=0)
    assert gathered["logits"].shape == (5, 7)
    assert gathered["branch_temperatures"].shape == (3,)
    assert torch.equal(gathered["branch_temperatures"], torch.tensor([2.0, 3.0, 4.0]))
    assert gathered["gate_regularization"].ndim == 0
    assert float(gathered["gate_regularization"]) == 2.0
    assert gathered["text_features"].shape == (7, 4)
