import torch

from models.base_model import BaseModel


def test_base_model_output_shape_for_flat_inputs() -> None:
    model = BaseModel(output_dim=256)
    features = torch.randn(5, 16)

    output = model(features)

    assert output.shape == (5, 256)


def test_base_model_output_shape_for_lstm_shaped_inputs() -> None:
    model = BaseModel(output_dim=128)
    features = torch.randn(4, 1, 16)

    output = model(features)

    assert output.shape == (4, 128)


def test_base_model_supports_additional_leading_dimensions() -> None:
    model = BaseModel(output_dim=64)
    features = torch.randn(2, 3, 1, 16)

    output = model(features)

    assert output.shape == (2, 3, 64)


def test_base_model_backpropagates() -> None:
    model = BaseModel(output_dim=32)
    features = torch.randn(3, 16, requires_grad=True)

    output = model(features)
    loss = output.sum()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
