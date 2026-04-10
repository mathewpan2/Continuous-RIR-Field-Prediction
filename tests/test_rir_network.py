import torch

from models.rir_network import RIRNetwork


def test_rir_network_output_shape() -> None:
    model = RIRNetwork()
    coordinates = torch.randn(5, 6)

    output = model(coordinates)

    assert output.shape == (5, 16000)


def test_rir_network_supports_batched_inputs_and_room_context() -> None:
    model = RIRNetwork(room_context_dim=4)
    coordinates = torch.randn(2, 3, 6)
    room_context = torch.randn(2, 3, 4)

    output = model(coordinates, room_context=room_context)

    assert output.shape == (2, 3, 16000)


def test_rir_network_accepts_room_context_none() -> None:
    model = RIRNetwork()
    coordinates = torch.randn(1, 6)

    output = model(coordinates, room_context=None)

    assert output.shape == (1, 16000)


def test_rir_network_backpropagates() -> None:
    model = RIRNetwork()
    coordinates = torch.randn(4, 6, requires_grad=True)

    output = model(coordinates)
    loss = output.sum()
    loss.backward()

    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
