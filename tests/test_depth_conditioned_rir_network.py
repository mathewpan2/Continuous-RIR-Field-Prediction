import torch

from models.depth_conditioned_rir_network import DepthConditionedRIRNetwork


def test_depth_conditioned_rir_network_output_shape() -> None:
    model = DepthConditionedRIRNetwork(output_dim=32)
    coordinates = torch.randn(3, 6)
    depth_map = torch.randn(3, 256, 512)

    output = model(coordinates, depth_map)

    assert output.shape == (3, 32)


def test_depth_conditioned_rir_network_supports_leading_dimensions() -> None:
    model = DepthConditionedRIRNetwork(output_dim=16)
    coordinates = torch.randn(2, 4, 6)
    depth_map = torch.randn(2, 4, 256, 512)

    output = model(coordinates, depth_map)

    assert output.shape == (2, 4, 16)


def test_depth_conditioned_rir_network_backpropagates() -> None:
    model = DepthConditionedRIRNetwork(output_dim=8)
    coordinates = torch.randn(2, 6, requires_grad=True)
    depth_map = torch.randn(2, 256, 512, requires_grad=True)

    output = model(coordinates, depth_map)
    loss = output.sum()
    loss.backward()

    assert coordinates.grad is not None
    assert depth_map.grad is not None
