import torch

from models.traditional_way_baseline import TraditionalWayBaseline


def test_traditional_way_baseline_output_shape() -> None:
    model = TraditionalWayBaseline(
        output_dim=128,
        sample_rate=8000,
        room_bounds=[[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]],
    )
    coordinates = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

    output = model(coordinates)

    assert output.shape == (1, 128)
    assert torch.isfinite(output).all()


def test_traditional_way_baseline_places_direct_path_impulse() -> None:
    model = TraditionalWayBaseline(
        output_dim=64,
        sample_rate=3430,
        room_bounds=[[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]],
        reflection_gain=0.0,
    )
    coordinates = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

    output = model(coordinates)

    assert int(torch.argmax(output[0]).item()) == 10
    assert torch.isclose(output[0, 10], torch.tensor(1.0))


def test_traditional_way_baseline_supports_leading_dimensions() -> None:
    model = TraditionalWayBaseline(output_dim=32, sample_rate=8000)
    coordinates = torch.randn(2, 3, 6)

    output = model(coordinates)

    assert output.shape == (2, 3, 32)
