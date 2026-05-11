from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class TraditionalWayBaseline(nn.Module):
    """Image-source early RIR baseline using direct path and first-order reflections."""

    def __init__(
        self,
        output_dim: int,
        sample_rate: int = 22050,
        room_bounds: Sequence[Sequence[float]] | Tensor | None = None,
        speed_of_sound: float = 343.0,
        reflection_gain: float = 0.6,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if speed_of_sound <= 0.0:
            raise ValueError("speed_of_sound must be positive")
        if reflection_gain < 0.0:
            raise ValueError("reflection_gain must be non-negative")

        if room_bounds is None:
            room_bounds_tensor = torch.tensor(
                [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]],
                dtype=torch.float32,
            )
        else:
            room_bounds_tensor = torch.as_tensor(room_bounds, dtype=torch.float32)
        if room_bounds_tensor.shape != (2, 3):
            raise ValueError(
                f"room_bounds must have shape (2, 3), got {tuple(room_bounds_tensor.shape)}"
            )
        if torch.any(room_bounds_tensor[1] <= room_bounds_tensor[0]):
            raise ValueError("room_bounds max corner must be greater than min corner")

        self.output_dim = int(output_dim)
        self.sample_rate = int(sample_rate)
        self.speed_of_sound = float(speed_of_sound)
        self.reflection_gain = float(reflection_gain)
        self.register_buffer("room_bounds", room_bounds_tensor)

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != 6:
            raise ValueError(
                f"Expected coordinates with last dimension 6, got {coordinates.shape[-1]}"
            )

        leading_shape = coordinates.shape[:-1]
        flat_coordinates = coordinates.reshape(-1, 6)
        sources = flat_coordinates[:, :3]
        receivers = flat_coordinates[:, 3:]
        room_bounds = self.room_bounds.to(device=coordinates.device, dtype=coordinates.dtype)
        output = torch.zeros(
            flat_coordinates.shape[0],
            self.output_dim,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

        path_points: list[tuple[Tensor, float]] = [(sources, 1.0)]
        for axis in range(3):
            low_images = sources.clone()
            low_images[:, axis] = 2.0 * room_bounds[0, axis] - sources[:, axis]
            high_images = sources.clone()
            high_images[:, axis] = 2.0 * room_bounds[1, axis] - sources[:, axis]
            path_points.append((low_images, self.reflection_gain))
            path_points.append((high_images, self.reflection_gain))

        batch_index = torch.arange(flat_coordinates.shape[0], device=coordinates.device)
        eps = torch.finfo(coordinates.dtype).eps
        for image_sources, gain in path_points:
            distances = torch.linalg.norm(image_sources - receivers, dim=-1).clamp_min(eps)
            delays = torch.round(distances / self.speed_of_sound * self.sample_rate).to(torch.long)
            amplitudes = gain / distances
            valid = (delays >= 0) & (delays < self.output_dim)
            if torch.any(valid):
                output[batch_index[valid], delays[valid]] += amplitudes[valid]

        peak = output.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        output = output / peak
        return output.reshape(*leading_shape, self.output_dim)
