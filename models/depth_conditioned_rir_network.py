from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .rir_network import FourierFeatureEncoding


class DepthMapEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if activation is None:
            activation = nn.SiLU()

        def act() -> nn.Module:
            return activation.__class__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            act(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            act(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            act(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embedding_dim),
            act(),
        )

    def forward(self, depth_map: Tensor) -> Tensor:
        if depth_map.ndim != 3:
            raise ValueError(
                f"Expected depth_map with shape (batch, height, width), got {depth_map.shape}"
            )
        x = depth_map.unsqueeze(1)
        x = self.features(x)
        return self.projection(x)


class DepthConditionedRIRNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 16000,
        hidden_dim: int = 384,
        num_hidden_layers: int = 4,
        num_frequencies: int = 8,
        depth_embedding_dim: int = 128,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        self.output_dim = output_dim
        self.encoding = FourierFeatureEncoding(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            include_input=True,
        )

        if activation is None:
            activation = nn.SiLU()

        self.depth_encoder = DepthMapEncoder(
            embedding_dim=depth_embedding_dim,
            activation=activation,
        )

        def act() -> nn.Module:
            return activation.__class__()

        mlp_input_dim = self.encoding.output_dim + depth_embedding_dim
        layers = []
        current_dim = mlp_input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(act())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coordinates: Tensor, depth_map: Tensor) -> Tensor:
        if coordinates.shape[-1] != self.encoding.input_dim:
            raise ValueError(
                f"Expected coordinates with last dimension {self.encoding.input_dim}, got {coordinates.shape[-1]}"
            )
        if depth_map.shape[:-2] != coordinates.shape[:-1]:
            raise ValueError(
                "depth_map leading dimensions must match coordinate leading dimensions"
            )

        leading_shape = coordinates.shape[:-1]
        flat_coordinates = coordinates.reshape(-1, coordinates.shape[-1])
        flat_depth_map = depth_map.reshape(-1, depth_map.shape[-2], depth_map.shape[-1])

        encoded_coordinates = self.encoding(flat_coordinates)
        encoded_depth = self.depth_encoder(flat_depth_map)
        features = torch.cat([encoded_coordinates, encoded_depth], dim=-1)
        output = self.mlp(features)
        return output.reshape(*leading_shape, self.output_dim)
