from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class FourierFeatureEncoding(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        num_frequencies: int = 8,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        frequencies = torch.pow(2.0, torch.arange(num_frequencies, dtype=torch.float32))
        frequencies = frequencies * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        base_dim = self.input_dim if self.include_input else 0
        return base_dim + self.input_dim * self.num_frequencies * 2

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension to be {self.input_dim}, got {coordinates.shape[-1]}"
            )

        encoded_parts = []
        if self.include_input:
            encoded_parts.append(coordinates)

        scaled = coordinates.unsqueeze(-1) * self.frequencies
        encoded_parts.append(torch.sin(scaled).reshape(*coordinates.shape[:-1], -1))
        encoded_parts.append(torch.cos(scaled).reshape(*coordinates.shape[:-1], -1))
        return torch.cat(encoded_parts, dim=-1)


class RIRNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 16000,
        hidden_dim: int = 512,
        num_hidden_layers: int = 4,
        num_frequencies: int = 8,
        room_context_dim: int = 0,
        room_context_hidden_dim: int = 128,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if room_context_dim < 0:
            raise ValueError("room_context_dim must be non-negative")

        self.output_dim = output_dim
        self.room_context_dim = room_context_dim

        self.encoding = FourierFeatureEncoding(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            include_input=True,
        )

        self.room_context_encoder: Optional[nn.Module]
        if room_context_dim > 0:
            self.room_context_encoder = nn.Sequential(
                nn.Linear(room_context_dim, room_context_hidden_dim),
                nn.SiLU(),
                nn.Linear(room_context_hidden_dim, room_context_hidden_dim),
                nn.SiLU(),
            )
            room_context_feature_dim = room_context_hidden_dim
        else:
            self.room_context_encoder = None
            room_context_feature_dim = 0

        if activation is None:
            activation = nn.SiLU()

        mlp_input_dim = self.encoding.output_dim + room_context_feature_dim
        layers = []
        current_dim = mlp_input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation.__class__())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Tensor,
        room_context: Optional[Tensor] = None,
    ) -> Tensor:
        if coordinates.shape[-1] != self.encoding.input_dim:
            raise ValueError(
                f"Expected coordinates with last dimension {self.encoding.input_dim}, got {coordinates.shape[-1]}"
            )

        leading_shape = coordinates.shape[:-1]
        flat_coordinates = coordinates.reshape(-1, coordinates.shape[-1])
        encoded_coordinates = self.encoding(flat_coordinates)

        features = encoded_coordinates
        if room_context is not None:
            if self.room_context_encoder is None:
                raise ValueError(
                    "room_context was provided but room_context_dim was set to 0"
                )
            if room_context.shape[:-1] != leading_shape:
                raise ValueError(
                    "room_context must have the same leading dimensions as coordinates"
                )
            if room_context.shape[-1] != self.room_context_dim:
                raise ValueError(
                    f"Expected room_context with last dimension {self.room_context_dim}, got {room_context.shape[-1]}"
                )

            flat_room_context = room_context.reshape(-1, room_context.shape[-1])
            room_context_features = self.room_context_encoder(flat_room_context)
            features = torch.cat([features, room_context_features], dim=-1)

        output = self.mlp(features)
        return output.reshape(*leading_shape, self.output_dim)