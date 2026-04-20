from __future__ import annotations

from torch import Tensor, nn


class BaseModel(nn.Module):
    """Baseline LSTM EDC predictor following the paper architecture.

    The paper fixes the LSTM hidden size and dense hidden layer width, while the
    final output length depends on the EDC target used during training.
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 2048,
        hidden_dim: int = 128,
        dense_hidden_dim: int = 2048,
        dropout: float = 0.3,
        num_lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if dense_hidden_dim <= 0:
            raise ValueError("dense_hidden_dim must be positive")
        if num_lstm_layers <= 0:
            raise ValueError("num_lstm_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dense_hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2:
            raise ValueError("features must have at least 2 dimensions")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension to be {self.input_dim}, got {features.shape[-1]}"
            )

        if features.ndim == 2:
            leading_shape = features.shape[:-1]
            lstm_input = features.reshape(-1, 1, self.input_dim)
        else:
            leading_shape = features.shape[:-2]
            seq_len = features.shape[-2]
            lstm_input = features.reshape(-1, seq_len, self.input_dim)

        _, (hidden_state, _) = self.lstm(lstm_input)
        output = hidden_state[-1]
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output.reshape(*leading_shape, self.output_dim)
