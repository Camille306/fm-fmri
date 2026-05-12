"""
LSTM binary classifier for time series: real (0) vs generated (1).
Input: (batch, seq_len, n_features); output: logits (batch, 2).
"""

import torch
import torch.nn as nn


class LSTMTimeSeriesClassifier(nn.Module):
    """Binary classifier: input (B, T, V) -> logits (B, 2). Label 0 = real, 1 = generated."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V)
        out, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            last_h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_h = h_n[-1]
        logits = self.classifier(last_h)
        return logits
