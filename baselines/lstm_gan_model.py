"""
LSTM-GAN Model for Rest-to-Task fMRI Prediction

A simpler GAN baseline than TimeGAN:
- Encoder: LSTM on rest sequence -> context (last hidden state)
- Generator: Autoregressive LSTM conditioned on context, outputs task sequence
- Discriminator: LSTM on task sequence -> real/fake (sequence-level)

Rest-conditioned: rest (B,L,V) -> context -> Generator -> task (B,T,V).
"""

import torch
import torch.nn as nn


class RestEncoder(nn.Module):
    """Encode rest sequence (B, L, V) -> (h_n, c_n) for Generator initial state."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        # x: (B, L, V)
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n  # each (num_layers, B, hidden_dim)


class Generator(nn.Module):
    """Autoregressive LSTM: (h0, c0) from encoder + start token -> task sequence (B, T, V)."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.start_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, h0, c0, prediction_length):
        """
        h0, c0: (num_layers, B, hidden_dim)
        Returns: (B, prediction_length, input_dim)
        """
        B = h0.size(1)
        device = h0.device
        start = self.start_token.expand(B, 1, self.input_dim)  # (B, 1, V)
        outputs = []
        h, c = h0, c0
        x = start  # (B, 1, V)
        for _ in range(prediction_length):
            out, (h, c) = self.lstm(x, (h, c))  # out: (B, 1, H)
            y = self.fc(out)  # (B, 1, V)
            outputs.append(y)
            x = y  # autoregressive: next input = current output
        return torch.cat(outputs, dim=1)  # (B, T, V)


class Discriminator(nn.Module):
    """LSTM on task sequence (B, T, V) -> sequence-level real/fake (B, 1)."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, V)
        out, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]  # (B, hidden_dim)
        return torch.sigmoid(self.fc(last_h))  # (B, 1)


class LSTMGAN(nn.Module):
    """
    LSTM-GAN for rest-to-task prediction.
    - Encoder(rest) -> (h0, c0)
    - Generator(h0, c0, T) -> task_pred (B, T, V)
    - Discriminator(task) -> real/fake (B, 1)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        max_prediction_length=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_prediction_length = max_prediction_length
        self.encoder = RestEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.generator = Generator(input_dim, hidden_dim, num_layers, dropout)
        self.discriminator = Discriminator(input_dim, hidden_dim, num_layers, dropout)

    def forward(self, x_rest, prediction_length=1, return_latent=False):
        """
        Generate task sequence from rest.
        x_rest: (B, L, V)
        prediction_length: int
        Returns: x_task (B, prediction_length, V)
        """
        h0, c0 = self.encoder(x_rest)
        x_task = self.generator(h0, c0, prediction_length)
        if return_latent:
            return x_task, None  # no latent in this model
        return x_task
