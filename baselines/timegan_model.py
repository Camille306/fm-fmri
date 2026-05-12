"""
TimeGAN Model for Rest-to-Task fMRI Prediction

TimeGAN (Time-series Generative Adversarial Networks) adapted for rest-to-task prediction.
This implementation includes:
- Embedder: Maps real data to latent space
- Recovery: Maps latent space back to data space  
- Generator: Generates synthetic task sequences from rest sequences
- Discriminator: Distinguishes real from synthetic task sequences
- Supervisor: Predicts next step in latent space

Based on the original TimeGAN paper: "Time-series Generative Adversarial Networks" (Yoon et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBlock(nn.Module):
    """LSTM block with optional bidirectional processing."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, hidden_dim) or (batch_size, seq_len, 2*hidden_dim) if bidirectional
        """
        output, (h_n, c_n) = self.lstm(x)
        return output


class Embedder(nn.Module):
    """
    Embedder network: Maps real data to latent space.
    Input: rest sequence (B, L, V)
    Output: latent sequence (B, L, H)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(Embedder, self).__init__()
        self.lstm = LSTMBlock(input_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - rest sequence
        Returns:
            h: (batch_size, seq_len, hidden_dim) - latent sequence
        """
        h = self.lstm(x)
        h = self.fc(h)
        return h


class Recovery(nn.Module):
    """
    Recovery network: Maps latent space back to data space.
    Input: latent sequence (B, L, H)
    Output: recovered data (B, L, V)
    """
    
    def __init__(self, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(Recovery, self).__init__()
        self.lstm = LSTMBlock(hidden_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        """
        Args:
            h: (batch_size, seq_len, hidden_dim) - latent sequence
        Returns:
            x_tilde: (batch_size, seq_len, output_dim) - recovered data
        """
        h = self.lstm(h)
        x_tilde = self.fc(h)
        return x_tilde


class Generator(nn.Module):
    """
    Generator network: Generates synthetic task sequences from rest sequences.
    Input: rest sequence (B, L, V) or rest latent (B, L, H)
    Output: task latent sequence (B, T, H)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1, use_embedder=True):
        super(Generator, self).__init__()
        self.use_embedder = use_embedder
        if use_embedder:
            # If input is raw data, embed it first
            self.embedder = Embedder(input_dim, hidden_dim, num_layers, dropout)
        self.lstm = LSTMBlock(hidden_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, prediction_length=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - rest sequence
            prediction_length: Length of task sequence to generate (if None, uses last hidden state)
        Returns:
            h_hat: (batch_size, prediction_length, hidden_dim) - generated task latent
        """
        if self.use_embedder:
            h = self.embedder(x)  # (batch_size, seq_len, hidden_dim)
        else:
            h = x  # Assume x is already in latent space
            
        # Use last hidden state to generate task sequence
        h_out, (h_n, c_n) = self.lstm.lstm(h)
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        if prediction_length is None or prediction_length == 1:
            h_hat = self.fc(last_hidden).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        else:
            # Generate sequence: use last hidden state and project for each time step
            # Simple approach: use the same base hidden state for all time steps
            # The LSTM in recovery will add temporal structure
            base_h = self.fc(last_hidden)  # (batch_size, hidden_dim)
            h_hat = base_h.unsqueeze(1).repeat(1, prediction_length, 1)  # (batch_size, prediction_length, hidden_dim)
            
        return h_hat


class Supervisor(nn.Module):
    """
    Supervisor network: Predicts next step in latent space.
    Input: latent sequence (B, L, H)
    Output: next latent step (B, L, H) (shifted by one)
    """
    
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super(Supervisor, self).__init__()
        self.lstm = LSTMBlock(hidden_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h):
        """
        Args:
            h: (batch_size, seq_len, hidden_dim) - latent sequence
        Returns:
            h_supervise: (batch_size, seq_len, hidden_dim) - predicted next step
        """
        h_out = self.lstm(h)
        h_supervise = self.fc(h_out)
        return h_supervise


class Discriminator(nn.Module):
    """
    Discriminator network: Distinguishes real from synthetic task sequences.
    Input: task latent sequence (B, T, H)
    Output: probability of being real (B, T, 1)
    """
    
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super(Discriminator, self).__init__()
        self.lstm = LSTMBlock(hidden_dim, hidden_dim, num_layers, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        """
        Args:
            h: (batch_size, seq_len, hidden_dim) - task latent sequence
        Returns:
            y: (batch_size, seq_len, 1) - probability of being real
        """
        h_out = self.lstm(h)
        y = torch.sigmoid(self.fc(h_out))
        return y


class TimeGAN(nn.Module):
    """
    Complete TimeGAN model for rest-to-task prediction.
    
    Architecture:
    1. Embedder: rest sequence -> latent space
    2. Generator: rest sequence -> task latent sequence
    3. Recovery: task latent -> task data
    4. Discriminator: distinguishes real vs synthetic task latents
    5. Supervisor: predicts next step in latent space
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        max_prediction_length=256
    ):
        super(TimeGAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_prediction_length = max_prediction_length
        
        # Components
        self.embedder = Embedder(input_dim, hidden_dim, num_layers, dropout)
        self.recovery = Recovery(hidden_dim, input_dim, num_layers, dropout)
        self.generator = Generator(input_dim, hidden_dim, num_layers, dropout, use_embedder=True)
        self.supervisor = Supervisor(hidden_dim, num_layers, dropout)
        self.discriminator = Discriminator(hidden_dim, num_layers, dropout)
        
    def forward(self, x_rest, prediction_length=1, return_latent=False):
        """
        Forward pass: Generate task sequence from rest sequence.
        
        Args:
            x_rest: (batch_size, seq_len, input_dim) - rest sequence
            prediction_length: Length of task sequence to generate
            return_latent: If True, also return latent representations
            
        Returns:
            x_task: (batch_size, prediction_length, input_dim) - generated task sequence
            (optional) h_task: (batch_size, prediction_length, hidden_dim) - task latent
        """
        # Embed rest sequence
        h_rest = self.embedder(x_rest)  # (batch_size, seq_len, hidden_dim)
        
        # Generate task latent sequence
        h_task = self.generator(x_rest, prediction_length=prediction_length)  # (batch_size, prediction_length, hidden_dim)
        
        # Recover task data from latent
        x_task = self.recovery(h_task)  # (batch_size, prediction_length, input_dim)
        
        if return_latent:
            return x_task, h_task
        return x_task
    
    def embed(self, x):
        """Embed data to latent space."""
        return self.embedder(x)
    
    def recover(self, h):
        """Recover data from latent space."""
        return self.recovery(h)
    
    def generate_latent(self, x_rest, prediction_length=1):
        """Generate task latent from rest sequence."""
        return self.generator(x_rest, prediction_length=prediction_length)
    
    def discriminate(self, h):
        """Discriminate real vs synthetic latent sequences."""
        return self.discriminator(h)
    
    def supervise(self, h):
        """Supervise: predict next step in latent space."""
        return self.supervisor(h)
