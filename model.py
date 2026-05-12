import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RestEncoder(nn.Module):
    """
    Encoder for resting state fMRI data.
    Maps rest data to a shared embedding space.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input dimension of rest data (e.g., 268*268 = 71724 for flattened correlation matrix)
            embedding_dim: Dimension of the shared embedding space
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout rate
        """
        super(RestEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer to embedding space
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Rest data tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.network(x)


class TaskEncoder(nn.Module):
    """
    Encoder for task state fMRI data.
    Maps task data to the same shared embedding space as rest encoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input dimension of task data
            embedding_dim: Dimension of the shared embedding space (must match RestEncoder)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout rate
        """
        super(TaskEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer to embedding space
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Task data tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.network(x)


class TaskDecoder(nn.Module):
    """
    Decoder that reconstructs task data from embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        output_dim: int = None,
        hidden_dims: list = None,
        dropout: float = 0.1
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding space
            output_dim: Output dimension (should match task input dimension for reconstruction)
            hidden_dims: List of hidden layer dimensions (default: [256, 512])
            dropout: Dropout rate
        """
        super(TaskDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        if output_dim is None:
            raise ValueError("output_dim must be specified")
        
        layers = []
        prev_dim = embedding_dim
        
        # Build hidden layers (reverse of encoder)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: Embedding tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Reconstructed task data of shape (batch_size, output_dim)
        """
        return self.network(embedding)


# ============================================================================
# Transformer-based Encoders and Decoder
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRestEncoder(nn.Module):
    """
    Transformer-based encoder for resting state fMRI data.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: Optional[int] = None,
        activation: str = 'gelu'
    ):
        """
        Args:
            input_dim: Input dimension of rest data (flattened)
            embedding_dim: Dimension of the shared embedding space
            d_model: Dimension of transformer model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            patch_size: Size of patches to split input into (if None, uses single feature per token)
            activation: Activation function ('relu' or 'gelu')
        """
        super(TransformerRestEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        
        # Determine patch size and sequence length
        if patch_size is None:
            # Use each feature as a token, but might be too long
            # Use a reasonable patch size if input is very large
            if input_dim > 5000:
                patch_size = max(64, input_dim // 200)  # ~200 tokens max
            else:
                patch_size = 1
        
        self.patch_size = patch_size
        seq_len = (input_dim + patch_size - 1) // patch_size  # Ceiling division
        
        # Input projection: project patches to d_model
        self.input_projection = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Pooling: use CLS token or mean pooling
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            seq_len += 1
        
        # Output projection to embedding space
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Rest data tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        batch_size = x.size(0)
        
        # Reshape to patches
        # Pad if necessary
        remainder = x.size(1) % self.patch_size
        if remainder != 0:
            padding = self.patch_size - remainder
            x = F.pad(x, (0, padding))
        
        # Reshape: (batch_size, input_dim) -> (batch_size, seq_len, patch_size)
        seq_len = x.size(1) // self.patch_size
        x = x.view(batch_size, seq_len, self.patch_size)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len+1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len+1, d_model)
        
        # Pool: use CLS token or mean pooling
        if self.use_cls_token:
            x = x[:, 0, :]  # Use CLS token (batch_size, d_model)
        else:
            x = x.mean(dim=1)  # Mean pooling (batch_size, d_model)
        
        # Project to embedding space
        x = self.output_projection(x)  # (batch_size, embedding_dim)
        
        return x


class TransformerTaskEncoder(nn.Module):
    """
    Transformer-based encoder for task state fMRI data.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: Optional[int] = None,
        activation: str = 'gelu'
    ):
        """
        Args:
            input_dim: Input dimension of task data (flattened)
            embedding_dim: Dimension of the shared embedding space (must match RestEncoder)
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            patch_size: Size of patches to split input into
            activation: Activation function ('relu' or 'gelu')
        """
        super(TransformerTaskEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        
        # Determine patch size and sequence length
        if patch_size is None:
            if input_dim > 5000:
                patch_size = max(64, input_dim // 200)
            else:
                patch_size = 1
        
        self.patch_size = patch_size
        seq_len = (input_dim + patch_size - 1) // patch_size
        
        # Input projection
        self.input_projection = nn.Linear(patch_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Pooling
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            seq_len += 1
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Task data tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        batch_size = x.size(0)
        
        # Reshape to patches
        remainder = x.size(1) % self.patch_size
        if remainder != 0:
            padding = self.patch_size - remainder
            x = F.pad(x, (0, padding))
        
        seq_len = x.size(1) // self.patch_size
        x = x.view(batch_size, seq_len, self.patch_size)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Pool
        if self.use_cls_token:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        
        # Project to embedding space
        x = self.output_projection(x)
        
        return x


class TransformerTaskDecoder(nn.Module):
    """
    Transformer-based decoder that reconstructs task data from embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        output_dim: int = None,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        patch_size: Optional[int] = None,
        activation: str = 'gelu'
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding space
            output_dim: Output dimension (should match task input dimension)
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            patch_size: Size of patches for output (should match encoder patch size)
            activation: Activation function ('relu' or 'gelu')
        """
        super(TransformerTaskDecoder, self).__init__()
        
        if output_dim is None:
            raise ValueError("output_dim must be specified")
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Determine patch size and sequence length
        if patch_size is None:
            if output_dim > 5000:
                patch_size = max(64, output_dim // 200)
            else:
                patch_size = 1
        
        self.patch_size = patch_size
        seq_len = (output_dim + patch_size - 1) // patch_size
        
        # Input projection: expand embedding to sequence
        self.input_projection = nn.Linear(embedding_dim, d_model)
        
        # Learnable query tokens for decoder (we'll expand embedding to sequence)
        self.num_query_tokens = seq_len
        self.query_tokens = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Positional encoding for queries
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection: from d_model to patch_size
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_size)
        )
        
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: Embedding tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Reconstructed task data of shape (batch_size, output_dim)
        """
        batch_size = embedding.size(0)
        
        # Expand embedding to d_model and use as memory for cross-attention
        memory = self.input_projection(embedding).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Get query tokens
        query = self.query_tokens.expand(batch_size, -1, -1)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding to queries
        query = self.pos_encoder(query)
        
        # Transformer decoder
        # memory: (batch_size, 1, d_model) - the embedding
        # query: (batch_size, seq_len, d_model) - learnable queries
        output = self.transformer_decoder(query, memory)  # (batch_size, seq_len, d_model)
        
        # Project to output patches
        output = self.output_projection(output)  # (batch_size, seq_len, patch_size)
        
        # Flatten and crop to output_dim
        output = output.view(batch_size, -1)  # (batch_size, seq_len * patch_size)
        output = output[:, :self.output_dim]  # (batch_size, output_dim)
        
        return output


class RestTaskModel(nn.Module):
    """
    Complete model combining rest encoder, task encoder, and task decoder.
    Supports both MLP-based and Transformer-based architectures.
    """
    
    def __init__(
        self,
        rest_input_dim: int,
        task_input_dim: int,
        embedding_dim: int = 128,
        architecture: str = 'mlp',  # 'mlp' or 'transformer'
        # MLP-specific parameters
        rest_hidden_dims: list = None,
        task_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        # Transformer-specific parameters
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        patch_size: Optional[int] = None,
        activation: str = 'gelu',
        # Common parameters
        dropout: float = 0.1
    ):
        """
        Args:
            rest_input_dim: Input dimension of rest data
            task_input_dim: Input dimension of task data
            embedding_dim: Dimension of shared embedding space
            architecture: Architecture type ('mlp' or 'transformer')
            rest_hidden_dims: Hidden dimensions for rest encoder (MLP only)
            task_hidden_dims: Hidden dimensions for task encoder (MLP only)
            decoder_hidden_dims: Hidden dimensions for decoder (MLP only)
            d_model: Transformer model dimension (Transformer only)
            nhead: Number of attention heads (Transformer only)
            num_layers: Number of transformer layers (Transformer only)
            dim_feedforward: Feedforward dimension (Transformer only)
            patch_size: Patch size for input/output (Transformer only, auto-determined if None)
            activation: Activation function ('relu' or 'gelu', Transformer only)
            dropout: Dropout rate
        """
        super(RestTaskModel, self).__init__()
        
        self.architecture = architecture.lower()
        
        if self.architecture == 'mlp':
            # MLP-based architecture
            self.rest_encoder = RestEncoder(
                input_dim=rest_input_dim,
                embedding_dim=embedding_dim,
                hidden_dims=rest_hidden_dims,
                dropout=dropout
            )
            
            self.task_encoder = TaskEncoder(
                input_dim=task_input_dim,
                embedding_dim=embedding_dim,
                hidden_dims=task_hidden_dims,
                dropout=dropout
            )
            
            self.task_decoder = TaskDecoder(
                embedding_dim=embedding_dim,
                output_dim=task_input_dim,
                hidden_dims=decoder_hidden_dims,
                dropout=dropout
            )
            
        elif self.architecture == 'transformer':
            # Transformer-based architecture
            self.rest_encoder = TransformerRestEncoder(
                input_dim=rest_input_dim,
                embedding_dim=embedding_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patch_size=patch_size,
                activation=activation
            )
            
            self.task_encoder = TransformerTaskEncoder(
                input_dim=task_input_dim,
                embedding_dim=embedding_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patch_size=patch_size,
                activation=activation
            )
            
            self.task_decoder = TransformerTaskDecoder(
                embedding_dim=embedding_dim,
                output_dim=task_input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patch_size=patch_size,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Must be 'mlp' or 'transformer'")
        
    def forward(
        self,
        rest_data: torch.Tensor,
        task_data: torch.Tensor
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            rest_data: Resting state data
            task_data: Task state data
            
        Returns:
            Dictionary containing:
                - rest_embedding: Embedding from rest encoder
                - task_embedding: Embedding from task encoder
                - reconstructed_task: Reconstructed task from rest embedding via decoder
        """
        rest_embedding = self.rest_encoder(rest_data)
        task_embedding = self.task_encoder(task_data)
        
        # Decode from rest embedding to reconstruct task
        reconstructed_task = self.task_decoder(rest_embedding)
        
        return {
            'rest_embedding': rest_embedding,
            'task_embedding': task_embedding,
            'reconstructed_task': reconstructed_task
        }

