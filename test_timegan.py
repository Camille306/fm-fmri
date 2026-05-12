"""
Simple test script to verify TimeGAN model can be instantiated and run forward pass.
Run from project root: python baselines/test_timegan.py  or from baselines/: python test_timegan.py
"""

import sys
from pathlib import Path

_base = Path(__file__).resolve().parent
_root = _base.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_base))

import torch
from timegan_model import TimeGAN


def test_timegan():
    """Test TimeGAN model instantiation and forward pass."""
    print("Testing TimeGAN model...")

    # Model parameters
    input_dim = 166  # Number of ROIs
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1
    prediction_length = 176

    # Create model
    print(f"Creating TimeGAN model with input_dim={input_dim}, hidden_dim={hidden_dim}...")
    model = TimeGAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_prediction_length=prediction_length
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 4
    lookback_length = 512

    print(f"\nTesting forward pass...")
    print(f"  Input shape: ({batch_size}, {lookback_length}, {input_dim})")
    print(f"  Prediction length: {prediction_length}")

    x_rest = torch.randn(batch_size, lookback_length, input_dim)

    # Forward pass
    with torch.no_grad():
        x_task = model(x_rest, prediction_length=prediction_length)
        print(f"  Output shape: {x_task.shape}")
        assert x_task.shape == (batch_size, prediction_length, input_dim), \
            f"Expected shape ({batch_size}, {prediction_length}, {input_dim}), got {x_task.shape}"

    print("✓ Forward pass successful!")

    # Test with return_latent
    print("\nTesting forward pass with return_latent=True...")
    with torch.no_grad():
        x_task, h_task = model(x_rest, prediction_length=prediction_length, return_latent=True)
        print(f"  Output shape: {x_task.shape}")
        print(f"  Latent shape: {h_task.shape}")
        assert h_task.shape == (batch_size, prediction_length, hidden_dim), \
            f"Expected latent shape ({batch_size}, {prediction_length}, {hidden_dim}), got {h_task.shape}"

    print("✓ Forward pass with latent successful!")

    # Test individual components
    print("\nTesting individual components...")

    # Embedder
    with torch.no_grad():
        h_rest = model.embed(x_rest)
        print(f"  Embedder output shape: {h_rest.shape}")
        assert h_rest.shape == (batch_size, lookback_length, hidden_dim)

    # Generator
    with torch.no_grad():
        h_task = model.generate_latent(x_rest, prediction_length=prediction_length)
        print(f"  Generator output shape: {h_task.shape}")
        assert h_task.shape == (batch_size, prediction_length, hidden_dim)

    # Recovery
    with torch.no_grad():
        x_recovered = model.recover(h_task)
        print(f"  Recovery output shape: {x_recovered.shape}")
        assert x_recovered.shape == (batch_size, prediction_length, input_dim)

    # Discriminator
    with torch.no_grad():
        y = model.discriminate(h_task)
        print(f"  Discriminator output shape: {y.shape}")
        assert y.shape == (batch_size, prediction_length, 1)

    # Supervisor
    with torch.no_grad():
        h_supervise = model.supervise(h_task[:, :-1, :])
        print(f"  Supervisor output shape: {h_supervise.shape}")
        assert h_supervise.shape == (batch_size, prediction_length - 1, hidden_dim)

    print("✓ All components working correctly!")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_timegan()
