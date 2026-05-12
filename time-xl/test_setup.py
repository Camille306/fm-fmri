"""
Quick test script to verify the dataset and model setup before training.

This script loads a small sample of data and tests the model forward pass
to ensure everything is configured correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from dataset import HCPRestingFCDataset

try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Error: transformers library not found. Install with: pip install transformers==4.40.1")
    sys.exit(1)


def test_dataset(data_root: str, num_subjects: int = 3):
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    try:
        dataset = HCPRestingFCDataset(data_root=data_root)
        print(f"✓ Found {len(dataset)} subjects")
        
        if len(dataset) == 0:
            print("✗ No subjects found in dataset!")
            return False
        
        # Test loading a few subjects
        print(f"\nTesting data loading for first {min(num_subjects, len(dataset))} subjects...")
        shapes = []
        for i in range(min(num_subjects, len(dataset))):
            subject_id, timeseries = dataset[i]
            shapes.append(timeseries.shape)
            print(f"  Subject {subject_id}: shape {timeseries.shape}")
        
        # Check consistency
        if len(set(shapes)) > 1:
            print(f"⚠ Warning: Subjects have different shapes: {set(shapes)}")
        else:
            print(f"✓ All subjects have consistent shape: {shapes[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model(model_name: str = 'thuml/timer-base-84m', device: str = 'cuda'):
    """Test model loading and forward pass."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        print("✓ Model loaded successfully")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        batch_size = 2
        seq_length = 100
        
        # Create dummy univariate input
        dummy_input = torch.randn(batch_size, seq_length).to(device)
        print(f"  Input shape: {dummy_input.shape}")
        
        try:
            outputs = model(dummy_input, return_dict=True)
            print("✓ Forward pass successful")
            
            # Check output structure
            if hasattr(outputs, 'logits'):
                print(f"  Output logits shape: {outputs.logits.shape}")
            if hasattr(outputs, 'last_hidden_state'):
                print(f"  Last hidden state shape: {outputs.last_hidden_state.shape}")
            
            return True, model
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            return False, model
            
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, None


def test_multivariate_forward(model, device: str = 'cuda', num_variables: int = 268):
    """Test if model can handle multivariate input (flattened)."""
    print("\n" + "=" * 60)
    print("Testing Multivariate Input Handling")
    print("=" * 60)
    
    try:
        batch_size = 2
        lookback_length = 100
        num_vars = num_variables
        
        # Create multivariate input: (batch, time, variables)
        multivariate_input = torch.randn(batch_size, lookback_length, num_vars)
        print(f"  Multivariate input shape: {multivariate_input.shape}")
        
        # Flatten to (batch, time * variables)
        flattened_input = multivariate_input.reshape(batch_size, -1).to(device)
        print(f"  Flattened input shape: {flattened_input.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(flattened_input, return_dict=True)
            
        print("✓ Multivariate forward pass successful")
        
        if hasattr(outputs, 'logits'):
            print(f"  Output logits shape: {outputs.logits.shape}")
            # Try to extract prediction for num_variables
            if outputs.logits.dim() == 3:
                # (batch, seq, hidden) - take last time step
                pred = outputs.logits[:, -1, :num_vars]
                print(f"  Extracted prediction shape: {pred.shape}")
            elif outputs.logits.dim() == 2:
                pred = outputs.logits[:, :num_vars]
                print(f"  Extracted prediction shape: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Multivariate forward pass failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset and model setup')
    parser.add_argument('--data_root', type=str,
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--model_name', type=str, default='thuml/timer-base-84m',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_variables', type=int, default=268,
                       help='Expected number of variables (brain regions)')
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {args.device}\n")
    
    # Test dataset
    dataset_ok = test_dataset(args.data_root)
    
    if not dataset_ok:
        print("\n✗ Dataset test failed. Please check your data path.")
        return
    
    # Test model
    model_ok, model = test_model(args.model_name, args.device)
    
    if not model_ok:
        print("\n✗ Model test failed. Please check model loading.")
        return
    
    if model is None:
        return
    
    # Test multivariate handling
    multivariate_ok = test_multivariate_forward(model, args.device, args.num_variables)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Dataset: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"Model Loading: {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Multivariate Handling: {'✓ PASS' if multivariate_ok else '✗ FAIL'}")
    
    if dataset_ok and model_ok and multivariate_ok:
        print("\n✓ All tests passed! You can proceed with training.")
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")


if __name__ == '__main__':
    main()

