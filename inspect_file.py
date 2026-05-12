#!/usr/bin/env python3
"""
Script to inspect a data file and determine if it's PyTorch or NumPy format.
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Will only check NumPy format.")


def inspect_file(file_path: str):
    """
    Inspect a file to determine if it's PyTorch or NumPy format.
    
    Args:
        file_path: Path to the file to inspect
    """
    file_path = Path(file_path)
    
    # Check if file exists
    try:
        exists = file_path.exists()
    except Exception as e:
        print(f"Error checking file existence: {type(e).__name__}: {e}")
        return
    
    if not exists:
        print(f"Error: File does not exist: {file_path}")
        print("Note: This may be on a remote filesystem that's not accessible from this machine.")
        return
    
    try:
        file_size = file_path.stat().st_size / (1024**2)
        is_file = file_path.is_file()
        is_dir = file_path.is_dir()
    except Exception as e:
        print(f"Error getting file stats: {type(e).__name__}: {e}")
        return
    
    print(f"Inspecting file: {file_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Is file: {is_file}")
    print(f"Is directory: {is_dir}")
    print("-" * 60)
    
    # If it's a directory, list contents
    if file_path.is_dir():
        print("This is a directory. Contents:")
        try:
            contents = list(file_path.iterdir())
            for item in contents[:20]:  # Show first 20 items
                print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
            if len(contents) > 20:
                print(f"  ... and {len(contents) - 20} more items")
        except PermissionError:
            print("  Permission denied to list directory contents")
        return
    
    # Try to load as NumPy
    print("\nAttempting to load as NumPy...")
    numpy_success = False
    numpy_data = None
    try:
        # Use mmap_mode='r' for large files to avoid loading everything into memory
        numpy_data = np.load(str(file_path), allow_pickle=False, mmap_mode='r')
        print("✓ Successfully loaded as NumPy!")
        print(f"  Type: {type(numpy_data)}")
        
        if isinstance(numpy_data, np.ndarray):
            print(f"  Shape: {numpy_data.shape}")
            print(f"  Dtype: {numpy_data.dtype}")
            print(f"  Size: {numpy_data.size}")
            print(f"  Memory size: {numpy_data.nbytes / (1024**2):.2f} MB")
        elif isinstance(numpy_data, np.lib.npyio.NpzFile):
            print("  This is a .npz file (compressed NumPy archive)")
            keys = list(numpy_data.keys())
            print(f"  Keys: {keys}")
            
            # Check if this looks like a PyTorch file
            pytorch_keys = ['.format_version', '.storage_alignment', 'byteorder', 'version', '.data/serialization_id', 'data.pkl']
            has_pytorch_keys = any(key.endswith(tuple(pytorch_keys)) or any(pk in key for pk in pytorch_keys) for key in keys)
            
            if has_pytorch_keys:
                print("  ⚠ WARNING: This appears to be a PyTorch tensor file!")
                print("    (PyTorch uses .npz-like format internally, but this is NOT a standard NumPy file)")
            
            for key in keys[:10]:  # Limit to first 10 keys
                try:
                    arr = numpy_data[key]
                    if isinstance(arr, np.ndarray):
                        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
                    else:
                        print(f"    {key}: {type(arr).__name__} (not a numpy array)")
                except Exception as e:
                    print(f"    {key}: Error loading - {e}")
        else:
            print(f"  Content type: {type(numpy_data)}")
            if hasattr(numpy_data, '__len__'):
                print(f"  Length: {len(numpy_data)}")
        
        numpy_success = True
    except (OSError, IOError, ValueError) as e:
        print(f"✗ Failed to load as NumPy: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"✗ Unexpected error loading as NumPy: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to load as PyTorch
    torch_success = False
    if TORCH_AVAILABLE:
        print("\nAttempting to load as PyTorch...")
        try:
            torch_data = torch.load(str(file_path), map_location='cpu', weights_only=False)
            print("✓ Successfully loaded as PyTorch!")
            print(f"  Type: {type(torch_data)}")
            
            if isinstance(torch_data, torch.Tensor):
                print(f"  Shape: {torch_data.shape}")
                print(f"  Dtype: {torch_data.dtype}")
                print(f"  Device: {torch_data.device}")
                print(f"  Requires grad: {torch_data.requires_grad}")
                print(f"  Number of elements: {torch_data.numel()}")
                print(f"  Memory size: {torch_data.element_size() * torch_data.numel() / (1024**2):.2f} MB")
            elif isinstance(torch_data, np.ndarray):
                print(f"  ⚠ PyTorch loaded this as a NumPy array (likely converted during load)")
                print(f"  Shape: {torch_data.shape}")
                print(f"  Dtype: {torch_data.dtype}")
                print(f"  Size: {torch_data.size}")
                print(f"  Memory size: {torch_data.nbytes / (1024**2):.2f} MB")
                # Try to convert to tensor to get more info
                try:
                    tensor_version = torch.from_numpy(torch_data)
                    print(f"  As PyTorch tensor: shape={tensor_version.shape}, dtype={tensor_version.dtype}")
                except:
                    pass
            elif isinstance(torch_data, dict):
                print(f"  Dictionary with {len(torch_data)} keys:")
                for key, value in list(torch_data.items())[:10]:
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: Tensor shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, np.ndarray):
                        print(f"    {key}: NumPy array shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: {type(value)}")
            else:
                print(f"  Content type: {type(torch_data)}")
                if hasattr(torch_data, '__len__'):
                    print(f"  Length: {len(torch_data)}")
                if hasattr(torch_data, 'shape'):
                    print(f"  Shape: {torch_data.shape}")
                if hasattr(torch_data, 'dtype'):
                    print(f"  Dtype: {torch_data.dtype}")
            
            torch_success = True
        except (OSError, IOError, RuntimeError, EOFError) as e:
            print(f"✗ Failed to load as PyTorch: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error loading as PyTorch: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    # Check for PyTorch indicators in NumPy keys
    pytorch_indicator = False
    if numpy_success and isinstance(numpy_data, np.lib.npyio.NpzFile):
        keys = list(numpy_data.keys())
        pytorch_keys = ['.format_version', '.storage_alignment', 'byteorder', 'version', '.data/serialization_id']
        pytorch_indicator = any(any(pk in key for pk in pytorch_keys) for key in keys)
    
    if torch_success:
        print("✓ File format: PyTorch (.pt or .pth)")
        print("  This is a PyTorch tensor file saved using torch.save()")
        if pytorch_indicator:
            print("  (PyTorch uses a zip-based format that NumPy can partially read)")
        if isinstance(torch_data, np.ndarray):
            print("  Note: PyTorch loaded this as a NumPy array (automatic conversion)")
    elif numpy_success and pytorch_indicator:
        print("✓ File format: PyTorch (.pt or .pth)")
        print("  (Detected PyTorch storage format keys in the file)")
    elif numpy_success and not torch_success:
        print("✓ File format: NumPy (.npy or .npz)")
    elif numpy_success and torch_success and not pytorch_indicator:
        print("⚠ File can be loaded as BOTH NumPy and PyTorch!")
        print("  (This is unusual - may be a NumPy array saved with PyTorch)")
    else:
        print("✗ Could not determine file format")
        print("  File may be in a different format or corrupted")


if __name__ == "__main__":
    file_path = "./data/data/task-based/emotion/roi_data_RL/196952"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    inspect_file(file_path)
