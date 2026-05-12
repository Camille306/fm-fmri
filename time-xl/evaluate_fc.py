"""
Evaluation script to load the best checkpoint and plot task FC (functional connectivity)
for the closest subject (subject with highest FC similarity between predicted and real).

This script:
1. Loads the best checkpoint from training
2. Evaluates the model on test data
3. Computes task FC matrices for both predicted and real task data
4. Finds the subject with the highest FC similarity
5. Plots both FC matrices side by side
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset classes
from dataset import HCPRestingFCDataset

# Import model and dataset classes
sys.path.append(str(Path(__file__).parent))
from train import FMRIWindowDataset, forward_with_projection, MultivariateProjectionHead
from lstm_baseline import LSTMBaseline, compute_functional_connectivity, compute_fc_similarity

try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model from checkpoint.
    
    Supports both LSTM baseline and Timer-XL models.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, model_type, args_dict)
        model_type: 'lstm' or 'timer'
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args_dict = checkpoint.get('args', {})
    
    # Determine model type based on checkpoint keys
    if 'model_state_dict' in checkpoint:
        # LSTM baseline model
        model_type = 'lstm'
        num_variables = checkpoint.get('num_variables', args_dict.get('input_dim', 268))
        hidden_dim = args_dict.get('hidden_dim', 128)
        num_layers = args_dict.get('num_layers', 2)
        dropout = args_dict.get('dropout', 0.1)
        
        model = LSTMBaseline(
            input_dim=num_variables,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=num_variables,
            dropout=dropout
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded LSTM baseline model: {num_variables} variables, {hidden_dim} hidden dim, {num_layers} layers")
        
        return model, model_type, args_dict
        
    elif 'base_model_state_dict' in checkpoint:
        # Timer-XL model
        model_type = 'timer'
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for Timer-XL models")
        
        model_name = args_dict.get('model_name', 'thuml/timer-base-84m')
        num_variables = checkpoint.get('num_variables', 268)
        model_output_dim = checkpoint.get('model_output_dim', None)
        
        print(f"Loading Timer-XL model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        base_model = base_model.to(device)
        base_model.load_state_dict(checkpoint['base_model_state_dict'])
        
        # Create and load projection head
        if model_output_dim is None:
            # Try to infer from checkpoint
            # Get a sample to determine output dim
            print("Warning: model_output_dim not found in checkpoint, attempting to infer...")
            model_output_dim = 512  # Default fallback
        
        projection_head = MultivariateProjectionHead(model_output_dim, num_variables).to(device)
        if 'projection_head_state_dict' in checkpoint:
            projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        
        print(f"Loaded Timer-XL model: {num_variables} variables, output dim: {model_output_dim}")
        
        # Return as tuple for compatibility
        return (base_model, projection_head), model_type, args_dict
    
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")


def predict_full_task_sequence(model, model_type, dataloader, device, prediction_length=1):
    """
    Generate predictions for full task sequences, aggregated by subject.
    
    Args:
        model: Model (LSTM or (base_model, projection_head) tuple)
        model_type: 'lstm' or 'timer'
        dataloader: DataLoader for test data
        device: Device to run on
        prediction_length: Number of time steps per prediction
        
    Returns:
        Dictionary mapping subject_id to (predictions, targets) tuples
        predictions: (num_windows * prediction_length, num_variables) or aggregated task sequence
        targets: (num_windows * prediction_length, num_variables) or aggregated task sequence
    """
    if model_type == 'lstm':
        model.eval()
    else:
        base_model, projection_head = model
        base_model.eval()
        if projection_head is not None:
            projection_head.eval()
    
    subject_predictions = {}
    subject_targets = {}
    subject_inputs = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_seq = batch['input'].to(device).float()
            target = batch['target'].to(device).float()
            subject_ids = batch['subject_id']
            
            batch_size = input_seq.shape[0]
            
            # Generate predictions
            if model_type == 'lstm':
                pred = model(input_seq, prediction_length=prediction_length)
            else:
                base_model, projection_head = model
                batch_size, lookback_length, num_variables = input_seq.shape
                input_flat = input_seq.reshape(batch_size, -1)
                num_variables = input_seq.shape[2]
                pred = forward_with_projection(
                    base_model, projection_head, input_flat, num_variables, 
                    device, prediction_length=prediction_length
                )
            
            # Flatten sequence predictions if needed
            if pred.dim() == 3:
                pred_flat = pred.reshape(-1, pred.shape[-1])  # (batch * seq_len, num_variables)
            else:
                pred_flat = pred
            
            if target.dim() == 3:
                target_flat = target.reshape(-1, target.shape[-1])
            else:
                target_flat = target
            
            # Aggregate by subject
            for i in range(batch_size):
                subject_id = subject_ids[i]
                
                if subject_id not in subject_predictions:
                    subject_predictions[subject_id] = []
                    subject_targets[subject_id] = []
                
                # Get predictions and targets for this sample
                if prediction_length > 1:
                    pred_sample = pred_flat[i * prediction_length:(i + 1) * prediction_length]
                    target_sample = target_flat[i * prediction_length:(i + 1) * prediction_length]
                else:
                    pred_sample = pred_flat[i:i+1]
                    target_sample = target_flat[i:i+1]
                
                subject_predictions[subject_id].append(pred_sample.cpu().numpy())
                subject_targets[subject_id].append(target_sample.cpu().numpy())
    
    # Concatenate predictions for each subject
    subject_data = {}
    for subject_id in subject_predictions.keys():
        preds = np.concatenate(subject_predictions[subject_id], axis=0)
        targets = np.concatenate(subject_targets[subject_id], axis=0)
        subject_data[subject_id] = (preds, targets)
    
    return subject_data


def find_closest_subject(subject_data):
    """
    Find the subject with the highest FC similarity between predicted and real task FC.
    
    Args:
        subject_data: Dictionary mapping subject_id to (predictions, targets) tuples
        
    Returns:
        Tuple of (best_subject_id, best_fc_sim, fc_pred, fc_target)
    """
    best_subject_id = None
    best_fc_sim = -1.0
    best_fc_pred = None
    best_fc_target = None
    
    print("\nComputing FC similarity for each subject...")
    for subject_id, (predictions, targets) in tqdm(subject_data.items(), desc="Finding closest subject"):
        # Compute FC similarity
        fc_sim = compute_fc_similarity(predictions, targets)
        
        if fc_sim > best_fc_sim:
            best_fc_sim = fc_sim
            best_subject_id = subject_id
            best_fc_pred = compute_functional_connectivity(predictions)
            best_fc_target = compute_functional_connectivity(targets)
    
    print(f"\nClosest subject: {best_subject_id}")
    print(f"FC Similarity: {best_fc_sim:.4f}")
    
    return best_subject_id, best_fc_sim, best_fc_pred, best_fc_target


def plot_fc_matrices(fc_pred, fc_target, subject_id, fc_similarity, save_path):
    """
    Plot predicted and real task FC matrices side by side.
    
    Args:
        fc_pred: Predicted FC matrix (num_variables, num_variables)
        fc_target: Real FC matrix (num_variables, num_variables)
        subject_id: Subject ID string
        fc_similarity: FC similarity score
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Determine common color scale
    vmin = min(fc_pred.min(), fc_target.min())
    vmax = max(fc_pred.max(), fc_target.max())
    
    # Plot predicted FC
    im1 = axes[0].imshow(fc_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Predicted Task FC\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('ROI Index', fontsize=12)
    axes[0].set_ylabel('ROI Index', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    
    # Plot real FC
    im2 = axes[1].imshow(fc_target, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Real Task FC\nSubject: {subject_id}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('ROI Index', fontsize=12)
    axes[1].set_ylabel('ROI Index', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Correlation')
    
    # Add overall title
    fig.suptitle(
        f'Task Functional Connectivity Comparison\nFC Similarity: {fc_similarity:.4f}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FC plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate best checkpoint and plot task FC for closest subject')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to best checkpoint file (best_model.pth)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing checkpoints (will look for best_model.pth if checkpoint_path not provided)')
    
    # Data arguments
    parser.add_argument('--data_root', type=str,
                       default='./data/hcp-resting-fc',
                       help='Root directory containing subject folders')
    parser.add_argument('--task_root', type=str, default=None,
                       help='Root directory for task data')
    parser.add_argument('--task_name', type=str, default='emotion',
                       help='Name of the task (e.g., "emotion")')
    parser.add_argument('--lookback_length', type=int, default=512,
                       help='Number of time steps to use as input context')
    parser.add_argument('--prediction_length', type=int, default=None,
                       help='Number of time steps to predict (None = infer from checkpoint)')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride for sliding window')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data (should match training)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training (should match training)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion of data for validation (should match training)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results and plots')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                       help='Maximum windows per subject (None = all)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")
    
    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    elif args.checkpoint_dir:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    else:
        raise ValueError("Must provide either --checkpoint_path or --checkpoint_dir")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model from checkpoint
    model, model_type, checkpoint_args = load_model_from_checkpoint(checkpoint_path, args.device)
    
    # Override args with checkpoint args if not provided
    if args.prediction_length is None:
        args.prediction_length = checkpoint_args.get('prediction_length', 1)
    if args.lookback_length is None:
        args.lookback_length = checkpoint_args.get('lookback_length', 512)
    if args.stride is None:
        args.stride = checkpoint_args.get('stride', 100)
    # normalize is a boolean with default, so we use checkpoint value if available
    if 'normalize' in checkpoint_args:
        args.normalize = checkpoint_args.get('normalize', True)
    
    print(f"\nModel configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Prediction length: {args.prediction_length}")
    print(f"  Lookback length: {args.lookback_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Normalize: {args.normalize}")
    
    # Load dataset
    print("\nLoading dataset...")
    hcp_dataset = HCPRestingFCDataset(
        data_root=args.data_root,
        task_root=args.task_root,
        task_name=args.task_name
    )
    print(f"Found {len(hcp_dataset)} subjects")
    
    # Create test dataset
    # Note: For proper evaluation, normalization stats should match training.
    # If normalization stats were saved in checkpoint, load them here.
    # Otherwise, test dataset will compute its own stats (which may differ slightly).
    test_dataset = FMRIWindowDataset(
        dataset=hcp_dataset,
        lookback_length=args.lookback_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        normalize=args.normalize,
        split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_subject=args.max_samples_per_subject,
        use_task_target=True,
        norm_sample_size=1000,
        norm_batch_size=100
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print(f"Test dataset size: {len(test_dataset)} windows")
    
    # Generate predictions for all subjects
    print("\nGenerating predictions...")
    subject_data = predict_full_task_sequence(
        model, model_type, test_loader, args.device, args.prediction_length
    )
    
    print(f"\nGenerated predictions for {len(subject_data)} subjects")
    
    # Find closest subject
    best_subject_id, best_fc_sim, fc_pred, fc_target = find_closest_subject(subject_data)
    
    # Plot FC matrices
    plot_path = os.path.join(args.output_dir, f'fc_comparison_{best_subject_id}.png')
    plot_fc_matrices(fc_pred, fc_target, best_subject_id, best_fc_sim, plot_path)
    
    # Save results summary
    results_path = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(results_path, 'w') as f:
        f.write("Task FC Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Number of subjects evaluated: {len(subject_data)}\n")
        f.write(f"\nClosest subject: {best_subject_id}\n")
        f.write(f"FC Similarity: {best_fc_sim:.4f}\n")
        f.write(f"\nFC plot saved to: {plot_path}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_path}")
    print(f"FC plot saved to: {plot_path}")


if __name__ == '__main__':
    main()

