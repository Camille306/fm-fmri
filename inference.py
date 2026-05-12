import torch
import numpy as np
from pathlib import Path
import argparse

from dataset import RestTaskDataset
from model import RestTaskModel
from torch.utils.data import DataLoader


def load_model(
    checkpoint_path: str,
    device: str = 'cpu',
    architecture: str = None,
    rest_input_dim: int = None,
    task_input_dim: int = None,
    **kwargs
):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        architecture: Architecture type ('mlp' or 'transformer', auto-detected from checkpoint if None)
        rest_input_dim: Rest input dimension (auto-detected from checkpoint if None)
        task_input_dim: Task input dimension (auto-detected from checkpoint if None)
        **kwargs: Additional model arguments
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get architecture from checkpoint or use provided/default
    arch = architecture or checkpoint.get('architecture', 'mlp')
    
    # Get model dimensions from checkpoint or use provided/defaults
    model_kwargs = {
        'rest_input_dim': rest_input_dim or checkpoint.get('rest_input_dim', 268*268),
        'task_input_dim': task_input_dim or checkpoint.get('task_input_dim', 268*268),
        'embedding_dim': checkpoint.get('embedding_dim', 128),
        'architecture': arch
    }
    
    # Add transformer-specific parameters if needed
    if arch == 'transformer':
        model_kwargs.update({
            'd_model': checkpoint.get('d_model', 256),
            'nhead': checkpoint.get('nhead', 8),
            'num_layers': checkpoint.get('num_layers', 4),
            'dim_feedforward': checkpoint.get('dim_feedforward', 1024),
            'patch_size': checkpoint.get('patch_size', None),
            'activation': checkpoint.get('activation', 'gelu')
        })
    
    # Override with any provided kwargs
    model_kwargs.update(kwargs)
    
    model = RestTaskModel(**model_kwargs)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def inference_single(model: RestTaskModel, rest_data: torch.Tensor, device: str = 'cpu'):
    """
    Perform inference on a single rest sample.
    
    Args:
        model: Trained RestTaskModel
        rest_data: Rest data tensor of shape (1, input_dim) or (input_dim,)
        device: Device to run on
        
    Returns:
        Dictionary with:
            - embedding: Embedding vector
            - reconstructed_task: Reconstructed task data
    """
    model.eval()
    
    # Ensure correct shape
    if rest_data.dim() == 1:
        rest_data = rest_data.unsqueeze(0)
    
    rest_data = rest_data.to(device)
    
    with torch.no_grad():
        # Encode rest data
        rest_embedding = model.rest_encoder(rest_data)
        
        # Decode to reconstruct task
        reconstructed_task = model.task_decoder(rest_embedding)
    
    return {
        'embedding': rest_embedding.cpu(),
        'reconstructed_task': reconstructed_task.cpu()
    }


def compute_embeddings(
    model: RestTaskModel,
    data_loader: DataLoader,
    device: str = 'cpu',
    save_path: str = None
):
    """
    Compute embeddings for all samples in a dataset.
    
    Args:
        model: Trained RestTaskModel
        data_loader: DataLoader for dataset
        device: Device to run on
        save_path: Optional path to save embeddings
        
    Returns:
        Dictionary with embeddings and metadata
    """
    model.eval()
    
    rest_embeddings = []
    task_embeddings = []
    subject_ids = []
    
    with torch.no_grad():
        for rest_data, task_data, subject_id_batch in data_loader:
            rest_data = rest_data.to(device)
            task_data = task_data.to(device)
            
            # Get embeddings
            rest_emb = model.rest_encoder(rest_data)
            task_emb = model.task_encoder(task_data)
            
            rest_embeddings.append(rest_emb.cpu())
            task_embeddings.append(task_emb.cpu())
            subject_ids.extend(subject_id_batch)
    
    rest_embeddings = torch.cat(rest_embeddings, dim=0)
    task_embeddings = torch.cat(task_embeddings, dim=0)
    
    results = {
        'rest_embeddings': rest_embeddings.numpy(),
        'task_embeddings': task_embeddings.numpy(),
        'subject_ids': subject_ids
    }
    
    if save_path:
        np.savez(save_path, **results)
        print(f"Saved embeddings to {save_path}")
    
    return results


def compute_similarity_matrix(
    rest_embeddings: torch.Tensor,
    task_embeddings: torch.Tensor,
    subject_ids: list
):
    """
    Compute similarity matrix between rest and task embeddings.
    
    Args:
        rest_embeddings: Rest embeddings tensor (N, embedding_dim)
        task_embeddings: Task embeddings tensor (M, embedding_dim)
        subject_ids: List of subject IDs
        
    Returns:
        Similarity matrix and labels for same/different subjects
    """
    # Normalize embeddings
    rest_embeddings = torch.nn.functional.normalize(rest_embeddings, p=2, dim=1)
    task_embeddings = torch.nn.functional.normalize(task_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(rest_embeddings, task_embeddings.t())
    
    # Create label matrix (1 for same subject, 0 for different)
    labels = torch.zeros(len(rest_embeddings), len(task_embeddings))
    for i in range(len(rest_embeddings)):
        for j in range(len(task_embeddings)):
            if subject_ids[i] == subject_ids[j]:
                labels[i, j] = 1.0
    
    return similarity_matrix, labels


def main():
    parser = argparse.ArgumentParser(description='Inference with Rest-to-Task Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--rest_csv', type=str, default='rest_info.csv',
                        help='Path to rest CSV file')
    parser.add_argument('--task_csv', type=str, default='hcp_task_info.csv',
                        help='Path to task CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--architecture', type=str, default=None,
                        choices=['mlp', 'transformer'],
                        help='Architecture type (auto-detected from checkpoint if not specified)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(
        args.checkpoint,
        device=device,
        architecture=args.architecture
    )
    print(f"Model loaded successfully! Architecture: {model.architecture}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = RestTaskDataset(
        rest_csv=args.rest_csv,
        task_csv=args.task_csv,
        device=device
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Compute embeddings
    print("Computing embeddings...")
    results = compute_embeddings(
        model,
        data_loader,
        device=device,
        save_path=str(output_dir / 'embeddings.npz')
    )
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    rest_embeddings = torch.from_numpy(results['rest_embeddings'])
    task_embeddings = torch.from_numpy(results['task_embeddings'])
    similarity_matrix, labels = compute_similarity_matrix(
        rest_embeddings,
        task_embeddings,
        results['subject_ids']
    )
    
    # Save similarity matrix
    np.save(output_dir / 'similarity_matrix.npy', similarity_matrix.numpy())
    np.save(output_dir / 'labels.npy', labels.numpy())
    
    # Print statistics
    same_subject_sim = similarity_matrix[labels.bool()].mean().item()
    diff_subject_sim = similarity_matrix[~labels.bool()].mean().item()
    
    print(f"\nSimilarity Statistics:")
    print(f"Same subject similarity: {same_subject_sim:.4f}")
    print(f"Different subject similarity: {diff_subject_sim:.4f}")
    print(f"Separation: {same_subject_sim - diff_subject_sim:.4f}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()

