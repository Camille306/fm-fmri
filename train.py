import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt  # <-- for plotting

from dataset import RestTaskDataset
from model import RestTaskModel
from loss import CombinedLoss


class Trainer:
    """
    Trainer class for the rest-to-task model.
    """
    
    def __init__(
        self,
        model: RestTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        use_negative_samples: bool = True,
        num_negatives: int = 1,
        checkpoint_dir: str = './',
        log_interval: int = 10
    ):
        """
        Args:
            model: RestTaskModel instance
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            use_negative_samples: Whether to use explicit negative samples
            num_negatives: Number of negative samples per positive pair
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_negative_samples = use_negative_samples
        self.num_negatives = num_negatives
        self.log_interval = log_interval
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = CombinedLoss(
            contrastive_weight=contrastive_weight,
            reconstruction_weight=reconstruction_weight
        ).to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_reconstruction_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_reconstruction_loss': []
        }
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (rest_data, task_data, subject_ids) in enumerate(pbar):
            # Move to device
            rest_data = rest_data.to(self.device)
            task_data = task_data.to(self.device)
            
            # Forward pass
            outputs = self.model(rest_data, task_data)
            
            # Get negative samples if needed
            negative_rest_embeddings = None
            negative_task_embeddings = None
            negative_subject_ids = None
            
            if self.use_negative_samples and self.train_loader.dataset:
                # Sample negatives from batch
                batch_size = rest_data.size(0)
                neg_rest_list = []
                neg_task_list = []
                neg_subject_list = []
                
                for i in range(batch_size):
                    # Get negative samples from dataset
                    negs = self.train_loader.dataset.dataset.get_negative_samples(
                        subject_ids[i], 
                        num_samples=self.num_negatives
                    )
                    if len(negs) > 0:
                        neg_rest, neg_task, neg_subj = negs[0]
                        neg_rest_list.append(neg_rest)
                        neg_task_list.append(neg_task)
                        neg_subject_list.append(neg_subj)
                
                if len(neg_rest_list) > 0:
                    # Stack raw negative data into batches and move to device
                    neg_rest_batch = torch.stack(neg_rest_list).to(self.device)
                    neg_task_batch = torch.stack(neg_task_list).to(self.device)
                    negative_subject_ids = torch.as_tensor(neg_subject_list, device=self.device)
        
                    # Encode negatives with the same model to get embeddings
                    neg_outputs = self.model(neg_rest_batch, neg_task_batch)
                    negative_rest_embeddings = neg_outputs["rest_embedding"]
                    negative_task_embeddings = neg_outputs["task_embedding"]
            
            
            # Compute loss
            loss_dict = self.criterion(
                rest_embeddings=outputs['rest_embedding'],
                task_embeddings=outputs['task_embedding'],
                reconstructed_task=outputs['reconstructed_task'],
                original_task=task_data,
                subject_ids=subject_ids,
                negative_rest_embeddings=negative_rest_embeddings,
                negative_task_embeddings=negative_task_embeddings,
                negative_subject_ids=negative_subject_ids
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_contrastive_loss += loss_dict['contrastive_loss'].item()
            total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'contrastive': loss_dict['contrastive_loss'].item(),
                    'reconstruction': loss_dict['reconstruction_loss'].item()
                })
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        
        return {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'reconstruction_loss': avg_reconstruction_loss
        }
    
    def validate(self) -> dict:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for rest_data, task_data, subject_ids in tqdm(self.val_loader, desc='Validation'):
                rest_data = rest_data.to(self.device)
                task_data = task_data.to(self.device)
                
                # Forward pass
                outputs = self.model(rest_data, task_data)
                
                # Compute loss (without negative samples for validation speed)
                loss_dict = self.criterion(
                    rest_embeddings=outputs['rest_embedding'],
                    task_embeddings=outputs['task_embedding'],
                    reconstructed_task=outputs['reconstructed_task'],
                    original_task=task_data,
                    subject_ids=subject_ids,
                    negative_rest_embeddings=None,
                    negative_task_embeddings=None,
                    negative_subject_ids=None
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_contrastive_loss += loss_dict['contrastive_loss'].item()
                total_reconstruction_loss += loss_dict['reconstruction_loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        
        return {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'reconstruction_loss': avg_reconstruction_loss
        }

    def log_reconstructions(self, epoch: int, max_samples: int = 5):
        """
        Save a few reconstructed vs. original task samples for inspection.

        Stores a JSON file with:
        - epoch
        - subject_id
        - original_task (flattened list)
        - reconstructed_task (flattened list)
        """
        if self.val_loader is None:
            return  # nothing to log

        self.model.eval()
        samples = []

        with torch.no_grad():
            for rest_data, task_data, subject_ids in self.val_loader:
                rest_data = rest_data.to(self.device)
                task_data = task_data.to(self.device)

                outputs = self.model(rest_data, task_data)
                reconstructed = outputs['reconstructed_task'].detach().cpu()
                original = task_data.detach().cpu()

                # Convert subject_ids to plain Python ints
                if torch.is_tensor(subject_ids):
                    subject_ids_list = subject_ids.detach().cpu().tolist()
                else:
                    subject_ids_list = list(subject_ids)

                batch_size = original.shape[0]

                for i in range(batch_size):
                    sample = {
                        "epoch": int(epoch),
                        "subject_id": int(subject_ids_list[i]),
                        "original_task": original[i].view(-1).tolist(),
                        "reconstructed_task": reconstructed[i].view(-1).tolist(),
                    }
                    samples.append(sample)

                    if len(samples) >= max_samples:
                        break

                if len(samples) >= max_samples:
                    break

        out_path = self.checkpoint_dir / f"reconstructions_epoch_{epoch}.json"
        with open(out_path, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"Saved {len(samples)} reconstruction samples to {out_path}")

    def plot_history(self):
        """Plot training/validation loss curves and save as PNGs."""
        epochs = range(1, len(self.history['train_loss']) + 1)

        if len(self.history['train_loss']) == 0:
            return

        # Total loss
        plt.figure()
        plt.plot(epochs, self.history['train_loss'], label='Train')
        if len(self.history['val_loss']) == len(self.history['train_loss']):
            plt.plot(epochs, self.history['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'loss_total.png')
        plt.close()

        # Contrastive loss
        plt.figure()
        plt.plot(epochs, self.history['train_contrastive_loss'], label='Train Contrastive')
        if len(self.history['val_contrastive_loss']) == len(self.history['train_contrastive_loss']):
            plt.plot(epochs, self.history['val_contrastive_loss'], label='Val Contrastive')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('Contrastive Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'loss_contrastive.png')
        plt.close()

        # Reconstruction loss
        plt.figure()
        plt.plot(epochs, self.history['train_reconstruction_loss'], label='Train Reconstruction')
        if len(self.history['val_reconstruction_loss']) == len(self.history['train_reconstruction_loss']):
            plt.plot(epochs, self.history['val_reconstruction_loss'], label='Val Reconstruction')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'loss_reconstruction.png')
        plt.close()
    
    def train(self, num_epochs: int, save_best: bool = True):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_contrastive_loss'].append(train_metrics['contrastive_loss'])
            self.history['train_reconstruction_loss'].append(train_metrics['reconstruction_loss'])
            
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(Contrastive: {train_metrics['contrastive_loss']:.4f}, "
                  f"Reconstruction: {train_metrics['reconstruction_loss']:.4f})")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_contrastive_loss'].append(val_metrics['contrastive_loss'])
                self.history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                
                print(f"Val Loss: {val_metrics['loss']:.4f} "
                      f"(Contrastive: {val_metrics['contrastive_loss']:.4f}, "
                      f"Reconstruction: {val_metrics['reconstruction_loss']:.4f})")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if save_best and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"Saved best model (val_loss: {best_val_loss:.4f})")

                # Log a few reconstructions each epoch
                self.log_reconstructions(epoch, max_samples=5)
            else:
                self.scheduler.step(train_metrics['loss'])
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("-" * 50)
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs, is_best=False)
        
        # Save training history and plots
        self.save_history()
        try:
            self.plot_history()
        except Exception as e:
            print(f"Could not plot history: {e}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'architecture': getattr(self.model, 'architecture', 'mlp')
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']


def main():
    """
    Main training script.
    Modify these parameters based on your data dimensions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Rest-to-Task Model')
    parser.add_argument('--rest_csv', type=str, default='rest_info.csv',
                        help='Path to rest CSV file')
    parser.add_argument('--task_csv', type=str, default='hcp_task_info.csv',
                        help='Path to task CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--architecture', type=str, default='mlp',
                        choices=['mlp', 'transformer'],
                        help='Architecture type: mlp or transformer')
    parser.add_argument('--rest_input_dim', type=int, default=268*268,
                        help='Rest input dimension (flattened)')
    parser.add_argument('--task_input_dim', type=int, default=268*268,
                        help='Task input dimension (flattened)')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for contrastive loss')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./',
                        help='Checkpoint directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    # Transformer-specific arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Feedforward dimension in transformer')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for transformer (auto-determined if None)')
    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'gelu'],
                        help='Activation function for transformer')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = RestTaskDataset(
        rest_csv=args.rest_csv,
        task_csv=args.task_csv,
        device=device
    )
    
    # Split into train/val by subjects (better for generalization)
    # Get unique subjects
    unique_subjects = full_dataset.get_unique_subjects()
    num_val_subjects = int(args.val_split * len(unique_subjects))
    
    # Shuffle subjects deterministically
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    val_subjects = set(unique_subjects[:num_val_subjects])
    train_subjects = set(unique_subjects[num_val_subjects:])
    
    # Get indices for train and val subjects
    train_indices = full_dataset.get_indices_for_subjects(train_subjects)
    val_indices = full_dataset.get_indices_for_subjects(val_subjects)
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Train subjects: {len(train_subjects)}, Train pairs: {len(train_indices)}")
    print(f"Val subjects: {len(val_subjects)}, Val pairs: {len(val_indices)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Inspect first batch to get actual dimensions
    print("Inspecting data dimensions...")
    sample_rest, sample_task, _ = full_dataset[0]
    actual_rest_dim = sample_rest.shape[0]
    actual_task_dim = sample_task.shape[0]
    
    print(f"Detected rest input dimension: {actual_rest_dim}")
    print(f"Detected task input dimension: {actual_task_dim}")
    
    # Use detected dimensions if they differ from defaults
    rest_input_dim = actual_rest_dim if actual_rest_dim != args.rest_input_dim else args.rest_input_dim
    task_input_dim = actual_task_dim if actual_task_dim != args.task_input_dim else args.task_input_dim
    
    # Create model
    print(f"Creating model with {args.architecture} architecture...")
    
    model_kwargs = {
        'rest_input_dim': rest_input_dim,
        'task_input_dim': task_input_dim,
        'embedding_dim': args.embedding_dim,
        'architecture': args.architecture,
        'dropout': 0.1
    }
    
    if args.architecture == 'transformer':
        model_kwargs.update({
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'patch_size': args.patch_size,
            'activation': args.activation
        })
    
    model = RestTaskModel(**model_kwargs)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=args.epochs)
    print("Training completed!")


if __name__ == '__main__':
    main()
