import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning shared embeddings.
    
    Pulls together embeddings of the same subject (rest and task).
    Pushes apart embeddings of different subjects.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for softmax (controls softness)
            margin: Margin for contrastive loss (for different subjects)
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self,
        rest_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor,
        subject_ids: list,
        negative_rest_embeddings: torch.Tensor = None,
        negative_task_embeddings: torch.Tensor = None,
        negative_subject_ids: list = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            rest_embeddings: Embeddings from rest encoder (batch_size, embedding_dim)
            task_embeddings: Embeddings from task encoder (batch_size, embedding_dim)
            subject_ids: List of subject IDs for each sample in the batch
            negative_rest_embeddings: Optional negative rest embeddings (different subjects)
            negative_task_embeddings: Optional negative task embeddings (different subjects)
            negative_subject_ids: Optional negative subject IDs
            
        Returns:
            Contrastive loss value
        """
        batch_size = rest_embeddings.size(0)
        
        # Normalize embeddings to unit vectors
        rest_embeddings = F.normalize(rest_embeddings, p=2, dim=1)
        task_embeddings = F.normalize(task_embeddings, p=2, dim=1)
        
        # Positive pairs: same subject's rest and task should be similar
        # Compute similarity between rest and task embeddings for same subjects
        positive_similarities = (rest_embeddings * task_embeddings).sum(dim=1)  # (batch_size,)
        
        # InfoNCE-style loss: maximize positive similarity relative to negatives
        if negative_rest_embeddings is not None and negative_task_embeddings is not None:
            # We have explicit negative samples
            negative_rest_embeddings = F.normalize(negative_rest_embeddings, p=2, dim=1)
            negative_task_embeddings = F.normalize(negative_task_embeddings, p=2, dim=1)
            
            # Compute similarities with negatives
            neg_similarities_rest = (rest_embeddings.unsqueeze(1) * negative_rest_embeddings.unsqueeze(0)).sum(dim=2)
            neg_similarities_task = (task_embeddings.unsqueeze(1) * negative_task_embeddings.unsqueeze(0)).sum(dim=2)
            
            # Contrastive loss: -log(exp(pos_sim / temp) / (exp(pos_sim / temp) + sum(exp(neg_sim / temp))))
            pos_logits = positive_similarities / self.temperature
            neg_logits = torch.cat([neg_similarities_rest, neg_similarities_task], dim=1) / self.temperature
            
            # Compute log probabilities
            log_probs = pos_logits - torch.logsumexp(torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1), dim=1)
            
            loss = -log_probs.mean()
        else:
            # Use batch negatives: different subjects in the same batch
            # Create labels: 1 for same subject, 0 for different
            labels = torch.zeros(batch_size, batch_size, device=rest_embeddings.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if subject_ids[i] == subject_ids[j]:
                        labels[i, j] = 1.0
            
            # Compute similarity matrix: rest_i vs task_j
            similarity_matrix = torch.matmul(rest_embeddings, task_embeddings.t()) / self.temperature
            
            # Mask for positive pairs (same subject)
            mask = labels.bool()
            
            # InfoNCE loss: for each rest embedding, classify the correct task embedding
            # Remove diagonal if we want to exclude self-pairs
            loss_rest_to_task = self._infonce_loss(similarity_matrix, mask)
            
            # Symmetric loss: for each task embedding, classify the correct rest embedding
            loss_task_to_rest = self._infonce_loss(similarity_matrix.t(), mask.t())
            
            loss = (loss_rest_to_task + loss_task_to_rest) / 2.0
        
        return loss
    
    def _infonce_loss(self, similarity_matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            similarity_matrix: (batch_size, batch_size) similarity matrix
            mask: (batch_size, batch_size) boolean mask for positive pairs
            
        Returns:
            InfoNCE loss
        """
        batch_size = similarity_matrix.size(0)
        
        # For each row, we want to maximize similarity to positive pairs
        # and minimize similarity to negative pairs
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of positive similarities for each row
        pos_sum = (exp_sim * mask.float()).sum(dim=1)
        
        # Total sum (positives + negatives)
        total_sum = exp_sim.sum(dim=1)
        
        # Log probability of positive
        log_probs = torch.log(pos_sum / (total_sum + 1e-8))
        
        # Negative log likelihood
        loss = -log_probs.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: contrastive loss + reconstruction loss.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        temperature: float = 0.07,
        reconstruction_loss_fn: str = 'mse'
    ):
        """
        Args:
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            temperature: Temperature for contrastive loss
            reconstruction_loss_fn: Type of reconstruction loss ('mse' or 'mae')
        """
        super(CombinedLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        if reconstruction_loss_fn == 'mse':
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss_fn == 'mae':
            self.reconstruction_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss_fn}")
    
    def forward(
        self,
        rest_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor,
        reconstructed_task: torch.Tensor,
        original_task: torch.Tensor,
        subject_ids: list,
        negative_rest_embeddings: torch.Tensor = None,
        negative_task_embeddings: torch.Tensor = None,
        negative_subject_ids: list = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            rest_embeddings: Embeddings from rest encoder
            task_embeddings: Embeddings from task encoder
            reconstructed_task: Reconstructed task from decoder
            original_task: Original task data
            subject_ids: Subject IDs for contrastive loss
            negative_rest_embeddings: Optional negative rest embeddings
            negative_task_embeddings: Optional negative task embeddings
            negative_subject_ids: Optional negative subject IDs
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(
            rest_embeddings,
            task_embeddings,
            subject_ids,
            negative_rest_embeddings,
            negative_task_embeddings,
            negative_subject_ids
        )
        
        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed_task, original_task)
        
        # Combined loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.reconstruction_weight * reconstruction_loss
        )
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'reconstruction_loss': reconstruction_loss
        }

