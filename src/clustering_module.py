import torch
import torch.nn as nn
import torch.nn.functional as F
from kmeans_pytorch import kmeans  # GPU-accelerated K-means
from torch import Tensor

class DEC_Clustering(nn.Module):
    def __init__(self, input_dim=512, n_clusters=10, hidden_dim=256, alpha=0.5, dropout=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize cluster centers on the same device as the model
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers)
        
    @torch.amp.autocast(device_type='cuda')
    def forward(self, embeddings: torch.Tensor) -> tuple:
        # Ensure cluster centers are on same device as input
        if self.cluster_centers.device != embeddings.device:
            self.cluster_centers.data = self.cluster_centers.data.to(embeddings.device)
            
        # Project embeddings
        z_hidden = self.projection[:-1](embeddings)
        z_recon = self.projection(embeddings)
        
        # Student's t-distribution
        q = 1.0 / (1.0 + (torch.cdist(z_hidden, self.cluster_centers)**2 / self.alpha))
        q = q**((self.alpha + 1.0) / 2.0)
        assignments = (q.t() / torch.sum(q, dim=1)).t()
        
        # Losses
        p = target_distribution(assignments.detach())
        kl_loss = F.kl_div(assignments.log(), p, reduction='batchmean')
        recon_loss = F.mse_loss(z_recon, embeddings)
        collapse_loss = torch.std(assignments.sum(0)) / torch.mean(assignments.sum(0))
        
        total_loss = kl_loss + 0.1 * recon_loss + 0.01 * collapse_loss
        pooled = torch.mm(assignments.t(), z_hidden)
        
        return assignments, pooled, kl_loss, recon_loss, total_loss, z_hidden
    
    def initialize_clusters(self, embeddings: torch.Tensor):
        """GPU-accelerated K-means initialization"""
        with torch.no_grad():
            z = self.projection[:-1](embeddings)
            
            # Convert to numpy only if absolutely necessary
            if not embeddings.is_cuda:
                z = z.cuda()
                
            cluster_ids, centers = kmeans(
                X=z, 
                num_clusters=self.n_clusters,
                device=z.device  # Ensure same device
            )
            self.cluster_centers.data = centers.float()

def target_distribution(q: Tensor) -> Tensor:
    """Compute target distribution for KL divergence loss"""
    weight = (q**2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()