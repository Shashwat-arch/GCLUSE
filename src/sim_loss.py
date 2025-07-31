import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, temperature=0.07, gamma=0.5, scale_by_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma  # RBF bandwidth parameter
        self.scale_by_temperature = scale_by_temperature

    def forward(self, out, mask):
        device = out.device
        
        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col = row.to(device), col.to(device)
        batch_size = out.shape[0]

        # Compute pairwise squared Euclidean distances
        pairwise_dist = torch.cdist(out, out, p=2).pow(2)  # [batch, batch]
        
        # Compute RBF kernel similarity
        sim_matrix = torch.exp(-self.gamma * pairwise_dist)
        
        # Apply temperature scaling
        sim_matrix = torch.div(sim_matrix, self.temperature)

        # Numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        # Mask self-comparisons
        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size, device=device),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )

        # Compute probabilities with epsilon for numerical stability
        exp_logits = torch.exp(sim_matrix) * logits_mask
        sum_exp = exp_logits.sum(1, keepdim=True) + 1e-8  # Prevent log(0)
        log_probs = sim_matrix - torch.log(sum_exp)

        # Calculate loss using positive pairs
        log_probs_pos = log_probs[row, col]
        loss = -log_probs_pos.mean()

        if self.scale_by_temperature:
            loss *= self.temperature
            
        return loss