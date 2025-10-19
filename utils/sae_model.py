"""
Sparse Autoencoder (SAE) for decomposing neural activations
Implements encoder-decoder with Top-K sparsity enforcement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder using Top-K sparsity
    
    Architecture:
        x -> Encoder -> TopK -> z (sparse codes) -> Decoder -> x_reconstructed
    
    Loss = MSE(x, x_reconstructed)
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_k: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_k = sparsity_k
        
        self.encoder = nn.Linear(input_dim, dict_size, bias=bias)
        self.decoder = nn.Linear(dict_size, input_dim, bias=bias)
        
        self._init_weights()
        
        logger.info(f"SAE (Top-K) initialized: {input_dim} -> {dict_size} -> {input_dim}")
        logger.info(f"  Sparsity K: {self.sparsity_k}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Encode
        pre_relu_activations = self.encoder(x)
        
        # 2. Enforce Batch Top-K Sparsity
        batch_size, dict_size = pre_relu_activations.shape
        num_to_keep = self.sparsity_k
        
        if num_to_keep <= 0:
            # Handle case where k=0 (all sparse)
            z = torch.zeros_like(pre_relu_activations)
        
        else:
            # Find the k-th largest value per sample
            kth_values, _ = torch.topk(
                pre_relu_activations, k=num_to_keep, dim=-1, sorted=False
            )
            
            # Get the minimum of these top-k values as the threshold for each sample
            threshold, _ = torch.min(kth_values, dim=-1, keepdim=True)
            
            # Create mask: 1.0 where activation >= threshold, 0.0 otherwise
            mask = (pre_relu_activations >= threshold).float()
            
            # 3. Apply mask and ReLU
            z = F.relu(pre_relu_activations * mask)
        
        # 4. Decode
        x_recon = self.decode(z)
        
        return x_recon, z

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity is enforced in forward pass, not with L1 loss
        sparsity_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Total loss is just reconstruction loss
        total_loss = recon_loss
        
        # L0 sparsity (fraction of non-zero activations) for logging
        l0_sparsity = (z > 0).float().mean()
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'l0_sparsity': l0_sparsity
        }

    def get_top_activating_features(
        self,
        x: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        pre_relu_activations = self.encoder(x)
        z = F.relu(pre_relu_activations)
        return torch.topk(z, k, dim=-1)


class SAEEnsemble(nn.Module):
    """
    Ensemble of SAEs for different layers
    """
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        num_layers: int,
        sparsity_k: int
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.saes = nn.ModuleList([
            SparseAutoencoder(input_dim, dict_size, sparsity_k)
            for _ in range(num_layers)
        ])
        
        logger.info(f"SAE Ensemble initialized with {num_layers} SAEs")
    
    def forward(self, layer_idx: int, x: torch.Tensor):
        return self.saes[layer_idx](x)
    
    def compute_loss(self, layer_idx: int, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor):
        return self.saes[layer_idx].compute_loss(x, x_recon, z)
