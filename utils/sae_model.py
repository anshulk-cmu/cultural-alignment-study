import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_k: int,
        aux_k: int = 512,
        aux_coef: float = 0.03
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_k = sparsity_k
        self.aux_k = aux_k
        self.aux_coef = aux_coef
        
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        
        self._init_weights()
        
        logger.info(f"SAE initialized: {input_dim} -> {dict_size} -> {input_dim}")
        logger.info(f"  Sparsity K: {sparsity_k}")
        logger.info(f"  Aux K: {aux_k}")
        logger.info(f"  Aux coefficient: {aux_coef}")
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        
        logger.info("Encoder initialized as decoder transpose")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        compute_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = x.shape[0]
        
        pre_relu = self.encode(x)
        
        k_values = torch.topk(pre_relu, self.sparsity_k, dim=-1).values
        threshold = k_values[:, -1:]
        
        mask = (pre_relu >= threshold).float()
        z = F.relu(pre_relu * mask)
        
        x_recon = self.decode(z)
        
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if compute_aux_loss:
            recon_error = x - x_recon
            dead_mask = (mask.sum(dim=0) == 0).float()
            
            if dead_mask.sum() > 0:
                top_dead_indices = torch.topk(
                    dead_mask * torch.randn_like(dead_mask),
                    min(self.aux_k, int(dead_mask.sum().item())),
                    dim=-1
                ).indices
                
                dead_pre_relu = pre_relu[:, top_dead_indices]
                dead_z = F.relu(dead_pre_relu)
                
                dead_decoder_weights = self.decoder.weight[:, top_dead_indices]
                aux_recon = torch.matmul(dead_z, dead_decoder_weights.T) + self.decoder.bias
                
                aux_loss = F.mse_loss(aux_recon, recon_error)
        
        return x_recon, z, aux_loss
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        aux_loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        recon_loss = F.mse_loss(x_recon, x)
        
        total_loss = recon_loss + self.aux_coef * aux_loss
        
        l0_sparsity = (z > 0).float().mean()
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'aux_loss': aux_loss,
            'l0_sparsity': l0_sparsity
        }
    
    def get_top_activating_features(
        self,
        x: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        pre_relu = self.encode(x)
        z = F.relu(pre_relu)
        return torch.topk(z, k, dim=-1)


class SAEEnsemble(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        num_layers: int,
        sparsity_k: int,
        aux_k: int = 512,
        aux_coef: float = 0.03
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.saes = nn.ModuleList([
            SparseAutoencoder(
                input_dim,
                dict_size,
                sparsity_k,
                aux_k,
                aux_coef
            )
            for _ in range(num_layers)
        ])
        
        logger.info(f"SAE Ensemble with {num_layers} SAEs")
    
    def forward(
        self,
        layer_idx: int,
        x: torch.Tensor,
        compute_aux_loss: bool = True
    ):
        return self.saes[layer_idx](x, compute_aux_loss)
    
    def compute_loss(
        self,
        layer_idx: int,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        aux_loss: torch.Tensor
    ):
        return self.saes[layer_idx].compute_loss(x, x_recon, z, aux_loss)
