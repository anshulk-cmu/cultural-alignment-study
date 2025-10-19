"""
Sparse Autoencoder (SAE) for decomposing neural activations
Implements encoder-decoder with L1 sparsity regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features
    
    Architecture:
        x -> Encoder -> z (sparse codes) -> Decoder -> x_reconstructed
    
    Loss = MSE(x, x_reconstructed) + lambda * L1(z)
    """
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_coef: float = 1e-3,
        bias: bool = True
    ):
        """
        Initialize SAE
        
        Args:
            input_dim: Dimension of input activations (e.g., 2048)
            dict_size: Size of learned dictionary (e.g., 16384)
            sparsity_coef: L1 regularization coefficient
            bias: Use bias in encoder/decoder
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_coef = sparsity_coef
        
        # Encoder: input_dim -> dict_size
        self.encoder = nn.Linear(input_dim, dict_size, bias=bias)
        
        # Decoder: dict_size -> input_dim
        self.decoder = nn.Linear(dict_size, input_dim, bias=bias)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"SAE initialized: {input_dim} -> {dict_size} -> {input_dim}")
        logger.info(f"  Sparsity coefficient: {sparsity_coef}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse codes with ReLU activation
        
        Args:
            x: Input activations [batch_size, input_dim]
        
        Returns:
            Sparse codes [batch_size, dict_size]
        """
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse codes to reconstructed activations
        
        Args:
            z: Sparse codes [batch_size, dict_size]
        
        Returns:
            Reconstructed activations [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode
        
        Args:
            x: Input activations [batch_size, input_dim]
        
        Returns:
            Tuple of (reconstructed, sparse_codes)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with reconstruction and sparsity terms
        
        Args:
            x: Original activations [batch_size, input_dim]
            x_recon: Reconstructed activations [batch_size, input_dim]
            z: Sparse codes [batch_size, dict_size]
        
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity loss (L1)
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss
        
        # L0 sparsity (fraction of non-zero activations)
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
        """
        Get top-k activating features for input
        
        Args:
            x: Input activations [batch_size, input_dim]
            k: Number of top features to return
        
        Returns:
            Tuple of (values, indices) for top-k features
        """
        z = self.encode(x)
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
        sparsity_coef: float = 1e-3
    ):
        """
        Initialize SAE ensemble for multiple layers
        
        Args:
            input_dim: Input dimension
            dict_size: Dictionary size
            num_layers: Number of layers (e.g., 3 for layers 6, 12, 18)
            sparsity_coef: Sparsity coefficient
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # Create SAE for each layer
        self.saes = nn.ModuleList([
            SparseAutoencoder(input_dim, dict_size, sparsity_coef)
            for _ in range(num_layers)
        ])
        
        logger.info(f"SAE Ensemble initialized with {num_layers} SAEs")
    
    def forward(self, layer_idx: int, x: torch.Tensor):
        """Forward pass for specific layer"""
        return self.saes[layer_idx](x)
    
    def compute_loss(self, layer_idx: int, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor):
        """Compute loss for specific layer"""
        return self.saes[layer_idx].compute_loss(x, x_recon, z)
