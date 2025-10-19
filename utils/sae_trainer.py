"""
Multi-GPU SAE Training with PyTorch DataParallel
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np

from configs.config import (
    SAE_LEARNING_RATE,
    SAE_WEIGHT_DECAY,
    SAE_WARMUP_STEPS,
    SAE_CHECKPOINT_EVERY,
    SAE_EVAL_EVERY
)

logger = logging.getLogger(__name__)


class SAETrainer:
    """
    Multi-GPU trainer for Sparse Autoencoders
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device_ids: List[int],
        learning_rate: float = SAE_LEARNING_RATE,
        weight_decay: float = SAE_WEIGHT_DECAY,
        warmup_steps: int = SAE_WARMUP_STEPS,
        save_dir: Path = None
    ):
        """
        Initialize trainer
        
        Args:
            model: SAE model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device_ids: List of GPU IDs to use (e.g., [0, 1, 2])
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            save_dir: Directory to save checkpoints
        """
        self.device_ids = device_ids
        self.primary_device = f"cuda:{device_ids[0]}"
        
        # Move model to primary device
        self.model = model.to(self.primary_device)
        
        # Wrap with DataParallel for multi-GPU
        if len(device_ids) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=device_ids
            )
            logger.info(f"Using DataParallel on GPUs: {device_ids}")
        else:
            logger.info(f"Using single GPU: {device_ids[0]}")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        # Optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Warmup steps: {warmup_steps}")
    
    def _get_lr_scale(self):
        """Get learning rate scale for warmup"""
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        return 1.0
    
    def _update_lr(self):
        """Update learning rate with warmup"""
        lr_scale = self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = SAE_LEARNING_RATE * lr_scale
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict of average metrics
        """
        self.model.train()
        
        metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'sparsity_loss': 0.0,
            'l0_sparsity': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x = batch['activations'].to(self.primary_device)
            
            # Forward pass
            x_recon, z = self.model(x)
            
            # Compute loss
            loss_dict = self.model.module.compute_loss(x, x_recon, z) if isinstance(
                self.model, nn.DataParallel
            ) else self.model.compute_loss(x, x_recon, z)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update learning rate
            self.current_step += 1
            self._update_lr()
            
            # Accumulate metrics
            for key in metrics.keys():
                metrics[key] += loss_dict[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'recon': loss_dict['recon_loss'].item(),
                'l0': loss_dict['l0_sparsity'].item()
            })
        
        # Average metrics
        for key in metrics.keys():
            metrics[key] /= num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dict of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'sparsity_loss': 0.0,
            'l0_sparsity': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            x = batch['activations'].to(self.primary_device)
            
            # Forward pass
            x_recon, z = self.model(x)
            
            # Compute loss
            loss_dict = self.model.module.compute_loss(x, x_recon, z) if isinstance(
                self.model, nn.DataParallel
            ) else self.model.compute_loss(x, x_recon, z)
            
            # Accumulate
            for key in metrics.keys():
                metrics[key] += loss_dict[key].item()
        
        # Average
        for key in metrics.keys():
            metrics[key] /= num_batches
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if self.save_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(
                self.model, nn.DataParallel
            ) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self, num_epochs: int):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        logger.info(f"  Device: {self.primary_device}")
        logger.info(f"  Multi-GPU: {len(self.device_ids) > 1}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch}/{num_epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            logger.info(f"\nTraining metrics:")
            logger.info(f"  Total loss: {train_metrics['total_loss']:.6f}")
            logger.info(f"  Recon loss: {train_metrics['recon_loss']:.6f}")
            logger.info(f"  Sparsity loss: {train_metrics['sparsity_loss']:.6f}")
            logger.info(f"  L0 sparsity: {train_metrics['l0_sparsity']:.4f}")
            
            # Validate
            if epoch % SAE_EVAL_EVERY == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                
                logger.info(f"\nValidation metrics:")
                logger.info(f"  Total loss: {val_metrics['total_loss']:.6f}")
                logger.info(f"  Recon loss: {val_metrics['recon_loss']:.6f}")
                logger.info(f"  L0 sparsity: {val_metrics['l0_sparsity']:.4f}")
                
                # Check if best model
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    logger.info(f"  ✓ New best validation loss!")
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % SAE_CHECKPOINT_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
