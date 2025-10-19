"""
Multi-GPU SAE Training with PyTorch DataParallel
Includes Top-K sparsity and dead neuron resetting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np

# Updated config imports
from configs.config import (
    SAE_LEARNING_RATE,
    SAE_WEIGHT_DECAY,
    SAE_WARMUP_STEPS,
    SAE_CHECKPOINT_EVERY,
    SAE_EVAL_EVERY,
    # NEW: Import params for dead neuron reset
    SAE_DICT_SIZE,
    SAE_DEAD_NEURON_CHECK_EVERY,
    SAE_DEAD_NEURON_MONITOR_STEPS,
    SAE_DEAD_NEURON_THRESHOLD
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
        save_dir: Path = None,
        # --- NEW: Dead neuron parameters ---
        dead_neuron_check_every: int = SAE_DEAD_NEURON_CHECK_EVERY,
        dead_neuron_monitor_steps: int = SAE_DEAD_NEURON_MONITOR_STEPS,
        dead_neuron_threshold: float = SAE_DEAD_NEURON_THRESHOLD
    ):
        """
        Initialize trainer
        """
        self.device_ids = device_ids
        self.primary_device = f"cuda:{device_ids[0]}"
        
        self.model = model.to(self.primary_device)
        
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
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        
        # --- NEW: Dead Neuron Tracking ---
        self.dead_neuron_check_every = dead_neuron_check_every
        self.dead_neuron_monitor_steps = dead_neuron_monitor_steps
        self.dead_neuron_threshold = dead_neuron_threshold
        
        # Buffer for tracking feature activations
        self.feature_activation_counts = torch.zeros(
            SAE_DICT_SIZE, 
            dtype=torch.float32, 
            device=self.primary_device
        )
        self.steps_since_last_monitor_reset = 0
        # --- End New ---

        logger.info(f"Trainer initialized:")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Dead Neuron Check Every: {self.dead_neuron_check_every} steps")
        logger.info(f"  Dead Neuron Threshold: {self.dead_neuron_threshold}")

    
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
    
    # --- NEW: Method to log activations ---
    @torch.no_grad()
    def _log_activations(self, z: torch.Tensor):
        """Logs the non-zero activations for dead neuron monitoring."""
        # Sum up non-zero activations for each feature across the batch
        self.feature_activation_counts += (z.detach() > 0).sum(dim=0)
        self.steps_since_last_monitor_reset += z.shape[0] # Add batch size

    # --- NEW: Method to reset dead neurons ---
    @torch.no_grad()
    def _reset_dead_neurons(self):
        """Check for and reset dead neurons."""
        logger.info(f"--- Running Dead Neuron Check (Step {self.current_step}) ---")
        
        if self.steps_since_last_monitor_reset == 0:
            logger.info("  Monitor steps is 0, skipping dead neuron check.")
            return

        # Calculate activation frequency
        activation_freq = self.feature_activation_counts / self.steps_since_last_monitor_reset
        
        dead_neuron_indices = torch.where(activation_freq < self.dead_neuron_threshold)[0]
        
        if len(dead_neuron_indices) == 0:
            logger.info(f"  No dead neurons found (min freq: {activation_freq.min():.6f}).")
            # Reset buffers
            self.feature_activation_counts.zero_()
            self.steps_since_last_monitor_reset = 0
            return

        logger.warning(f"  Found {len(dead_neuron_indices)} dead neurons (min freq: {activation_freq.min():.6f}). Resetting...")

        # Get the underlying model (handles DataParallel wrapper)
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # --- 1. Re-initialize encoder weights (Xavier uniform) ---
        new_encoder_weights = model_module.encoder.weight.data[dead_neuron_indices]
        nn.init.xavier_uniform_(new_encoder_weights)
        
        # --- 2. Re-initialize encoder bias (zeros) ---
        if model_module.encoder.bias is not None:
            model_module.encoder.bias.data[dead_neuron_indices] = 0.0

        # --- 3. Re-initialize decoder weights (Xavier uniform) ---
        new_decoder_weights = model_module.decoder.weight.data[:, dead_neuron_indices]
        nn.init.xavier_uniform_(new_decoder_weights)
        
        # --- 4. Reset the entire optimizer state ---
        # This is the safest way to clear old momentum/Adam states
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'], # Keep current LR
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
        )
        
        logger.info(f"  ✓ Reset {len(dead_neuron_indices)} neurons and re-initialized optimizer.")

        # --- 5. Reset monitoring buffers ---
        self.feature_activation_counts.zero_()
        self.steps_since_last_monitor_reset = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
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
            x = batch['activations'].to(self.primary_device)
            
            x_recon, z = self.model(x)
            
            loss_dict = self.model.module.compute_loss(x, x_recon, z) if isinstance(
                self.model, nn.DataParallel
            ) else self.model.compute_loss(x, x_recon, z)
            
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # --- UPDATED: Log activations before optimizer step ---
            self._log_activations(z)
            
            self.optimizer.step()
            
            self.current_step += 1
            self._update_lr()
            
            # --- UPDATED: Check for dead neuron reset ---
            if self.current_step > 0 and self.current_step % self.dead_neuron_check_every == 0:
                self._reset_dead_neurons()

            # Accumulate metrics
            for key in metrics.keys():
                metrics[key] += loss_dict[key].item()
            
            pbar.set_postfix({
                'loss': loss_dict['total_loss'].item(),
                'recon': loss_dict['recon_loss'].item(),
                'l0': loss_dict['l0_sparsity'].item()
            })
        
        for key in metrics.keys():
            metrics[key] /= num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model
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
            
            x_recon, z = self.model(x)
            
            loss_dict = self.model.module.compute_loss(x, x_recon, z) if isinstance(
                self.model, nn.DataParallel
            ) else self.model.compute_loss(x, x_recon, z)
            
            for key in metrics.keys():
                metrics[key] += loss_dict[key].item()
        
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
            'best_val_loss': self.best_val_loss,
            'current_step': self.current_step # Save global step
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        else:
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self, num_epochs: int):
        """
        Full training loop
        """
        logger.info(f"\nStarting training for {num_epochs} epochs...")
        logger.info(f"  Device: {self.primary_device}")
        logger.info(f"  Multi-GPU: {len(self.device_ids) > 1}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch}/{num_epochs} (Global Step: {self.current_step})")
            logger.info(f"{'='*80}")
            
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            logger.info(f"\nTraining metrics:")
            logger.info(f"  Total loss: {train_metrics['total_loss']:.6f}")
            logger.info(f"  Recon loss: {train_metrics['recon_loss']:.6f}")
            logger.info(f"  Sparsity loss (L1): {train_metrics['sparsity_loss']:.6f}")
            logger.info(f"  L0 sparsity: {train_metrics['l0_sparsity']:.4f}")
            
            is_best = False
            if self.val_loader and epoch % SAE_EVAL_EVERY == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                
                logger.info(f"\nValidation metrics:")
                logger.info(f"  Total loss: {val_metrics['total_loss']:.6f}")
                logger.info(f"  Recon loss: {val_metrics['recon_loss']:.6f}")
                logger.info(f"  L0 sparsity: {val_metrics['l0_sparsity']:.4f}")
                
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    logger.info(f"  ✓ New best validation loss!")
            
            if epoch % SAE_CHECKPOINT_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
