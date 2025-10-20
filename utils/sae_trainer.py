import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from tqdm import tqdm

from configs.config import (
    SAE_LEARNING_RATE,
    SAE_WEIGHT_DECAY,
    SAE_WARMUP_STEPS,
    SAE_CHECKPOINT_EVERY,
    SAE_EVAL_EVERY,
    SAE_DICT_SIZE,
    SAE_DEAD_NEURON_CHECK_EVERY,
    SAE_DEAD_NEURON_MONITOR_STEPS,
    SAE_DEAD_NEURON_THRESHOLD
)

logger = logging.getLogger(__name__)


class SAETrainer:
    
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
        dead_neuron_check_every: int = SAE_DEAD_NEURON_CHECK_EVERY,
        dead_neuron_monitor_steps: int = SAE_DEAD_NEURON_MONITOR_STEPS,
        dead_neuron_threshold: float = SAE_DEAD_NEURON_THRESHOLD
    ):
        
        self.device_ids = device_ids
        self.primary_device = f"cuda:{device_ids[0]}"
        
        self.model = model.to(self.primary_device)
        
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            logger.info(f"DataParallel on GPUs: {device_ids}")
        
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
        
        self.dead_neuron_check_every = dead_neuron_check_every
        self.dead_neuron_monitor_steps = dead_neuron_monitor_steps
        self.dead_neuron_threshold = dead_neuron_threshold
        
        self.feature_activation_counts = torch.zeros(
            SAE_DICT_SIZE,
            dtype=torch.float32,
            device=self.primary_device
        )
        self.steps_since_last_monitor_reset = 0
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Dead neuron check every: {dead_neuron_check_every}")
    
    def _get_lr_scale(self):
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        return 1.0
    
    def _update_lr(self):
        lr_scale = self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = SAE_LEARNING_RATE * lr_scale
    
    @torch.no_grad()
    def _log_activations(self, z: torch.Tensor):
        self.feature_activation_counts += (z.detach() > 0).sum(dim=0)
        self.steps_since_last_monitor_reset += z.shape[0]
    
    @torch.no_grad()
    def _reset_dead_neurons(self):
        logger.info(f"Dead neuron check at step {self.current_step}")
        
        if self.steps_since_last_monitor_reset == 0:
            logger.info("  No steps since last reset, skipping")
            return
        
        activation_freq = self.feature_activation_counts / self.steps_since_last_monitor_reset
        dead_neuron_indices = torch.where(activation_freq < self.dead_neuron_threshold)[0]
        
        if len(dead_neuron_indices) == 0:
            logger.info(f"  No dead neurons (min freq: {activation_freq.min():.6f})")
            self.feature_activation_counts.zero_()
            self.steps_since_last_monitor_reset = 0
            return
        
        logger.warning(f"  Found {len(dead_neuron_indices)} dead neurons")
        
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        new_encoder_weights = model_module.encoder.weight.data[dead_neuron_indices]
        nn.init.xavier_uniform_(new_encoder_weights)
        
        if model_module.encoder.bias is not None:
            model_module.encoder.bias.data[dead_neuron_indices] = 0.0
        
        new_decoder_weights = model_module.decoder.weight.data[:, dead_neuron_indices]
        nn.init.xavier_uniform_(new_decoder_weights)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer.param_groups[0]['lr'],
            weight_decay=self.optimizer.param_groups[0]['weight_decay']
        )
        
        logger.info(f"  Reset complete, optimizer reinitialized")
        
        self.feature_activation_counts.zero_()
        self.steps_since_last_monitor_reset = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_aux_loss = 0.0
        total_l0 = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            x = batch['activations'].to(self.primary_device)
            
            self.optimizer.zero_grad()
            
            x_recon, z, aux_loss = self.model(x, compute_aux_loss=True)
            loss_dict = self.model.module.compute_loss(x, x_recon, z, aux_loss) if isinstance(self.model, nn.DataParallel) else self.model.compute_loss(x, x_recon, z, aux_loss)
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self._log_activations(z)
            
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_aux_loss += loss_dict['aux_loss'].item()
            total_l0 += loss_dict['l0_sparsity'].item()
            num_batches += 1
            
            self.current_step += 1
            self._update_lr()
            
            if self.current_step % self.dead_neuron_check_every == 0:
                self._reset_dead_neurons()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'recon': f"{loss_dict['recon_loss'].item():.6f}",
                'aux': f"{loss_dict['aux_loss'].item():.6f}",
                'l0': f"{loss_dict['l0_sparsity'].item():.4f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches,
            'l0_sparsity': total_l0 / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_aux_loss = 0.0
        total_l0 = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch['activations'].to(self.primary_device)
            
            x_recon, z, aux_loss = self.model(x, compute_aux_loss=False)
            loss_dict = self.model.module.compute_loss(x, x_recon, z, aux_loss) if isinstance(self.model, nn.DataParallel) else self.model.compute_loss(x, x_recon, z, aux_loss)
            
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_aux_loss += loss_dict['aux_loss'].item()
            total_l0 += loss_dict['l0_sparsity'].item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches,
            'l0_sparsity': total_l0 / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'current_step': self.current_step,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if self.save_dir:
            checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = self.save_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model: {best_path}")
            else:
                logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self, num_epochs: int):
        logger.info(f"\nStarting training for {num_epochs} epochs")
        logger.info(f"  Device: {self.primary_device}")
        logger.info(f"  Multi-GPU: {len(self.device_ids) > 1}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch}/{num_epochs} (Step: {self.current_step})")
            logger.info(f"{'='*80}")
            
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            logger.info(f"\nTraining:")
            logger.info(f"  Total loss: {train_metrics['total_loss']:.6f}")
            logger.info(f"  Recon loss: {train_metrics['recon_loss']:.6f}")
            logger.info(f"  Aux loss: {train_metrics['aux_loss']:.6f}")
            logger.info(f"  L0 sparsity: {train_metrics['l0_sparsity']:.4f}")
            
            is_best = False
            if self.val_loader and epoch % SAE_EVAL_EVERY == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                
                logger.info(f"\nValidation:")
                logger.info(f"  Total loss: {val_metrics['total_loss']:.6f}")
                logger.info(f"  Recon loss: {val_metrics['recon_loss']:.6f}")
                logger.info(f"  L0 sparsity: {val_metrics['l0_sparsity']:.4f}")
                
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                    logger.info(f"  New best validation loss!")
            
            if epoch % SAE_CHECKPOINT_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
