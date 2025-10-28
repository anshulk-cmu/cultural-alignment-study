"""
GPU-optimized activation extraction with memory management
Extracts sentence-level activations via mean pooling for SAE training
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import gc
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class MemoryEfficientActivationExtractor:
    """
    Extract sentence-level activations with automatic memory management
    Uses mean pooling over sequence length for SAE compatibility
    """
    
    def __init__(
        self, 
        model, 
        target_layers: List[int],
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize activation extractor
        
        Args:
            model: Transformer model
            target_layers: List of layer indices to extract from
            device: Device string (if None, inferred from model)
            use_fp16: Use mixed precision
        """
        self.model = model
        self.target_layers = target_layers
        self.use_fp16 = use_fp16
        self.hooks = []
        self.activations = {layer: [] for layer in target_layers}
        
        # Infer device from model
        self.device = device if device else str(next(model.parameters()).device)
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Activation extractor initialized:")
        logger.info(f"  Target layers: {target_layers}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision (FP16): {use_fp16}")
    
    def _register_hooks(self):
        """Register forward hooks on target layers"""
        
        def get_activation(layer_idx):
            def hook(module, input, output):
                # Extract hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Detach and keep on GPU temporarily (will aggregate later)
                activation = hidden_states.detach()
                self.activations[layer_idx].append(activation)
            return hook
        
        # Register hooks on Qwen layers
        try:
            for layer_idx in self.target_layers:
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(get_activation(layer_idx))
                self.hooks.append(hook)
            
            logger.info(f"  Registered {len(self.hooks)} forward hooks")
        except Exception as e:
            logger.error(f"Failed to register hooks: {e}")
            raise
    
    def clear_activations(self):
        """Clear stored activations to free memory"""
        for layer in self.target_layers:
            self.activations[layer] = []
        gc.collect()
    
    def extract_batch(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract sentence-level activations for a single batch
        Uses mean pooling over sequence length for SAE compatibility
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Dict[int, torch.Tensor]: Sentence-level activations per layer 
                                     [batch_size, hidden_dim]
        """
        try:
            # Get model device (handles multi-GPU device_map="auto")
            model_device = next(self.model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            
            # Clear previous activations
            self.clear_activations()
            
            # Forward pass with mixed precision
            with torch.no_grad():
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        _ = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask
                        )
                else:
                    _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            
            # Aggregate activations: mean pooling over sequence length
            batch_activations = {}
            for layer in self.target_layers:
                if self.activations[layer]:
                    # Concatenate all activations for this layer
                    acts = torch.cat(self.activations[layer], dim=0)  # [B, S, H]
                    
                    # Mean pooling with attention mask (ignore padding)
                    mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
                    mask_expanded = mask_expanded.to(acts.device)
                    
                    masked_acts = acts * mask_expanded  # Zero out padding
                    sum_acts = masked_acts.sum(dim=1)  # [B, H]
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                    seq_lengths = seq_lengths.to(acts.device)
                    
                    # Average over actual tokens only
                    aggregated = sum_acts / seq_lengths.clamp(min=1)  # [B, H]
                    
                    # Move to CPU and convert to float32 for numerical stability
                    batch_activations[layer] = aggregated.float().cpu()
            
            return batch_activations
            
        except Exception as e:
            logger.error(f"Error in extract_batch: {e}")
            raise
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Removed all forward hooks")


class ActivationSaver:
    """
    Incrementally save activations to disk
    Optimized: stores only indices, not full text strings
    """
    
    def __init__(
        self, 
        save_dir: Path, 
        model_type: str, 
        dataset_name: str,
        resume: bool = False
    ):
        """
        Initialize activation saver
        
        Args:
            save_dir: Base directory for saving
            model_type: 'base', 'chat', or 'delta'
            dataset_name: Name of dataset
            resume: Resume from existing chunks
        """
        self.save_dir = save_dir
        self.model_type = model_type
        self.dataset_name = dataset_name
        
        # Create save directory
        self.save_path = save_dir / dataset_name / model_type
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize buffers (optimized: no text storage)
        self.activation_buffers = {}
        self.metadata_buffer = {
            'languages': [],
            'categories': [],
            'indices': []
        }
        
        # Resume capability
        self.chunk_counter = 0
        if resume:
            existing_chunks = list(self.save_path.glob("chunk_*.npz"))
            if existing_chunks:
                self.chunk_counter = len(existing_chunks)
                logger.info(f"  Resuming from chunk {self.chunk_counter}")
        
        logger.info(f"Activation saver initialized:")
        logger.info(f"  Save path: {self.save_path}")
        logger.info(f"  Resume mode: {resume}")
    
    def add_batch(
        self, 
        activations: Dict[int, torch.Tensor],
        texts: List[str],
        languages: List[str],
        categories: List[str],
        indices: List[int]
    ):
        """
        Add a batch of activations to buffer
        
        Args:
            activations: Dict mapping layer -> tensor [batch_size, hidden_dim]
            texts: List of text strings (not stored, only for validation)
            languages: List of language labels
            categories: List of category labels
            indices: List of dataset indices
        """
        try:
            # Add activations
            for layer, acts in activations.items():
                if layer not in self.activation_buffers:
                    self.activation_buffers[layer] = []
                # Store as numpy for memory efficiency
                self.activation_buffers[layer].append(acts.numpy())
            
            # Add metadata (no text storage for memory efficiency)
            self.metadata_buffer['languages'].extend(languages)
            self.metadata_buffer['categories'].extend(categories)
            self.metadata_buffer['indices'].extend(indices)
            
        except Exception as e:
            logger.error(f"Error adding batch: {e}")
            raise
    
    def save_chunk(self):
        """Save current buffer to disk and clear"""
        
        if not self.activation_buffers:
            return
        
        try:
            logger.info(f"    Saving chunk {self.chunk_counter}...")
            
            # Concatenate all activations
            chunk_data = {
                'activations': {},
                'metadata': self.metadata_buffer.copy()
            }
            
            for layer, acts_list in self.activation_buffers.items():
                chunk_data['activations'][layer] = np.concatenate(acts_list, axis=0)
            
            # Log shapes for verification
            for layer, acts in chunk_data['activations'].items():
                logger.debug(f"      Layer {layer}: {acts.shape}")
            
            # Save
            chunk_file = self.save_path / f"chunk_{self.chunk_counter:04d}.npz"
            np.savez_compressed(chunk_file, **chunk_data)
            
            logger.info(f"    Saved {len(self.metadata_buffer['indices'])} samples to {chunk_file.name}")
            
            # Clear buffers
            self.activation_buffers = {}
            self.metadata_buffer = {
                'languages': [],
                'categories': [],
                'indices': []
            }
            self.chunk_counter += 1
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error saving chunk: {e}")
            raise
    
    def finalize(self):
        """Save any remaining data and create summary"""
        
        try:
            # Save remaining data
            if self.activation_buffers:
                self.save_chunk()
            
            # Create summary file
            summary = {
                'model_type': self.model_type,
                'dataset_name': self.dataset_name,
                'num_chunks': self.chunk_counter,
                'save_path': str(self.save_path)
            }
            
            summary_file = self.save_path / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"  ✓ Finalized: {self.chunk_counter} chunks saved")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in finalize: {e}")
            raise
