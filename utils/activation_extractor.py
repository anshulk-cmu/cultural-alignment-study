"""
GPU-optimized activation extraction with memory management
"""
import torch
import torch.nn as nn
from typing import Dict, List
import gc
from tqdm import tqdm
from pathlib import Path
import numpy as np


class MemoryEfficientActivationExtractor:
    """
    Extract activations with automatic memory management
    """
    
    def __init__(
        self, 
        model, 
        target_layers: List[int],
        device: str = "cuda",
        use_fp16: bool = True
    ):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.use_fp16 = use_fp16
        self.hooks = []
        self.activations = {layer: [] for layer in target_layers}
        
        # Register hooks
        self._register_hooks()
        
        print(f"Activation extractor initialized:")
        print(f"  Target layers: {target_layers}")
        print(f"  Device: {device}")
        print(f"  Mixed precision (FP16): {use_fp16}")
    
    def _register_hooks(self):
        """Register forward hooks on target layers"""
        
        def get_activation(layer_idx):
            def hook(module, input, output):
                # Extract hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]  # First element is usually hidden states
                else:
                    hidden_states = output
                
                # Detach and move to CPU immediately to save GPU memory
                # Store as float32 for numerical stability in SAE training
                activation = hidden_states.detach().float().cpu()
                self.activations[layer_idx].append(activation)
            return hook
        
        # Register hooks on Qwen layers
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(get_activation(layer_idx))
            self.hooks.append(hook)
        
        print(f"  Registered {len(self.hooks)} forward hooks")
    
    def clear_activations(self):
        """Clear stored activations to free memory"""
        for layer in self.target_layers:
            self.activations[layer] = []
        gc.collect()
    
    def extract_batch(self, input_ids, attention_mask):
        """
        Extract activations for a single batch
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Dict[int, torch.Tensor]: Activations per layer
        """
        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Clear previous activations
        self.clear_activations()
        
        # Forward pass with mixed precision
        with torch.no_grad():
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    _ = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
            else:
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        # Concatenate activations (they're already on CPU)
        batch_activations = {}
        for layer in self.target_layers:
            if self.activations[layer]:
                batch_activations[layer] = torch.cat(self.activations[layer], dim=0)
        
        return batch_activations
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("Removed all forward hooks")


class ActivationSaver:
    """
    Incrementally save activations to disk to avoid memory overflow
    """
    
    def __init__(self, save_dir: Path, model_type: str, dataset_name: str):
        self.save_dir = save_dir
        self.model_type = model_type
        self.dataset_name = dataset_name
        
        # Create save directory
        self.save_path = save_dir / dataset_name / model_type
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize buffers
        self.activation_buffers = {}
        self.metadata_buffer = {
            'texts': [],
            'languages': [],
            'categories': [],
            'indices': []
        }
        
        self.chunk_counter = 0
        
        print(f"Activation saver initialized:")
        print(f"  Save path: {self.save_path}")
    
    def add_batch(
        self, 
        activations: Dict[int, torch.Tensor],
        texts: List[str],
        languages: List[str],
        categories: List[str],
        indices: List[int]
    ):
        """Add a batch of activations to buffer"""
        
        # Add activations
        for layer, acts in activations.items():
            if layer not in self.activation_buffers:
                self.activation_buffers[layer] = []
            # Store on CPU as numpy for memory efficiency
            self.activation_buffers[layer].append(acts.numpy())
        
        # Add metadata
        self.metadata_buffer['texts'].extend(texts)
        self.metadata_buffer['languages'].extend(languages)
        self.metadata_buffer['categories'].extend(categories)
        self.metadata_buffer['indices'].extend(indices)
    
    def save_chunk(self):
        """Save current buffer to disk and clear"""
        
        if not self.activation_buffers:
            return
        
        print(f"    Saving chunk {self.chunk_counter}...")
        
        # Concatenate all activations
        chunk_data = {
            'activations': {},
            'metadata': self.metadata_buffer.copy()
        }
        
        for layer, acts_list in self.activation_buffers.items():
            chunk_data['activations'][layer] = np.concatenate(acts_list, axis=0)
        
        # Save
        chunk_file = self.save_path / f"chunk_{self.chunk_counter:04d}.npz"
        np.savez_compressed(chunk_file, **chunk_data)
        
        print(f"    Saved {len(self.metadata_buffer['texts'])} samples to {chunk_file}")
        
        # Clear buffers
        self.activation_buffers = {}
        self.metadata_buffer = {
            'texts': [],
            'languages': [],
            'categories': [],
            'indices': []
        }
        self.chunk_counter += 1
        
        # Force garbage collection
        gc.collect()
    
    def finalize(self):
        """Save any remaining data and create summary"""
        
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
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Finalized: {self.chunk_counter} chunks saved")
        
        return summary
