"""
Dataset for loading activation chunks for SAE training
"""
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    """Initialize each dataloader worker with unique but reproducible seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ActivationDataset(Dataset):
    """
    Dataset for loading pre-extracted activations from .npz chunks
    """
    
    def __init__(
        self,
        activation_dir: Path,
        layer_idx: int,
        model_type: str = "base"
    ):
        """
        Initialize activation dataset
        
        Args:
            activation_dir: Directory containing activation chunks
            layer_idx: Layer index to load (6, 12, or 18)
            model_type: 'base', 'chat', or 'delta'
        """
        self.activation_dir = activation_dir
        self.layer_idx = layer_idx
        self.model_type = model_type
        
        # Find all chunk files
        self.chunk_files = sorted(activation_dir.glob("chunk_*.npz"))
        
        if not self.chunk_files:
            raise ValueError(f"No chunks found in {activation_dir}")
        
        # Load all activations into memory (they're already compressed)
        logger.info(f"Loading {len(self.chunk_files)} chunks for layer {layer_idx}...")
        self.activations = []
        self.metadata = []
        
        for chunk_file in self.chunk_files:
            data = np.load(chunk_file, allow_pickle=True)
            acts = data['activations'].item()[layer_idx]
            self.activations.append(acts)
            self.metadata.extend([
                {
                    'language': lang,
                    'category': cat,
                    'index': idx
                }
                for lang, cat, idx in zip(
                    data['metadata'].item()['languages'],
                    data['metadata'].item()['categories'],
                    data['metadata'].item()['indices']
                )
            ])
        
        # Concatenate all activations
        self.activations = np.concatenate(self.activations, axis=0)
        
        logger.info(f"  Loaded {len(self.activations)} activations")
        logger.info(f"  Shape: {self.activations.shape}")
        logger.info(f"  Memory: {self.activations.nbytes / 1024**3:.2f} GB")
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        """
        Get single activation vector
        
        Returns:
            Dict with 'activation' tensor and metadata
        """
        return {
            'activation': torch.from_numpy(self.activations[idx]).float(),
            'metadata': self.metadata[idx]
        }


def create_activation_dataloaders(
    run_dir: Path,
    dataset_names: List[str],
    layer_idx: int,
    model_type: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for all datasets with reproducibility
    
    Args:
        run_dir: Run directory from Phase 1 (e.g., run_20251019_192554)
        dataset_names: List of dataset names (e.g., ['train', 'test'])
        layer_idx: Layer index
        model_type: 'base', 'chat', or 'delta'
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Shuffle data
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping dataset_name -> DataLoader
    """
    # Set generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloaders = {}
    
    for dataset_name in dataset_names:
        activation_dir = run_dir / dataset_name / model_type
        
        if not activation_dir.exists():
            logger.warning(f"Activation dir not found: {activation_dir}")
            continue
        
        dataset = ActivationDataset(activation_dir, layer_idx, model_type)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_activations,
            generator=g,
            worker_init_fn=worker_init_fn
        )
        
        dataloaders[dataset_name] = dataloader
        logger.info(f"  Created dataloader for {dataset_name}: {len(dataset)} samples")
    
    return dataloaders


def collate_activations(batch):
    """Collate function for activation batches"""
    activations = torch.stack([item['activation'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'activations': activations,
        'metadata': metadata
    }
