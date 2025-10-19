"""
Memory-optimized data loading for activation extraction
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import gc
import logging

# Import from config
from configs.config import MAX_LENGTH, DATASETS

logger = logging.getLogger(__name__)


class StreamingTextDataset(Dataset):
    """Memory-efficient dataset that loads data on-the-fly"""
    
    def __init__(
        self, 
        data_path: Path, 
        tokenizer, 
        max_length: Optional[int] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (uses config default if None)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else MAX_LENGTH
        
        try:
            logger.info(f"Loading from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"  Loaded {len(self.data)} samples")
            logger.info(f"  Max sequence length: {self.max_length}")
        except Exception as e:
            logger.error(f"Failed to load dataset from {data_path}: {e}")
            raise
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['text']
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'text': text,
                'language': item.get('language', 'unknown'),
                'category': item.get('category', 'unknown'),
                'idx': idx
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise


def collate_fn_optimized(batch):
    """Custom collate function for efficient batching"""
    try:
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'texts': [item['text'] for item in batch],
            'languages': [item['language'] for item in batch],
            'categories': [item['category'] for item in batch],
            'indices': [item['idx'] for item in batch]
        }
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        raise


def load_all_datasets(
    tokenizer, 
    batch_size: int = 32, 
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Load all datasets with optimized settings
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size per dataset
        num_workers: Number of data loading workers
    
    Returns:
        Dict mapping dataset_name -> DataLoader
    """
    dataloaders = {}
    total_samples = 0
    
    logger.info("="*80)
    logger.info("LOADING ALL DATASETS")
    logger.info("="*80)
    
    for dataset_name, config in DATASETS.items():
        logger.info(f"\nLoading {dataset_name}...")
        
        if not config['path'].exists():
            logger.warning(f"  Dataset not found: {config['path']}, skipping")
            continue
        
        try:
            dataset = StreamingTextDataset(config['path'], tokenizer)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn_optimized,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            
            dataloaders[dataset_name] = dataloader
            total_samples += len(dataset)
            
            logger.info(f"  ✓ Loaded {len(dataset)} samples")
            logger.info(f"  Batches: {len(dataloader)}")
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue
    
    logger.info(f"\nTotal samples: {total_samples}")
    logger.info("="*80)
    
    if not dataloaders:
        logger.error("No datasets loaded successfully!")
        raise RuntimeError("Failed to load any datasets")
    
    return dataloaders
