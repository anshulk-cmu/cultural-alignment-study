"""
Memory-optimized data loading for activation extraction
"""
import json
import torch
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import gc

class StreamingTextDataset(Dataset):
    """Memory-efficient dataset that loads data on-the-fly"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"  Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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


def collate_fn_optimized(batch):
    """Custom collate function for efficient batching"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'languages': [item['language'] for item in batch],
        'categories': [item['category'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }


def load_all_datasets(tokenizer, batch_size: int = 32, num_workers: int = 4):
    """Load all datasets with optimized settings"""
    from configs.config import DATASETS
    
    dataloaders = {}
    total_samples = 0
    
    print("="*80)
    print("LOADING ALL DATASETS")
    print("="*80)
    
    for dataset_name, config in DATASETS.items():
        print(f"\nLoading {dataset_name}...")
        
        if not config['path'].exists():
            print(f"  ⚠️  WARNING: {config['path']} not found, skipping")
            continue
        
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
        
        print(f"  ✓ Loaded {len(dataset)} samples")
        print(f"  Batches: {len(dataloader)}")
    
    print(f"\nTotal samples: {total_samples}")
    print("="*80)
    
    return dataloaders
