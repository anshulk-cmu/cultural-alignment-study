"""
Phase 1.5: Prepare text data for Phase 2.5 feature labeling

Consolidates Phase 1 activations and texts by layer, creating:
- layer{X}_texts.json: Text samples for each layer
- layer{X}_metadata.json: Language, category, indices
- layer{X}_{model_type}_activations.pt: Activation tensors

Saves to: /data/user_data/anshulk/cultural-alignment-study/activations/run_TIMESTAMP/
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import os
import torch
import numpy as np
import json
import random
from pathlib import Path
from collections import defaultdict

from configs.config import (
    ACTIVATION_ROOT,
    TARGET_LAYERS,
    DATASETS,
    setup_logger
)

SEED = 42


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def find_latest_run(activation_root: Path) -> Path:
    """Find the most recent Phase 1 activation run directory"""
    run_dirs = sorted(activation_root.glob("run_*"))
    if not run_dirs:
        raise ValueError(f"No run directories found in {activation_root}")
    return run_dirs[-1]


def load_original_datasets(logger):
    """Load original text datasets from train/test JSON files"""
    logger.info("Loading original datasets from JSON files...")
    
    datasets_loaded = {}
    
    for dataset_name, config in DATASETS.items():
        try:
            logger.info(f"  Loading {dataset_name} from {config['path']}...")
            
            if not config['path'].exists():
                raise FileNotFoundError(f"Dataset file not found: {config['path']}")
            
            with open(config['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = [item['text'] for item in data]
            datasets_loaded[dataset_name] = texts
            logger.info(f"    ✓ Loaded {len(texts)} texts")
            
        except Exception as e:
            logger.error(f"    ✗ Failed to load {dataset_name}: {e}")
            raise
    
    return datasets_loaded


def load_dataset_activations(run_dir: Path, dataset_name: str, model_type: str, 
                            layer_idx: int, dataset_texts: list, logger):
    """Load and consolidate activation chunks for a dataset"""
    activation_dir = run_dir / dataset_name / model_type
    
    if not activation_dir.exists():
        logger.warning(f"    Skipping {model_type} (directory not found)")
        return None, [], []
    
    chunk_files = sorted(activation_dir.glob("chunk_*.npz"))
    logger.info(f"    Loading {len(chunk_files)} chunks for {model_type}...")
    
    activations_list = []
    texts = []
    metadata = []
    
    for chunk_file in chunk_files:
        data = np.load(chunk_file, allow_pickle=True)
        
        # Extract activations for this layer
        acts = data['activations'].item()[layer_idx]
        activations_list.append(acts)
        
        # Extract metadata
        meta = data['metadata'].item()
        indices = meta['indices']
        languages = meta['languages']
        categories = meta['categories']
        
        # Retrieve texts using indices
        for idx, lang, cat in zip(indices, languages, categories):
            texts.append(dataset_texts[idx])
            metadata.append({
                'dataset': dataset_name,
                'language': lang,
                'category': cat,
                'original_index': idx
            })
    
    # Concatenate all chunks
    activations = np.concatenate(activations_list, axis=0)
    logger.info(f"      {model_type}: {activations.shape[0]} samples")
    
    return activations, texts, metadata


def consolidate_layer(run_dir: Path, datasets: dict, layer_idx: int, 
                      output_dir: Path, logger):
    """Consolidate activations and texts for a single layer"""
    logger.info(f"\n{'─'*80}")
    logger.info(f"Processing Layer {layer_idx}")
    logger.info(f"{'─'*80}")
    
    model_types = ['base', 'chat', 'delta']
    dataset_names = list(datasets.keys())
    
    # Storage for layer data
    layer_activations = defaultdict(list)
    layer_texts = []
    layer_metadata = {
        'dataset_names': [],
        'languages': [],
        'categories': [],
        'original_indices': []
    }
    
    # Process each dataset
    for dataset_name in dataset_names:
        logger.info(f"  Processing dataset: {dataset_name}")
        dataset_texts = datasets[dataset_name]
        
        # Load activations for each model type
        dataset_texts_loaded = None
        dataset_meta_loaded = None
        
        for model_type in model_types:
            acts, texts, metadata = load_dataset_activations(
                run_dir, dataset_name, model_type, layer_idx, 
                dataset_texts, logger
            )
            
            if acts is not None:
                layer_activations[model_type].append(acts)
                
                # Store texts and metadata once (same for all model types)
                if dataset_texts_loaded is None:
                    dataset_texts_loaded = texts
                    dataset_meta_loaded = metadata
        
        # Add to layer collections
        if dataset_texts_loaded:
            layer_texts.extend(dataset_texts_loaded)
            layer_metadata['dataset_names'].extend([dataset_name] * len(dataset_texts_loaded))
            layer_metadata['languages'].extend([m['language'] for m in dataset_meta_loaded])
            layer_metadata['categories'].extend([m['category'] for m in dataset_meta_loaded])
            layer_metadata['original_indices'].extend([m['original_index'] for m in dataset_meta_loaded])
    
    # Concatenate activations across datasets
    for model_type in model_types:
        if layer_activations[model_type]:
            layer_activations[model_type] = np.concatenate(
                layer_activations[model_type], axis=0
            )
    
    # Save texts
    text_file = output_dir / f"layer{layer_idx}_texts.json"
    logger.info(f"\n  Saving texts to: {text_file}")
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump(layer_texts, f, ensure_ascii=False, indent=2)
    logger.info(f"    ✓ Saved {len(layer_texts)} texts")
    
    # Save metadata
    metadata_file = output_dir / f"layer{layer_idx}_metadata.json"
    logger.info(f"  Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(layer_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"    ✓ Saved metadata")
    
    # Save activations (one file per model type)
    for model_type in model_types:
        if model_type in layer_activations and len(layer_activations[model_type]) > 0:
            acts_tensor = torch.from_numpy(layer_activations[model_type]).float()
            
            act_file = output_dir / f"layer{layer_idx}_{model_type}_activations.pt"
            logger.info(f"  Saving {model_type} activations to: {act_file}")
            torch.save(acts_tensor, act_file)
            
            size_mb = acts_tensor.element_size() * acts_tensor.nelement() / 1024**2
            logger.info(f"    ✓ Saved {acts_tensor.shape} tensor ({size_mb:.1f} MB)")
    
    logger.info(f"\n  ✓ Layer {layer_idx} complete")
    return len(layer_texts)


def main():
    """Main execution"""
    set_seed(SEED)
    
    logger = setup_logger('phase1_5_prepare', 'phase1_5_prepare.log')
    
    try:
        logger.info("="*80)
        logger.info("PHASE 1.5: PREPARING TEXT DATA FOR PHASE 2.5")
        logger.info("="*80)
        logger.info(f"Random seed: {SEED}")
        
        # Find latest Phase 1 run
        run_dir = find_latest_run(ACTIVATION_ROOT)
        logger.info(f"\nUsing activation data from: {run_dir}")
        
        # Output to same run directory
        output_dir = run_dir
        logger.info(f"Output directory: {output_dir}")
        
        # Load original datasets
        datasets = load_original_datasets(logger)
        logger.info(f"\nLoaded {len(datasets)} datasets:")
        for name, texts in datasets.items():
            logger.info(f"  {name}: {len(texts)} texts")
        
        # Process each layer
        logger.info("\n" + "="*80)
        logger.info("CONSOLIDATING ACTIVATIONS AND TEXTS BY LAYER")
        logger.info("="*80)
        
        total_samples = 0
        for layer_idx in TARGET_LAYERS:
            num_samples = consolidate_layer(run_dir, datasets, layer_idx, output_dir, logger)
            total_samples = max(total_samples, num_samples)
        
        # Create summary
        model_types = ['base', 'chat', 'delta']
        summary = {
            'run_dir': str(run_dir),
            'output_dir': str(output_dir),
            'seed': SEED,
            'layers': TARGET_LAYERS,
            'model_types': model_types,
            'datasets': list(datasets.keys()),
            'total_samples': total_samples,
            'files_created': {
                'texts': [f"layer{l}_texts.json" for l in TARGET_LAYERS],
                'metadata': [f"layer{l}_metadata.json" for l in TARGET_LAYERS],
                'activations': [f"layer{l}_{mt}_activations.pt" for l in TARGET_LAYERS for mt in model_types]
            }
        }
        
        summary_file = output_dir / "consolidation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("CONSOLIDATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total samples: {summary['total_samples']}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Summary saved to: {summary_file}")
        
        logger.info("\n" + "="*80)
        logger.info("✓ PHASE 1.5 COMPLETE")
        logger.info("="*80)
        logger.info("\nNext step: Run Phase 2.5 scripts")
        logger.info("  python scripts/phase2_5_extract_examples.py")
        
    except Exception as e:
        logger.error(f"Phase 1.5 failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
