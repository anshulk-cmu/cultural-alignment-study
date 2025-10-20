"""
Phase 1.5: Post-processing to prepare text data for Phase 2.5

This script bridges Phase 1 (activation extraction) and Phase 2.5 (feature labeling) by:
1. Loading activation chunks from Phase 1 output
2. Extracting sample indices from metadata
3. Reloading original datasets to retrieve corresponding texts
4. Creating consolidated layer{X}_texts.json files
5. Optionally consolidating activations to layer{X}_activations.pt format

This is a one-time CPU/single-GPU operation that avoids re-running expensive Phase 1.
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging

from configs.config import (
    ACTIVATION_ROOT,
    TARGET_LAYERS,
    DATASETS,
    setup_logger
)
from utils.data_loader import load_all_datasets
from transformers import AutoTokenizer

logger = setup_logger('phase1_5_prepare', 'phase1_5_prepare.log')


def find_latest_run(activation_root: Path) -> Path:
    """Find the latest activation run directory."""
    run_dirs = sorted(activation_root.glob("run_*"))
    if not run_dirs:
        raise ValueError(f"No run directories found in {activation_root}")
    return run_dirs[-1]


def extract_assistant_response(messages):
    """Extract only the assistant's response from messages list."""
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            return message.get('content', '')
    return None


def load_original_datasets():
    """Load original datasets to retrieve text content."""
    logger.info("Loading original datasets...")

    from datasets import load_dataset

    datasets_loaded = {}

    for dataset_name, config in DATASETS.items():
        try:
            logger.info(f"  Loading {dataset_name}...")

            # Special handling for Updesh_beta (has multiple splits and special config)
            if dataset_name == "updesh_beta":
                all_texts = []

                # Load both English and Hindi splits
                for split in config['split']:
                    logger.info(f"    Loading split: {split}...")
                    dataset = load_dataset(
                        config['path'],
                        config['config'],
                        split=split
                    )

                    # Extract assistant responses from messages
                    for sample in dataset:
                        messages = sample.get(config['text_field'], [])
                        assistant_text = extract_assistant_response(messages)
                        if assistant_text:
                            all_texts.append(assistant_text)

                    logger.info(f"      Extracted {len(all_texts)} texts from {split}")

                # Truncate to max_samples
                max_samples = config.get('max_samples', len(all_texts))
                texts = all_texts[:max_samples]

            else:
                # Standard dataset loading
                dataset = load_dataset(
                    config['path'],
                    split=config['split']
                )

                # Extract text field
                if '.' in config['text_field']:
                    # Handle nested fields like 'translation.hi'
                    parts = config['text_field'].split('.')
                    texts = []
                    for sample in dataset:
                        value = sample
                        for part in parts:
                            value = value[part]
                        texts.append(value)
                else:
                    texts = [sample[config['text_field']] for sample in dataset]

                # Truncate to max_samples
                max_samples = config.get('max_samples', len(texts))
                texts = texts[:max_samples]

            datasets_loaded[dataset_name] = texts
            logger.info(f"    ✓ Loaded {len(texts)} texts for {dataset_name}")

        except Exception as e:
            logger.error(f"    ✗ Failed to load {dataset_name}: {e}")
            raise

    return datasets_loaded


def consolidate_activations_and_texts(run_dir: Path, datasets: dict, output_dir: Path):
    """
    Consolidate activations and texts by layer.

    Args:
        run_dir: Phase 1 output directory (e.g., run_20251019_192554)
        datasets: Dict mapping dataset_name -> list of texts
        output_dir: Output directory for consolidated files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_types = ['base', 'chat', 'delta']
    dataset_names = list(datasets.keys())

    logger.info("="*80)
    logger.info("CONSOLIDATING ACTIVATIONS AND TEXTS BY LAYER")
    logger.info("="*80)

    for layer_idx in TARGET_LAYERS:
        logger.info(f"\n{'─'*80}")
        logger.info(f"Processing Layer {layer_idx}")
        logger.info(f"{'─'*80}")

        # Dictionary to store all activations and texts for this layer
        layer_data = {
            'activations': defaultdict(list),  # model_type -> list of activation arrays
            'texts': [],                        # consolidated texts
            'metadata': {
                'dataset_names': [],
                'languages': [],
                'categories': [],
                'original_indices': [],
                'model_type_offsets': {}       # model_type -> start index
            }
        }

        # Track global sample index
        global_sample_idx = 0

        # Process each dataset
        for dataset_name in dataset_names:
            logger.info(f"  Processing dataset: {dataset_name}")

            dataset_texts = datasets[dataset_name]

            # Load activations for all model types
            for model_type in model_types:
                activation_dir = run_dir / dataset_name / model_type

                if not activation_dir.exists():
                    logger.warning(f"    Skipping {model_type} (directory not found)")
                    continue

                # Load all chunks for this dataset/model/layer
                chunk_files = sorted(activation_dir.glob("chunk_*.npz"))
                logger.info(f"    Loading {len(chunk_files)} chunks for {model_type}...")

                model_activations = []
                chunk_texts = []
                chunk_metadata = []

                for chunk_file in chunk_files:
                    data = np.load(chunk_file, allow_pickle=True)

                    # Extract activations for this layer
                    acts = data['activations'].item()[layer_idx]
                    model_activations.append(acts)

                    # Extract metadata
                    metadata = data['metadata'].item()
                    indices = metadata['indices']
                    languages = metadata['languages']
                    categories = metadata['categories']

                    # Get texts using indices
                    for idx, lang, cat in zip(indices, languages, categories):
                        chunk_texts.append(dataset_texts[idx])
                        chunk_metadata.append({
                            'dataset': dataset_name,
                            'language': lang,
                            'category': cat,
                            'original_index': idx
                        })

                # Concatenate all chunks for this model type
                model_activations = np.concatenate(model_activations, axis=0)
                layer_data['activations'][model_type].append(model_activations)

                logger.info(f"      {model_type}: {model_activations.shape[0]} samples")

            # Store texts and metadata (same for all model types)
            layer_data['texts'].extend(chunk_texts)
            layer_data['metadata']['dataset_names'].extend([dataset_name] * len(chunk_texts))
            layer_data['metadata']['languages'].extend([m['language'] for m in chunk_metadata])
            layer_data['metadata']['categories'].extend([m['category'] for m in chunk_metadata])
            layer_data['metadata']['original_indices'].extend([m['original_index'] for m in chunk_metadata])

            global_sample_idx += len(chunk_texts)

        # Concatenate activations across datasets for each model type
        for model_type in model_types:
            if layer_data['activations'][model_type]:
                layer_data['activations'][model_type] = np.concatenate(
                    layer_data['activations'][model_type],
                    axis=0
                )

        # Save texts as JSON
        text_file = output_dir / f"layer{layer_idx}_texts.json"
        logger.info(f"\n  Saving texts to: {text_file}")
        with open(text_file, 'w', encoding='utf-8') as f:
            json.dump(layer_data['texts'], f, ensure_ascii=False, indent=2)
        logger.info(f"    ✓ Saved {len(layer_data['texts'])} texts")

        # Save metadata as JSON
        metadata_file = output_dir / f"layer{layer_idx}_metadata.json"
        logger.info(f"  Saving metadata to: {metadata_file}")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(layer_data['metadata'], f, ensure_ascii=False, indent=2)
        logger.info(f"    ✓ Saved metadata")

        # Save activations as PyTorch tensors (one file per model type)
        for model_type in model_types:
            if model_type in layer_data['activations'] and len(layer_data['activations'][model_type]) > 0:
                acts = layer_data['activations'][model_type]
                acts_tensor = torch.from_numpy(acts).float()

                act_file = output_dir / f"layer{layer_idx}_{model_type}_activations.pt"
                logger.info(f"  Saving {model_type} activations to: {act_file}")
                torch.save(acts_tensor, act_file)
                logger.info(f"    ✓ Saved {acts_tensor.shape} tensor ({acts_tensor.element_size() * acts_tensor.nelement() / 1024**2:.1f} MB)")

        logger.info(f"\n  ✓ Layer {layer_idx} complete")

    # Create summary file
    summary = {
        'run_dir': str(run_dir),
        'output_dir': str(output_dir),
        'layers': TARGET_LAYERS,
        'model_types': model_types,
        'datasets': dataset_names,
        'total_samples': len(layer_data['texts']),
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


def main():
    """Main execution."""
    try:
        logger.info("="*80)
        logger.info("PHASE 1.5: PREPARING TEXT DATA FOR PHASE 2.5")
        logger.info("="*80)

        # Find latest activation run
        run_dir = find_latest_run(ACTIVATION_ROOT)
        logger.info(f"\nUsing activation data from: {run_dir}")

        # Output to the same run directory for easy access
        output_dir = run_dir
        logger.info(f"Output directory: {output_dir}")

        # Load original datasets
        datasets = load_original_datasets()
        logger.info(f"\nLoaded {len(datasets)} datasets:")
        for name, texts in datasets.items():
            logger.info(f"  {name}: {len(texts)} texts")

        # Consolidate activations and texts
        consolidate_activations_and_texts(run_dir, datasets, output_dir)

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
