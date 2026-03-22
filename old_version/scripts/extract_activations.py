#!/usr/bin/env python3
"""
Extract hidden state activations from Qwen2-1.5B base and instruct models.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import json
from typing import Dict, List, Tuple
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "/data/user_data/anshulk/cultural-alignment-study/sanskriti_data/data/generated_sentences_12k_batch.csv"
OUTPUT_DIR = Path("/data/user_data/anshulk/cultural-alignment-study/activations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
BASE_MODEL_PATH = "Qwen/Qwen2-1.5B"
INSTRUCT_MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"

# Layer configuration
TARGET_LAYERS = [8, 16, 24, 28]
TOTAL_LAYERS = 28

# GPU configuration
BASE_GPU = 0
INSTRUCT_GPU = 1

# Batch size
BATCH_SIZE = 512
MAX_LENGTH = 256


def load_sentences():
    """Load sentences from CSV."""
    print(f"Reading from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"Total rows: {len(df)}")
    
    # Verify required columns
    required_cols = ['sentence_1', 'sentence_2', 'sentence_3', 'group_type', 'state', 'attribute']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Collect all sentences with metadata
    sentences_data = []
    
    for idx, row in df.iterrows():
        for sent_num in [1, 2, 3]:
            sent_col = f'sentence_{sent_num}'
            if pd.notna(row[sent_col]) and row[sent_col].strip():
                sentences_data.append({
                    'row_id': idx,
                    'sentence_num': sent_num,
                    'sentence': row[sent_col],
                    'group_type': row['group_type'],
                    'state': row['state'],
                    'attribute': row['attribute'],
                    'base_correct': row.get('base_correct', None),
                    'instruct_correct': row.get('instruct_correct', None)
                })
    
    print(f"Total sentences extracted: {len(sentences_data)}")
    
    return sentences_data


def extract_layer_activations(
    model, 
    tokenizer, 
    sentences: List[str], 
    layer_idx: int,
    device: int,
    batch_size: int,
    model_name: str
) -> np.ndarray:
    """
    Extract mean-pooled activations from a single layer.
    
    Returns:
        numpy array of shape (n_sentences, hidden_size)
    """
    n_sentences = len(sentences)
    hidden_size = model.config.hidden_size
    
    # Initialize storage
    activations = np.zeros((n_sentences, hidden_size), dtype=np.float32)
    
    # Process in batches
    n_batches = (n_sentences + batch_size - 1) // batch_size
    
    print(f"[{model_name} GPU:{device}] Processing layer {layer_idx}: {n_batches} batches")
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_sentences)
            batch_sentences = sentences[start_idx:end_idx]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(f"cuda:{device}")
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract hidden states for target layer
            layer_hidden_states = outputs.hidden_states[layer_idx]
            
            # Mean pool across sequence dimension
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = layer_hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            mean_hidden = sum_hidden / sum_mask
            
            # Store in numpy array
            activations[start_idx:end_idx] = mean_hidden.cpu().numpy()
            
            # Clear GPU cache
            del inputs, outputs
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Progress update
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
                print(f"[{model_name} GPU:{device}] Layer {layer_idx}: {batch_idx + 1}/{n_batches} batches")
    
    return activations


def process_model(model_type: str, model_path: str, device: int, sentences_data: List[Dict]):
    """
    Process one model on one GPU - extracts all layers sequentially.
    This function runs in a separate process.
    """
    print(f"\n[{model_type.upper()}] Starting on GPU {device}")
    
    # Set GPU memory limit
    torch.cuda.set_per_process_memory_fraction(0.70, device)
    
    # Load model
    print(f"[{model_type.upper()}] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=f"cuda:{device}",
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()
    
    print(f"[{model_type.upper()}] Model loaded on GPU {device}")
    print(f"[{model_type.upper()}] Architecture: {model.config.num_hidden_layers} layers, hidden_size: {model.config.hidden_size}")
    
    # Extract sentences
    sentences = [item['sentence'] for item in sentences_data]
    
    print(f"[{model_type.upper()}] Extracting activations for {len(sentences)} sentences")
    print(f"[{model_type.upper()}] Target layers: {TARGET_LAYERS}")
    
    # Extract each layer sequentially
    activations = {}
    
    for layer_idx in TARGET_LAYERS:
        print(f"\n[{model_type.upper()}] Extracting layer {layer_idx}...")
        
        activations[layer_idx] = extract_layer_activations(
            model, tokenizer, sentences, layer_idx, device, BATCH_SIZE, model_type.upper()
        )
        
        print(f"[{model_type.upper()}] Layer {layer_idx} complete: shape {activations[layer_idx].shape}")
        
        # Save immediately after extraction
        output_file = OUTPUT_DIR / f"{model_type}_layer{layer_idx}_activations.npy"
        np.save(output_file, activations[layer_idx])
        print(f"[{model_type.upper()}] Saved layer {layer_idx} to {output_file}")
    
    # Save metadata
    metadata_file = OUTPUT_DIR / f"{model_type}_metadata.json"
    metadata = {
        'model_type': model_type,
        'model_path': model_path,
        'n_sentences': len(sentences_data),
        'layers': TARGET_LAYERS,
        'hidden_size': model.config.hidden_size,
        'batch_size': BATCH_SIZE,
        'max_length': MAX_LENGTH,
        'device': device,
        'sentences_metadata': sentences_data
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[{model_type.upper()}] Saved metadata to {metadata_file}")
    
    # Memory report
    print(f"\n[{model_type.upper()}] GPU {device} memory usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
    print(f"[{model_type.upper()}] COMPLETE")


def create_master_index():
    """Create a master index CSV mapping sentences to activation indices."""
    
    print(f"\n{'='*80}")
    print("CREATING MASTER INDEX")
    print(f"{'='*80}")
    
    # Load base metadata
    base_metadata_file = OUTPUT_DIR / "base_metadata.json"
    with open(base_metadata_file, 'r') as f:
        base_metadata = json.load(f)
    
    sentences_data = base_metadata['sentences_metadata']
    
    # Create index dataframe
    index_df = pd.DataFrame(sentences_data)
    index_df['activation_idx'] = range(len(index_df))
    
    # Save index
    index_file = OUTPUT_DIR / "activation_index.csv"
    index_df.to_csv(index_file, index=False)
    print(f"✓ Saved master index to {index_file}")
    print(f"  Total entries: {len(index_df)}")
    
    return index_df


def verify_activations():
    """Verify that saved activations match expected dimensions."""
    
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")
    
    for model_type in ['base', 'instruct']:
        print(f"\n{model_type.upper()} model:")
        for layer_idx in TARGET_LAYERS:
            file_path = OUTPUT_DIR / f"{model_type}_layer{layer_idx}_activations.npy"
            activations = np.load(file_path)
            print(f"  Layer {layer_idx}: {activations.shape} - {activations.dtype}")
            print(f"    Mean: {activations.mean():.4f}, Std: {activations.std():.4f}")


def main():
    """Main execution with parallel processing."""
    
    print("="*80)
    print("PARALLEL ACTIVATION EXTRACTION FOR PROBING ANALYSIS")
    print("Qwen2-1.5B Base vs Instruct - Dual GPU Execution")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models: Qwen2-1.5B (base & instruct)")
    print(f"  Layers: {TARGET_LAYERS} (out of {TOTAL_LAYERS} total)")
    print(f"  GPUs: {BASE_GPU} (base), {INSTRUCT_GPU} (instruct)")
    print(f"  Execution: PARALLEL - both models run simultaneously")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max sequence length: {MAX_LENGTH}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Load sentences (shared by both processes)
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    sentences_data = load_sentences()
    
    # Show group distribution
    group_counts = {}
    for item in sentences_data:
        group = item['group_type']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print(f"\nGroup distribution:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count} sentences")
    
    # Create parallel processes for each model
    print(f"\n{'='*80}")
    print("STARTING PARALLEL EXTRACTION")
    print(f"{'='*80}")
    
    # Start base model process
    base_process = mp.Process(
        target=process_model,
        args=("base", BASE_MODEL_PATH, BASE_GPU, sentences_data)
    )
    
    # Start instruct model process
    instruct_process = mp.Process(
        target=process_model,
        args=("instruct", INSTRUCT_MODEL_PATH, INSTRUCT_GPU, sentences_data)
    )
    
    # Launch both processes
    base_process.start()
    instruct_process.start()
    
    print("\n✓ Both processes started - running in parallel")
    print("  Base model: GPU 0")
    print("  Instruct model: GPU 1")
    
    # Wait for both to complete
    print("\nWaiting for processes to complete...")
    base_process.join()
    print("✓ Base model process completed")
    
    instruct_process.join()
    print("✓ Instruct model process completed")
    
    # Create master index
    index_df = create_master_index()
    
    # Verify saved activations
    verify_activations()
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print(f"  - base_layer{{8,16,24, 28}}_activations.npy")
    print(f"  - instruct_layer{{8,16,24, 28}}_activations.npy")
    print(f"  - base_metadata.json")
    print(f"  - instruct_metadata.json")
    print(f"  - activation_index.csv")
    print(f"\nTotal sentences processed: {len(sentences_data)}")
    print(f"Activation shape per layer: ({len(sentences_data)}, 1536)")


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()