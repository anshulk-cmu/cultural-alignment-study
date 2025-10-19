"""
Phase 1: Memory-Optimized Activation Extraction
Extracts activations from Qwen base and chat models at layers 6, 12, 18
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import gc
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import (
    MODELS, TARGET_LAYERS, DEVICE, 
    ACTIVATION_ROOT, MODEL_CACHE,
    BATCH_SIZE_PER_GPU, NUM_WORKERS, USE_FP16,
    EMPTY_CACHE_EVERY_N, SAVE_EVERY_N_BATCHES,
    NUM_GPUS
)
from utils.data_loader import load_all_datasets
from utils.activation_extractor import (
    MemoryEfficientActivationExtractor,
    ActivationSaver
)


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def load_model_and_tokenizer(model_name: str):
    """Load model with memory-optimized settings"""
    
    print(f"\nLoading model: {model_name}")
    print(f"  Cache dir: {MODEL_CACHE}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map="auto" if NUM_GPUS > 1 else DEVICE,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    print(f"  ✓ Model loaded")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"  Num layers: {len(model.model.layers)}")
    print_gpu_memory()
    
    return model, tokenizer


def extract_activations_for_model(
    model_name: str,
    model_type: str,
    dataloaders: dict,
    run_output_dir: Path
):
    """Extract activations for one model across all datasets"""
    
    print("\n" + "="*80)
    print(f"EXTRACTING ACTIVATIONS: {model_type.upper()} MODEL")
    print("="*80)
    
    # Load model
    model, _ = load_model_and_tokenizer(model_name)
    
    # Create extractor
    extractor = MemoryEfficientActivationExtractor(
        model,
        TARGET_LAYERS,
        device=DEVICE,
        use_fp16=USE_FP16
    )
    
    # Process each dataset
    for dataset_name, dataloader in dataloaders.items():
        print(f"\n{'─'*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'─'*80}")
        
        # Create saver
        saver = ActivationSaver(
            save_dir=run_output_dir,
            model_type=model_type,
            dataset_name=dataset_name
        )
        
        # Process batches
        batch_counter = 0
        
        for batch_idx, batch in enumerate(dataloader):
            
            # Extract activations
            activations = extractor.extract_batch(
                batch['input_ids'],
                batch['attention_mask']
            )
            
            # Add to saver
            saver.add_batch(
                activations,
                batch['texts'],
                batch['languages'],
                batch['categories'],
                batch['indices']
            )
            
            batch_counter += 1
            
            # Progress update
            if batch_counter % 10 == 0:
                print(f"  Processed {batch_counter}/{len(dataloader)} batches", end='\r')
            
            # Save intermediate results
            if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                saver.save_chunk()
                print_gpu_memory()
            
            # Clear GPU cache periodically
            if batch_counter % EMPTY_CACHE_EVERY_N == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"  Processed {batch_counter}/{len(dataloader)} batches - Complete")
        
        # Finalize dataset
        saver.finalize()
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Remove hooks and clear model
    extractor.remove_hooks()
    del model
    del extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✓ Completed {model_type} model extraction")


def compute_deltas(run_output_dir: Path):
    """Compute delta activations (chat - base)"""
    import numpy as np
    import json
    
    print("\nComputing delta activations...")
    
    # Get list of datasets
    base_dir = run_output_dir
    dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        
        print(f"\n  Processing: {dataset_name}")
        
        base_path = dataset_dir / "base"
        chat_path = dataset_dir / "chat"
        delta_path = dataset_dir / "delta"
        delta_path.mkdir(exist_ok=True)
        
        if not base_path.exists() or not chat_path.exists():
            print(f"    ⚠️  Skipping (missing base or chat)")
            continue
        
        # Get chunk files
        base_chunks = sorted(base_path.glob("chunk_*.npz"))
        chat_chunks = sorted(chat_path.glob("chunk_*.npz"))
        
        if len(base_chunks) != len(chat_chunks):
            print(f"    ⚠️  Warning: Mismatch in chunk count")
        
        # Process each chunk
        for idx, (base_chunk, chat_chunk) in enumerate(zip(base_chunks, chat_chunks)):
            if idx % 5 == 0:
                print(f"    Chunk {idx+1}/{len(base_chunks)}", end='\r')
            
            # Load chunks
            base_data = np.load(base_chunk, allow_pickle=True)
            chat_data = np.load(chat_chunk, allow_pickle=True)
            
            # Compute deltas
            delta_data = {
                'activations': {},
                'metadata': dict(base_data['metadata'].item())
            }
            
            for layer in TARGET_LAYERS:
                base_acts = base_data['activations'].item()[layer]
                chat_acts = chat_data['activations'].item()[layer]
                delta_data['activations'][layer] = chat_acts - base_acts
            
            # Save
            delta_file = delta_path / base_chunk.name
            np.savez_compressed(delta_file, **delta_data)
        
        print(f"    Chunk {len(base_chunks)}/{len(base_chunks)} - Complete")
        
        # Create summary
        summary = {
            'dataset_name': dataset_name,
            'num_chunks': len(base_chunks),
            'model_type': 'delta'
        }
        
        with open(delta_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"    ✓ Saved {len(base_chunks)} delta chunks")


def main():
    """Main execution"""
    
    print("="*80)
    print("PHASE 1: MEMORY-OPTIMIZED ACTIVATION EXTRACTION")
    print("="*80)
    
    # Print system info
    print(f"\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {NUM_GPUS}")
    
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"\nConfiguration:")
    print(f"  Target layers: {TARGET_LAYERS}")
    print(f"  Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    print(f"  Total batch size: {BATCH_SIZE_PER_GPU * NUM_GPUS if NUM_GPUS > 1 else BATCH_SIZE_PER_GPU}")
    print(f"  Mixed precision (FP16): {USE_FP16}")
    print(f"  Save every N batches: {SAVE_EVERY_N_BATCHES}")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = ACTIVATION_ROOT / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {run_output_dir}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["base"],
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  ✓ Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Supports Hindi: Yes (Qwen is multilingual)")
    
    # Load all datasets
    dataloaders = load_all_datasets(
        tokenizer,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS
    )
    
    if not dataloaders:
        print("\n❌ ERROR: No datasets loaded!")
        return
    
    # Extract from base model
    extract_activations_for_model(
        MODELS["base"],
        "base",
        dataloaders,
        run_output_dir
    )
    
    # Extract from chat model
    extract_activations_for_model(
        MODELS["chat"],
        "chat",
        dataloaders,
        run_output_dir
    )
    
    # Compute delta activations
    print("\n" + "="*80)
    print("COMPUTING DELTA ACTIVATIONS (chat - base)")
    print("="*80)
    compute_deltas(run_output_dir)
    
    print("\n" + "="*80)
    print("✓ PHASE 1 COMPLETE!")
    print("="*80)
    print(f"\nAll activations saved to: {run_output_dir}")
    print("\nNext steps:")
    print("  - Review activation statistics")
    print("  - Proceed to Phase 2: SAE Training")


if __name__ == "__main__":
    main()
