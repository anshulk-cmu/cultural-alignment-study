"""
Phase 1: Memory-Optimized Activation Extraction
Extracts sentence-level activations from Qwen base and chat models
"""

import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import torch
import gc
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from configs.config import (
    MODELS, TARGET_LAYERS, DEVICE, 
    ACTIVATION_ROOT, MODEL_CACHE,
    BATCH_SIZE_PER_GPU, NUM_WORKERS, USE_FP16,
    EMPTY_CACHE_EVERY_N, SAVE_EVERY_N_BATCHES,
    NUM_GPUS, setup_logger
)
from utils.data_loader import load_all_datasets
from utils.activation_extractor import (
    MemoryEfficientActivationExtractor,
    ActivationSaver
)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(
    'phase1_extraction', 
    f'phase1_extraction_{timestamp}.log'
)


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def load_model_and_tokenizer(model_name: str, use_single_gpu: bool = True):
    """
    Load model with memory-optimized settings
    
    Args:
        model_name: Model name or path
        use_single_gpu: If True, load on single GPU (recommended for stability)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"\nLoading model: {model_name}")
        logger.info(f"  Cache dir: {MODEL_CACHE}")
        logger.info(f"  Single GPU mode: {use_single_gpu}")
        
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
        if use_single_gpu:
            # Single GPU - more stable, recommended
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE,
                trust_remote_code=True,
                torch_dtype=torch.float16 if USE_FP16 else torch.float32,
                device_map=None,  # Don't auto-distribute
                low_cpu_mem_usage=True
            )
            model = model.to(DEVICE)
        else:
            # Multi-GPU distribution
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE,
                trust_remote_code=True,
                torch_dtype=torch.float16 if USE_FP16 else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        model.eval()
        
        logger.info(f"  ✓ Model loaded")
        logger.info(f"  Device: {next(model.parameters()).device}")
        logger.info(f"  Dtype: {next(model.parameters()).dtype}")
        logger.info(f"  Num layers: {len(model.model.layers)}")
        print_gpu_memory()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def extract_activations_for_model(
    model_name: str,
    model_type: str,
    dataloaders: dict,
    run_output_dir: Path
):
    """
    Extract activations for one model across all datasets
    
    Args:
        model_name: Model identifier
        model_type: 'base' or 'chat'
        dataloaders: Dict of dataset_name -> DataLoader
        run_output_dir: Output directory for this run
    """
    try:
        logger.info("\n" + "="*80)
        logger.info(f"EXTRACTING ACTIVATIONS: {model_type.upper()} MODEL")
        logger.info("="*80)
        
        # Load model (use single GPU for stability)
        model, _ = load_model_and_tokenizer(model_name, use_single_gpu=True)
        
        # Create extractor
        extractor = MemoryEfficientActivationExtractor(
            model,
            TARGET_LAYERS,
            device=None,  # Infer from model
            use_fp16=USE_FP16
        )
        
        # Process each dataset
        for dataset_name, dataloader in dataloaders.items():
            logger.info(f"\n{'─'*80}")
            logger.info(f"Processing: {dataset_name}")
            logger.info(f"{'─'*80}")
            
            # Create saver
            saver = ActivationSaver(
                save_dir=run_output_dir,
                model_type=model_type,
                dataset_name=dataset_name
            )
            
            # Process batches
            batch_counter = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
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
                        logger.info(f"  Processed {batch_counter}/{len(dataloader)} batches")
                    
                    # Save intermediate results
                    if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                        saver.save_chunk()
                        print_gpu_memory()
                    
                    # Clear GPU cache periodically
                    if batch_counter % EMPTY_CACHE_EVERY_N == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
            
            logger.info(f"  Processed {batch_counter}/{len(dataloader)} batches - Complete")
            
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
        
        logger.info(f"\n✓ Completed {model_type} model extraction")
        
    except Exception as e:
        logger.error(f"Failed to extract activations for {model_type}: {e}")
        raise


def compute_deltas(run_output_dir: Path):
    """
    Compute delta activations (chat - base)
    
    Args:
        run_output_dir: Directory containing base and chat activations
    """
    import numpy as np
    import json
    
    try:
        logger.info("\n" + "="*80)
        logger.info("COMPUTING DELTA ACTIVATIONS (chat - base)")
        logger.info("="*80)
        
        # Get list of datasets
        base_dir = run_output_dir
        dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            
            logger.info(f"\n  Processing: {dataset_name}")
            
            base_path = dataset_dir / "base"
            chat_path = dataset_dir / "chat"
            delta_path = dataset_dir / "delta"
            delta_path.mkdir(exist_ok=True)
            
            if not base_path.exists() or not chat_path.exists():
                logger.warning(f"    Skipping (missing base or chat)")
                continue
            
            # Get chunk files
            base_chunks = sorted(base_path.glob("chunk_*.npz"))
            chat_chunks = sorted(chat_path.glob("chunk_*.npz"))
            
            if len(base_chunks) != len(chat_chunks):
                logger.warning(f"    Mismatch in chunk count: base={len(base_chunks)}, chat={len(chat_chunks)}")
            
            # Process each chunk
            for idx, (base_chunk, chat_chunk) in enumerate(zip(base_chunks, chat_chunks)):
                if idx % 5 == 0:
                    logger.info(f"    Chunk {idx+1}/{len(base_chunks)}")
                
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
            
            logger.info(f"    ✓ Saved {len(base_chunks)} delta chunks")
            
            # Create summary
            summary = {
                'dataset_name': dataset_name,
                'num_chunks': len(base_chunks),
                'model_type': 'delta'
            }
            
            with open(delta_path / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
    except Exception as e:
        logger.error(f"Failed to compute deltas: {e}")
        raise


def main():
    """Main execution"""
    
    try:
        logger.info("="*80)
        logger.info("PHASE 1: MEMORY-OPTIMIZED ACTIVATION EXTRACTION")
        logger.info("="*80)
        
        # Print system info
        logger.info(f"\nSystem Information:")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  Number of GPUs: {NUM_GPUS}")
        
        for i in range(NUM_GPUS):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Target layers: {TARGET_LAYERS}")
        logger.info(f"  Batch size per GPU: {BATCH_SIZE_PER_GPU}")
        logger.info(f"  Mixed precision (FP16): {USE_FP16}")
        logger.info(f"  Save every N batches: {SAVE_EVERY_N_BATCHES}")
        
        # Create run directory
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = ACTIVATION_ROOT / f"run_{run_timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nOutput directory: {run_output_dir}")
        
        # Load tokenizer
        logger.info("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODELS["base"],
            cache_dir=MODEL_CACHE,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("  ✓ Tokenizer loaded")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        
        # Load all datasets
        dataloaders = load_all_datasets(
            tokenizer,
            batch_size=BATCH_SIZE_PER_GPU,
            num_workers=NUM_WORKERS
        )
        
        if not dataloaders:
            raise RuntimeError("No datasets loaded!")
        
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
        compute_deltas(run_output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("✓ PHASE 1 COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nAll activations saved to: {run_output_dir}")
        logger.info("\nNext steps:")
        logger.info("  - Review activation statistics")
        logger.info("  - Proceed to Phase 2: SAE Training")
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        raise


if __name__ == "__main__":
    main()
