"""
Phase 1: Activation Extraction
Extracts sentence-level activations from Qwen base and chat models
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import torch
import gc
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp

from configs.config import (
    MODELS, TARGET_LAYERS, 
    ACTIVATION_ROOT, MODEL_CACHE,
    BATCH_SIZE_PER_GPU, NUM_WORKERS, USE_FP16,
    EMPTY_CACHE_EVERY_N, SAVE_EVERY_N_BATCHES,
    setup_logger
)
from utils.data_loader import load_all_datasets
from utils.activation_extractor import (
    MemoryEfficientActivationExtractor,
    ActivationSaver
)


def print_gpu_memory(gpu_id):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"    GPU {gpu_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def load_model_and_tokenizer(model_name: str, device: str):
    print(f"\nLoading model: {model_name}")
    print(f"  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded on {device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def extract_activations_for_model(
    model_name: str,
    model_type: str,
    run_output_dir: str,
    gpu_id: int
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print("="*80)
    print(f"EXTRACTING ACTIVATIONS: {model_type.upper()} MODEL (GPU {gpu_id})")
    print("="*80)
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    extractor = MemoryEfficientActivationExtractor(
        model,
        TARGET_LAYERS,
        device=device,
        use_fp16=USE_FP16
    )
    
    dataloaders = load_all_datasets(
        tokenizer,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS
    )
    
    for dataset_name, dataloader in dataloaders.items():
        print(f"\n{'─'*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'─'*80}")
        
        saver = ActivationSaver(
            save_dir=Path(run_output_dir),
            model_type=model_type,
            dataset_name=dataset_name
        )
        
        batch_counter = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                activations = extractor.extract_batch(
                    batch['input_ids'],
                    batch['attention_mask']
                )
                
                saver.add_batch(
                    activations,
                    batch['texts'],
                    batch['languages'],
                    batch['categories'],
                    batch['indices']
                )
                
                batch_counter += 1
                
                if batch_counter % 10 == 0:
                    print(f"  Processed {batch_counter}/{len(dataloader)} batches")
                
                if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                    saver.save_chunk()
                    print_gpu_memory(gpu_id)
                
                if batch_counter % EMPTY_CACHE_EVERY_N == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        print(f"  Processed {batch_counter}/{len(dataloader)} batches - Complete")
        saver.finalize()
        
        torch.cuda.empty_cache()
        gc.collect()
    
    extractor.remove_hooks()
    del model
    del extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✓ Completed {model_type} model extraction")


def compute_deltas(run_output_dir: Path):
    import numpy as np
    import json
    
    print("\n" + "="*80)
    print("COMPUTING DELTA ACTIVATIONS (chat - base)")
    print("="*80)
    
    dataset_dirs = [d for d in run_output_dir.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\n  Processing: {dataset_name}")
        
        base_path = dataset_dir / "base"
        chat_path = dataset_dir / "chat"
        delta_path = dataset_dir / "delta"
        delta_path.mkdir(exist_ok=True)
        
        if not base_path.exists() or not chat_path.exists():
            print(f"    Skipping (missing base or chat)")
            continue
        
        base_chunks = sorted(base_path.glob("chunk_*.npz"))
        chat_chunks = sorted(chat_path.glob("chunk_*.npz"))
        
        if len(base_chunks) != len(chat_chunks):
            print(f"    Mismatch in chunk count: base={len(base_chunks)}, chat={len(chat_chunks)}")
        
        for idx, (base_chunk, chat_chunk) in enumerate(zip(base_chunks, chat_chunks)):
            if idx % 5 == 0:
                print(f"    Chunk {idx+1}/{len(base_chunks)}")
            
            base_data = np.load(base_chunk, allow_pickle=True)
            chat_data = np.load(chat_chunk, allow_pickle=True)
            
            delta_data = {
                'activations': {},
                'metadata': dict(base_data['metadata'].item())
            }
            
            for layer in TARGET_LAYERS:
                base_acts = base_data['activations'].item()[layer]
                chat_acts = chat_data['activations'].item()[layer]
                delta_data['activations'][layer] = chat_acts - base_acts
            
            delta_file = delta_path / base_chunk.name
            np.savez_compressed(delta_file, **delta_data)
        
        print(f"    ✓ Saved {len(base_chunks)} delta chunks")
        
        summary = {
            'dataset_name': dataset_name,
            'num_chunks': len(base_chunks),
            'model_type': 'delta'
        }
        
        with open(delta_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger('phase1_extraction', f'phase1_extraction_{timestamp}.log')
    
    logger.info("="*80)
    logger.info("PHASE 1: ACTIVATION EXTRACTION")
    logger.info("="*80)
    
    logger.info(f"\nSystem Information:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Target layers: {TARGET_LAYERS}")
    logger.info(f"  Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    logger.info(f"  Mixed precision (FP16): {USE_FP16}")
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = ACTIVATION_ROOT / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nOutput directory: {run_output_dir}")
    
    mp.set_start_method('spawn', force=True)
    
    p1 = mp.Process(target=extract_activations_for_model, 
                    args=(MODELS["base"], "base", str(run_output_dir), 0))
    p2 = mp.Process(target=extract_activations_for_model, 
                    args=(MODELS["chat"], "chat", str(run_output_dir), 1))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    compute_deltas(run_output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 1 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nAll activations saved to: {run_output_dir}")


if __name__ == "__main__":
    main()
