"""
Phase 1: Sequential Activation Extraction
Extracts activations from base model, then chat model, then computes deltas
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import torch
import gc
import time
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Reproducibility seed
SEED = 42


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def log_gpu_memory(logger, device):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def load_model_and_tokenizer(model_name: str, device: str, logger):
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Target device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizer loaded successfully")

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

    logger.info(f"Model loaded on {device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    log_gpu_memory(logger, device)

    return model, tokenizer


def cleanup_gpu(logger, device):
    logger.info("Starting GPU cleanup")
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu_memory(logger, device)
    logger.info("GPU cleanup complete")


def extract_activations_for_model(model_name: str, model_type: str, run_output_dir: Path, logger, seed: int):
    device = "cuda:0"
    torch.cuda.set_device(0)

    logger.info("="*80)
    logger.info(f"EXTRACTING ACTIVATIONS: {model_type.upper()} MODEL")
    logger.info("="*80)

    model, tokenizer = load_model_and_tokenizer(model_name, device, logger)

    logger.info("Initializing activation extractor")
    extractor = MemoryEfficientActivationExtractor(
        model,
        TARGET_LAYERS,
        device=device,
        use_fp16=USE_FP16
    )

    logger.info("Loading datasets")
    dataloaders = load_all_datasets(
        tokenizer,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS,
        seed=seed
    )
    logger.info(f"Loaded {len(dataloaders)} datasets")

    for dataset_name, dataloader in dataloaders.items():
        logger.info(f"\n{'─'*80}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Total batches: {len(dataloader)}")
        logger.info(f"{'─'*80}")

        saver = ActivationSaver(
            save_dir=run_output_dir,
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
                    logger.info(f"Processed {batch_counter}/{len(dataloader)} batches")

                if batch_counter % SAVE_EVERY_N_BATCHES == 0:
                    saver.save_chunk()
                    log_gpu_memory(logger, device)

                if batch_counter % EMPTY_CACHE_EVERY_N == 0:
                    cleanup_gpu(logger, device)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

        logger.info(f"Dataset complete: {batch_counter}/{len(dataloader)} batches processed")
        saver.finalize()
        cleanup_gpu(logger, device)

    logger.info("Removing hooks and cleaning up model")
    extractor.remove_hooks()
    del model, tokenizer, extractor, dataloaders
    cleanup_gpu(logger, device)

    logger.info(f"✓ Completed {model_type} model extraction")


def compute_deltas(run_output_dir: Path, logger):
    logger.info("\n" + "="*80)
    logger.info("COMPUTING DELTA ACTIVATIONS (chat - base)")
    logger.info("="*80)

    dataset_dirs = [d for d in run_output_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dataset_dirs)} datasets to process")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        logger.info(f"\nProcessing dataset: {dataset_name}")

        base_path = dataset_dir / "base"
        chat_path = dataset_dir / "chat"
        delta_path = dataset_dir / "delta"
        delta_path.mkdir(exist_ok=True)

        if not base_path.exists() or not chat_path.exists():
            logger.warning(f"Skipping {dataset_name} - missing base or chat directory")
            continue

        base_chunks = sorted(base_path.glob("chunk_*.npz"))
        chat_chunks = sorted(chat_path.glob("chunk_*.npz"))

        logger.info(f"Base chunks: {len(base_chunks)}, Chat chunks: {len(chat_chunks)}")

        if len(base_chunks) != len(chat_chunks):
            logger.warning(f"Chunk count mismatch: base={len(base_chunks)}, chat={len(chat_chunks)}")

        for idx, (base_chunk, chat_chunk) in enumerate(zip(base_chunks, chat_chunks)):
            if idx % 5 == 0:
                logger.info(f"Processing chunk {idx+1}/{len(base_chunks)}")

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

        logger.info(f"✓ Saved {len(base_chunks)} delta chunks for {dataset_name}")

        summary = {
            'dataset_name': dataset_name,
            'num_chunks': len(base_chunks),
            'model_type': 'delta'
        }

        with open(delta_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    # Set seed for reproducibility
    set_seed(SEED)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger('phase1_extraction', f'phase1_extraction_{timestamp}.log')

    logger.info("="*80)
    logger.info("PHASE 1: SEQUENTIAL ACTIVATION EXTRACTION")
    logger.info("="*80)
    
    logger.info(f"\nReproducibility:")
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Deterministic operations: Enabled")
    logger.info(f"CuDNN deterministic: True")
    logger.info(f"CuDNN benchmark: False")

    logger.info(f"\nSystem Information:")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU 0 Total Memory: {total_memory:.2f}GB")

    logger.info(f"\nConfiguration:")
    logger.info(f"Models - Base: {MODELS['base']}")
    logger.info(f"Models - Chat: {MODELS['chat']}")
    logger.info(f"Target layers: {TARGET_LAYERS}")
    logger.info(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    logger.info(f"Mixed precision (FP16): {USE_FP16}")
    logger.info(f"Save every N batches: {SAVE_EVERY_N_BATCHES}")
    logger.info(f"Empty cache every N batches: {EMPTY_CACHE_EVERY_N}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = ACTIVATION_ROOT / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {run_output_dir}")

    logger.info("\n" + "="*80)
    logger.info("STAGE 1: BASE MODEL EXTRACTION")
    logger.info("="*80)
    start_time = datetime.now()
    extract_activations_for_model(MODELS["base"], "base", run_output_dir, logger, SEED)
    end_time = datetime.now()
    logger.info(f"Base model extraction completed in {end_time - start_time}")

    logger.info("\nWaiting 10 seconds before next stage...")
    time.sleep(10)

    logger.info("\n" + "="*80)
    logger.info("STAGE 2: CHAT MODEL EXTRACTION")
    logger.info("="*80)
    start_time = datetime.now()
    extract_activations_for_model(MODELS["chat"], "chat", run_output_dir, logger, SEED)
    end_time = datetime.now()
    logger.info(f"Chat model extraction completed in {end_time - start_time}")

    logger.info("\nWaiting 10 seconds before delta computation...")
    time.sleep(10)

    logger.info("\n" + "="*80)
    logger.info("STAGE 3: DELTA COMPUTATION")
    logger.info("="*80)
    start_time = datetime.now()
    compute_deltas(run_output_dir, logger)
    end_time = datetime.now()
    logger.info(f"Delta computation completed in {end_time - start_time}")

    logger.info("\n" + "="*80)
    logger.info("✓ PHASE 1 COMPLETE!")
    logger.info("="*80)
    logger.info(f"All activations saved to: {run_output_dir}")


if __name__ == "__main__":
    main()
