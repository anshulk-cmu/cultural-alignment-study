"""
Phase 1: Sequential Activation Extraction
Base model → cleanup → Chat model → cleanup → Delta computation
"""

import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import torch
import gc
import time
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger('phase1_extraction', f'phase1_extraction_{timestamp}.log')


def log_gpu_memory():
    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        logger.info(f"GPU {i}: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")


def cleanup_gpu():
    logger.info("Cleaning GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_gpu_memory()


def load_model(model_name: str):
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Cache: {MODEL_CACHE} | Device: {DEVICE} | FP16: {USE_FP16}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)
    model.eval()

    logger.info(f"Model loaded: {len(model.model.layers)} layers | {next(model.parameters()).dtype}")
    log_gpu_memory()

    return model


def extract_activations_for_model(model_name: str, model_type: str,
                                   dataloaders: dict, run_output_dir: Path):
    logger.info("="*80)
    logger.info(f"STAGE: {model_type.upper()} MODEL EXTRACTION")
    logger.info("="*80)

    model = load_model(model_name)

    extractor = MemoryEfficientActivationExtractor(
        model, TARGET_LAYERS, device=None, use_fp16=USE_FP16
    )

    for dataset_name, dataloader in dataloaders.items():
        logger.info(f"\nDataset: {dataset_name} | Batches: {len(dataloader)}")

        saver = ActivationSaver(
            save_dir=run_output_dir,
            model_type=model_type,
            dataset_name=dataset_name
        )

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

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}")

                if (batch_idx + 1) % SAVE_EVERY_N_BATCHES == 0:
                    saver.save_chunk()
                    log_gpu_memory()

                if (batch_idx + 1) % EMPTY_CACHE_EVERY_N == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                continue

        saver.finalize()
        logger.info(f"  ✓ {dataset_name} complete")

    extractor.remove_hooks()
    del model, extractor
    cleanup_gpu()

    logger.info(f"✓ {model_type.upper()} extraction complete\n")


def compute_deltas(run_output_dir: Path):
    logger.info("="*80)
    logger.info("STAGE: DELTA COMPUTATION (chat - base)")
    logger.info("="*80)

    dataset_dirs = [d for d in run_output_dir.iterdir() if d.is_dir()]

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        logger.info(f"\nDataset: {dataset_name}")

        base_path = dataset_dir / "base"
        chat_path = dataset_dir / "chat"
        delta_path = dataset_dir / "delta"
        delta_path.mkdir(exist_ok=True)

        if not base_path.exists() or not chat_path.exists():
            logger.warning(f"  Missing base or chat, skipping")
            continue

        base_chunks = sorted(base_path.glob("chunk_*.npz"))
        chat_chunks = sorted(chat_path.glob("chunk_*.npz"))

        if len(base_chunks) != len(chat_chunks):
            logger.warning(f"  Chunk mismatch: base={len(base_chunks)} chat={len(chat_chunks)}")

        for idx, (base_chunk, chat_chunk) in enumerate(zip(base_chunks, chat_chunks)):
            if (idx + 1) % 5 == 0:
                logger.info(f"  Chunk {idx+1}/{len(base_chunks)}")

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

            np.savez_compressed(delta_path / base_chunk.name, **delta_data)

        logger.info(f"  ✓ {len(base_chunks)} delta chunks saved")

        summary = {
            'dataset_name': dataset_name,
            'num_chunks': len(base_chunks),
            'model_type': 'delta'
        }

        with open(delta_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    logger.info("✓ Delta computation complete\n")


def main():
    start_time = time.time()

    logger.info("="*80)
    logger.info("PHASE 1: SEQUENTIAL ACTIVATION EXTRACTION")
    logger.info("="*80)
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    logger.info(f"GPUs: {NUM_GPUS}")
    for i in range(NUM_GPUS):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info(f"\nConfig:")
    logger.info(f"  Layers: {TARGET_LAYERS}")
    logger.info(f"  Batch size: {BATCH_SIZE_PER_GPU}")
    logger.info(f"  FP16: {USE_FP16}")
    logger.info(f"  Save interval: {SAVE_EVERY_N_BATCHES} batches")
    logger.info(f"  Cache clear: {EMPTY_CACHE_EVERY_N} batches")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = ACTIVATION_ROOT / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput: {run_output_dir}")

    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["base"],
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded | Vocab size: {len(tokenizer)}")

    logger.info("\nLoading datasets...")
    dataloaders = load_all_datasets(
        tokenizer,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS
    )
    if not dataloaders:
        raise RuntimeError("No datasets loaded!")
    logger.info(f"Loaded {len(dataloaders)} datasets")

    extract_activations_for_model(MODELS["base"], "base", dataloaders, run_output_dir)
    logger.info("Waiting 10s before next stage...")
    time.sleep(10)

    extract_activations_for_model(MODELS["chat"], "chat", dataloaders, run_output_dir)
    logger.info("Waiting 10s before delta computation...")
    time.sleep(10)

    compute_deltas(run_output_dir)

    elapsed = time.time() - start_time
    logger.info("="*80)
    logger.info("✓ PHASE 1 COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {run_output_dir}")
    logger.info(f"Duration: {elapsed/60:.1f} minutes")
    logger.info("\nNext: Phase 2 - SAE Training")


if __name__ == "__main__":
    main()
