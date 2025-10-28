#!/usr/bin/env python3
"""
Phase 2.5: Parallel Qwen Labeling with 4x 48GB-class GPUs (e.g., L40S)
Using Qwen1.5-32B-Chat model
"""
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import subprocess
import os
import time
import gc
import random
import numpy as np
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

# GPU configuration - 4x GPUs
FREE_GPUS = [0, 1, 2, 3]

# REPRODUCIBILITY SEED
SEED = 42


def create_worker_script():
    """Create the worker script that will be run as a subprocess."""
    worker_code = '''#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import torch
import gc
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import SAE_OUTPUT_ROOT, setup_logger

# REPRODUCIBILITY SEED
SEED = 42


def set_seed(seed):
    """Set all random seeds for reproducibility."""
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
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def load_qwen():
    """Load Qwen1.5-32B-Chat with 8-bit quantization and memory optimization."""
    from transformers import BitsAndBytesConfig
    
    # Use 32B Model
    model_path = "/data/models/huggingface/qwen/Qwen1.5-32B-Chat"
    
    # Verify GPU
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"Worker initializing on GPU {device}: {device_name}")
        print(f"Total GPU memory: {total_memory:.1f} GB")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model with 8-bit quantization...")
    # 8-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},  # All on single GPU (fits in 48GB)
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True  # Prevent hanging
    )
    
    model.eval()  # Set to evaluation mode
    
    # Report memory usage
    alloc, reserved = get_gpu_memory_usage()
    print(f"Model loaded successfully")
    print(f"GPU Memory - Allocated: {alloc:.2f} GB, Reserved: {reserved:.2f} GB")
    
    return tokenizer, model


def create_labeling_prompt(feature_data):
    """Create a prompt for feature labeling with top examples."""
    examples_text = ""
    # Use 20 examples
    num_examples = min(20, len(feature_data['examples']))
    
    for i, ex in enumerate(feature_data['examples'][:num_examples], 1):
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        examples_text += f"{i}. {text} (activation: {ex['activation']:.3f})\\n\\n"
    
    prompt = f"""You are analyzing a sparse autoencoder feature from a language model trained on Indian English and Hindi text. Below are the top {num_examples} text examples where this feature activated most strongly.

EXAMPLES:
{examples_text}

TASK:
Analyze these examples and determine:
1. Do these examples form a COHERENT group with a shared pattern?
2. If YES: Provide a concise 1-2 sentence label describing the linguistic, cultural, or semantic pattern.
3. If NO: Output exactly "INCOHERENT"

Focus on:
- Cultural markers (Indian festivals, names, food, regional terms, honorifics)
- Linguistic patterns (Hindi words, code-switching, grammatical structures)
- Semantic themes (topics, sentiment, discourse patterns)

OUTPUT FORMAT:
If coherent: A clear, specific label (e.g., "Hindi honorifics and respectful address forms" or "References to Indian festivals and celebrations")
If incoherent: "INCOHERENT"

LABEL:"""
    return prompt


def generate_label(tokenizer, model, prompt, max_new_tokens=50):
    """Generate a label using Qwen model with GREEDY DECODING for reproducibility."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in linguistic analysis and cultural understanding."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # GREEDY DECODING - fully deterministic
            num_beams=1,  # No beam search
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up tensors
    del inputs, outputs, generated_ids
    
    return response.strip()


def save_checkpoint(results, output_file, gpu_id):
    """Save intermediate results as checkpoint."""
    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[GPU {gpu_id}] Checkpoint saved: {len(results)} results")


def load_checkpoint(output_file):
    """Load checkpoint if exists."""
    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--file_paths', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    file_paths = json.loads(args.file_paths)
    gpu_id = args.gpu_id
    output_file = Path(args.output_file)
    
    # SET SEED FOR REPRODUCIBILITY
    set_seed(SEED)
    
    logger = setup_logger(f'qwen_label_gpu{gpu_id}', f'phase2_5_qwen_gpu{gpu_id}.log')
    
    logger.info(f"[GPU {gpu_id}] Worker started with SEED={SEED}")
    logger.info(f"[GPU {gpu_id}] Deterministic mode: ENABLED")
    logger.info(f"[GPU {gpu_id}] Processing {len(file_paths)} files")
    logger.info(f"[GPU {gpu_id}] Output: {output_file}")
    
    # Check for checkpoint
    all_results = load_checkpoint(output_file)
    processed_features = set(r['feature_id'] for r in all_results)
    
    if all_results:
        logger.info(f"[GPU {gpu_id}] Resuming from checkpoint: {len(all_results)} features already processed")
    
    # Load model
    try:
        tokenizer, model = load_qwen()
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to load model: {e}")
        raise
    
    # Process each file in SORTED order for reproducibility
    for file_idx, file_path_str in enumerate(sorted(file_paths), 1):
        file_path = Path(file_path_str)
        logger.info(f"[GPU {gpu_id}] Processing file {file_idx}/{len(file_paths)}: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        # Only process top 400 features per file (SORTED by feature_id for consistency)
        features = sorted(features, key=lambda x: x.get('feature_id', ''))[:400]
        logger.info(f"[GPU {gpu_id}] Loaded {len(features)} features from {file_path.name}")
        
        # Process features with periodic cleanup
        for feat_idx, feature in enumerate(tqdm(features, desc=f"GPU {gpu_id} - {file_path.stem}"), 1):
            try:
                # Skip if already processed
                if feature['feature_id'] in processed_features:
                    continue
                
                prompt = create_labeling_prompt(feature)
                label = generate_label(tokenizer, model, prompt)
                
                is_coherent = 'INCOHERENT' not in label.upper()
                
                result = {
                    'feature_id': feature['feature_id'],
                    'sae_name': feature['sae_name'],
                    'feature_idx': feature['feature_idx'],
                    'num_examples': feature['num_examples'],
                    'label_qwen': label,
                    'is_coherent': is_coherent,
                    'processed_by_gpu': gpu_id,
                    'seed': SEED
                }
                
                # Copy over additional statistics
                for key in ['max_activation', 'mean_activation', 'sparsity']:
                    if key in feature:
                        result[key] = feature[key]
                
                all_results.append(result)
                processed_features.add(feature['feature_id'])
                
                # Periodic GPU cleanup (every 50 features)
                if feat_idx % 50 == 0:
                    clear_gpu_memory()
                    alloc, reserved = get_gpu_memory_usage()
                    logger.info(f"[GPU {gpu_id}] Progress: {feat_idx}/{len(features)} | Memory: {alloc:.1f}GB / {reserved:.1f}GB")
                
                # Checkpoint save (every 100 features)
                if feat_idx % 100 == 0:
                    save_checkpoint(all_results, output_file, gpu_id)
                
            except Exception as e:
                logger.error(f"[GPU {gpu_id}] Error labeling feature {feature['feature_id']}: {str(e)}")
                continue
        
        # Save checkpoint after each file
        save_checkpoint(all_results, output_file, gpu_id)
        
        # Aggressive cleanup between files
        clear_gpu_memory()
        logger.info(f"[GPU {gpu_id}] Completed file {file_idx}/{len(file_paths)}")
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[GPU {gpu_id}] ✓ Complete: {len(all_results)} labels saved to {output_file}")
    
    # Final memory report
    alloc, reserved = get_gpu_memory_usage()
    logger.info(f"[GPU {gpu_id}] Final GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB")


if __name__ == "__main__":
    main()
'''
    return worker_code


def main():
    # Set seed in main process
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Use 32B Model
    model_path = "/data/models/huggingface/qwen/Qwen1.5-32B-Chat"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_dir = SAE_OUTPUT_ROOT
    final_output_file = output_dir / "labels_qwen_initial.json"
    
    # Setup main logger
    logger = setup_logger('qwen_label_parallel', 'phase2_5_qwen_parallel.log')
    
    logger.info("=" * 80)
    logger.info("PARALLEL QWEN1.5-32B-CHAT FEATURE LABELING (REPRODUCIBLE)")
    logger.info(f"Using GPUs: {FREE_GPUS} (4x 48GB-class GPUs)")
    logger.info(f"SEED: {SEED}")
    logger.info(f"Generation: GREEDY DECODING (fully deterministic)")
    logger.info("=" * 80)
    logger.info(f"Input directory: {examples_dir}")
    logger.info(f"Output file: {final_output_file}")
    logger.info(f"Model path: {model_path}")
    
    # Verify model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please check the model path!")
        return
    
    logger.info(f"✓ Model path verified: {model_path}")
    
    # Get all example files and SORT for reproducibility
    examples_files = sorted(examples_dir.glob("*_examples.json"))
    
    if not examples_files:
        logger.error(f"No example files found in {examples_dir}")
        return
    
    logger.info(f"\nFound {len(examples_files)} example files to process")
    for f in examples_files:
        logger.info(f"  - {f.name}")
    
    # Split files between 4 GPUs (9 files → 2,2,3,2 distribution)
    files_per_gpu = len(examples_files) // len(FREE_GPUS)
    remainder = len(examples_files) % len(FREE_GPUS)
    
    file_splits = []
    start_idx = 0
    for i, gpu_id in enumerate(FREE_GPUS):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + files_per_gpu + extra
        file_splits.append((gpu_id, [str(f) for f in examples_files[start_idx:end_idx]]))
        start_idx = end_idx
    
    logger.info("\nWork distribution:")
    for gpu_id, files in file_splits:
        logger.info(f"  GPU {gpu_id}: {len(files)} files")
        for f in files:
            logger.info(f"    - {Path(f).name}")
    
    # Create worker script
    worker_script_path = Path("/tmp/qwen_worker_32b_8bit_reproducible.py")
    with open(worker_script_path, 'w') as f:
        f.write(create_worker_script())
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING PARALLEL PROCESSING")
    logger.info("=" * 80)
    logger.info("Features per file: ~400 (top features only)")
    logger.info("Estimated time: 6-10 hours")
    logger.info("Checkpoint saves: Every 100 features")
    logger.info("Memory cleanup: Every 50 features")
    logger.info("Generation mode: GREEDY (deterministic)")
    logger.info("")
    
    # Launch subprocesses with CUDA_VISIBLE_DEVICES set
    processes = []
    
    for gpu_id, file_list in file_splits:
        output_file = output_dir / f"labels_qwen_initial_gpu{gpu_id}.json"
        
        # Create environment with CUDA_VISIBLE_DEVICES set
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONHASHSEED'] = str(SEED)  # Ensure hash reproducibility
        
        # Prepare arguments
        cmd = [
            sys.executable,
            str(worker_script_path),
            '--gpu_id', str(gpu_id),
            '--file_paths', json.dumps(sorted(file_list)),  # Sort for consistency
            '--output_file', str(output_file)
        ]
        
        # Start subprocess
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        processes.append((gpu_id, proc))
        logger.info(f"  ✓ Started worker on GPU {gpu_id} (PID: {proc.pid})")
    
    logger.info("")
    
    # Monitor processes and stream output
    while any(proc.poll() is None for _, proc in processes):
        for gpu_id, proc in processes:
            if proc.poll() is None:  # Process still running
                # Read available output
                while True:
                    try:
                        line = proc.stdout.readline()
                        if line:
                            print(f"[GPU {gpu_id}] {line.rstrip()}")
                        else:
                            break
                    except:
                        break
        time.sleep(0.1)
    
    # Get final outputs
    for gpu_id, proc in processes:
        remaining = proc.stdout.read()
        if remaining:
            for line in remaining.split('\n'):
                if line.strip():
                    print(f"[GPU {gpu_id}] {line}")
    
    # Check return codes
    logger.info("\n" + "=" * 80)
    logger.info("WORKER STATUS")
    logger.info("=" * 80)
    all_success = True
    for gpu_id, proc in processes:
        if proc.returncode != 0:
            logger.error(f"✗ GPU {gpu_id} worker FAILED with code {proc.returncode}")
            all_success = False
        else:
            logger.info(f"✓ GPU {gpu_id} worker completed successfully")
    
    if not all_success:
        logger.error("\n⚠️  Some workers failed. Check individual GPU logs for details.")
        logger.error("Checkpoint files are saved - you can resume by rerunning this script.")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("MERGING RESULTS")
    logger.info("=" * 80)
    
    # Merge results from all GPUs in SORTED order
    all_results = []
    for gpu_id in sorted(FREE_GPUS):
        gpu_output_file = output_dir / f"labels_qwen_initial_gpu{gpu_id}.json"
        if gpu_output_file.exists():
            with open(gpu_output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
                logger.info(f"  GPU {gpu_id}: {len(results)} features")
        else:
            logger.warning(f"  GPU {gpu_id}: output file not found!")
    
    # Sort merged results by feature_id for consistency
    all_results = sorted(all_results, key=lambda x: x.get('feature_id', ''))
    
    # Save merged results
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Merged results saved to {final_output_file}")
    
    # Summary statistics
    if all_results:
        coherent = sum(1 for r in all_results if r['is_coherent'])
        coherent_pct = (coherent / len(all_results)) * 100
        
        logger.info("\n" + "=" * 80)
        logger.info("LABELING COMPLETE - FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"SEED: {SEED}")
        logger.info(f"Generation mode: GREEDY DECODING (deterministic)")
        logger.info(f"Total features labeled: {len(all_results)}")
        logger.info(f"Coherent features: {coherent} ({coherent_pct:.1f}%)")
        logger.info(f"Incoherent features: {len(all_results) - coherent} ({100 - coherent_pct:.1f}%)")
        
        # GPU distribution
        logger.info("\nProcessing distribution:")
        for gpu_id in sorted(FREE_GPUS):
            gpu_count = sum(1 for r in all_results if r.get('processed_by_gpu') == gpu_id)
            logger.info(f"  GPU {gpu_id}: {gpu_count} features")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PHASE 2.5 LABELING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nNext step: python scripts/phase2_5_qwen_validate.py")
    else:
        logger.error("⚠️  No features were successfully labeled")


if __name__ == "__main__":
    main()
