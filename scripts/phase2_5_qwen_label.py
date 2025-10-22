#!/usr/bin/env python3
"""
Phase 2.5: Parallel Qwen Labeling with GPU 2 & 3
GPU isolation using subprocess with environment variables
"""
import sys
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import subprocess
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

# GPU configuration - using only free GPUs
FREE_GPUS = [2, 3]


def create_worker_script():
    """Create the worker script that will be run as a subprocess."""
    worker_code = '''#!/usr/bin/env python3
import sys
import os
sys.path.append('/home/anshulk/cultural-alignment-study')

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import SAE_OUTPUT_ROOT, setup_logger

def load_qwen():
    """Load Qwen2.5-72B with 8-bit quantization."""
    from transformers import BitsAndBytesConfig
    
    model_path = "/user_data/anshulk/models/qwen2.5-72b-instruct"
    
    # Verify which GPU we're actually on
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        print(f"Worker initializing on GPU {device}: {device_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 8-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},  # Force all layers on device 0
        trust_remote_code=True
    )
    
    print(f"Model loaded successfully on device: {model.device}")
    return tokenizer, model


def create_labeling_prompt(feature_data):
    """Create a prompt for feature labeling with top examples."""
    examples_text = ""
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
    """Generate a label using Qwen2.5 model."""
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
            temperature=0.3,
            do_sample=False,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--file_paths', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    file_paths = json.loads(args.file_paths)
    gpu_id = args.gpu_id
    
    logger = setup_logger(f'qwen_label_gpu{gpu_id}', f'phase2_5_qwen_gpu{gpu_id}.log')
    
    logger.info(f"[GPU {gpu_id}] Worker started")
    logger.info(f"[GPU {gpu_id}] Processing {len(file_paths)} files")
    
    # Load model
    tokenizer, model = load_qwen()
    
    all_results = []
    
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        logger.info(f"[GPU {gpu_id}] Processing {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        # Only process top 400 features
        features = features[:400]
        
        logger.info(f"[GPU {gpu_id}] Loaded {len(features)} features from {file_path.name}")
        
        for feature in tqdm(features, desc=f"GPU {gpu_id} - {file_path.stem}"):
            try:
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
                    'processed_by_gpu': gpu_id
                }
                
                # Copy over additional statistics
                for key in ['max_activation', 'mean_activation', 'sparsity']:
                    if key in feature:
                        result[key] = feature[key]
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"[GPU {gpu_id}] Error labeling feature {feature['feature_id']}: {str(e)}")
                continue
    
    # Save results
    output_file = Path(args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[GPU {gpu_id}] Saved {len(all_results)} labels to {output_file}")


if __name__ == "__main__":
    main()
'''
    return worker_code


def main():
    model_path = "/user_data/anshulk/models/qwen2.5-72b-instruct"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_dir = SAE_OUTPUT_ROOT
    final_output_file = output_dir / "labels_qwen_initial.json"
    
    # Setup main logger
    logger = setup_logger('qwen_label_parallel', 'phase2_5_qwen_parallel.log')
    
    logger.info("=" * 80)
    logger.info("PARALLEL QWEN2.5-72B FEATURE LABELING")
    logger.info(f"Using GPUs: {FREE_GPUS}")
    logger.info("=" * 80)
    logger.info(f"Input directory: {examples_dir}")
    logger.info(f"Output file: {final_output_file}")
    logger.info(f"Model path: {model_path}")
    
    # Get all example files
    examples_files = sorted(examples_dir.glob("*_examples.json"))
    
    if not examples_files:
        logger.error(f"No example files found in {examples_dir}")
        return
    
    logger.info(f"\nFound {len(examples_files)} example files to process")
    for f in examples_files:
        logger.info(f"  - {f.name}")
    
    # Split files between GPUs
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
    worker_script_path = Path("/tmp/qwen_worker.py")
    with open(worker_script_path, 'w') as f:
        f.write(create_worker_script())
    
    logger.info("\nStarting parallel processing with subprocess approach...")
    
    # Launch subprocesses with CUDA_VISIBLE_DEVICES set
    processes = []
    
    for gpu_id, file_list in file_splits:
        output_file = output_dir / f"labels_qwen_initial_gpu{gpu_id}.json"
        
        # Create environment with CUDA_VISIBLE_DEVICES set
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Prepare arguments
        cmd = [
            sys.executable,
            str(worker_script_path),
            '--gpu_id', str(gpu_id),
            '--file_paths', json.dumps(file_list),
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
        logger.info(f"  Started worker on GPU {gpu_id} (PID: {proc.pid})")
    
    # Monitor processes and stream output
    import select
    import time
    
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
    all_success = True
    for gpu_id, proc in processes:
        if proc.returncode != 0:
            logger.error(f"GPU {gpu_id} worker failed with code {proc.returncode}")
            all_success = False
        else:
            logger.info(f"GPU {gpu_id} worker completed successfully")
    
    if not all_success:
        logger.error("Some workers failed. Check logs for details.")
        return
    
    logger.info("\nAll workers completed. Merging results...")
    
    # Merge results from all GPUs
    all_results = []
    for gpu_id in FREE_GPUS:
        gpu_output_file = output_dir / f"labels_qwen_initial_gpu{gpu_id}.json"
        if gpu_output_file.exists():
            with open(gpu_output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_results.extend(results)
                logger.info(f"  GPU {gpu_id}: {len(results)} features")
        else:
            logger.warning(f"  GPU {gpu_id}: output file not found!")
    
    # Save merged results
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nMerged results saved to {final_output_file}")
    
    # Summary statistics
    if all_results:
        coherent = sum(1 for r in all_results if r['is_coherent'])
        coherent_pct = (coherent / len(all_results)) * 100
        
        logger.info("=" * 80)
        logger.info("LABELING COMPLETE")
        logger.info(f"Total features labeled: {len(all_results)}")
        logger.info(f"Coherent features: {coherent} ({coherent_pct:.1f}%)")
        logger.info(f"Incoherent features: {len(all_results) - coherent} ({100 - coherent_pct:.1f}%)")
        
        # GPU distribution
        logger.info("\nProcessing distribution:")
        for gpu_id in FREE_GPUS:
            gpu_count = sum(1 for r in all_results if r.get('processed_by_gpu') == gpu_id)
            logger.info(f"  GPU {gpu_id}: {gpu_count} features")
        
        logger.info("=" * 80)
    else:
        logger.error("No features were successfully labeled")


if __name__ == "__main__":
    import os
    main()
