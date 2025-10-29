#!/usr/bin/env python3
"""
Phase 2.5: Parallel Qwen3 Validation with 4x 48GB-class GPUs (e.g., L40S)
Using Qwen3-30B-A3B-Instruct-2507 model
WITH CHECKPOINT SUPPORT, TQDM PROGRESS BARS, AND OPTIMIZED MEMORY MANAGEMENT
"""
import sys
project_root = '/home/anshulk/cultural-alignment-study'
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import subprocess
import os
import time
import gc
import random
import numpy as np
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

# GPU configuration - 4 GPUs
REQUESTED_GPUS = 4

# REPRODUCIBILITY SEED
SEED = 42


def create_worker_script():
    """Create the worker script with optimized memory management."""
    worker_code = '''#!/usr/bin/env python3
import sys
import os
project_root = '/home/anshulk/cultural-alignment-study'
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import torch
import gc
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs.config import SAE_OUTPUT_ROOT, setup_logger

SEED = 42


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def load_qwen3_30b():
    """Load Qwen3-30B with optimized 8-bit quantization."""
    model_path = "/data/user_data/anshulk/data/models/Qwen3-30B-A3B-Instruct-2507"
    
    assigned_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
    print(f"Worker GPU {assigned_gpu}: Loading model...")
    
    # Clear all GPU memory before loading
    clear_gpu_memory()
    
    # Set PyTorch memory allocator configs for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optimized 8-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,  # Reduce memory overhead
        bnb_4bit_use_double_quant=False
    )
    
    # Load with auto device_map for better memory management
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",  # Auto memory management instead of manual assignment
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        max_memory={0: "46GB"}  # Reserve 46GB of 48GB, leave 2GB buffer
    )
    model.eval()
    
    # Enable memory-efficient attention if available
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
    
    alloc, reserved = get_gpu_memory_usage()
    print(f"Model loaded: {alloc:.2f}GB / {reserved:.2f}GB")
    
    # Clear any residual memory after loading
    clear_gpu_memory()
    
    return tokenizer, model


def create_validation_prompt(feature_data, initial_label, examples):
    """Create validation prompt."""
    examples_text = ""
    num_examples = min(20, len(examples))
    
    for i, ex in enumerate(examples[:num_examples], 1):
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        safe_text = json.dumps(text)[1:-1]
        examples_text += f"{i}. {safe_text} (activation: {ex['activation']:.3f})\\n\\n"
    
    safe_label = json.dumps(initial_label)[1:-1]
    
    prompt = f"""You are validating a feature label from sparse autoencoder analysis of Indian English and Hindi text.

INITIAL LABEL: "{safe_label}"

EXAMPLES ({num_examples} total):
{examples_text}
TASK:
Evaluate whether the label accurately describes the pattern in these examples.

EVALUATION CRITERIA:
- KEEP: Label accurately describes ≥15 out of {num_examples} examples
- REVISE: Label is partially correct but needs refinement (provide improved label)
- INVALIDATE: Label doesn't match examples OR no coherent pattern exists (< 10 examples match)

OUTPUT FORMAT (strictly follow this):
ACTION: [KEEP|REVISE|INVALIDATE]
REVISED_LABEL: [Only if ACTION is REVISE - provide the improved label here, otherwise leave empty]
REASON: [Brief 1-2 sentence explanation of your decision]

Your response:"""
    return prompt


def generate_validation(tokenizer, model, prompt, max_new_tokens=70):
    """Generate validation with GREEDY DECODING and optimized memory usage."""
    messages = [
        {"role": "system", "content": "You are a rigorous linguistic analyst validating feature labels for interpretability research. Strictly follow the output format."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", padding=False, truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache reuse
            return_dict_in_generate=False  # Reduce memory overhead
        )
    
    response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # Immediately delete tensors and clear cache after each generation
    del inputs, outputs, response_ids
    torch.cuda.empty_cache()
    gc.collect()
    
    return response.strip()


def parse_validation_response(response):
    """Parse validation response."""
    action = 'KEEP'
    revised_label = None
    reason = "Parsing failed."
    
    lines = response.split('\\n')
    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith("ACTION:"):
            action_val = line.split(":", 1)[1].strip().upper()
            if action_val in ['KEEP', 'REVISE', 'INVALIDATE']:
                action = action_val
        elif line_upper.startswith("REVISED_LABEL:"):
            revised_label = line.split(":", 1)[1].strip()
            if not revised_label or revised_label.lower() in ['n/a', 'none', '']:
                revised_label = None
        elif line_upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    
    if action != 'REVISE':
        revised_label = None
    
    return action, revised_label, reason


def save_checkpoint(results, output_file, gpu_id):
    """Save checkpoint."""
    checkpoint_file = output_file.parent / f"{output_file.stem}_checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


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
    parser.add_argument('--labels_file', type=str, required=True)
    parser.add_argument('--examples_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    set_seed(SEED)
    
    with open(args.labels_file, 'r', encoding='utf-8') as f:
        label_subset = json.load(f)
    with open(args.examples_file, 'r', encoding='utf-8') as f:
        examples_map = json.load(f)
    
    gpu_id = args.gpu_id
    output_file = Path(args.output_file)
    
    logger = setup_logger(f'qwen3_validate_gpu{gpu_id}', f'phase2_5_validate_gpu{gpu_id}.log')
    
    print(f"[GPU {gpu_id}] Starting validation of {len(label_subset)} labels")
    
    validated = load_checkpoint(output_file)
    processed_features = set(v['feature_id'] for v in validated)
    
    if validated:
        print(f"[GPU {gpu_id}] Resuming: {len(validated)} already done")
    
    tokenizer, model = load_qwen3_30b()
    
    # TQDM Progress bar
    with tqdm(total=len(label_subset), desc=f"GPU {gpu_id}", initial=len(validated), position=gpu_id, leave=True) as pbar:
        
        for label_data in label_subset:
            feature_id = label_data.get('feature_id', 'unknown')
            
            if feature_id in processed_features:
                continue
            
            initial_label = label_data.get('label_qwen', '')
            
            # Quick invalidation for incoherent
            if not label_data.get('is_coherent', False) or not initial_label:
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'Initial label was INCOHERENT'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                label_data['seed'] = SEED
                validated.append(label_data)
                processed_features.add(feature_id)
                pbar.update(1)
                continue
            
            examples = examples_map.get(str(feature_id), [])
            if not examples:
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'No examples available'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                label_data['seed'] = SEED
                validated.append(label_data)
                processed_features.add(feature_id)
                pbar.update(1)
                continue
            
            # Actual validation
            try:
                prompt = create_validation_prompt(label_data, initial_label, examples)
                response = generate_validation(tokenizer, model, prompt)
                action, revised_label, reason = parse_validation_response(response)
                
                label_data['validation_action'] = action
                label_data['validation_response'] = response
                label_data['validation_reason'] = reason
                label_data['validated_by_gpu'] = gpu_id
                label_data['seed'] = SEED
                
                if action == 'KEEP':
                    label_data['final_label'] = initial_label
                elif action == 'REVISE':
                    label_data['final_label'] = revised_label if revised_label else initial_label
                else:
                    label_data['final_label'] = None
                
                validated.append(label_data)
                processed_features.add(feature_id)
                
                # More aggressive memory cleanup - every 25 features instead of 50
                if len(validated) % 25 == 0:
                    clear_gpu_memory()
                    alloc, reserved = get_gpu_memory_usage()
                    logger.info(f"GPU {gpu_id} memory after {len(validated)} features: {alloc:.2f}GB / {reserved:.2f}GB")
                
                # Save checkpoint more frequently - every 50 features instead of 100
                if len(validated) % 50 == 0:
                    save_checkpoint(validated, output_file, gpu_id)
                
            except Exception as e:
                logger.error(f"Error validating {feature_id}: {e}")
                label_data['validation_action'] = 'ERROR'
                label_data['validation_reason'] = str(e)
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                label_data['seed'] = SEED
                validated.append(label_data)
                processed_features.add(feature_id)
                clear_gpu_memory()  # Clear memory after error
            
            pbar.update(1)
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    
    print(f"[GPU {gpu_id}] ✓ Complete: {len(validated)} results saved")


if __name__ == "__main__":
    main()
'''
    return worker_code


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    model_path = "/data/user_data/anshulk/data/models/Qwen3-30B-A3B-Instruct-2507"
    initial_labels_file = SAE_OUTPUT_ROOT / "labels_qwen_initial.json"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_dir = SAE_OUTPUT_ROOT
    final_output_file = output_dir / "labels_qwen3_validated.json"
    
    logger = setup_logger('qwen3_validate_parallel', 'phase2_5_validate_parallel.log')
    
    logger.info("=" * 80)
    logger.info("PARALLEL QWEN3-30B LABEL VALIDATION (4 GPUs) - OPTIMIZED")
    logger.info(f"GPUs: {REQUESTED_GPUS} x L40S 48GB")
    logger.info(f"SEED: {SEED} (GREEDY DECODING)")
    logger.info(f"Memory Management: Optimized for 8-bit quantization")
    logger.info("=" * 80)
    
    # Load labels
    with open(initial_labels_file, 'r', encoding='utf-8') as f:
        initial_labels = json.load(f)
    logger.info(f"Loaded {len(initial_labels)} labels to validate")
    
    # Load examples
    examples_map = {}
    for file_path in sorted(examples_dir.glob("*_examples.json")):
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
            for feat in features:
                if 'feature_id' in feat and 'examples' in feat:
                    examples_map[str(feat['feature_id'])] = feat['examples']
    logger.info(f"Loaded examples for {len(examples_map)} features")
    
    # Get GPUs from SLURM (user confirmed always 0,1,2,3)
    assigned_gpu_ids_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if assigned_gpu_ids_str:
        assigned_gpu_ids = [int(gid.strip()) for gid in assigned_gpu_ids_str.split(',')]
    else:
        assigned_gpu_ids = list(range(REQUESTED_GPUS))
    
    logger.info(f"Using GPUs: {assigned_gpu_ids}")
    
    # Distribute work
    num_gpus = len(assigned_gpu_ids)
    labels_per_gpu = len(initial_labels) // num_gpus
    remainder = len(initial_labels) % num_gpus
    
    label_splits = []
    start_idx = 0
    for i in range(num_gpus):
        gpu_id = assigned_gpu_ids[i]
        extra = 1 if i < remainder else 0
        end_idx = start_idx + labels_per_gpu + extra
        label_splits.append((gpu_id, initial_labels[start_idx:end_idx]))
        logger.info(f"  GPU {gpu_id}: {end_idx - start_idx} labels")
        start_idx = end_idx
    
    # Write worker script
    worker_script_path = Path("/tmp/qwen3_validate_worker_4gpu_optimized.py")
    with open(worker_script_path, 'w') as f:
        f.write(create_worker_script())
    
    # Write examples map
    examples_temp_file = Path("/tmp") / f"qwen3_examples_{os.getpid()}.json"
    with open(examples_temp_file, 'w', encoding='utf-8') as f:
        json.dump(examples_map, f, ensure_ascii=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING 4 GPU WORKERS (OPTIMIZED)")
    logger.info("=" * 80)
    
    processes = []
    for gpu_id, label_list in label_splits:
        output_file = output_dir / f"labels_qwen3_validated_gpu{gpu_id}.json"
        labels_temp_file = Path("/tmp") / f"qwen3_labels_gpu{gpu_id}_{os.getpid()}.json"
        
        with open(labels_temp_file, 'w', encoding='utf-8') as f:
            json.dump(label_list, f, ensure_ascii=False)
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONHASHSEED'] = str(SEED)
        # Add memory optimization env vars
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        
        cmd = [
            sys.executable,
            str(worker_script_path),
            '--gpu_id', str(gpu_id),
            '--labels_file', str(labels_temp_file),
            '--examples_file', str(examples_temp_file),
            '--output_file', str(output_file)
        ]
        
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        processes.append((gpu_id, proc))
        logger.info(f"  ✓ Started GPU {gpu_id} worker (PID: {proc.pid})")
    
    # Monitor workers
    while any(proc.poll() is None for _, proc in processes):
        for gpu_id, proc in processes:
            try:
                line = proc.stdout.readline()
                if line:
                    print(f"[GPU {gpu_id}] {line.rstrip()}")
            except:
                pass
        time.sleep(0.1)
    
    # Merge results
    logger.info("\n" + "=" * 80)
    logger.info("MERGING RESULTS")
    logger.info("=" * 80)
    
    all_validated = []
    for gpu_id, _ in label_splits:
        gpu_file = output_dir / f"labels_qwen3_validated_gpu{gpu_id}.json"
        if gpu_file.exists():
            with open(gpu_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_validated.extend(results)
                logger.info(f"  GPU {gpu_id}: {len(results)} features")
    
    # Sort and save
    all_validated = sorted(all_validated, key=lambda x: x.get('feature_id', ''))
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validated, f, indent=2, ensure_ascii=False)
    
    # Stats
    stats = {
        'KEEP': sum(1 for r in all_validated if r.get('validation_action') == 'KEEP'),
        'REVISE': sum(1 for r in all_validated if r.get('validation_action') == 'REVISE'),
        'INVALIDATE': sum(1 for r in all_validated if r.get('validation_action') == 'INVALIDATE'),
        'ERROR': sum(1 for r in all_validated if r.get('validation_action') == 'ERROR'),
    }
    
    total = len(all_validated)
    final_valid = stats['KEEP'] + stats['REVISE']
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total: {total}")
    logger.info(f"KEEP:       {stats['KEEP']:>6} ({stats['KEEP']/total*100:>5.1f}%)")
    logger.info(f"REVISE:     {stats['REVISE']:>6} ({stats['REVISE']/total*100:>5.1f}%)")
    logger.info(f"INVALIDATE: {stats['INVALIDATE']:>6} ({stats['INVALIDATE']/total*100:>5.1f}%)")
    logger.info(f"ERROR:      {stats['ERROR']:>6} ({stats['ERROR']/total*100:>5.1f}%)")
    logger.info(f"Final Valid: {final_valid} ({final_valid/total*100:.1f}%)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
