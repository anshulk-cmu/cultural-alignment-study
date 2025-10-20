#!/usr/bin/env python3
"""
Phase 2.5: Parallel Qwen Validation with GPU 2 & 3
GPU isolation using subprocess with environment variables
"""
import sys
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

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
sys.path.append('/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features')

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config import SAE_OUTPUT_ROOT, setup_logger


def load_qwen():
    """Load Qwen2.5-72B with 8-bit quantization."""
    from transformers import BitsAndBytesConfig
    
    model_path = "/mnt/nfs-shared-centralus/anshulk/models/qwen2.5-72b-instruct"
    
    # Verify which GPU we're actually on
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        print(f"Worker initializing on GPU {device}: {device_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
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
    
    print(f"Model loaded successfully")
    return tokenizer, model


def create_validation_prompt(feature_data, initial_label, examples):
    """Create a validation prompt for label quality assessment."""
    examples_text = ""
    num_examples = min(20, len(examples))
    
    for i, ex in enumerate(examples[:num_examples], 1):
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        examples_text += f"{i}. {text}\\n\\n"
    
    prompt = f"""You are validating a feature label from sparse autoencoder analysis of Indian English and Hindi text.

INITIAL LABEL: "{initial_label}"

EXAMPLES ({num_examples} total):
{examples_text}

TASK:
Evaluate whether the label accurately describes the pattern in these examples.

EVALUATION CRITERIA:
- KEEP: Label accurately describes ≥15 out of {num_examples} examples
- REVISE: Label is partially correct but needs refinement (provide improved label)
- INVALIDATE: Label doesn't match examples OR no coherent pattern exists (< 10 examples match)

OUTPUT FORMAT (be explicit):
ACTION: [KEEP|REVISE|INVALIDATE]
REVISED_LABEL: [Only if ACTION is REVISE - provide the improved label here]
REASON: [Brief 1-2 sentence explanation of your decision]

Your response:"""
    return prompt


def generate_validation(tokenizer, model, prompt, max_new_tokens=50):
    """Generate validation response using Qwen2.5."""
    messages = [
        {"role": "system", "content": "You are a rigorous linguistic analyst validating feature labels for interpretability research."},
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
            temperature=0.2,
            do_sample=False,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def parse_validation_response(response):
    """Parse the validation response to extract action, revised label, and reason."""
    action = 'KEEP'  # Default
    revised_label = None
    reason = ""
    
    # Extract ACTION
    if 'INVALIDATE' in response.upper():
        action = 'INVALIDATE'
    elif 'REVISE' in response.upper():
        action = 'REVISE'
    elif 'KEEP' in response.upper():
        action = 'KEEP'
    
    # Extract REVISED_LABEL if action is REVISE
    if action == 'REVISE' and 'REVISED_LABEL:' in response:
        try:
            revised_part = response.split('REVISED_LABEL:')[1]
            if 'REASON:' in revised_part:
                revised_label = revised_part.split('REASON:')[0].strip()
            else:
                revised_label = revised_part.strip()
        except:
            pass
    
    # Extract REASON
    if 'REASON:' in response:
        try:
            reason = response.split('REASON:')[1].strip()
        except:
            pass
    
    return action, revised_label, reason


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--labels_data', type=str, required=True)
    parser.add_argument('--examples_map', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    label_subset = json.loads(args.labels_data)
    examples_map = json.loads(args.examples_map)
    gpu_id = args.gpu_id
    
    logger = setup_logger(f'qwen_validate_gpu{gpu_id}', f'phase2_5_validate_gpu{gpu_id}.log')
    
    logger.info(f"[GPU {gpu_id}] Worker started")
    logger.info(f"[GPU {gpu_id}] Validating {len(label_subset)} labels")
    
    # Load model
    tokenizer, model = load_qwen()
    
    validated = []
    
    for label_data in tqdm(label_subset, desc=f"GPU {gpu_id} - Validating"):
        try:
            # Skip incoherent features
            if not label_data.get('is_coherent', False):
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'Initial label was INCOHERENT'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                validated.append(label_data)
                continue
            
            # Get examples for this feature
            examples = examples_map.get(label_data['feature_id'], [])
            if not examples:
                logger.warning(f"[GPU {gpu_id}] No examples found for {label_data['feature_id']}")
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'No examples available'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                validated.append(label_data)
                continue
            
            # Create validation prompt
            prompt = create_validation_prompt(
                label_data,
                label_data['label_qwen'],
                examples
            )
            
            # Generate validation
            response = generate_validation(tokenizer, model, prompt)
            
            # Parse response
            action, revised_label, reason = parse_validation_response(response)
            
            # Store results
            label_data['validation_action'] = action
            label_data['validation_response'] = response
            label_data['validation_reason'] = reason
            label_data['validated_by_gpu'] = gpu_id
            
            # Determine final label
            if action == 'KEEP':
                label_data['final_label'] = label_data['label_qwen']
            elif action == 'REVISE':
                label_data['final_label'] = revised_label if revised_label else label_data['label_qwen']
            else:  # INVALIDATE
                label_data['final_label'] = None
            
            validated.append(label_data)
            
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error validating feature {label_data.get('feature_id', 'unknown')}: {str(e)}")
            label_data['validation_action'] = 'ERROR'
            label_data['validation_reason'] = str(e)
            label_data['final_label'] = None
            label_data['validated_by_gpu'] = gpu_id
            validated.append(label_data)
            continue
    
    # Save results
    output_file = Path(args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[GPU {gpu_id}] Validated {len(validated)} features")


if __name__ == "__main__":
    main()
'''
    return worker_code


def main():
    model_path = "/mnt/nfs-shared-centralus/anshulk/models/qwen2.5-72b-instruct"
    initial_labels_file = SAE_OUTPUT_ROOT / "labels_qwen_initial.json"
    examples_dir = SAE_OUTPUT_ROOT / "feature_examples"
    output_dir = SAE_OUTPUT_ROOT
    final_output_file = output_dir / "labels_qwen_validated.json"
    
    logger = setup_logger('qwen_validate_parallel', 'phase2_5_validate_parallel.log')
    
    logger.info("=" * 80)
    logger.info("PARALLEL QWEN2.5-72B LABEL VALIDATION")
    logger.info(f"Using GPUs: {FREE_GPUS}")
    logger.info("=" * 80)
    logger.info(f"Input file: {initial_labels_file}")
    logger.info(f"Examples directory: {examples_dir}")
    logger.info(f"Output file: {final_output_file}")
    
    # Load initial labels
    logger.info(f"\nLoading initial labels from {initial_labels_file}")
    with open(initial_labels_file, 'r', encoding='utf-8') as f:
        initial_labels = json.load(f)
    
    # Sample proportionally across all labels
    total_to_process = min(3600, len(initial_labels))
    initial_labels = initial_labels[:total_to_process]
    
    logger.info(f"Loaded {len(initial_labels)} initial labels")
    
    # Load all examples into a map
    examples_dir = Path(examples_dir)
    examples_map = {}
    
    logger.info("\nLoading examples...")
    for file_path in examples_dir.glob("*_examples.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
            for feat in features:
                examples_map[feat['feature_id']] = feat['examples']
    
    logger.info(f"Loaded examples for {len(examples_map)} features")
    
    # Split labels between GPUs
    labels_per_gpu = len(initial_labels) // len(FREE_GPUS)
    remainder = len(initial_labels) % len(FREE_GPUS)
    
    label_splits = []
    start_idx = 0
    for i, gpu_id in enumerate(FREE_GPUS):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + labels_per_gpu + extra
        label_splits.append((gpu_id, initial_labels[start_idx:end_idx]))
        start_idx = end_idx
    
    logger.info("\nWork distribution:")
    for gpu_id, labels in label_splits:
        logger.info(f"  GPU {gpu_id}: {len(labels)} labels")
    
    # Create worker script
    worker_script_path = Path("/tmp/qwen_validate_worker.py")
    with open(worker_script_path, 'w') as f:
        f.write(create_worker_script())
    
    logger.info("\nStarting parallel validation with subprocess approach...")
    
    # Launch subprocesses with CUDA_VISIBLE_DEVICES set
    processes = []
    
    for gpu_id, label_list in label_splits:
        output_file = output_dir / f"labels_qwen_validated_gpu{gpu_id}.json"
        
        # Create environment with CUDA_VISIBLE_DEVICES set
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Prepare arguments
        cmd = [
            sys.executable,
            str(worker_script_path),
            '--gpu_id', str(gpu_id),
            '--labels_data', json.dumps(label_list),
            '--examples_map', json.dumps(examples_map),
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
    all_validated = []
    for gpu_id in FREE_GPUS:
        gpu_output_file = output_dir / f"labels_qwen_validated_gpu{gpu_id}.json"
        if gpu_output_file.exists():
            with open(gpu_output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                all_validated.extend(results)
                logger.info(f"  GPU {gpu_id}: {len(results)} features")
        else:
            logger.warning(f"  GPU {gpu_id}: output file not found!")
    
    # Save merged results
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_validated, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nMerged results saved to {final_output_file}")
    
    # Compute statistics
    stats = {
        'KEEP': sum(1 for v in all_validated if v.get('validation_action') == 'KEEP'),
        'REVISE': sum(1 for v in all_validated if v.get('validation_action') == 'REVISE'),
        'INVALIDATE': sum(1 for v in all_validated if v.get('validation_action') == 'INVALIDATE'),
        'ERROR': sum(1 for v in all_validated if v.get('validation_action') == 'ERROR')
    }
    
    total = len(all_validated)
    final_valid = stats['KEEP'] + stats['REVISE']
    
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Total features validated: {total}")
    logger.info(f"KEEP: {stats['KEEP']} ({stats['KEEP']/total*100:.1f}%)")
    logger.info(f"REVISE: {stats['REVISE']} ({stats['REVISE']/total*100:.1f}%)")
    logger.info(f"INVALIDATE: {stats['INVALIDATE']} ({stats['INVALIDATE']/total*100:.1f}%)")
    logger.info(f"ERROR: {stats['ERROR']} ({stats['ERROR']/total*100:.1f}%)")
    logger.info(f"Final valid features: {final_valid} ({final_valid/total*100:.1f}%)")
    
    # GPU distribution
    logger.info("\nProcessing distribution:")
    for gpu_id in FREE_GPUS:
        gpu_count = sum(1 for v in all_validated if v.get('validated_by_gpu') == gpu_id)
        logger.info(f"  GPU {gpu_id}: {gpu_count} features")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    import os
    main()
