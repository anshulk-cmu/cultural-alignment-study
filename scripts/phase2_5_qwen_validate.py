#!/usr/bin/env python3
"""
Phase 2.5: Parallel Qwen3 Validation with 2x 48GB-class GPUs (e.g., L40S)
Using Qwen3-30B-A3B-Instruct-2507 model
WITH CHECKPOINT SUPPORT for fault tolerance
"""
import sys
# Ensure the project root is in the path
project_root = '/home/anshulk/cultural-alignment-study'
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import subprocess
import os
import time
import gc
from pathlib import Path
from configs.config import SAE_OUTPUT_ROOT, setup_logger

# GPU configuration - We will request 2 GPUs from Slurm
REQUESTED_GPUS = 2


def create_worker_script():
    """Create the worker script that will be run as a subprocess."""
    worker_code = '''#!/usr/bin/env python3
import sys
import os
# Ensure the project root is in the path for the worker too
project_root = '/home/anshulk/cultural-alignment-study'
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs.config import SAE_OUTPUT_ROOT, setup_logger


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


def load_qwen3_30b():
    """Load Qwen3-30B-A3B-Instruct-2507 with 8-bit quantization."""

    # Set model path to where you downloaded it
    model_path = "/data/user_data/anshulk/data/models/Qwen3-30B-A3B-Instruct-2507"

    # Verify which GPU we're actually on (should be ID 0 due to CUDA_VISIBLE_DEVICES)
    assigned_gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
    internal_gpu_id = 0 # Default if CUDA_VISIBLE_DEVICES maps to a single device
    device_name = "CPU"
    total_memory = 0
    if torch.cuda.is_available():
        try:
            internal_gpu_id = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(internal_gpu_id)
            total_memory = torch.cuda.get_device_properties(internal_gpu_id).total_memory / 1024**3
        except Exception as e:
            print(f"Error getting CUDA device info: {e}")

    print(f"Worker assigned physical GPU {assigned_gpu_id}, mapped to internal ID {internal_gpu_id}: {device_name} ({total_memory:.1f} GB)")

    clear_gpu_memory()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )

    # Qwen3 uses EOS token for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 8-bit quantization...")
    # 8-bit quantization config (suitable for MoE on L40S)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0 # Common setting for 8-bit
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": internal_gpu_id},  # Map to the single visible GPU (ID 0)
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # Use bfloat16 if supported, float16 otherwise
        low_cpu_mem_usage=True,
        local_files_only=True # Prevent hanging
    )

    model.eval()  # Set to evaluation mode

    # Report memory usage
    alloc, reserved = get_gpu_memory_usage()
    print(f"Model loaded successfully")
    print(f"GPU Memory - Allocated: {alloc:.2f} GB, Reserved: {reserved:.2f} GB")

    return tokenizer, model


def create_validation_prompt(feature_data, initial_label, examples):
    """Create a validation prompt for label quality assessment."""
    examples_text = ""
    num_examples = min(20, len(examples)) # Use 20 examples

    for i, ex in enumerate(examples[:num_examples], 1):
        text = ex['text'][:300]
        if len(ex['text']) > 300:
            text += "..."
        # Ensure newline characters are escaped for JSON and clarity
        safe_text = json.dumps(text)[1:-1] # Basic escaping
        examples_text += f"{i}. {safe_text} (activation: {ex['activation']:.3f})\\n\\n"

    # Escape initial label as well
    safe_initial_label = json.dumps(initial_label)[1:-1]

    prompt = f"""You are validating a feature label from sparse autoencoder analysis of Indian English and Hindi text.

INITIAL LABEL: "{safe_initial_label}"

EXAMPLES ({num_examples} total):
{examples_text}
TASK:
Evaluate whether the label accurately describes the pattern in these examples.

EVALUATION CRITERIA:
- KEEP: Label accurately describes ≥15 out of {num_examples} examples
- REVISE: Label is partially correct but needs refinement (provide improved label)
- INVALIDATE: Label doesn't match examples OR no coherent pattern exists (< 10 examples match)

OUTPUT FORMAT (strictly follow this, use newline characters between fields):
ACTION: [KEEP|REVISE|INVALIDATE]
REVISED_LABEL: [Only if ACTION is REVISE - provide the improved label here, otherwise leave empty]
REASON: [Brief 1-2 sentence explanation of your decision]

Your response:"""
    return prompt


def generate_validation(tokenizer, model, prompt, max_new_tokens=70): # Increased slightly for reason
    """Generate validation response using Qwen3."""
    messages = [
        {"role": "system", "content": "You are a rigorous linguistic analyst validating feature labels for interpretability research. Strictly follow the output format."},
        {"role": "user", "content": prompt}
    ]

    # Use apply_chat_template for Qwen models
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
            eos_token_id=tokenizer.eos_token_id, # Use EOS token id
            do_sample=False, # Use greedy decoding for consistency
            temperature=0.1, # Low temperature for validation
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id # Use assigned pad token
        )

    # Decode only the newly generated tokens
    response_ids = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Clean up tensors
    del inputs, outputs, response_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


def parse_validation_response(response):
    """Parse the validation response to extract action, revised label, and reason."""
    action = 'KEEP'  # Default
    revised_label = None
    reason = "Parsing failed or reason not provided." # Default reason

    lines = response.split('\\n')
    action_found = False
    revised_found = False
    reason_found = False

    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith("ACTION:"):
            action_val = line.split(":", 1)[1].strip().upper()
            if action_val in ['KEEP', 'REVISE', 'INVALIDATE']:
                action = action_val
                action_found = True
        elif line_upper.startswith("REVISED_LABEL:"):
            revised_label = line.split(":", 1)[1].strip()
            # Handle cases where the label might be empty or placeholder
            if not revised_label or revised_label.lower() in ['n/a', 'none', '']:
                revised_label = None
            revised_found = True
        elif line_upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
            reason_found = True

    # Simple keyword check as fallback if structured parsing failed
    if not action_found:
        response_upper = response.upper()
        if 'INVALIDATE' in response_upper: action = 'INVALIDATE'
        elif 'REVISE' in response_upper: action = 'REVISE'
        # Keep remains default

    # If action is REVISE but no revised label was found via tag, try to extract contextually
    if action == 'REVISE' and not revised_found and revised_label is None:
         # Simplistic: assume label follows action if reason tag exists
         if reason_found:
              potential_label = response.split("REASON:")[0].split("ACTION:")[1].strip().split('\\n')[0].strip()
              if potential_label and potential_label.upper() != 'REVISE':
                   revised_label = potential_label
         # If no reason tag, maybe it's just the rest of the string after ACTION: REVISE
         elif not reason_found and action_found and action == 'REVISE':
             parts = response.split("ACTION: REVISE", 1)
             if len(parts) > 1 and parts[1].strip():
                 revised_label = parts[1].strip().split('\\n')[0].strip()

    # Ensure revised_label is None if action is not REVISE
    if action != 'REVISE':
        revised_label = None

    return action, revised_label, reason


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
    parser.add_argument('--gpu_id', type=int, required=True, help="Assigned physical GPU ID")
    parser.add_argument('--labels_data', type=str, required=True)
    parser.add_argument('--examples_map', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    label_subset = json.loads(args.labels_data)
    # Deserialize the map values as well
    examples_map_serial = json.loads(args.examples_map)
    examples_map = {k: v for k, v in examples_map_serial.items()}

    gpu_id = args.gpu_id # This is the assigned physical GPU ID
    output_file = Path(args.output_file)

    logger = setup_logger(f'qwen3_validate_gpu{gpu_id}', f'phase2_5_validate_gpu{gpu_id}.log')

    logger.info(f"[GPU {gpu_id}] Worker started")
    logger.info(f"[GPU {gpu_id}] Validating {len(label_subset)} labels")
    logger.info(f"[GPU {gpu_id}] Output: {output_file}")

    # Check for checkpoint and resume
    validated = load_checkpoint(output_file)
    processed_features = set(v['feature_id'] for v in validated)
    
    if validated:
        logger.info(f"[GPU {gpu_id}] Resuming from checkpoint: {len(validated)} features already validated")
    
    # Load model
    try:
        tokenizer, model = load_qwen3_30b()
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to load model: {e}", exc_info=True)
        # Attempt to save partial results indicating failure
        error_results = []
        for label_data in label_subset:
             if label_data.get('feature_id') not in processed_features:
                 label_data['validation_action'] = 'ERROR'
                 label_data['validation_reason'] = f"Model loading failed: {e}"
                 label_data['final_label'] = None
                 label_data['validated_by_gpu'] = gpu_id
                 error_results.append(label_data)
        
        if error_results:
            validated.extend(error_results)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
        sys.exit(1) # Ensure the main script knows this worker failed

    # Process labels
    features_processed_this_run = 0
    
    for idx, label_data in enumerate(tqdm(label_subset, desc=f"GPU {gpu_id} - Validating")):
        try:
            feature_id = label_data.get('feature_id', 'unknown')
            
            # Skip if already processed in checkpoint
            if feature_id in processed_features:
                continue
            
            initial_label = label_data.get('label_qwen', '')

            # Skip incoherent/invalid features more robustly
            if not label_data.get('is_coherent', False) or not initial_label or initial_label.upper() == 'INCOHERENT':
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'Initial label was INCOHERENT or missing'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                validated.append(label_data)
                processed_features.add(feature_id)
                features_processed_this_run += 1
                continue

            # Get examples for this feature
            feature_id_str = str(feature_id) # Ensure key is string
            examples = examples_map.get(feature_id_str, [])
            if not examples:
                logger.warning(f"[GPU {gpu_id}] No examples found for {feature_id}")
                label_data['validation_action'] = 'INVALIDATE'
                label_data['validation_reason'] = 'No examples available'
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                validated.append(label_data)
                processed_features.add(feature_id)
                features_processed_this_run += 1
                continue

            # Create validation prompt
            prompt = create_validation_prompt(
                label_data,
                initial_label,
                examples
            )

            # Generate validation
            response = generate_validation(tokenizer, model, prompt)

            # Parse response
            action, revised_label, reason = parse_validation_response(response)

            # Store results
            label_data['validation_action'] = action
            label_data['validation_response'] = response # Store raw response
            label_data['validation_reason'] = reason
            label_data['validated_by_gpu'] = gpu_id

            # Determine final label
            if action == 'KEEP':
                label_data['final_label'] = initial_label
            elif action == 'REVISE':
                # Use revised only if valid, else keep original as fallback
                label_data['final_label'] = revised_label if revised_label else initial_label
                if not revised_label:
                    logger.warning(f"[GPU {gpu_id}] Action REVISE but failed to parse revised_label for {feature_id}. Raw response: {response}")
            else:  # INVALIDATE or ERROR
                label_data['final_label'] = None

            validated.append(label_data)
            processed_features.add(feature_id)
            features_processed_this_run += 1

            # Memory cleanup every 50 features
            if features_processed_this_run % 50 == 0:
                clear_gpu_memory()
                alloc, reserved = get_gpu_memory_usage()
                logger.info(f"[GPU {gpu_id}] Progress: {len(validated)}/{len(label_subset)} | Memory: {alloc:.1f}GB / {reserved:.1f}GB")

            # Checkpoint save every 100 features
            if features_processed_this_run % 100 == 0:
                save_checkpoint(validated, output_file, gpu_id)

        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Error validating feature {feature_id}: {e}", exc_info=True)
            # Ensure label_data exists and is a dict before modifying
            if isinstance(label_data, dict):
                label_data['validation_action'] = 'ERROR'
                label_data['validation_reason'] = str(e)
                label_data['final_label'] = None
                label_data['validated_by_gpu'] = gpu_id
                validated.append(label_data) # Append even on error
                processed_features.add(feature_id)
            else:
                 logger.error(f"[GPU {gpu_id}] Invalid label_data format during error handling: {label_data}")

            # Attempt to clear memory after error
            clear_gpu_memory()
            continue # Move to next feature

    # Save final results for this worker
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)
        logger.info(f"[GPU {gpu_id}] ✓ Complete: {len(validated)} validation results saved to {output_file}")
    except Exception as e:
         logger.error(f"[GPU {gpu_id}] Failed to save final results to {output_file}: {e}")

    # Final memory report
    alloc, reserved = get_gpu_memory_usage()
    logger.info(f"[GPU {gpu_id}] Final GPU Memory - Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB")


if __name__ == "__main__":
    main()
'''
    return worker_code


def main():
    # Set Model Path to the downloaded Qwen3 model
    model_path = "/data/user_data/anshulk/data/models/Qwen3-30B-A3B-Instruct-2507"

    # Define input/output paths using the symlinks in /home
    home_base = Path("/home/anshulk/cultural-alignment-study/outputs")
    initial_labels_file = home_base / "sae_models/labels_qwen_initial.json"
    examples_dir = home_base / "sae_models/feature_examples"
    output_dir = home_base / "sae_models" # Output goes next to inputs
    final_output_file = output_dir / "labels_qwen3_validated.json" # New output filename

    logger = setup_logger('qwen3_validate_parallel', 'phase2_5_validate_parallel.log')

    logger.info("=" * 80)
    logger.info("PARALLEL QWEN3-30B LABEL VALIDATION (WITH CHECKPOINTS)")
    logger.info(f"Expecting {REQUESTED_GPUS} GPUs (e.g., 2x L40S)")
    logger.info("=" * 80)
    logger.info(f"Input file: {initial_labels_file}")
    logger.info(f"Examples directory: {examples_dir}")
    logger.info(f"Output file: {final_output_file}")
    logger.info(f"Model path: {model_path}")
    logger.info("Checkpoint saves: Every 100 features")
    logger.info("Memory cleanup: Every 50 features")

    # Verify model exists at the new path
    if not Path(model_path).exists():
       logger.error(f"Model not found at {model_path}")
       logger.error("Please ensure the model download completed successfully!")
       return
    logger.info(f"✓ Model path verified: {model_path}")

    # Load initial labels
    if not initial_labels_file.exists():
        logger.error(f"Initial labels file not found: {initial_labels_file}")
        logger.error("Ensure the previous labeling step completed successfully.")
        return
    logger.info(f"\nLoading initial labels from {initial_labels_file}")
    try:
        with open(initial_labels_file, 'r', encoding='utf-8') as f:
            initial_labels = json.load(f)
    except Exception as e:
        logger.error(f"Error loading initial labels: {e}")
        return

    # Filter out any non-dict items just in case
    initial_labels = [item for item in initial_labels if isinstance(item, dict)]
    logger.info(f"Loaded {len(initial_labels)} initial labels to validate")
    if not initial_labels:
        logger.warning("No labels loaded. Nothing to validate.")
        return

    # Load all examples into a map
    if not examples_dir.exists():
        logger.error(f"Examples directory not found: {examples_dir}")
        return
    examples_map = {}

    logger.info("\nLoading examples...")
    example_files_found = list(examples_dir.glob("*_examples.json"))
    if not example_files_found:
         logger.error(f"No example json files found in {examples_dir}")
         return

    for file_path in example_files_found:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                features = json.load(f)
                for feat in features:
                    # Use string keys for JSON compatibility when passing via cmd line
                    if 'feature_id' in feat and 'examples' in feat:
                         examples_map[str(feat['feature_id'])] = feat['examples']
                    else:
                         logger.warning(f"Skipping feature in {file_path} due to missing 'feature_id' or 'examples'")
        except Exception as e:
            logger.error(f"Error loading examples from {file_path}: {e}")
            continue # Skip corrupted files

    if not examples_map:
        logger.error("Failed to load any examples. Cannot proceed.")
        return

    logger.info(f"Loaded examples for {len(examples_map)} features")

    # Determine available GPUs from Slurm environment or default to 0, 1
    assigned_gpu_ids_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if assigned_gpu_ids_str:
        try:
            assigned_gpu_ids = [int(gid.strip()) for gid in assigned_gpu_ids_str.split(',')]
            if len(assigned_gpu_ids) != REQUESTED_GPUS:
                 logger.warning(f"Expected {REQUESTED_GPUS} GPUs but CUDA_VISIBLE_DEVICES={assigned_gpu_ids_str}. Adjusting.")
                 assigned_gpu_ids = assigned_gpu_ids[:REQUESTED_GPUS] # Take what we can get
            logger.info(f"Using GPUs assigned by Slurm: {assigned_gpu_ids}")
        except ValueError:
            logger.error(f"Could not parse CUDA_VISIBLE_DEVICES='{assigned_gpu_ids_str}'. Defaulting to GPUs 0, 1.")
            assigned_gpu_ids = list(range(REQUESTED_GPUS))
    else:
        logger.warning("CUDA_VISIBLE_DEVICES not set. Defaulting to GPUs 0, 1. Ensure Slurm script requests GPUs.")
        assigned_gpu_ids = list(range(REQUESTED_GPUS))

    num_gpus = len(assigned_gpu_ids)
    if num_gpus == 0:
        logger.error("No GPUs available or assigned. Cannot proceed.")
        return

    # Split labels between the *available* GPUs
    labels_per_gpu = len(initial_labels) // num_gpus
    remainder = len(initial_labels) % num_gpus

    label_splits = []
    start_idx = 0
    for i in range(num_gpus):
        gpu_id = assigned_gpu_ids[i] # Use the actual assigned GPU ID
        extra = 1 if i < remainder else 0
        end_idx = start_idx + labels_per_gpu + extra
        # Ensure end_idx does not exceed list bounds
        end_idx = min(end_idx, len(initial_labels))
        if start_idx < end_idx: # Only add if there are labels for this GPU
             label_splits.append((gpu_id, initial_labels[start_idx:end_idx]))
        start_idx = end_idx

    if not label_splits:
        logger.error("Failed to distribute labels among GPUs. No work to do.")
        return

    logger.info("\nWork distribution:")
    total_distributed = 0
    for gpu_id, labels in label_splits:
        logger.info(f"  Worker assigned GPU {gpu_id}: {len(labels)} labels")
        total_distributed += len(labels)
    logger.info(f"Total labels distributed: {total_distributed}")

    # Create worker script
    worker_script_path = Path("/tmp/qwen3_validate_worker_checkpoints.py")
    try:
        with open(worker_script_path, 'w') as f:
            f.write(create_worker_script())
        logger.info(f"Worker script written to {worker_script_path}")
    except Exception as e:
        logger.error(f"Failed to write worker script: {e}")
        return

    logger.info("\n" + "=" * 80)
    logger.info("STARTING PARALLEL VALIDATION")
    logger.info("=" * 80)
    logger.info("Estimated time: 3-5 hours")
    logger.info("")

    # Launch subprocesses with CUDA_VISIBLE_DEVICES set correctly
    processes = []

    # Serialize examples map once
    try:
        examples_map_serial = json.dumps({k: v for k, v in examples_map.items()})
    except Exception as e:
        logger.error(f"Failed to serialize examples map: {e}")
        return

    for assigned_gpu_id, label_list in label_splits:
        output_file = output_dir / f"labels_qwen3_validated_gpu{assigned_gpu_id}.json"

        # Create environment with CUDA_VISIBLE_DEVICES set for this specific worker
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu_id)

        # Prepare arguments, passing serialized data
        try:
             labels_data_serial = json.dumps(label_list)
        except Exception as e:
            logger.error(f"Failed to serialize labels for GPU {assigned_gpu_id}: {e}")
            continue # Skip this worker

        cmd = [
            sys.executable, # Use the same python interpreter
            str(worker_script_path),
            '--gpu_id', str(assigned_gpu_id),
            '--labels_data', labels_data_serial,
            '--examples_map', examples_map_serial,
            '--output_file', str(output_file)
        ]

        # Start subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            processes.append((assigned_gpu_id, proc))
            logger.info(f"  ✓ Started worker for GPU {assigned_gpu_id} (PID: {proc.pid})")
        except Exception as e:
            logger.error(f"Failed to start worker for GPU {assigned_gpu_id}: {e}")
            continue

    if not processes:
        logger.error("No worker processes were started successfully.")
        return

    logger.info("")

    # Monitor processes and stream output
    active_processes = list(processes)
    while active_processes:
        for gpu_id, proc in active_processes[:]:
            if proc.poll() is None:  # Process still running
                try:
                    line = proc.stdout.readline()
                    if line:
                        print(f"[GPU {gpu_id}] {line.rstrip()}")
                    else:
                        time.sleep(0.05)
                except Exception as e:
                    logger.warning(f"Error reading stdout for GPU {gpu_id}: {e}")
            else: # Process finished
                # Process any remaining output
                try:
                    for line in proc.stdout:
                         print(f"[GPU {gpu_id}] {line.rstrip()}")
                except Exception as e:
                     logger.warning(f"Error reading remaining stdout for GPU {gpu_id}: {e}")
                active_processes.remove((gpu_id, proc))

    # Check final return codes
    logger.info("\n" + "=" * 80)
    logger.info("WORKER STATUS")
    logger.info("=" * 80)
    all_success = True
    for gpu_id, proc in processes:
        if proc.returncode != 0:
            logger.error(f"✗ GPU {gpu_id} worker failed with exit code {proc.returncode}")
            all_success = False
        else:
            logger.info(f"✓ GPU {gpu_id} worker completed successfully")

    if not all_success:
        logger.warning("\n⚠️  Some workers failed. Check individual GPU logs for details.")
        logger.warning("Checkpoint files are saved - you can resume by rerunning this script.")

    logger.info("\n" + "=" * 80)
    logger.info("MERGING RESULTS")
    logger.info("=" * 80)

    # Merge results from all GPUs
    all_validated = []
    for gpu_id, _ in label_splits:
        gpu_output_file = output_dir / f"labels_qwen3_validated_gpu{gpu_id}.json"
        if gpu_output_file.exists():
            try:
                with open(gpu_output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_validated.extend(results)
                        logger.info(f"  GPU {gpu_id}: Merged {len(results)} features")
                    else:
                        logger.error(f"  GPU {gpu_id}: Invalid format in {gpu_output_file}")
            except Exception as e:
                 logger.error(f"  GPU {gpu_id}: Error reading {gpu_output_file}: {e}")
        else:
             worker_failed = any(p.returncode != 0 for gid, p in processes if gid == gpu_id)
             if worker_failed:
                  logger.warning(f"  GPU {gpu_id}: output file not found, likely due to worker failure")
             else:
                  logger.error(f"  GPU {gpu_id}: output file not found but worker seemed successful?")

    if not all_validated:
        logger.error("\nNo results were successfully merged. Final output file will be empty.")
        try:
            with open(final_output_file, 'w', encoding='utf-8') as f:
                 json.dump([], f)
            logger.info(f"Created empty results file: {final_output_file}")
        except Exception as e:
            logger.error(f"Failed to create empty results file: {e}")
        return

    # Save merged results
    try:
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_validated, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✓ Merged results saved to {final_output_file}")
    except Exception as e:
        logger.error(f"Failed to save final merged results: {e}")
        return

    # Compute statistics
    stats = {
        'KEEP': 0, 'REVISE': 0, 'INVALIDATE': 0, 'ERROR': 0
    }
    
    for v in all_validated:
        action = v.get('validation_action', 'ERROR')
        if action in stats:
            stats[action] += 1
        else:
            logger.warning(f"Unknown validation action '{action}' found in results.")
            stats['ERROR'] += 1

    total = len(all_validated)
    if total == 0:
        logger.warning("Total validated features is zero after merge.")
        return

    final_valid = stats['KEEP'] + stats['REVISE']

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total features processed: {total}")
    logger.info(f"KEEP:       {stats['KEEP']:>6} ({stats['KEEP']/total*100:>5.1f}%)")
    logger.info(f"REVISE:     {stats['REVISE']:>6} ({stats['REVISE']/total*100:>5.1f}%)")
    logger.info(f"INVALIDATE: {stats['INVALIDATE']:>6} ({stats['INVALIDATE']/total*100:>5.1f}%)")
    logger.info(f"ERROR:      {stats['ERROR']:>6} ({stats['ERROR']/total*100:>5.1f}%)")
    logger.info("-" * 40)
    logger.info(f"Final Valid (Keep+Revise): {final_valid:>6} ({final_valid/total*100:>5.1f}%)")

    # GPU distribution
    logger.info("\nProcessing distribution:")
    gpu_counts = {}
    for v in all_validated:
        gid = v.get('validated_by_gpu', 'Unknown')
        gpu_counts[gid] = gpu_counts.get(gid, 0) + 1

    for gpu_id in assigned_gpu_ids:
        count = gpu_counts.get(gpu_id, 0)
        logger.info(f"  GPU {gpu_id}: {count:>6} features")
    if 'Unknown' in gpu_counts:
         logger.warning(f"  Unknown GPU: {gpu_counts['Unknown']:>6} features")

    logger.info("=" * 80)
    logger.info("✓ PHASE 2.5 VALIDATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
