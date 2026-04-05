"""
Step 1: Behavioral Evaluation + Activation Extraction
=====================================================
Runs LLaMA-3.1-8B base and instruct on all 21,726 Sanskriti questions.
Single forward pass per question: logit-based answer extraction + activation hooks.

Usage:
    python eval_step1.py                    # runs both models in parallel (2 GPUs)
    python eval_step1.py --model base       # run only base model
    python eval_step1.py --debug            # first 100 questions only
"""

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Logging
# ============================================================

def setup_logger(model_type=None):
    """Setup file + console logger. Each model type gets its own log file."""
    log_dir = "/home/anshulk/cultural-mi/logs"
    os.makedirs(log_dir, exist_ok=True)

    suffix = f"_{model_type}" if model_type else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"step1{suffix}_{timestamp}.log")

    logger = logging.getLogger(f"step1{suffix}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — everything
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================
# Config
# ============================================================

def load_config():
    with open("/home/anshulk/cultural-mi/configs/config.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# Constants
# ============================================================

FIVE_SHOT_PREFIX = """\
The following are multiple choice questions about Indian culture.

Question: What is the national animal of India?
A) Tiger
B) Lion
C) Elephant
D) Peacock
Answer: A

Question: Which city is the capital of India?
A) Mumbai
B) Kolkata
C) New Delhi
D) Chennai
Answer: C

Question: What is the most widely spoken language in India?
A) Tamil
B) Hindi
C) Telugu
D) Bengali
Answer: B

Question: Which festival is known as the festival of lights in India?
A) Holi
B) Eid
C) Diwali
D) Pongal
Answer: C

Question: What is the national sport of India?
A) Cricket
B) Hockey
C) Kabaddi
D) Badminton
Answer: B

"""

INSTRUCT_SYSTEM_MSG = (
    "You are a helpful assistant. Answer the following multiple choice question "
    "about Indian culture by responding with only the letter of the correct answer "
    "(A, B, C, or D). Do not explain your answer."
)

HOOK_LAYER_INDICES = [4, 8, 14, 20, 26, 30, 31]
HOOK_NAMES = ["embed"] + [f"layer_{i:02d}" for i in HOOK_LAYER_INDICES]

CSV_FIELDS = [
    "question_id", "ground_truth_letter", "predicted_letter", "correct",
    "logprob_A", "logprob_B", "logprob_C", "logprob_D",
    "forced_choice", "top1_token_id",
]


# ============================================================
# Dataset Loading
# ============================================================

def load_sanskriti(config):
    """Load Sanskriti, derive ground truth letters, exclude 127 broken rows."""
    ds = load_dataset(
        "13ari/Sanskriti", split="train",
        cache_dir=config["dataset"]["local_dir"],
    )
    df = ds.to_pandas()

    option_cols = ["option1", "option2", "option3", "option4"]
    letters = ["A", "B", "C", "D"]

    def get_gt_letter(row):
        ans = str(row["answer"]).strip().lower()
        for col, letter in zip(option_cols, letters):
            if str(row[col]).strip().lower() == ans:
                return letter
        return None

    df["ground_truth_letter"] = df.apply(get_gt_letter, axis=1)
    usable = df[df["ground_truth_letter"].notna()].reset_index(drop=True)
    # question_id assigned AFTER filtering — 0 to 21725 with no gaps
    # This ensures question_id == positional index == activation array row index
    usable["question_id"] = range(len(usable))

    # Attach entity_key from EDA output (if available) for merge_step1.py
    eda_usable_path = "/data/user_data/anshulk/cultural-mi/analysis/sanskriti_usable.csv"
    if os.path.exists(eda_usable_path):
        eda = pd.read_csv(eda_usable_path)
        if "entity_key" in eda.columns:
            # Join on question content (not positional index) to handle ordering differences
            join_cols = ["question", "state", "attribute"]
            usable = usable.merge(
                eda[join_cols + ["entity_key"]].drop_duplicates(subset=join_cols),
                on=join_cols, how="left",
            )
            n_matched = usable["entity_key"].notna().sum()
            n_total = len(usable)
            if n_matched < n_total:
                print(f"Warning: entity_key matched {n_matched}/{n_total} questions")

    return usable


# ============================================================
# Prompt Formatting
# ============================================================

def format_base_prompt(row):
    question_block = (
        f"Question: {row['question']}\n"
        f"A) {row['option1']}\n"
        f"B) {row['option2']}\n"
        f"C) {row['option3']}\n"
        f"D) {row['option4']}\n"
        f"Answer:"
    )
    return FIVE_SHOT_PREFIX + question_block


def format_instruct_prompt(row, tokenizer):
    user_msg = (
        f"Question: {row['question']}\n\n"
        f"A) {row['option1']}\n"
        f"B) {row['option2']}\n"
        f"C) {row['option3']}\n"
        f"D) {row['option4']}"
    )
    messages = [
        {"role": "system", "content": INSTRUCT_SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# Activation Hooks
# ============================================================

def setup_hooks(model):
    """Register forward hooks at embed_tokens + 7 transformer layers.
    Returns hooks list and activation cache dict."""
    cache = {}
    hooks = []

    # Embedding layer (true input embeddings, pre-transformer)
    def embed_hook(module, input, output):
        cache["embed"] = output.detach().float().cpu()

    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # Transformer layers
    for layer_idx in HOOK_LAYER_INDICES:
        name = f"layer_{layer_idx:02d}"

        def make_hook(n):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                cache[n] = hidden.detach().float().cpu()
            return hook_fn

        hooks.append(
            model.model.layers[layer_idx].register_forward_hook(make_hook(name))
        )

    return hooks, cache


def compute_pooled(cache, attention_mask, target_start_tokens=None):
    """Compute mean-pooled and last-token representations from hook cache.

    Args:
        cache: dict of hook_name -> (batch, seq, 4096) float32 tensors
        attention_mask: (batch, seq) tensor
        target_start_tokens: optional (batch,) tensor for base model —
            masks out padding + 5-shot prefix tokens per example
    """
    results = {}
    for name, hidden in cache.items():
        mask = attention_mask.clone().float()

        if target_start_tokens is not None:
            for i in range(mask.size(0)):
                mask[i, :target_start_tokens[i]] = 0.0

        # Mean pool over unmasked positions
        mask_3d = mask.unsqueeze(-1)  # (batch, seq, 1)
        h_mean = (hidden * mask_3d).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Last non-padding token
        last_pos = attention_mask.sum(dim=1) - 1
        h_last = hidden[torch.arange(hidden.size(0)), last_pos]

        results[name] = {"mean_pool": h_mean, "last_token": h_last}

    return results


# ============================================================
# Logit-based Answer Extraction
# ============================================================

def extract_from_logits(logits, attention_mask, answer_token_ids):
    """Extract predicted letter and logprobs from last-position logits.

    Returns:
        predicted_letters: list[str]
        answer_log_probs: (batch, 4) tensor
        forced_flags: list[bool]
        top1_ids: (batch,) tensor
    """
    seq_lengths = attention_mask.sum(dim=1) - 1
    last_logits = logits[torch.arange(len(seq_lengths)), seq_lengths]  # (batch, vocab)

    ids = torch.tensor(
        [answer_token_ids[l] for l in "ABCD"], device=last_logits.device
    )
    answer_logits = last_logits[:, ids]  # (batch, 4)

    # Log probs over full vocabulary, then slice A/B/C/D
    log_probs = torch.log_softmax(last_logits, dim=-1)
    answer_log_probs = log_probs[:, ids]  # (batch, 4)

    predicted_idx = answer_logits.argmax(dim=-1)
    letters = ["A", "B", "C", "D"]
    predicted_letters = [letters[i] for i in predicted_idx]

    top1_ids = last_logits.argmax(dim=-1)
    ids_set = set(ids.tolist())
    forced = [top1_ids[i].item() not in ids_set for i in range(len(top1_ids))]

    return predicted_letters, answer_log_probs.cpu(), forced, top1_ids.cpu()


# ============================================================
# Checkpoint Helpers
# ============================================================

def get_resume_info(config, model_type, batch_size):
    """Derive resume point from CSV row count (not meta.json) to avoid double-writes.

    Returns (start_batch, n_completed_rows). Also truncates the CSV to a clean
    batch boundary so partially-written batches get overwritten cleanly.
    """
    csv_path = os.path.join(config["checkpoints"], f"step1_{model_type}_results_partial.csv")
    if not os.path.exists(csv_path):
        return 0, 0

    # Count existing data rows (subtract header)
    with open(csv_path) as f:
        n_rows = sum(1 for _ in f) - 1
    if n_rows <= 0:
        return 0, 0

    # Truncate to clean batch boundary
    clean_rows = (n_rows // batch_size) * batch_size
    start_batch = clean_rows // batch_size

    if clean_rows < n_rows:
        # Read header + clean_rows, rewrite file
        df_partial = pd.read_csv(csv_path, nrows=clean_rows)
        df_partial.to_csv(csv_path, index=False)

    return start_batch, clean_rows


# ============================================================
# Main Evaluation Loop
# ============================================================

def run_evaluation(model_type, device, df, config, batch_size, debug, preempt_event):
    """Run full evaluation for one model type on one GPU."""
    log = setup_logger(model_type)

    # Ensure output directories exist
    os.makedirs(config["checkpoints"], exist_ok=True)
    os.makedirs(config["results"]["step1"], exist_ok=True)

    log.info(f"Loading model from {config['models'][model_type]['local_dir']}")
    log.info(f"Device: {device}, dtype: bfloat16")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config["models"][model_type]["local_dir"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    log.debug(f"Model config: {model.config.num_hidden_layers} layers, "
              f"hidden={model.config.hidden_size}, vocab={model.config.vocab_size}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"][model_type]["local_dir"]
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    log.debug(f"Tokenizer: pad_token={tokenizer.pad_token}, "
              f"pad_token_id={tokenizer.pad_token_id}, padding_side=left")

    # Get A/B/C/D token IDs — space-prefixed because prompt ends with "Answer:"
    # and the model predicts " A", " B", " C", " D" (with leading space)
    answer_token_ids = {
        l: tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in "ABCD"
    }
    log.info(f"Answer token IDs: {answer_token_ids}")

    # Five-shot prefix token length (constant)
    five_shot_token_len = len(tokenizer.encode(FIVE_SHOT_PREFIX, add_special_tokens=False))
    log.info(f"Five-shot prefix: {five_shot_token_len} tokens")

    n = len(df) if not debug else min(100, len(df))
    log.info(f"Questions to process: {n} (debug={debug})")

    # Verify prompt formatting on first question
    sample_row = df.iloc[0]
    if model_type == "base":
        sample_prompt = format_base_prompt(sample_row)
    else:
        sample_prompt = format_instruct_prompt(sample_row, tokenizer)
    sample_tokens = tokenizer.encode(sample_prompt, add_special_tokens=(model_type == "base"))
    log.debug(f"Sample prompt ({len(sample_tokens)} tokens):\n{sample_prompt[:500]}")

    # Setup hooks
    hooks, activation_cache = setup_hooks(model)
    log.info(f"Registered {len(hooks)} hooks: {HOOK_NAMES}")

    # Pre-allocate or re-open activation arrays
    act_base_dir = os.path.join(config["activations"], model_type)
    act_arrays = {}
    new_arrays = 0
    resumed_arrays = 0
    for hname in HOOK_NAMES:
        for ptype in ["mean_pool", "last_token"]:
            d = os.path.join(act_base_dir, ptype)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{hname}.npy")
            if os.path.exists(path):
                mode = "r+"
                resumed_arrays += 1
            else:
                mode = "w+"
                new_arrays += 1
            act_arrays[(hname, ptype)] = np.lib.format.open_memmap(
                path, mode=mode, dtype=np.float32, shape=(n, 4096)
            )
    log.info(f"Activation arrays: {new_arrays} new, {resumed_arrays} resumed "
             f"({len(act_arrays)} total, {n}x4096 float32 each)")
    log.info(f"Activation dir: {act_base_dir}")

    # Resume logic — derive from CSV row count (not meta.json) to avoid double-writes
    csv_path = os.path.join(config["checkpoints"], f"step1_{model_type}_results_partial.csv")
    start_batch, n_completed = get_resume_info(config, model_type, batch_size)

    # CSV output
    write_header = (start_batch == 0) or not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    log.debug(f"CSV output: {csv_path} (header={'written' if write_header else 'skipped'})")

    # Counters
    correct_count = 0
    forced_count = 0
    total_done = 0
    start_time = time.time()

    # If resuming, restore counters from truncated CSV
    if start_batch > 0:
        total_batches_est = (n + batch_size - 1) // batch_size
        log.info(f"Resuming from batch {start_batch} (of {total_batches_est}), "
                 f"{n_completed} rows already completed")
        try:
            partial = pd.read_csv(csv_path)
            correct_count = int(partial["correct"].sum())
            forced_count = int(partial["forced_choice"].sum())
            total_done = len(partial)
            log.info(f"Restored counters: {total_done} done, "
                     f"acc={correct_count/max(total_done,1)*100:.1f}%, "
                     f"forced={forced_count}")
        except Exception as e:
            log.warning(f"Could not restore counters from partial CSV: {e}")
    else:
        log.info("Starting fresh (no checkpoint found)")

    total_batches = (n + batch_size - 1) // batch_size
    log.info(f"Batch size: {batch_size}, total batches: {total_batches}")

    # GPU memory info
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        log.info(f"GPU: {mem.name}, {mem.total_memory / 1e9:.1f}GB total, "
                 f"{allocated:.1f}GB allocated after model load")

    for batch_idx in range(total_batches):
        if batch_idx < start_batch:
            continue

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n)
        batch = df.iloc[batch_start:batch_end]
        actual_bs = len(batch)

        # Format prompts
        if model_type == "base":
            prompts = [format_base_prompt(row) for _, row in batch.iterrows()]
        else:
            prompts = [format_instruct_prompt(row, tokenizer) for _, row in batch.iterrows()]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=(model_type == "base"),
        ).to(device)

        seq_len = inputs["input_ids"].shape[1]

        # Log first batch details for debugging
        if batch_idx == start_batch:
            log.debug(f"First batch: input_ids shape={inputs['input_ids'].shape}, "
                      f"seq_len={seq_len}")
            log.debug(f"First prompt tokens (first 20): "
                      f"{inputs['input_ids'][0, :20].tolist()}")
            log.debug(f"Attention mask sum (first example): "
                      f"{inputs['attention_mask'][0].sum().item()}")

        # Single forward pass — hooks fire once
        with torch.no_grad():
            outputs = model(**inputs)

        # Verify hooks captured data (first batch only)
        if batch_idx == start_batch:
            for hname in HOOK_NAMES:
                if hname in activation_cache:
                    h = activation_cache[hname]
                    log.debug(f"Hook '{hname}': shape={h.shape}, "
                              f"dtype={h.dtype}, "
                              f"norm={h.float().norm(dim=-1).mean():.2f}")
                else:
                    log.error(f"Hook '{hname}' NOT in cache — check hook setup!")

        # Extract answers from logits
        pred_letters, answer_lp, forced, top1 = extract_from_logits(
            outputs.logits, inputs["attention_mask"], answer_token_ids
        )

        # Log forced-choice details for debugging
        if any(forced):
            for i, is_forced in enumerate(forced):
                if is_forced:
                    qid = batch.iloc[i]["question_id"]
                    top1_tok = tokenizer.decode([top1[i].item()])
                    log.debug(
                        f"Forced choice: qid={qid}, "
                        f"top1='{top1_tok}' (id={top1[i].item()}), "
                        f"pred={pred_letters[i]}, "
                        f"gt={batch.iloc[i]['ground_truth_letter']}, "
                        f"logprobs=[A={answer_lp[i,0]:.3f}, B={answer_lp[i,1]:.3f}, "
                        f"C={answer_lp[i,2]:.3f}, D={answer_lp[i,3]:.3f}]"
                    )

        # Compute pooled activations
        if model_type == "base":
            example_lengths = inputs["attention_mask"].sum(dim=1)
            max_len = inputs["attention_mask"].shape[1]
            padding_lengths = max_len - example_lengths
            target_starts = (padding_lengths + 1 + five_shot_token_len).cpu()
            pooled = compute_pooled(
                activation_cache, inputs["attention_mask"].cpu(), target_starts
            )
            if batch_idx == start_batch:
                log.debug(f"Base model target_starts (first 3): {target_starts[:3].tolist()}")
                log.debug(f"Pooling mask tokens per example (first 3): "
                          f"{[(example_lengths[i] - 1 - five_shot_token_len).item() for i in range(min(3, actual_bs))]}")
        else:
            pooled = compute_pooled(activation_cache, inputs["attention_mask"].cpu())

        # Write activations to pre-allocated arrays
        for hname in HOOK_NAMES:
            for ptype in ["mean_pool", "last_token"]:
                act_arrays[(hname, ptype)][batch_start:batch_start + actual_bs] = (
                    pooled[hname][ptype].numpy()
                )

        # Write results to CSV
        for i, (_, row) in enumerate(batch.iterrows()):
            is_correct = 1 if pred_letters[i] == row["ground_truth_letter"] else 0
            writer.writerow({
                "question_id": row["question_id"],
                "ground_truth_letter": row["ground_truth_letter"],
                "predicted_letter": pred_letters[i],
                "correct": is_correct,
                "logprob_A": f"{answer_lp[i, 0].item():.6f}",
                "logprob_B": f"{answer_lp[i, 1].item():.6f}",
                "logprob_C": f"{answer_lp[i, 2].item():.6f}",
                "logprob_D": f"{answer_lp[i, 3].item():.6f}",
                "forced_choice": int(forced[i]),
                "top1_token_id": top1[i].item(),
            })
            correct_count += is_correct
            forced_count += forced[i]
            total_done += 1

        # Clear hook cache
        activation_cache.clear()

        # Checkpoint
        if preempt_event.is_set() or (batch_idx + 1) % 100 == 0:
            csv_file.flush()
            # (resume derived from CSV row count, no meta.json needed)
            for arr in act_arrays.values():
                arr.flush()
            if (batch_idx + 1) % 100 == 0:
                log.info(f"Checkpoint saved at batch {batch_idx + 1}/{total_batches}")
            if preempt_event.is_set():
                log.warning(f"PREEMPTION detected at batch {batch_idx}. "
                            f"Checkpointed {total_done} questions. Exiting.")
                csv_file.close()
                return

        # Progress logging (every 10 batches)
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = n - total_done
            eta = remaining / rate if rate > 0 else 0
            gpu_mem = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            log.info(
                f"{total_done}/{n} ({total_done/n*100:.1f}%) | "
                f"{elapsed:.0f}s | {rate:.1f} q/s | ETA {eta:.0f}s | "
                f"acc: {correct_count/max(total_done,1)*100:.1f}% | "
                f"forced: {forced_count}/{total_done} | "
                f"GPU mem: {gpu_mem:.1f}GB | seq_len: {seq_len}"
            )

    # Final flush
    csv_file.close()
    # (no meta.json needed — resume derives from CSV row count)
    for arr in act_arrays.values():
        arr.flush()

    # Remove hooks
    for h in hooks:
        h.remove()

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info(f"COMPLETED: {total_done} questions in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Accuracy: {correct_count}/{total_done} ({correct_count/max(total_done,1)*100:.2f}%)")
    log.info(f"Forced choice: {forced_count}/{total_done} ({forced_count/max(total_done,1)*100:.2f}%)")
    log.info(f"Throughput: {total_done/elapsed:.1f} questions/sec")
    log.info("=" * 60)

    # Verify activation files
    for hname in HOOK_NAMES:
        for ptype in ["mean_pool", "last_token"]:
            arr = act_arrays[(hname, ptype)]
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            if has_nan or has_inf:
                log.error(f"ACTIVATION CHECK FAILED: {hname}/{ptype} has NaN={has_nan}, Inf={has_inf}")
            else:
                log.debug(f"Activation OK: {hname}/{ptype} shape={arr.shape}, "
                          f"mean_norm={np.linalg.norm(arr, axis=1).mean():.2f}")

    # Rename partial CSV to final
    final_csv = os.path.join(
        config["results"]["step1"], f"{model_type}_results.csv"
    )
    partial_csv = os.path.join(
        config["checkpoints"], f"step1_{model_type}_results_partial.csv"
    )
    shutil.move(partial_csv, final_csv)
    log.info(f"Results saved to {final_csv}")

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    log.info("Model unloaded, GPU cache cleared.")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Behavioral Evaluation")
    parser.add_argument("--model", choices=["base", "instruct"], default=None,
                        help="Run only one model (default: both in parallel)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16, reduce to 8 if OOM)")
    parser.add_argument("--debug", action="store_true",
                        help="Run on first 100 questions only")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    log = setup_logger("main")

    log.info("=" * 60)
    log.info("Step 1: Behavioral Evaluation + Activation Extraction")
    log.info("=" * 60)
    log.info(f"Config loaded from {os.path.abspath('/home/anshulk/cultural-mi/configs/config.yaml')}")
    log.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    log.info(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'N/A')}")
    log.info(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log.info(f"  cuda:{i} = {props.name}, {props.total_memory / 1e9:.1f}GB")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"Args: model={args.model}, batch_size={args.batch_size}, debug={args.debug}")

    df = load_sanskriti(config)
    log.info(f"Dataset loaded: {len(df)} usable questions")
    has_ek = "entity_key" in df.columns
    log.info(f"Entity keys: {'attached' if has_ek else 'not available'} "
             f"({'%d matched' % df['entity_key'].notna().sum() if has_ek else 'N/A'})")

    # Save prepared CSV for merge_step1.py
    prep_path = os.path.join(config["results"]["step1"], "sanskriti_prepared.csv")
    os.makedirs(config["results"]["step1"], exist_ok=True)
    df.to_csv(prep_path, index=False)
    log.info(f"Prepared dataset saved: {prep_path}")

    if args.model:
        # Single model mode — run directly (for testing)
        preempt_event = mp.Event()
        signal.signal(signal.SIGUSR1, lambda s, f: preempt_event.set())
        device = "cuda:0"
        log.info(f"Single model mode: running {args.model} on {device}")
        run_evaluation(args.model, device, df, config, args.batch_size, args.debug, preempt_event)
    else:
        # Dual model mode — spawn two processes
        mp.set_start_method("spawn", force=True)
        preempt_event = mp.Event()
        signal.signal(signal.SIGUSR1, lambda s, f: preempt_event.set())

        log.info("Dual model mode: spawning base (cuda:0) and instruct (cuda:1)")

        p_base = mp.Process(
            target=run_evaluation,
            args=("base", "cuda:0", df, config, args.batch_size, args.debug, preempt_event),
        )
        p_inst = mp.Process(
            target=run_evaluation,
            args=("instruct", "cuda:1", df, config, args.batch_size, args.debug, preempt_event),
        )

        p_base.start()
        p_inst.start()
        log.info(f"Processes started: base pid={p_base.pid}, instruct pid={p_inst.pid}")

        p_base.join()
        p_inst.join()
        log.info(f"Processes finished: base exit={p_base.exitcode}, instruct exit={p_inst.exitcode}")

        if preempt_event.is_set():
            job_id = os.environ.get("SLURM_JOB_ID", "")
            if job_id:
                log.warning(f"Preemption detected. Requeuing job {job_id}...")
                os.system(f"scontrol requeue {job_id}")
            else:
                log.warning("Preemption detected but no SLURM_JOB_ID found.")
        else:
            log.info("Both models completed successfully.")


if __name__ == "__main__":
    main()
