# Step 1: Behavioral Evaluation — Final Validated Plan

**Status:** Validated, ready for implementation
**Date:** 2026-03-21

---

## Goal

Run both LLaMA-3.1-8B models (base and instruct) on all 21,853 Sanskriti questions. Record per-question correctness. Assign behavioral labels: **suppression**, **enhancement**, **control_both_correct**, **control_both_wrong**.

---

## What Was Validated

| Claim | Status | Notes |
|-------|--------|-------|
| Sanskriti has 21,853 rows | Confirmed | Single `train` split |
| 10 columns including answer as full text | Confirmed | Column `short explaination / source link` has a typo — must match exactly |
| 36 states, 16 attributes, 4 question types | Confirmed | |
| LLaMA-3.1-8B: 32 layers, 4096 hidden, BF16 | Confirmed | Both base and instruct identical architecture |
| ~16GB VRAM per model in BF16 | Confirmed | |
| Instruct model has built-in chat template | Confirmed | `tokenizer.apply_chat_template()` works out of the box |
| Instruct model has 3 EOS token IDs | Confirmed | `[128001, 128008, 128009]` — important for generation stopping |
| HuggingFace license accepted | Confirmed | Token: in config.yaml |
| SLURM cluster has L40S/A100/A6000 GPUs | Confirmed | L40S (48GB) most widely available |
| LLaMA-3.1-70B-Instruct scored 0.86 on Sanskriti | Confirmed | From ACL 2025 paper (arxiv:2506.15355) |
| LLaMA-3.2-3B-Instruct scored 0.52 on Sanskriti | Confirmed | Our 8B should fall between these |

---

## Behavioral Labeling Logic

For each of the 21,853 questions:

```
base_correct    instruct_correct    label
─────────────   ────────────────    ─────────────────────
     1                0             suppression
     0                1             enhancement
     1                1             control_both_correct
     0                0             control_both_wrong
```

**Suppression** = RLHF broke knowledge the base model had.
**Enhancement** = RLHF added knowledge the base model lacked.

---

## Prompt Design (Validated)

### Base Model — Completion Style

The base model was never trained on chat formats. It only understands text completion. Using chat tokens on it will produce garbage.

```
The following is a multiple choice question about Indian culture.

Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Answer:
```

The model completes after `"Answer: "` with a single token.

### Instruct Model — Chat Template

The instruct model was trained with the Llama 3.1 chat format. Use `tokenizer.apply_chat_template()` which inserts all special tokens automatically.

**System message:**
```
You are a helpful assistant. Answer the following multiple choice question about Indian culture by responding with only the letter of the correct answer (A, B, C, or D). Do not explain your answer.
```

**User message:**
```
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}
```

### Why Different Prompts

This is the most critical design decision. The base model treats `<|start_header_id|>` as literal text and gets confused. The instruct model without its chat template underperforms because it expects the structure it was trained on. Each model must receive the format it was trained with.

---

## Decoding Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0 | Greedy decoding, deterministic output |
| max_new_tokens | 5 | Only need 1 letter, cap prevents runaway |
| do_sample | False | Must be False for greedy |

No sampling. Labels must be stable across runs.

---

## Answer Extraction

1. Get model output string
2. Strip whitespace, take first character
3. If in `{A, B, C, D}` → that's the prediction
4. Otherwise → record as `null`

**Ground truth derivation:**
- Compare `answer` column text against `option1`, `option2`, `option3`, `option4`
- Use `.strip().lower()` before comparing (dataset has whitespace/casing inconsistencies)
- Matching option → ground truth letter (A=option1, B=option2, C=option3, D=option4)
- If no match after stripping → flag the row for manual inspection

---

## Detailed TODO

### Phase A: Environment & Data Setup

```
A1. Create conda env `cultural` from configs/environment.yml
A2. Activate env, verify: torch, transformers, datasets, accelerate all import
A3. Login to HuggingFace: huggingface-cli login --token <token>
A4. Download Sanskriti dataset: load_dataset("13ari/Sanskriti", cache_dir=<dataset_dir>)
A5. Inspect: confirm 21,853 rows, print column names, check dtypes
A6. Check for missing values in question, option1-4, answer columns
A7. Build ground_truth_letter column:
    - Strip whitespace, lowercase both answer and option1-4
    - Match answer to one of the four options
    - Record which option matched as A/B/C/D
A8. Count rows where answer matches no option → clean or flag these
A9. Assign question_id (0 to 21852) as stable index
A10. Save prepared dataset as sanskriti_prepared.csv to results/step1/
```

### Phase B: Download Models

```
B1. Download Llama-3.1-8B to /data/.../models/base/
    Use: snapshot_download("meta-llama/Llama-3.1-8B", local_dir=..., token=...)
B2. Download Llama-3.1-8B-Instruct to /data/.../models/instruct/
B3. Verify both directories contain config.json + safetensors files
B4. Check total disk usage (~16GB each, ~32GB total)
```

### Phase C: Base Model Evaluation

```
C1. Write evaluation script: scripts/eval_base.py
C2. Load model from local_dir in BF16, device_map="auto"
C3. Load tokenizer from same path
C4. Run 1 test question manually, print raw output, verify extraction works
C5. Format all 21,853 prompts using completion template
C6. Run batched inference (batch_size from config, default 16)
    - Pad from left (decoder-only model)
    - Decode only the newly generated tokens
C7. After each batch of 500 questions, save checkpoint CSV:
    checkpoints/base_checkpoint_{question_id}.csv
C8. Extract prediction letter from each output
C9. Compare against ground_truth_letter → base_correct (0/1)
C10. Log null/invalid prediction count
C11. Save full results: results/step1/base_results.csv
     Columns: question_id, ground_truth_letter, raw_output_base, predicted_base, base_correct
C12. Unload model from GPU: del model; torch.cuda.empty_cache()
```

### Phase D: Instruct Model Evaluation

```
D1. Write evaluation script: scripts/eval_instruct.py
    (or combine with base script using a flag)
D2. Load instruct model from local_dir in BF16, device_map="auto"
D3. Load tokenizer, verify tokenizer.chat_template is not None
D4. Run 1 test question with apply_chat_template, print formatted prompt, verify
D5. Format all 21,853 prompts using apply_chat_template:
    messages = [
      {"role": "system", "content": <system_msg>},
      {"role": "user", "content": <user_msg>}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
D6. Run batched inference with same decoding settings
    - Same left-padding, same batch size
    - IMPORTANT: set add_special_tokens=False when tokenizing the already-templated string
      to avoid duplicating <|begin_of_text|>
D7. Checkpoint every 500 questions
D8. Extract predictions, compare against ground truth → instruct_correct
D9. Log null/invalid count
D10. Save: results/step1/instruct_results.csv
     Columns: question_id, ground_truth_letter, raw_output_instruct, predicted_instruct, instruct_correct
```

### Phase E: Merge & Label

```
E1. Load base_results.csv and instruct_results.csv
E2. Join on question_id
E3. Assign behavioral_label using the 2x2 table
E4. Merge back all original Sanskriti columns (state, attribute, question_type, etc.)
E5. Save master CSV: results/step1/sanskriti_behavioral_labels.csv
    Columns: question_id, state, attribute, question, option1-4, answer,
             question_type, ground_truth_letter, predicted_base, predicted_instruct,
             base_correct, instruct_correct, behavioral_label
E6. Print group counts:
    - suppression: N (X%)
    - enhancement: N (X%)
    - control_both_correct: N (X%)
    - control_both_wrong: N (X%)
E7. Print null rates for both models
```

### Phase F: Sanity Checks

These must all pass before proceeding to Step 2.

```
F1. Overall accuracy check
    - Base accuracy should be 40-75%
    - If <30% or >80%: prompt or extraction is broken
    - Reference: Llama-3.2-3B-Instruct scored 52%, Llama-3.1-70B-Instruct scored 86%
    - Expected 8B base: roughly 45-65%

F2. Instruct > Base overall
    - If instruct scores lower than base, the chat template or prompt is wrong
    - This is the single strongest signal of a prompt format bug

F3. Option distribution
    - Count predictions of A, B, C, D for each model
    - Each letter should be 15-35% of predictions
    - If any letter >50%: position bias → note as limitation
    - Save as: results/step1/step1_option_distribution.csv

F4. Question type breakdown
    - Compute accuracy per question_type for both models
    - Expected ordering (easiest → hardest):
      General Awareness > Country Prediction > Association > State Prediction
    - If this ranking is inverted, something is off

F5. State & attribute breakdown
    - Compute suppression rate per state and per attribute
    - No single state should account for >15% of all suppression cases
    - If it does: data imbalance, not a real cultural effect

F6. Suppression-enhancement balance
    - Report both rates
    - Previous work (Qwen2-1.5B): suppression ~8%, enhancement ~7%
    - LLaMA 8B may differ — report whatever the data shows

F7. Null prediction audit
    - If null rate >2% for either model: investigate prompt
    - Manually inspect 10-20 null examples
    - Common cause: model outputting full explanation instead of letter
    - Fix: shorten system prompt to "Respond with only the letter. Nothing else."

F8. Save aggregate stats: results/step1/step1_aggregate_stats.json
    Contents:
    {
      "base_accuracy": float,
      "instruct_accuracy": float,
      "group_counts": {"suppression": N, "enhancement": N, ...},
      "null_rate_base": float,
      "null_rate_instruct": float,
      "accuracy_by_question_type": {...},
      "accuracy_by_attribute_top5": {...},
      "accuracy_by_attribute_bottom5": {...},
      "accuracy_by_state_top5": {...},
      "accuracy_by_state_bottom5": {...}
    }
```

### Phase G: SLURM Job Script

```
G1. Write scripts/run_step1.sh (SLURM batch script)
    - Partition: array (or debug for testing)
    - GPUs: 1x L40S (48GB) — sufficient for 8B BF16 + batch inference
    - Time: 4-6 hours estimated for 21,853 questions x 2 models
    - Memory: 64GB RAM
    - Activate cultural conda env
    - Run eval_base.py then eval_instruct.py then merge script
G2. Test on debug partition with 100 questions first
G3. Run full evaluation on array partition
```

---

## Output Artifacts

| File | Location | Description |
|------|----------|-------------|
| `sanskriti_behavioral_labels.csv` | `results/step1/` | Master CSV, 21,853 rows, all columns + labels |
| `base_results.csv` | `results/step1/` | Raw base model predictions |
| `instruct_results.csv` | `results/step1/` | Raw instruct model predictions |
| `step1_aggregate_stats.json` | `results/step1/` | Accuracy, group counts, breakdowns |
| `step1_option_distribution.csv` | `results/step1/` | A/B/C/D prediction counts per model |
| `step1_run_log.txt` | `logs/` | Timestamps, GPU, errors |
| `sanskriti_prepared.csv` | `results/step1/` | Dataset with question_id and ground_truth_letter |

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong prompt format for base model | Never use chat template on base; test 1 question first |
| Answer text mismatch | Strip + lowercase before matching; flag unmatched rows |
| Model outputs explanation instead of letter | Short system prompt; check null rate; fallback to stricter prompt |
| OOM on GPU | Start with batch_size=8; L40S has 48GB, plenty for 16GB model |
| Run crashes mid-evaluation | Checkpoint every 500 questions; resume from last checkpoint |
| Position bias in MCQ | Check option distribution; note as limitation if skewed |
| Instruct tokenizer duplicates BOS | Use `add_special_tokens=False` after `apply_chat_template` |

---

## Expected Results (Rough Ranges)

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Base accuracy | 45-65% | Interpolated: 3B-Instruct=52%, 70B-Instruct=86% |
| Instruct accuracy | 55-75% | Should be higher than base |
| Suppression rate | 5-12% | Prior work: ~8% on Qwen2-1.5B |
| Enhancement rate | 4-10% | Prior work: ~7% on Qwen2-1.5B |
| control_both_correct | 50-65% | Bulk of questions |
| control_both_wrong | 15-30% | Hard questions neither model gets |
| Null rate | <2% | If higher, fix the prompt |

---

## What Comes Next

The master CSV from this step feeds directly into:
- **Step 2:** Extract hidden-state activations at selected layers for suppression/enhancement/control questions
- **Step 3:** Run probing experiments (linear, MDL, KL divergence) on those activations
- **Step 4:** Circuit-level analysis to identify specific attention heads and MLPs responsible

The behavioral labels are the foundation. Everything downstream inherits from this file.
