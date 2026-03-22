# Step 1: Behavioral Evaluation — Final Plan

**Status:** Resources downloaded, validated, ready for implementation
**Date:** 2026-03-22

---

## Goal

Run both LLaMA-3.1-8B models (base and instruct) on 21,726 valid Sanskriti questions. Record per-question correctness. Assign behavioral labels: **suppression**, **enhancement**, **control_both_correct**, **control_both_wrong**.

---

## Dataset: Sanskriti

**Source:** `13ari/Sanskriti` on HuggingFace (ACL 2025 Findings, arxiv:2506.15355)
**Local cache:** `/data/user_data/anshulk/cultural-mi/dataset/`

### Verified Statistics

| Property | Value |
|----------|-------|
| Total rows | 21,853 |
| Usable rows | **21,726** (127 excluded — see below) |
| Split | Single `train` split |
| Missing values | **Zero** across all columns |
| Columns | `state`, `attribute`, `question`, `option1`, `option2`, `option3`, `option4`, `answer`, `short explaination / source link`, `question_type` |

Note: column name `short explaination / source link` has a misspelling. Code must match it exactly.

### 127 Excluded Rows (0.58%)

The `answer` column does not match any of the four options for 127 rows. These were manually inspected and categorized:

| Category | Count | Action |
|----------|-------|--------|
| Fixable typos (>80% string similarity) | 10 | Could fuzzy-match, but safer to exclude |
| Substring matches | 6 | Ambiguous — exclude |
| Truly broken (answer unrelated to options) | 111 | Cannot determine ground truth |
| **Total excluded** | **127** | |

Key finding: **59 of the 111 broken rows are from Karnataka** (rows ~13580-13886), appearing to be a bad data batch where the answer column contains question fragments (e.g., answer="Primary ingredient in Dhuska" when options are "rice and lentils", "wheat flour", etc.).

**Decision:** Exclude all 127 from evaluation. Ground truth cannot be determined. This is 0.58% of the dataset — negligible impact.

### Ground Truth Letter Distribution (21,726 usable rows)

| Letter | Count | Percentage |
|--------|-------|------------|
| A (option1) | 5,885 | 27.1% |
| B (option2) | 6,308 | 29.0% |
| C (option3) | 5,008 | 23.1% |
| D (option4) | 4,525 | 20.8% |

Slight B-bias, D-underrepresented. Not extreme — acceptable.

### States (36 unique)

Highly imbalanced: Telangana (1,705) to Lakshadweep (131) — 13x range.

Top 5: Telangana (1,705), Karnataka (1,450), Andhra Pradesh (1,128), Delhi (1,082), Arunachal Pradesh (1,023)
Bottom 5: Lakshadweep (131), Mizoram (210), Meghalaya (267), Maharashtra (283), Ladakh (286)

### Attributes (16 unique)

Extremely imbalanced: Tourism (3,808) to Nightlife (42) — 90x range.

| Attribute | Count | Note |
|-----------|-------|------|
| Tourism | 3,808 | |
| History | 2,637 | |
| Festivals | 2,260 | |
| Cultural_Common_Sense | 2,109 | |
| Art | 2,075 | |
| Dance_and_Music | 2,024 | |
| Cuisine | 1,686 | |
| Costume | 1,515 | |
| Rituals_and_Ceremonies | 1,007 | |
| Personalities | 990 | |
| Language | 906 | |
| Religion | 483 | |
| Sports | 162 | Too sparse for reliable per-attribute stats |
| Transport | 77 | Too sparse |
| Medicine | 72 | Too sparse |
| Nightlife | 42 | Too sparse |

Per-attribute breakdowns should use percentages, not raw counts. Bottom 4 attributes (<200 rows) should be grouped or flagged as low-confidence.

### Question Types (4 unique, roughly balanced)

| Type | Count | Nature |
|------|-------|--------|
| Country Prediction | 5,563 | **Trivial: answer is always "India"** |
| Association | 5,454 | Match cultural element to region/entity |
| General Awareness | 5,449 | Identify what is associated with a place |
| State Prediction | 5,387 | Identify which state — hardest type |

**Critical finding:** All 5,563 Country Prediction answers are "India". India's position across A/B/C/D is balanced (A=26.8%, B=29.0%, C=22.6%, D=21.6%). Distractors are generic countries (Japan, Brazil, Italy, Canada, etc.). Both models should score near 100% on this type. **The interesting behavioral differences will come from the other 3 types.**

Expected difficulty ordering (from Sanskriti paper):
General Awareness (easiest) > Country Prediction > Association > State Prediction (hardest)

---

## Models

**Local paths:**
- Base: `/data/user_data/anshulk/cultural-mi/models/base/`
- Instruct: `/data/user_data/anshulk/cultural-mi/models/instruct/`

### Architecture (Identical Between Both)

| Property | Value |
|----------|-------|
| Architecture | `LlamaForCausalLM` |
| Total parameters | 8,030,261,248 (8.03B) |
| Hidden size | 4,096 |
| Layers | 32 |
| Attention heads | 32 |
| KV heads (GQA) | 8 (4 queries per KV group) |
| Head dimension | 128 |
| Intermediate size (MLP) | 14,336 |
| Activation | SiLU (SwiGLU) |
| Vocab size | 128,256 |
| Max positions | 131,072 (128K context) |
| RoPE | llama3 type, factor=8.0 |
| Dtype | bfloat16 |
| `tie_word_embeddings` | false |

### Per-Layer Parameter Breakdown

| Component | Params | Detail |
|-----------|--------|--------|
| Attention (Q+K+V+O) | 41.9M | Q: 4096x4096, K: 4096x1024, V: 4096x1024, O: 4096x4096 |
| MLP (gate+up+down) | 176.2M | SwiGLU: gate(4096x14336) + up(4096x14336) + down(14336x4096) |
| LayerNorm (x2) | 8,192 | RMSNorm, eps=1e-5 |
| **Layer total** | **218.1M** | MLP is 4.2x larger than attention |

### Global Components

| Component | Params |
|-----------|--------|
| Embedding | 525.3M (128256 x 4096) |
| LM Head | 525.3M (128256 x 4096, not tied) |
| Final LayerNorm | 4,096 |

### Weight Shards (4 safetensors files per model)

| Shard | Layers | Size |
|-------|--------|------|
| model-00001-of-00004 | 0–8 (+ embedding) | 4.7 GB |
| model-00002-of-00004 | 9–20 | 4.7 GB |
| model-00003-of-00004 | 20–31 | 4.6 GB |
| model-00004-of-00004 | 31 (+ lm_head + final norm) | 1.1 GB |

Total: ~15.1 GB per model in safetensors. Each directory is ~30 GB on disk because HF also downloaded the `original/` directory (Meta's consolidated .pth format, ~15 GB). Can delete `original/` to save 30 GB total.

### Differences Between Base and Instruct

Only three differences in configuration:

| Property | Base | Instruct |
|----------|------|----------|
| `eos_token_id` | `128001` | `[128001, 128008, 128009]` |
| `eos_token` | `<\|end_of_text\|>` (128001) | `<\|eot_id\|>` (128009) |
| Chat template | **None** | Full Jinja2 template in tokenizer_config.json |

All architecture parameters, RoPE config, and activation functions are identical. The weight differences come solely from SFT + RLHF post-training.

### Instruct Chat Template Details

The template auto-injects before the system message:
```
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
```

This is baked into the Jinja2 template — not something we control. It's how the model was trained. The full rendered prompt for a sample question:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant. Answer the following multiple choice question about Indian culture by responding with only the letter of the correct answer (A, B, C, or D). Do not explain your answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: Which of the given regions is home to the Jarawa body painting?

A) Surguja district
B) South Andaman and Middle Andaman Islands
C) Buddha Marg, Patna
D) Telangana<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

### Prompt Token Lengths (measured on 500-question sample)

| Model | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Base (completion) | 66 | 64 | 48 | 99 |
| Instruct (chat) | 126 | 124 | 108 | 159 |

Instruct prompts are ~2x longer due to chat template overhead. Both are very short — inference will be fast.

---

## Behavioral Labeling Logic

For each of the 21,726 usable questions:

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

## Prompt Design

### Base Model — Completion Style

The base model has **no chat template** in its tokenizer. It was never trained on chat formats. Using chat tokens on it will produce garbage — the base model treats `<|start_header_id|>` as literal text.

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

Use `tokenizer.apply_chat_template()` with `add_generation_prompt=True`. This inserts all special tokens automatically.

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

**When tokenizing the formatted string:** use `add_special_tokens=False` to avoid duplicating `<|begin_of_text|>` (the template already includes it).

---

## Decoding Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0 | Greedy decoding, deterministic output |
| max_new_tokens | 5 | Only need 1 letter; cap prevents runaway |
| do_sample | False | Must be False for greedy |

Note: both model configs ship with `do_sample=True, temperature=0.6, top_p=0.9` in generation_config.json. We **must override** these in the generate() call.

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
- If no match after stripping → exclude row (127 total, already identified)

---

## Detailed TODO

### Phase A: Environment & Data Preparation

```
A1. Create conda env `cultural` — DONE
A2. Install packages: torch, transformers, datasets, accelerate, etc. — DONE
A3. HuggingFace login already configured via housing env
A4. Sanskriti dataset already downloaded to /data/.../dataset/
A5. Build ground_truth_letter for all 21,853 rows:
    - Strip whitespace, lowercase both answer and option1-4
    - Match answer to option → A/B/C/D
    - Mark 127 no-match rows with ground_truth_letter = null
A6. Assign question_id (0 to 21852) as stable index
A7. Exclude 127 no-match rows → 21,726 usable rows
A8. Save as results/step1/sanskriti_prepared.csv
```

### Phase B: Models — DONE

```
B1. Base model downloaded to /data/.../models/base/ (30 GB with original/)
B2. Instruct model downloaded to /data/.../models/instruct/ (30 GB with original/)
B3. Both have 4 safetensors shards + config.json + tokenizer files verified
B4. Config diff verified: only eos_token_id and chat_template differ
```

### Phase C: Base Model Evaluation

```
C1. Write evaluation script: scripts/eval_step1.py (single script, mode flag)
C2. Load model from /data/.../models/base/ in BF16, device_map="auto"
C3. Load tokenizer — set padding_side="left", add pad_token if missing
    (LLaMA tokenizers have no pad_token by default — set pad_token = eos_token)
C4. Run 1 test question manually, print raw output, verify letter extraction
C5. Format all 21,726 prompts using completion template
C6. Run batched inference (batch_size=16, reduce to 8 if OOM)
    - Left-pad to equal length within each batch
    - Generate with: temperature=0, max_new_tokens=5, do_sample=False
    - Decode only the newly generated tokens (slice off input length)
C7. Checkpoint every 500 questions to checkpoints/base_checkpoint_{n}.csv
C8. Extract prediction letter, compare against ground_truth_letter
C9. Save: results/step1/base_results.csv
    Columns: question_id, ground_truth_letter, raw_output_base, predicted_base, base_correct
C10. Log null/invalid count — expect <2%
C11. Unload model: del model; torch.cuda.empty_cache(); gc.collect()
```

### Phase D: Instruct Model Evaluation

```
D1. Load instruct model from /data/.../models/instruct/ in BF16, device_map="auto"
D2. Load tokenizer — verify chat_template is not None, set padding_side="left"
D3. Run 1 test question:
    - apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    - Tokenize with add_special_tokens=False
    - Print formatted prompt, verify special tokens present
D4. Format all 21,726 prompts using apply_chat_template
D5. Run batched inference with identical decoding settings
    - Same left-padding, same batch size
    - Instruct prompts are ~126 tokens (vs base ~66) — still fits easily
D6. Checkpoint every 500 questions
D7. Extract predictions → instruct_correct
D8. Save: results/step1/instruct_results.csv
    Columns: question_id, ground_truth_letter, raw_output_instruct, predicted_instruct, instruct_correct
D9. Log null/invalid count
```

### Phase E: Merge & Label

```
E1. Load base_results.csv and instruct_results.csv
E2. Join on question_id
E3. Assign behavioral_label using the 2x2 table
E4. Merge back all original Sanskriti columns
E5. Save master CSV: results/step1/sanskriti_behavioral_labels.csv
    Columns: question_id, state, attribute, question, option1-4, answer,
             question_type, ground_truth_letter, predicted_base, predicted_instruct,
             base_correct, instruct_correct, behavioral_label
E6. Print group counts and percentages
E7. Print null rates for both models
```

### Phase F: Sanity Checks

These must all pass before proceeding to Step 2.

```
F1. Overall accuracy check
    - Base accuracy should be 40-75%
    - If <30% or >80%: prompt or extraction is broken
    - Reference: Llama-3.2-3B-Instruct=52%, Llama-3.1-70B-Instruct=86%
    - Expected 8B base: roughly 45-65%
    - Country Prediction type alone should be near 100% for both models
      (all answers are "India") — factor this in when interpreting overall accuracy

F2. Instruct > Base overall
    - If instruct scores lower than base, chat template or prompt is wrong
    - Single strongest signal of a prompt format bug

F3. Option distribution
    - Count predictions of A, B, C, D for each model
    - Compare against ground truth distribution (A=27.1%, B=29.0%, C=23.1%, D=20.8%)
    - If any letter >50% in predictions: severe position bias
    - Save as: results/step1/step1_option_distribution.csv

F4. Question type breakdown
    - Compute accuracy per question_type for both models
    - Country Prediction should be ~95-100% for both models
    - Expected difficulty (excluding Country Prediction):
      General Awareness (easiest) > Association > State Prediction (hardest)

F5. State & attribute breakdown
    - Compute suppression rate per state and per attribute
    - No single state should account for >15% of all suppression cases
    - Bottom 4 attributes (Sports, Transport, Medicine, Nightlife — <200 rows each)
      will have noisy per-attribute stats; group or flag them

F6. Suppression-enhancement balance
    - Report both rates honestly
    - Previous work (Qwen2-1.5B): suppression ~8%, enhancement ~7%
    - LLaMA 8B may differ — report whatever the data shows

F7. Null prediction audit
    - If null rate >2% for either model: investigate prompt
    - Manually inspect 10-20 null examples
    - Common cause: model outputting full explanation instead of letter
    - Fallback: shorten system prompt to "Respond with only the letter. Nothing else."

F8. Save aggregate stats: results/step1/step1_aggregate_stats.json
```

### Phase G: SLURM Job Script

```
G1. Write scripts/run_step1.sh (SLURM batch script)
    - Partition: array (or debug for testing)
    - GPUs: 1x L40S (48GB) — sufficient for 8B BF16 (~16GB weights + batch overhead)
    - Time: 2-4 hours estimated
      (21,726 questions x ~126 tokens max x 2 models,
       at ~1000 tokens/sec on L40S ≈ ~2.7 hours + overhead)
    - Memory: 64GB RAM
    - Activate cultural conda env
    - Run eval_step1.py --mode base, then --mode instruct, then merge
G2. Test on debug partition with 100 questions first
G3. Run full evaluation on array partition
```

---

## Output Artifacts

| File | Location | Description |
|------|----------|-------------|
| `sanskriti_prepared.csv` | `results/step1/` | Dataset with question_id, ground_truth_letter, 127 rows marked |
| `base_results.csv` | `results/step1/` | Raw base model predictions |
| `instruct_results.csv` | `results/step1/` | Raw instruct model predictions |
| `sanskriti_behavioral_labels.csv` | `results/step1/` | Master CSV, 21,726 rows with behavioral labels |
| `step1_aggregate_stats.json` | `results/step1/` | Accuracy, group counts, per-type/attribute/state breakdowns |
| `step1_option_distribution.csv` | `results/step1/` | A/B/C/D prediction counts per model |
| `step1_run_log.txt` | `logs/` | Timestamps, GPU, batch size, errors |

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong prompt for base model | Base has no chat template — use plain completion. Test 1 question first. |
| generation_config.json overrides | Both models ship with `do_sample=True, temp=0.6`. Must explicitly pass `do_sample=False, temperature=0` to `generate()`. |
| Instruct tokenizer duplicates BOS | `apply_chat_template` already prepends `<\|begin_of_text\|>`. Use `add_special_tokens=False` when tokenizing. |
| LLaMA has no pad_token | Set `tokenizer.pad_token = tokenizer.eos_token` for batched inference. |
| Answer text mismatch | 127 rows already identified and excluded. Use `.strip().lower()` for matching. |
| Model outputs explanation instead of letter | Short, direct system prompt. Check null rate. Fallback prompt if >2%. |
| OOM on GPU | L40S has 48GB, model is ~16GB. Batch size 16 at ~160 tokens leaves ample headroom. Reduce to 8 if needed. |
| Run crash mid-evaluation | Checkpoint every 500 questions. Resume from last checkpoint. |
| Country Prediction inflates accuracy | All answers are "India" — will inflate both models equally. Report accuracy with and without Country Prediction. |
| Sparse attributes | Bottom 4 attributes have <200 rows. Per-attribute stats for these are unreliable. |

---

## Expected Results (Rough Ranges)

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Base accuracy (overall) | 50-70% | Country Prediction inflates; excluding it: 40-60% |
| Instruct accuracy (overall) | 60-80% | Should be higher than base |
| Country Prediction accuracy | ~95-100% | Both models (answer is always "India") |
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
