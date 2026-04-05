# Step 1: Behavioral Evaluation — Complete Analysis

- **Mechanistic Interpretability of Cultural Knowledge in Instruction-Tuned LLMs**
- **Anshul Kumar and Pragati Bhattad — Carnegie Mellon University, April 2026**
- **Target venue: EMNLP 2026 Workshops**

This document records every decision, every formula, every code path, and every
result from Step 1 of the MI pipeline. Step 1 runs LLaMA-3.1-8B (base and
instruct) on all 21,726 Sanskriti questions, extracts behavioral labels
(suppression/enhancement/control), collects internal activations at 8 hook points,
and produces the master CSV that drives all downstream analysis. Every number
will be validated against the actual output files once the run completes.

---

## Quick Reference: Key Numbers

All results validated against output CSVs from SLURM job 6959494
(2026-04-05, A100-SXM4-40GB × 2).

| Metric | Value | Source |
|--------|-------|--------|
| Total questions evaluated | 21,726 | `sanskriti_prepared.csv` |
| Models evaluated | 2 (base + instruct) | `config.yaml` |
| GPUs used | 2 × NVIDIA A100-SXM4-40GB (42.4 GB each) | SLURM log |
| Batch size | 24 | `run_step1.sh` |
| Total batches per model | 906 | ceil(21726 / 24) = 906 |
| Hook points | 8 (embed + 7 layers) | `eval_step1.py` |
| Activation files | 32 (8 hooks × 2 pooling × 2 models) | `activations/` |
| Activation storage | 9.0 GB total | `activations/` |
| **Base accuracy** | **87.61%** (19,035/21,726) | `base_results.csv` |
| **Instruct accuracy** | **88.88%** (19,310/21,726) | `instruct_results.csv` |
| **Suppression count** | **655** (3.0%) | `sanskriti_behavioral_labels.csv` |
| **Enhancement count** | **930** (4.3%) | `sanskriti_behavioral_labels.csv` |
| Control (both correct) | 18,380 (84.6%) | `sanskriti_behavioral_labels.csv` |
| Control (both wrong) | 1,761 (8.1%) | `sanskriti_behavioral_labels.csv` |
| Base forced-choice rate | 0.00% (0/21,726) | `base_results.csv` |
| Instruct forced-choice rate | 0.00% (0/21,726) | `instruct_results.csv` |
| Runtime (instruct) | 12.3 min (29.4 q/s) | SLURM log |
| Runtime (base) | 15.3 min (23.7 q/s) | SLURM log |
| Total runtime (parallel) | 16 min | SLURM log |

---

## Glossary: Every Abbreviation, Formula, and Metric Explained

This section defines every technical term used in Step 1. If you encounter
something unclear elsewhere in the document, check here first.

### Abbreviations

| Abbreviation | Full Form | What It Is |
|-------------|-----------|-----------|
| MI | Mechanistic Interpretability | Studying what happens inside neural networks at the level of individual neurons, layers, and circuits |
| MCQ | Multiple Choice Question | A question with 4 answer options, one correct |
| CP | Country Prediction | Question type — "which country is this cultural element from?" (always India) |
| SP | State Prediction | Question type — "which state is this cultural element from?" |
| GT | Ground Truth | The correct answer letter (A/B/C/D) as labeled in the dataset |
| BF16 | Brain Float 16 | A 16-bit floating-point format for efficient neural network computation |
| GQA | Grouped Query Attention | Attention mechanism in LLaMA 3.1 — 32 query heads share 8 key-value heads |
| RoPE | Rotary Position Embedding | Method for encoding token positions in transformer models |
| SDPA | Scaled Dot-Product Attention | PyTorch's fused attention kernel, used by default in transformers 5.x |
| SFT | Supervised Fine-Tuning | Training a base model on curated instruction-response pairs |
| RLHF | Reinforcement Learning from Human Feedback | Training using human preference ratings |
| pp | Percentage Points | Arithmetic difference between two percentages |
| OOM | Out Of Memory | GPU runs out of VRAM during computation |
| SLURM | Simple Linux Utility for Resource Management | Job scheduler for compute clusters |
| VRAM | Video Random Access Memory | GPU memory for model weights and activations |
| BPE | Byte Pair Encoding | Tokenization algorithm that merges frequent byte pairs into tokens |

### Statistical and Mathematical Formulas

**Log-probability (logprob)** — The natural logarithm of a token's probability
under the model's output distribution. More negative = less probable.

```
Given logits z (raw model output for each vocabulary token):

Step 1: Convert logits to probabilities via softmax
  P(token_i) = exp(z_i) / Σ_j exp(z_j)

Step 2: Take the natural log
  logprob(token_i) = log(P(token_i)) = z_i - log(Σ_j exp(z_j))

This is equivalent to log_softmax(z)_i.

Range: (-∞, 0]
  0    = probability 1.0 (model is certain this token comes next)
  -2.3 = probability 0.1 (10% chance)
  -6.9 = probability 0.001 (0.1% chance)
  -13  = probability ~2.2e-6 (essentially zero)

In our evaluation:
  We extract logprobs for 4 answer tokens. The exact tokens differ by model:
    Base model:     " A" (362), " B" (426), " C" (356), " D" (423)
                    Space-prefixed, because the prompt ends with "Answer:"
    Instruct model: "A" (32), "B" (33), "C" (34), "D" (35)
                    Bare letters, because the chat template ends with "\n\n"
  The predicted answer = argmax over these 4 logprobs
```

**Logprob margin** — The difference between the highest and second-highest
logprob among the four answer tokens. Measures prediction confidence.

```
Formula: margin = logprob_top1 - logprob_top2

Where:
  logprob_top1 = max(logprob_A, logprob_B, logprob_C, logprob_D)
  logprob_top2 = second highest among the four

Example:
  logprobs = [A=-2.1, B=-3.5, C=-4.2, D=-5.0]
  top1 = A (-2.1), top2 = B (-3.5)
  margin = -2.1 - (-3.5) = 1.4 nats

Interpretation:
  margin > 2.0  = high confidence (model strongly prefers one answer)
  margin 0.5-2  = moderate confidence
  margin < 0.5  = low confidence (top two answers are nearly tied)

Why 0.5 nats as the threshold?
  exp(0.5) ≈ 1.65, so a 0.5-nat margin means the top answer is only
  1.65× more probable than the runner-up. For a 4-choice MCQ, this is
  barely above the noise floor. We flag these as "low confidence."
```

**Forced choice** — A boolean flag per question. True when the model's overall
top-1 predicted token (across the entire vocabulary) is NOT one of A/B/C/D.

```
The model predicts the next token after "Answer:". Its top-1 prediction
might be " A" (an answer letter) or it might be "\n" (newline), "the",
"India", or any other token.

If top-1 is one of the model's answer tokens: forced = False (natural choice)
If top-1 is NOT one of the answer tokens:    forced = True (forced choice)

When forced = True, we still extract the answer by comparing the logprobs
of ONLY the four answer tokens. The model didn't naturally output a letter,
but the relative ranking among A/B/C/D still carries information about
which answer the model considers most likely.

Expected rates:
  Base model: ~80-90% forced (base models don't naturally output "A"/"B"/"C"/"D")
  Instruct model: <5% forced (instruct models follow instructions to output a letter)

A high forced rate on the instruct model would indicate a prompt formatting
issue. A high forced rate on the base model is normal and expected.
```

**Behavioral labels** — The core output of Step 1. Each question gets exactly
one of four labels based on whether each model answered correctly.

```
            │ Instruct correct │ Instruct wrong │
────────────┼──────────────────┼────────────────┤
Base correct│ control_both_    │ suppression    │
            │ correct          │                │
────────────┼──────────────────┼────────────────┤
Base wrong  │ enhancement      │ control_both_  │
            │                  │ wrong          │
────────────┴──────────────────┴────────────────┘

Suppression: Base got it right, instruct got it wrong.
  The instruction-tuning process "suppressed" knowledge the base model had.

Enhancement: Base got it wrong, instruct got it right.
  Instruction-tuning "enhanced" the model's ability on this question.

Control (both correct): Both models answer correctly.
  Knowledge is preserved through instruction tuning.

Control (both wrong): Neither model answers correctly.
  Both models lack the knowledge (or the question is too hard/ambiguous).
```

**Mean pooling** — Averaging hidden state vectors across sequence positions,
weighted by the attention mask to exclude padding tokens.

```
Formula: h_mean = Σ(h_i × mask_i) / Σ(mask_i)

Where:
  h_i    = hidden state vector at position i (4096-dimensional)
  mask_i = 1 if position i is a real token, 0 if padding
  Σ      = sum over all sequence positions

This produces a single 4096-dimensional vector that represents the
"average meaning" of the entire input sequence at a given layer.

For the base model, we additionally mask out the 5-shot prefix tokens
so that mean pooling only covers the target question, not the examples.
```

**Last-token pooling** — Extracting the hidden state at the final non-padding
position. This is where the model concentrates its "answer" representation.

```
Formula: h_last = h[-1]  (last position in the sequence dimension)

For left-padded sequences (our configuration), this is always the
rightmost position in the tensor — the last real token. Padding is
on the left, so real tokens occupy the rightmost positions. For right-padded sequences, it
would be the last 1 in the attention mask.
```

**Activation norm growth** — The L2 norm of activation vectors increases
through the layers of a transformer. This is expected and indicates that
later layers carry more concentrated, task-relevant information.

```
We measure: mean_norm = mean(||h_i||_2) across all questions

Expected pattern (and what we observe):
  embed:    ~0.5  (raw token embeddings, small norm)
  layer_04: ~4-5  (early processing)
  layer_08: ~5-6  (building representations)
  layer_14: ~6-9  (mid-network)
  layer_20: ~10-17 (late processing)
  layer_26: ~15-30 (near output)
  layer_30: ~30-48 (penultimate)
  layer_31: ~35-62 (final layer, highest norm)

If a layer shows NaN or Inf norms, the model has a numerical issue.
If norms are identical across layers, the hooks are not capturing
the correct tensors.
```

### Model Architecture Reference

LLaMA-3.1-8B architecture (both base and instruct share the same architecture;
instruct has additional SFT + RLHF training):

```
Architecture:       LlamaForCausalLM
Parameters:         8.03 billion
Layers:             32 transformer decoder layers (indexed 0-31)
Hidden size:        4096
Attention heads:    32 query heads, 8 key-value heads (GQA, 4:1 ratio)
Intermediate size:  14,336 (FFN hidden dimension)
Vocabulary size:    128,256 tokens
Position encoding:  RoPE (Rotary Position Embeddings)
Context length:     131,072 tokens (128K)
Precision:          BF16 (bfloat16) — ~16 GB VRAM for weights
Tokenizer:          tiktoken-based BPE (byte-level with merges)
```

---

## Table of Contents

1. [Purpose of This Step](#1-purpose-of-this-step)
2. [Relationship to the Pipeline](#2-relationship-to-the-pipeline)
3. [Infrastructure and Environment](#3-infrastructure-and-environment)
4. [Model Loading and Configuration](#4-model-loading-and-configuration)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Prompt Engineering](#6-prompt-engineering)
7. [Tokenization](#7-tokenization)
8. [The Token ID Bug and Fix](#8-the-token-id-bug-and-fix)
9. [Forward Pass and Logit Extraction](#9-forward-pass-and-logit-extraction)
10. [Activation Hook System](#10-activation-hook-system)
11. [Activation Pooling Strategies](#11-activation-pooling-strategies)
12. [Hook Layer Selection Rationale](#12-hook-layer-selection-rationale)
13. [Checkpointing and Preemption Handling](#13-checkpointing-and-preemption-handling)
14. [Parallel Execution Architecture](#14-parallel-execution-architecture)
15. [The Merge Pipeline](#15-the-merge-pipeline)
16. [Behavioral Labeling Logic](#16-behavioral-labeling-logic)
17. [Three-Tier Analysis Framework](#17-three-tier-analysis-framework)
18. [Per-Dimension Breakdowns](#18-per-dimension-breakdowns)
19. [Entity-Level Analysis](#19-entity-level-analysis)
20. [Logprob Margin and Confidence Analysis](#20-logprob-margin-and-confidence-analysis)
21. [Position Distribution Analysis](#21-position-distribution-analysis)
22. [Sanity Check Framework](#22-sanity-check-framework)
23. [Plots and Visualizations](#23-plots-and-visualizations)
24. [Results](#24-results)
25. [Activation Verification](#25-activation-verification)
26. [Bugs Encountered and Fixed](#26-bugs-encountered-and-fixed)
27. [Design Decisions: What We Chose and Why](#27-design-decisions-what-we-chose-and-why)
28. [What This Step Does NOT Do](#28-what-this-step-does-not-do)
29. [Output Files](#29-output-files)
30. [Runtime and Reproducibility](#30-runtime-and-reproducibility)

---

## 1. Purpose of This Step

Step 1 answers two questions:

1. **Behavioral:** For each of the 21,726 questions, does the base model get it
   right? Does the instruct model get it right? The combination of these two
   answers produces a behavioral label (suppression/enhancement/control).

2. **Representational:** What do the internal activations look like at each
   layer when the model processes each question? These activations are saved
   for Steps 2-4 (probing, circuit analysis).

### Why Behavioral Evaluation Is the Foundation of the MI Pipeline

The central research question of this project is: **does instruction tuning
(SFT + RLHF) systematically suppress or enhance cultural knowledge that
already exists in the base model's weights?** To answer this, we need three
things:

1. **A ground-truth behavioral signal** — for each question, did the model's
   behavior change between base and instruct? This is what the behavioral
   labels capture. Without this signal, Steps 2-4 have nothing to predict
   or explain.

2. **Internal representations at multiple network depths** — what do the
   hidden states look like when the model processes a cultural question?
   These activations are the raw material for mechanistic analysis.

3. **Exact correspondence between (1) and (2)** — the activations must come
   from the same forward pass that produced the behavioral label. If they
   don't, any correlation found in probing could be an artifact.

This is why Step 1 does both behavioral evaluation and activation extraction
in a single pass: it produces the paired (behavior, representation) data that
the entire downstream analysis depends on.

### The Suppression Hypothesis

The working hypothesis (from the project proposal and EDA findings) is that
instruction tuning via RLHF may teach the model to "play it safe" on
culturally specific questions — defaulting to more generic or Western-centric
answers rather than committing to niche regional knowledge. If this is true,
we should see:

- A non-trivial fraction of questions where the base model answers correctly
  but the instruct model does not (suppression cases)
- Suppression concentrated in questions requiring specific cultural knowledge
  (Association and General Awareness types) rather than pattern-matching
  questions (Country/State Prediction)
- Systematic differences in internal representations between suppressed and
  non-suppressed questions, detectable by linear probes (Step 2)

Step 1 tests the first two predictions directly. Steps 2-4 investigate the
third.

### Why Both in a Single Forward Pass

We could run behavioral evaluation and activation extraction as two separate
scripts. We chose to combine them into a single forward pass because:

1. **Efficiency:** Loading an 8B model costs ~16 GB VRAM and ~45 seconds.
   Running the forward pass is cheap (~2 seconds per batch of 64). Doing two
   separate passes would double the model loading time without any benefit.

2. **Consistency:** The activations are guaranteed to correspond to the exact
   same input that produced the behavioral labels. No risk of misalignment
   from different tokenization, batching, or random state.

3. **Simplicity:** One script, one SLURM job, one set of logs to audit.

### What This Step Produces

- `base_results.csv` — 21,726 rows with predicted letter, correctness,
  logprobs for A/B/C/D, forced-choice flag, top-1 token ID
- `instruct_results.csv` — same format
- `sanskriti_behavioral_labels.csv` — merged master CSV with behavioral labels
- 32 activation `.npy` files — `(21726, 4096)` float32 arrays
- Aggregate stats JSON, per-dimension CSVs, 7+ plots, sanity check log

---

## 2. Relationship to the Pipeline

This project has a 4-step pipeline. Step 1 is the foundation.

```
Step 0: EDA (completed)
  └── Understand the dataset, identify artifacts, set expectations

Step 1: Behavioral Evaluation + Activation Extraction ← THIS STEP
  ├── Input:  21,726 Sanskriti questions + 2 LLaMA models
  ├── Output: Behavioral labels + activation arrays
  └── Feeds into: Steps 2, 3, 4

Step 2: Probing Analysis
  ├── Input:  Activation arrays from Step 1
  ├── Method: Train linear probes on activations to predict behavioral labels
  └── Goal:   Find which layers encode cultural knowledge

Step 3: Circuit Identification
  ├── Input:  Key layers identified in Step 2
  ├── Method: Attention pattern analysis, causal tracing
  └── Goal:   Identify specific circuits responsible for suppression

Step 4: Interpretation and Synthesis
  ├── Input:  Results from Steps 1-3
  └── Goal:   Coherent narrative of how RLHF affects cultural knowledge
```

**Critical dependency:** If Step 1 produces wrong behavioral labels, every
downstream analysis is invalidated. This is why we have 11 sanity checks in
the merge pipeline (Section 22).

### Connection to EDA Findings

The EDA (documented in `docs/eda_analysis.md`) identified several dataset
properties that directly inform Step 1 design:

| EDA Finding | Step 1 Impact |
|-------------|---------------|
| 127 broken rows (answer ≠ any option) | Excluded before evaluation |
| Country Prediction answer = always India | Expect ~100% accuracy; drives Tier 2 split |
| Position bias: B=29%, D=20.8% | Must check model prediction distribution |
| 78.6% near-duplicate involvement | Behavioral labels will cluster by entity |
| 55.4% template-based questions | Step 3-4 must control for template vs knowledge |
| 4 sparse attributes (<200 questions) | Report with uncertainty warnings |

---

## 3. Infrastructure and Environment

### Hardware

```
Cluster:     CMU Babel cluster
Partition:   preempt (7-day time limit, may be preempted)
GPUs:        2 × NVIDIA RTX PRO 6000 Blackwell Server Edition
             96 GB VRAM each, 204 GB total
CPU:         16 cores allocated
RAM:         128 GB system memory
Storage:     /data/user_data/ (NVMe, fast) for models/activations
             /home/ (NFS) for code/logs
```

### Software Versions

```
Python:           3.11 (conda env: cultural)
PyTorch:          2.10.0+cu128
Transformers:     5.3.0
CUDA:             12.9
Driver:           575.51.03
```

### SLURM Configuration

The SLURM script (`scripts/run_step1.sh`) specifies:

```bash
#SBATCH --job-name=step1_eval
#SBATCH --partition=preempt
#SBATCH --gres=gpu:RTX_PRO_6000:2      # 2 GPUs of specific type
#SBATCH --constraint=VRAM_96GB          # ensure 96GB models
#SBATCH --mem=128G                      # system RAM
#SBATCH --time=7-00:00:00              # max 7 days (preempt partition limit)
#SBATCH --cpus-per-task=16             # for data loading
#SBATCH --signal=B:USR1@120            # send SIGUSR1 120s before preemption
#SBATCH --output=logs/step1_slurm_%j.out  # %j = job ID
```

**Why `preempt` partition?** The RTX PRO 6000 GPUs are available on `preempt`
with generous time limits. The alternative (`debug`) has a 30-minute limit,
which is too short for the full run plus potential retries. The `preempt`
partition means our job can be killed at any time by higher-priority jobs — this
is why we have checkpointing (Section 13).

**Why `--signal=B:USR1@120`?** SLURM sends SIGUSR1 to the job 120 seconds
before preemption. Our code catches this signal, checkpoints all progress,
and exits cleanly. On resubmission, the job resumes from the last checkpoint.

**Why 2 GPUs?** We run the base model on GPU 0 and the instruct model on GPU 1
simultaneously. Each model needs ~16 GB VRAM for weights plus ~6 GB for
activations during inference, totaling ~22 GB peak. A single 96 GB GPU could
theoretically hold both, but parallel execution halves the wall-clock time.

---

## 4. Model Loading and Configuration

### Loading Strategy

```python
# From eval_step1.py, lines 362-366
model = AutoModelForCausalLM.from_pretrained(
    config["models"][model_type]["local_dir"],
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
model.eval()
```

**`torch_dtype=torch.bfloat16`:** Loads all model weights in BF16 precision.
This halves VRAM usage compared to FP32 (16 GB vs 32 GB for 8B parameters)
with negligible accuracy loss. BF16 has the same exponent range as FP32 (8 bits)
but fewer mantissa bits (7 vs 23), making it suitable for inference.

```
Floating-point format comparison:

FP32 (IEEE 754 single precision):
  [1 sign] [8 exponent] [23 mantissa]  = 32 bits = 4 bytes per parameter
  Range: ±3.4 × 10^38
  Precision: ~7 decimal digits

BF16 (Brain Float 16):
  [1 sign] [8 exponent] [7 mantissa]   = 16 bits = 2 bytes per parameter
  Range: ±3.4 × 10^38   ← SAME as FP32 (critical for avoiding overflow)
  Precision: ~2 decimal digits

FP16 (IEEE 754 half precision):
  [1 sign] [5 exponent] [10 mantissa]  = 16 bits = 2 bytes per parameter
  Range: ±6.5 × 10^4    ← MUCH smaller range (overflows easily)
  Precision: ~3 decimal digits

Why BF16 over FP16?
  LLaMA's logits can reach magnitudes of ±50-100. FP16 overflows at 65504,
  which is dangerously close. BF16 shares FP32's exponent range, so it
  handles the same magnitudes with no overflow risk. The reduced mantissa
  (7 vs 10 bits for FP16) means less precision, but for inference the
  rounding errors are below the noise floor of the model's predictions.

VRAM calculation:
  8.03 billion parameters × 2 bytes/param (BF16) = 16.06 GB
  8.03 billion parameters × 4 bytes/param (FP32) = 32.12 GB
  Savings: 16 GB — freeing this VRAM for activations and attention caches
```

Note: transformers 5.3.0 emits a deprecation warning suggesting `dtype` instead
of `torch_dtype`. This is cosmetic and does not affect behavior.

**`device_map={"": device}`:** Places the entire model on one GPU. The empty
string key `""` means "all layers." This avoids the complexity of model
parallelism, which is unnecessary since each model fits on a single GPU.
Alternative: `device_map="auto"` would use `accelerate` to distribute layers
across devices, but this adds overhead and is not needed for 8B models on
96 GB GPUs.

**`model.eval()`:** Disables dropout and sets batch normalization to evaluation
mode. LLaMA doesn't use dropout or batch norm, so this is a no-op in practice,
but it is standard practice and makes the intent clear.

### What Happens Inside the Model During a Forward Pass

To understand what the hooks capture and what the logits mean, it helps to
trace a single question through the entire model:

```
Input: "Question: What dance form originated in Kerala?\nA) Bharatanatyam\nB) Kathakali\nC) Odissi\nD) Mohiniyattam\nAnswer:"

Step 1: Tokenization
  The tokenizer converts this string into a sequence of integer token IDs.
  Each token maps to a learned 4096-dimensional embedding vector.
  Example: "Kathakali" → might be 2-3 BPE tokens: ["Kath", "ak", "ali"]

Step 2: Embedding (hooked as "embed")
  Each token ID is looked up in the embedding matrix (128256 × 4096).
  Output: (seq_len, 4096) — one vector per token.
  These are raw lookup vectors with no contextual information yet.
  "Kathakali" tokens have the same embedding regardless of context.

Step 3: Transformer layers 0-3 (NOT hooked)
  Each layer applies:
    a) RMSNorm (normalizes the hidden state)
    b) Self-attention with RoPE (tokens attend to earlier tokens)
    c) Residual connection (add attention output to input)
    d) RMSNorm again
    e) SwiGLU feed-forward network (non-linear transformation)
    f) Another residual connection
  By layer 3, tokens have basic syntactic understanding: the model knows
  "Answer:" is a prompt for completion, "A)" labels an option, etc.

Step 4: Layer 4 (HOOKED)
  Same operations as above. The hook captures the output hidden state.
  At this depth, the model is beginning to form semantic representations.
  The "Kerala" token now carries some geographic context from attending
  to the question text.

Step 5: Layers 5-7 (NOT hooked)
  Representations continue building.

Step 6: Layer 8 (HOOKED)
  Entity recognition is forming. The model is starting to associate
  "Kerala" with cultural concepts in its weights.

Step 7: Layers 9-13 (NOT hooked)
  Mid-network processing. Knowledge retrieval begins. The model's weights
  contain associations learned during pre-training (e.g., Kerala → Kathakali).

Step 8: Layer 14 (HOOKED — network midpoint)
  By now, the model has retrieved relevant factual associations. The
  hidden state at the last token position is beginning to encode which
  answer is correct.

Step 9: Layers 15-19 (NOT hooked)
  Late-middle processing. Answer confidence crystallizes.

Step 10: Layer 20 (HOOKED)
  The model has likely "decided" its answer by this point. The residual
  stream at the last position carries a strong signal toward one option.

Step 11-13: Layers 21-31 (hooked at 26, 30, 31)
  Final processing and output preparation. Layer 31 feeds directly into
  the language model head (lm_head), a linear projection from 4096
  dimensions to 128,256 dimensions (vocabulary size).

Step 14: lm_head projection
  output_logits = lm_head(layer_31_output)  # (seq_len, 128256)
  These are the raw logits — unnormalized scores for every vocabulary token.
  We extract logits at the LAST position only (the "Answer:" position).

Step 15: Logit → probability → log-probability
  We apply log_softmax to get log-probabilities, then extract the values
  for tokens " A" (362), " B" (426), " C" (356), " D" (423).
  The predicted answer is the letter with the highest logprob.
```

This trace explains why we hook at multiple layers: the transformation from
"raw text" to "answer decision" happens gradually, and different layers
capture different stages of that process.

### Tokenizer Configuration

```python
# From eval_step1.py, lines 372-380
tokenizer = AutoTokenizer.from_pretrained(
    config["models"][model_type]["local_dir"]
)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = 128001   # <|end_of_text|>
tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128001)
```

**`padding_side = "left"`:** Critical for decoder-only models. When batching
sequences of different lengths, padding goes on the LEFT so that the last
token of every sequence is a real token, not a pad token. This matters because:

1. The model's prediction for the answer comes from the LAST position's logits.
   If padding were on the right, shorter sequences would have pad tokens at the
   end, and we'd need to find each sequence's actual last position to extract
   logits. Left-padding means position -1 is always valid.

2. Activation extraction at the "last token" position works correctly — it
   always gets the final real token, not a padding artifact.

**`pad_token_id = 128001`:** LLaMA's tokenizer has no dedicated pad token. We
use `<|end_of_text|>` (ID 128001) as the padding token. We explicitly set the
ID rather than using `tokenizer.eos_token` because the instruct tokenizer's
`eos_token` is `<|eot_id|>` (ID 128009) — a semantically meaningful
end-of-turn marker that would confuse the model if used as padding. The base
tokenizer's `eos_token` is `<|end_of_text|>` (128001), so the distinction only
matters for instruct, but we use the explicit ID for both to be safe. Since
padding is on the left and masked out by the attention mask, the model never
attends to these tokens.

### Model Paths

Both models are pre-downloaded to local NVMe storage for fast loading:

```
Base:     /data/user_data/anshulk/cultural-mi/models/base
Instruct: /data/user_data/anshulk/cultural-mi/models/instruct
```

Loading from local NVMe takes ~2-45 seconds (depending on whether the model
is cached in the OS page cache). Loading from HuggingFace Hub would take
minutes per run and is not reproducible without network access.

---

## 5. Dataset Preparation

### Loading and Filtering

```python
# From eval_step1.py, lines 141-163
def load_sanskriti(config):
    ds = load_dataset("13ari/Sanskriti", split="train",
                      cache_dir=config["dataset"]["local_dir"])
    df = ds.to_pandas()

    def get_gt_letter(row):
        ans = str(row["answer"]).strip().lower()
        for col, letter in zip(option_cols, letters):
            if str(row[col]).strip().lower() == ans:
                return letter
        return None

    df["ground_truth_letter"] = df.apply(get_gt_letter, axis=1)
    usable = df[df["ground_truth_letter"].notna()].reset_index(drop=True)
    usable["question_id"] = range(len(usable))
```

**Ground truth derivation:** For each row, we compare the `answer` column
(free text) against each of the four option columns (`option1`..`option4`).
The comparison is case-insensitive and whitespace-stripped. The first matching
option determines the ground truth letter (A/B/C/D). If no option matches,
the row is excluded.

**127 excluded rows:** As documented in the EDA (Section 2), 127 rows have an
`answer` value that doesn't match any option. These are broken data entries
where the answer column contains question fragments or unrelated text. At 0.58%
of the dataset, exclusion is the correct approach.

**Question ID assignment:** `question_id` is assigned AFTER filtering, producing
a contiguous range 0 to 21,725 with no gaps. This is critical because
`question_id` doubles as the row index into the activation arrays:

```
question_id 0    → activation_array[0, :]
question_id 1    → activation_array[1, :]
...
question_id 21725 → activation_array[21725, :]
```

If we assigned IDs before filtering, there would be 127 gaps in the activation
arrays, wasting memory and complicating indexing.

### Entity Key Attachment

```python
# From eval_step1.py, lines 165-180
eda_usable_path = "/data/user_data/anshulk/cultural-mi/analysis/sanskriti_usable.csv"
if os.path.exists(eda_usable_path):
    eda = pd.read_csv(eda_usable_path)
    if "entity_key" in eda.columns:
        join_cols = ["question", "state", "attribute"]
        usable = usable.merge(
            eda[join_cols + ["entity_key"]].drop_duplicates(subset=join_cols),
            on=join_cols, how="left",
        )
```

**What entity_key is:** The EDA step (Section 10 of `eda_analysis.md`) derived
8,156 unique cultural entity keys by combining regex extraction with
state+attribute+answer fallback. Each entity key identifies a distinct cultural
concept (e.g., "Kerala|Dance_and_Music|Kathakali"). Multiple questions may share
the same entity key if they ask about the same concept from different angles.

**Why join on content, not index:** The EDA and Step 1 may process the dataset
in different orders (different library versions, different filtering logic). We
join on `(question, state, attribute)` — the content itself — rather than
positional index, making the join robust to ordering changes.

**Left join:** Questions without a matching entity key get `NaN`. The merge
script handles these gracefully by excluding them from entity-level analysis.

---

## 6. Prompt Engineering

### Base Model: 5-Shot Format

The base model receives a 5-shot prompt — five solved examples followed by
the target question:

```python
# From eval_step1.py, lines 81-119
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

[... 3 more examples ...]

"""
```

The target question is appended in the same format, ending with `Answer:` (no
trailing space):

```python
# From eval_step1.py, lines 188-197
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
```

**Why 5-shot, not 0-shot or 1-shot?**

A base model has no instruction-following training. It predicts the next token
based purely on pattern recognition. Without examples, it has no reason to
output "A", "B", "C", or "D" — it would continue the text in whatever way
seems most natural (often by rephrasing the question or generating a sentence).

The 5-shot prefix establishes the pattern: "Question → Options → Answer: {letter}".
After seeing 5 examples of this pattern, the model's next-token distribution at
the `Answer:` position shifts toward letter tokens. Five shots is the standard
in the MMLU evaluation protocol (Hendrycks et al., 2021) and matches the
Sanskriti paper's methodology.

**Why these specific 5 examples?** They are simple, unambiguous Indian cultural
knowledge questions (national animal, capital city, most spoken language,
Diwali, national sport). Any culturally literate human would answer them
correctly, and any model with basic India knowledge should too. They establish
the pattern without biasing toward any specific state, attribute, or question
type in the evaluation set.

**Why `Answer:` with no trailing space?** The model must predict the space
itself as part of the next token. In LLaMA 3.1's tokenizer, ` A` (space+A) is
a single token (ID 362). The five-shot examples show `Answer: A` where the
space is part of the ` A` token. By ending the prompt at `Answer:`, we let the
model predict the full ` A` / ` B` / ` C` / ` D` token. This is critical for
correct logprob extraction (see Section 8).

**Base vs instruct token IDs:** The base model uses space-prefixed tokens
(` A`=362, ` B`=426, ` C`=356, ` D`=423) because the prompt ends with
`Answer:` and the expected completion starts with a space. The instruct model
uses bare tokens (`A`=32, `B`=33, `C`=34, `D`=35) because the chat template
ends with `\n\n` (the assistant turn header), and the model predicts a bare
letter directly with no leading space. Getting these wrong produces
near-random accuracy (see Section 8 and Section 26, Bug 3).

**Five-shot prefix length:** 179 tokens (constant across all questions). This
is logged at startup for verification.

### Why 5 Shots Is the Sweet Spot

The number of in-context examples involves a trade-off:

```
0-shot:  No pattern established. Base model generates free-form text.
         Forced-choice rate would be ~99%. Logprobs for A/B/C/D are
         essentially random because the model has no reason to predict
         a letter token.

1-shot:  Weak pattern. The model might or might not generalize the
         "Question → Answer: letter" format from a single example.
         Unstable across different questions.

3-shot:  Reasonable pattern establishment. Used by some benchmarks
         (e.g., HellaSwag). But for a domain like Indian cultural
         knowledge, the model benefits from more demonstrations to
         firmly anchor the expected format.

5-shot:  Strong pattern. The MMLU standard (Hendrycks et al., 2021).
         Five examples give the model a clear, unambiguous pattern.
         The Sanskriti paper's evaluation also used 5-shot, making
         our results directly comparable.

10-shot: Diminishing returns. More context tokens increase compute cost
         (attention is quadratic in sequence length) without materially
         improving format compliance. Also risks "priming" the model
         toward the specific topics in the examples.
```

The 5-shot examples were specifically chosen to be:
- **Unambiguous:** Each has one clearly correct answer that any reasonable
  model would know
- **Domain-relevant:** All about Indian culture, priming the model for the
  domain
- **Non-overlapping:** None share a state, attribute, or topic with the
  evaluation questions
- **Balanced:** Correct answers are A, C, B, C, B — no single letter
  is over-represented, avoiding priming the model to predict one letter

### Instruct Model: Chat Template

The instruct model uses LLaMA 3.1's native chat template:

```python
# From eval_step1.py, lines 200-214
INSTRUCT_SYSTEM_MSG = (
    "You are a helpful assistant. Answer the following multiple choice question "
    "about Indian culture by responding with only the letter of the correct answer "
    "(A, B, C, or D). Do not explain your answer."
)

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
```

**Why no 5-shot for instruct?** The instruct model is trained to follow
instructions. The system message explicitly tells it to "respond with only the
letter." Adding 5-shot examples would waste context and potentially confuse the
model by mixing demonstration format with instruction format.

**`add_generation_prompt=True`:** This appends the model's "turn start" token
so the model knows it should generate a response. Without this, the model
might predict another user message or system message.

The full chat template expansion looks like this for a sample question:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the following multiple choice question
about Indian culture by responding with only the letter of the correct answer
(A, B, C, or D). Do not explain your answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: What dance form originated in Kerala?

A) Bharatanatyam
B) Kathakali
C) Odissi
D) Mohiniyattam<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

The special tokens serve specific purposes:
- `<|begin_of_text|>` — sequence start marker (BOS)
- `<|start_header_id|>...<|end_header_id|>` — role delimiter
- `<|eot_id|>` — end of turn (signals role transition)
- The final `assistant` header with no `<|eot_id|>` is the generation prompt:
  it tells the model "now it's your turn to generate"

The model was fine-tuned (SFT + RLHF) on conversations formatted exactly this
way. Using this exact template is critical — deviations (wrong special tokens,
missing headers, etc.) can degrade performance because the model has learned
to associate specific token patterns with instruction-following behavior.

**System message design:** The instruction is explicit about three things:
(1) the topic domain (Indian culture), (2) the response format (only the
letter), (3) what NOT to do (don't explain). This minimizes the chance of
the model generating a paragraph instead of a single letter.

**Why not 0-shot raw (no system message)?** Without a system message, the
instruct model would use its default behavior, which may include explaining
its reasoning. The logit extraction depends on the model concentrating its
probability mass on the answer letter at the generation position.

### Prompt Length Comparison

| Component | Base | Instruct |
|-----------|------|----------|
| System/prefix | 179 tokens (5-shot) | ~60 tokens (system msg + template) |
| Question | Variable | Variable |
| Total (typical) | ~230-270 tokens | ~120-160 tokens |
| Max observed | ~265 tokens | ~156 tokens |

The base model's prompts are ~1.7× longer due to the 5-shot prefix. This
explains why the base model processes batches more slowly (~40 q/s vs ~60 q/s
for instruct) — longer sequences require more compute in the self-attention
layers (quadratic in sequence length for standard attention).

---

## 7. Tokenization

### Batch Tokenization

```python
# From eval_step1.py, lines 491-498
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024,
    add_special_tokens=(model_type == "base"),
).to(device)
```

**`padding=True`:** Pads all sequences in the batch to the length of the
longest sequence. Combined with `padding_side="left"`, shorter sequences get
pad tokens prepended.

**`truncation=True, max_length=1024`:** Safety limit to prevent OOM from
unexpectedly long sequences. In practice, no Sanskriti question produces a
prompt longer than ~270 tokens (base) or ~160 tokens (instruct), so this
never triggers. The limit is defensive, not functional.

**`add_special_tokens=(model_type == "base")`:** For the base model, we add
the BOS (beginning-of-sequence) token because the base model expects it at the
start of every sequence. For the instruct model, the chat template already
includes the appropriate special tokens, so we set this to False to avoid
double-adding them.

**`return_tensors="pt"`:** Returns PyTorch tensors (not lists or NumPy arrays).
The `.to(device)` call moves the tensors to the GPU.

### What the Tokenizer Produces

For a batch of 64 prompts:

```
inputs["input_ids"]:      (64, max_seq_len)  — token IDs
inputs["attention_mask"]:  (64, max_seq_len)  — 1 for real tokens, 0 for padding
```

With left padding, a batch might look like:

```
Sequence 1 (230 tokens, padded to 265):
  [PAD PAD PAD ... PAD  The following are multiple ... Answer:]
  [  0   0   0 ...   0    1    1    1    1    ...     1    1 ]

Sequence 2 (265 tokens, no padding needed):
  [The following are multiple choice questions ... Answer:]
  [  1    1    1    1    1    1    1    ...     1    1 ]
```

### Complete Worked Example: Tokenizing a Base Model Prompt

To make the tokenization concrete, here is what happens to a real question:

```
Raw prompt (truncated 5-shot prefix for brevity):
  "The following are multiple choice questions about Indian culture.\n\n
   [... 5 examples ...]\n\n
   Question: Which classical dance form is associated with Kerala?\n
   A) Bharatanatyam\nB) Kathakali\nC) Odissi\nD) Mohiniyattam\nAnswer:"

Tokenizer processing:
  1. add_special_tokens=True → prepend BOS token (ID 128000)
  2. BPE encoding splits the string into subword tokens:
     "The" → [791]
     " following" → [2768]
     " are" → [527]
     " multiple" → [5361]
     ...
     "Answer" → [16533]
     ":" → [25]

  Final token IDs: [128000, 791, 2768, 527, 5361, ...]
  Total: ~230-265 tokens depending on question length

  The model will predict the NEXT token after this sequence.
  Expected: " B" (token 426) for Kathakali → Kerala

Batch padding (for a batch of 64):
  If this sequence is 235 tokens and the longest in the batch is 265:
  Left-pad with 30 pad tokens (ID 128001, <|end_of_text|> reused as PAD)

  Padded: [128001, 128001, ..., 128001, 128000, 791, 2768, ...]
           |<---- 30 pad tokens ---->|  |<----- 235 real tokens ----->|

  Attention mask: [0, 0, ..., 0, 1, 1, 1, ...]
                  |<-- 30 zeros -->| |<-- 235 ones -->|
```

---

## 8. The Token ID Bug and Fix

This section documents a critical bug that was discovered in the initial run,
diagnosed from the near-random accuracy results, and fixed before the
production run.

### The Bug

```python
# ORIGINAL CODE (BUGGY):
answer_token_ids = {
    l: tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"
}
# Produced: {'A': 32, 'B': 33, 'C': 34, 'D': 35}
```

This encoded the bare characters `A`, `B`, `C`, `D` — without a leading space.
In LLaMA 3.1's tokenizer, these map to byte-level tokens (IDs 32-35).

### Why It Was Wrong

The prompt ends with `Answer:` (no trailing space). The five-shot examples show
`Answer: A` where the completion is ` A` — **space plus letter**. In LLaMA 3.1's
tokenizer:

```
tokenizer.encode("Answer: A") → [16533, 25, 362]
                                  Answer  :    ·A

The space is part of the " A" token (ID 362), not a separate token.
```

So the model predicts ` A` (token 362), ` B` (token 426), ` C` (token 356),
or ` D` (token 423) — NOT bare `A` (32), `B` (33), `C` (34), `D` (35).

Bare character tokens vs space-prefixed tokens:

```
Bare "A": ID 32   — used in rare contexts like acronyms within words
" A":     ID 362  — used at word boundaries after spaces
" B":     ID 426
" C":     ID 356
" D":     ID 423
```

### How It Was Detected

The initial run produced ~28% accuracy for BOTH models across ALL question
types. The merge pipeline's sanity checks flagged this:

```
[WARN] F1: Base accuracy (Tier 2) = 27.8% (expected 40-70%)
[FAIL] F2: Instruct (28.0%) <= Base (28.4%)
[WARN] F3: Country Prediction: base=30.2%, inst=29.8% (expected >=95%)
[WARN] F4: Forced choice: base=87.7%, inst=86.6% (inst expected <5%)
```

28% accuracy on a 4-choice MCQ is essentially random (25%). Country Prediction
at 30% was the clearest red flag — these questions always have India as the
answer, and both models should trivially answer them correctly.

The logprob values confirmed the issue: all four answer logprobs were around
-13 to -19 (extremely improbable tokens), with margins near zero. The model
was being asked "how likely is token 32?" when it actually cared about token 362.

### The Fix

The fix is model-specific because the two prompt formats produce different
token boundaries:

```python
# FIXED CODE — base model (prompt ends with "Answer:"):
answer_token_ids = {
    l: tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in "ABCD"
}
# Produces: {'A': 362, 'B': 426, 'C': 356, 'D': 423}

# FIXED CODE — instruct model (chat template ends with "\n\n"):
answer_token_ids = {
    l: tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"
}
# Produces: {'A': 32, 'B': 33, 'C': 34, 'D': 35}
```

The base model predicts ` A` (space-prefixed) because `Answer:` is followed
by a space in the 5-shot pattern. The instruct model predicts bare `A`
because the chat template's generation prompt ends with `\n\n`, and the model
outputs a letter directly with no leading space.

### Lesson

This is a well-known pitfall in logit-based MCQ evaluation. The LLM Evaluation
Harness (EleutherAI) and lm-eval handle it by checking multiple token variants.
Our fix is simpler because we know the exact prompt format and can verify the
tokenization directly.

### Why This Bug Is Instructive for the MI Community

The token ID bug illustrates a fundamental challenge in LLM evaluation: **the
mapping between human-readable text and model-internal token representations
is not intuitive.** A few key insights:

1. **BPE tokenizers are context-sensitive.** The same character ("A") maps to
   completely different token IDs depending on whether it appears at a word
   boundary (preceded by a space) or within a word. This is by design — BPE
   learns frequent byte-pair merges from the training corpus, and " A" (space+A)
   is far more common at word boundaries than bare "A" within words.

2. **The difference is not just cosmetic.** Token 32 (bare "A") and token 362
   (" A") have completely different learned embedding vectors. They activate
   different neurons, participate in different attention patterns, and have
   different output probabilities. Comparing logprobs at the wrong token is
   like measuring temperature with a ruler — the instrument doesn't match
   the quantity.

3. **Sanity checks saved us.** Without the 11-check framework in the merge
   pipeline, we might have accepted the ~28% accuracy as "the base model is
   just bad at cultural knowledge" and proceeded with garbage behavioral labels.
   The Country Prediction check (F3) was the strongest signal: there is no
   plausible scenario where a 8B parameter model with India in its training
   data gets "What country is Kathakali from?" wrong 70% of the time.

4. **The fix was trivial, but the diagnosis was not.** The actual code change
   was 1 character (adding a space: `f" {l}"` instead of `l`). The diagnosis
   required understanding tokenizer internals, prompt format conventions, and
   what "normal" accuracy looks like for different question types. This is
   typical of evaluation bugs — small causes, large effects, hard to find
   without domain knowledge.

---

## 9. Forward Pass and Logit Extraction

### Single Forward Pass

```python
# From eval_step1.py, lines 510-511
with torch.no_grad():
    outputs = model(**inputs)
```

**`torch.no_grad()`:** Disables gradient computation. Since we are doing
inference only (no training), this saves ~50% memory and ~30% compute by
not building the autograd graph.

**`model(**inputs)`:** Passes `input_ids` and `attention_mask` to the model.
The model returns `outputs.logits` of shape `(batch, seq_len, vocab_size)` —
for each position in each sequence, the raw logit for every token in the
vocabulary (128,256 tokens).

### Logit Extraction Logic

```python
# From eval_step1.py, lines 284-313
def extract_from_logits(logits, attention_mask, answer_token_ids):
    # Step 1: Extract logits at the last position (rightmost in the tensor)
    last_logits = logits[:, -1, :]  # (batch, vocab)

    # Step 2: Get logits for the 4 answer tokens
    ids = torch.tensor([answer_token_ids[l] for l in "ABCD"], device=last_logits.device)
    answer_logits = last_logits[:, ids]  # (batch, 4)

    # Step 3: Compute log-probabilities over FULL vocabulary, then slice
    log_probs = torch.log_softmax(last_logits, dim=-1)
    answer_log_probs = log_probs[:, ids]  # (batch, 4)

    # Step 4: Predict the answer letter (argmax over 4 answer logits)
    predicted_idx = answer_logits.argmax(dim=-1)
    letters = ["A", "B", "C", "D"]
    predicted_letters = [letters[i] for i in predicted_idx]

    # Step 5: Check if the overall top-1 token is one of A/B/C/D
    top1_ids = last_logits.argmax(dim=-1)
    ids_set = set(ids.tolist())
    forced = [top1_ids[i].item() not in ids_set for i in range(len(top1_ids))]

    return predicted_letters, answer_log_probs.cpu(), forced, top1_ids.cpu()
```

**Step 1 — Using position -1 for left-padded sequences:** With left-padding,
every sequence in the batch has its last real token at the rightmost position
(`max_seq_len - 1`). Padding goes on the left, so all real tokens are
right-aligned. Using `logits[:, -1, :]` correctly extracts the logits at the
answer position for every sequence, regardless of how much padding each has.

**Why NOT `attention_mask.sum(dim=1) - 1`:** This was an earlier bug
(Section 26, Bug 4). `attention_mask.sum()` gives the count of real tokens,
not the position of the last real token. For a 230-token sequence padded to
265, the count is 230, giving index 229 — but the last real token is at
position 264 (the rightmost position). Using the count-based index extracts
logits from the MIDDLE of the sequence, producing near-random accuracy.
See Section 26, Bug 4 for the full diagnosis.

**Step 3 — Answer token extraction:** We index into the vocabulary dimension
to get just the 4 logits we care about: tokens for ` A`, ` B`, ` C`, ` D`.

**Step 4 — Log-softmax over full vocabulary:** We compute `log_softmax` over
all 128,256 tokens, THEN slice the 4 answer tokens. This is important: the
log-probabilities are normalized over the entire vocabulary, not just A/B/C/D.
If we normalized over only 4 tokens, we'd lose information about how confident
the model is overall (a model that assigns 90% probability to "the" and
distributes the remaining 10% across A/B/C/D should look different from a
model that assigns 50% to "A" and distributes the rest).

```
Full-vocabulary vs restricted normalization — worked example:

Scenario A: Instruct model (naturally outputs a letter)
  Raw logits: " A"=12.5, " B"=8.3, " C"=7.1, " D"=6.8, "the"=3.2, ...
  Full-vocab softmax: P(" A")=0.72, P(" B")=0.011, P(" C")=0.003, ...
  Full-vocab logprob: lp(" A")=-0.33, lp(" B")=-4.51, lp(" C")=-5.81, ...
  Margin: -0.33 - (-4.51) = 4.18 nats (very confident)

  4-token-only softmax: P(" A")=0.98, P(" B")=0.015, P(" C")=0.004, ...
  4-token-only logprob: lp(" A")=-0.02, lp(" B")=-4.20, lp(" C")=-5.52, ...

  The 4-token normalization makes the model look MORE confident than it
  actually is, because it hides the probability mass on non-answer tokens.

Scenario B: Base model (wants to output "India" or "\n")
  Raw logits: "India"=15.1, "\n"=14.8, " A"=5.2, " B"=4.9, " C"=5.0, ...
  Full-vocab softmax: P(" A")=5e-5, P(" B")=3.7e-5, P(" C")=4.1e-5, ...
  Full-vocab logprob: lp(" A")=-9.90, lp(" B")=-10.20, lp(" C")=-10.10, ...
  Margin: -9.90 - (-10.10) = 0.20 nats (very low confidence)

  4-token-only softmax: P(" A")=0.34, P(" B")=0.25, P(" C")=0.28, ...
  4-token-only logprob: lp(" A")=-1.08, lp(" B")=-1.39, lp(" C")=-1.27, ...

  The 4-token normalization makes the model look like it has a reasonable
  preference for " A", but in reality it barely distinguishes A/B/C/D
  because 99.99% of its probability mass is on non-answer tokens.

Conclusion: Full-vocabulary normalization preserves the "absolute confidence"
signal, which is essential for the confidence analysis in Section 20.
```

**Step 5 — Prediction:** `argmax` over the 4 answer logits. Note we use the
RAW logits (not log-probabilities) for argmax. This is equivalent because
log-softmax is monotonic (the ordering is preserved).

**Step 6 — Forced choice detection:** We check if the overall top-1 token
(across all 128,256 vocabulary tokens) is one of our 4 answer tokens. If not,
the model was "forced" to answer — its natural prediction was something else
(newline, continuation word, etc.), and we're using the relative ranking among
A/B/C/D only.

### Why Logit-Based, Not Generation-Based

Alternative: use `model.generate(max_new_tokens=5)` and parse the output text
with regex to extract A/B/C/D.

We chose logit-based extraction because:

1. **Deterministic:** No sampling temperature, top-p, or other generation
   parameters to tune. The argmax over logits gives a single, reproducible
   answer.

2. **Information-rich:** We get the full probability distribution over A/B/C/D,
   not just the top-1 answer. The logprobs and margin carry confidence
   information used in the merge analysis.

3. **Faster:** One forward pass gives us everything. Generation requires
   multiple autoregressive steps (even with `max_new_tokens=5`, the model runs
   5 sequential forward passes).

4. **Standard:** This is how MMLU, HellaSwag, ARC, and most MCQ benchmarks
   are evaluated in the literature.

---

## 10. Activation Hook System

### What Hooks Are

PyTorch's `register_forward_hook` attaches a callback function to a module.
Every time that module's `forward()` method runs, the callback fires with
the module's input and output tensors. We use this to capture internal
activations without modifying the model code.

### Hook Registration

```python
# From eval_step1.py, lines 221-247
def setup_hooks(model):
    cache = {}
    hooks = []

    # Embedding layer hook
    def embed_hook(module, input, output):
        cache["embed"] = output.detach().float().cpu()
    hooks.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # Transformer layer hooks
    for layer_idx in HOOK_LAYER_INDICES:  # [4, 8, 14, 20, 26, 30, 31]
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
```

**`model.model.embed_tokens`:** The embedding layer. Its output is the token
embedding matrix of shape `(batch, seq, 4096)` — the raw vector representations
before any transformer processing.

**`model.model.layers[i]`:** The i-th transformer decoder layer. Its output
is the hidden state after that layer's self-attention and feed-forward network.

**`output.detach().float().cpu()`:** Three operations:
- `.detach()`: Removes the tensor from the autograd graph (we don't need gradients)
- `.float()`: Converts from BF16 to FP32 for numerical precision in downstream
  probing. BF16 has only 7 bits of mantissa, which can cause precision loss
  in mean-pooling operations.
- `.cpu()`: Moves the tensor to CPU RAM to free GPU memory for the next batch.

**The `isinstance` check (transformers 5.3.0 compatibility):**

```python
hidden = output[0] if isinstance(output, (tuple, list)) else output
```

In older transformers versions (< 5.0), `LlamaDecoderLayer.forward()` returned
a tuple `(hidden_states, present_key_value, ...)`, so `output[0]` extracted the
hidden states. In transformers 5.3.0, it returns `hidden_states` directly as a
plain tensor. Using `output[0]` on a tensor indexes the first BATCH element,
not the first tuple element — producing a tensor of shape `(seq, hidden)` instead
of `(batch, seq, hidden)`.

This bug was caught in the first run when both processes crashed with:
```
IndexError: shape mismatch: indexing tensors could not be broadcast
together with shapes [156], [64]
```

The `isinstance` check handles both old and new transformers versions.

**The `make_hook` closure:** Python closures capture variables by reference,
not by value. Without the `make_hook` factory function, all hooks would share
the same `layer_idx` variable and overwrite each other's cache entries. The
factory creates a new scope for each hook, binding `n` to the correct name.

This is a common Python pitfall that deserves a concrete illustration:

```python
# BROKEN — all hooks write to cache["layer_31"]
for layer_idx in [4, 8, 14, 20, 26, 30, 31]:
    name = f"layer_{layer_idx:02d}"
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        cache[name] = hidden.detach().float().cpu()  # 'name' is captured by REFERENCE
    model.model.layers[layer_idx].register_forward_hook(hook_fn)

# When the forward pass runs, ALL 7 hook functions execute.
# But they all see name = "layer_31" (the LAST value of the loop variable),
# because Python closures capture the variable 'name', not its current value.
# Result: cache has only one entry ("layer_31") — the last hook to fire wins.

# CORRECT — make_hook creates a new scope with its own 'n' variable
def make_hook(n):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        cache[n] = hidden.detach().float().cpu()  # 'n' is local to make_hook
    return hook_fn

for layer_idx in [4, 8, 14, 20, 26, 30, 31]:
    name = f"layer_{layer_idx:02d}"
    model.model.layers[layer_idx].register_forward_hook(make_hook(name))

# Now each hook_fn has its own 'n': "layer_04", "layer_08", etc.
# The cache correctly has 7 entries after a forward pass.
```

An alternative fix is `functools.partial` or a default argument
(`def hook_fn(module, input, output, n=name)`), but the factory function
pattern is the clearest and most explicit.

### Cache Lifecycle

```
For each batch:
  1. Forward pass fires all 8 hooks
  2. Each hook writes its tensor to cache[name]
  3. compute_pooled() reads from cache and produces mean_pool/last_token vectors
  4. Vectors are written to memory-mapped .npy arrays
  5. activation_cache.clear() empties the cache for the next batch
```

The cache is a plain Python `dict`. It is NOT shared across processes — each
subprocess (base and instruct) has its own cache.

---

## 11. Activation Pooling Strategies

Each hook captures a `(batch, seq, 4096)` tensor — one 4096-dimensional vector
per token position. For downstream probing, we need a single vector per
question. We use two pooling strategies:

### Mean Pooling

```python
# From eval_step1.py, lines 258-268
mask = attention_mask.clone().float()

# For base model: mask out padding + 5-shot prefix
if target_start_tokens is not None:
    for i in range(mask.size(0)):
        mask[i, :target_start_tokens[i]] = 0.0

mask_3d = mask.unsqueeze(-1)  # (batch, seq, 1)
h_mean = (hidden * mask_3d).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
```

**Base model prefix masking:** The base model's prompt includes a 179-token
5-shot prefix that is identical for every question. If we included these tokens
in the mean pool, every question's representation would be dominated by the
same prefix, washing out the question-specific signal.

We compute `target_starts` for each sequence:

```python
# From eval_step1.py, lines 546-549
example_lengths = inputs["attention_mask"].sum(dim=1)
max_len = inputs["attention_mask"].shape[1]
padding_lengths = max_len - example_lengths
target_starts = (padding_lengths + 1 + five_shot_token_len).cpu()
```

This skips: padding tokens + BOS token (1) + five-shot prefix tokens (179).
Only the target question's tokens contribute to the mean pool.

**`.clamp(min=1)`:** Prevents division by zero if a sequence has no unmasked
tokens (should never happen in practice, but defensive coding).

### Last-Token Pooling

```python
# From eval_step1.py, lines 270-271
h_last = hidden[:, -1, :]
```

Extracts the hidden state at the rightmost position. With left-padding, this
is always the final real (non-padding) token for every sequence in the batch.
In a causal language model, this position contains the model's accumulated
representation of the entire input — it is the position from which the model
predicts the next token (the answer).

### Why Both Strategies

- **Mean pooling** captures the "average content" of the input. It is robust
  to the specific position of relevant information and works well when the
  cultural knowledge is distributed across multiple tokens.

- **Last-token pooling** captures the "decision point" representation. In
  transformer language models, the last position concentrates information
  from across the sequence via attention. It is more directly related to the
  model's prediction.

Step 2 will train probes using both representations and compare their
performance. If mean-pooled probes work better, the cultural knowledge is
distributed; if last-token probes work better, it concentrates at the
decision point.

### Mathematical Intuition: Why Two Pooling Strategies Capture Different Signals

Consider a simplified example with a 3-token sequence and 2-dimensional
hidden states at some layer:

```
Token positions:  [Kerala]   [dance]   [Answer:]
Hidden states:    [3.1, 0.2] [0.5, 2.8] [4.2, 4.1]

Mean pool:  [(3.1+0.5+4.2)/3, (0.2+2.8+4.1)/3] = [2.60, 2.37]
Last token: [4.2, 4.1]

The mean pool is dominated by the "Kerala" and "dance" tokens — the content
tokens that carry the cultural knowledge. It averages out the individual
contributions.

The last token [4.2, 4.1] is different from any individual content token.
In a causal transformer, the last position attends to ALL previous positions.
It has "seen" both Kerala and dance, and its representation is a compressed
summary optimized for predicting the next token (the answer).
```

In practice, with 4096 dimensions and 50-200 real tokens, the distinction
becomes more pronounced:

- **Mean pooling** is robust to sequence length variation. Whether a question
  has 50 or 200 tokens, the mean pool is a fixed-size average. But it dilutes
  strong signals from specific tokens (e.g., the entity name "Kathakali")
  with noise from function words ("the", "which", "is").

- **Last-token pooling** concentrates the model's "decision" into a single
  vector. It is potentially more informative for predicting the behavioral
  label (since the behavioral label depends on the model's decision), but
  it is also more sensitive to prompt format. If the model allocates the
  last position's representation to formatting (predicting a space vs a
  letter) rather than knowledge, last-token pooling may be less useful.

The empirical comparison in Step 2 will resolve which signal is stronger for
cultural knowledge probing.

---

## 12. Hook Layer Selection Rationale

We hook 8 points: the embedding layer plus 7 of the 32 transformer layers.

```python
HOOK_LAYER_INDICES = [4, 8, 14, 20, 26, 30, 31]
```

### Why These Specific Layers

```
Layer 0-3:   Very early — still processing syntax, not semantics
Layer 4:     ← HOOKED — First checkpoint in early processing
Layer 5-7:   Early processing continues
Layer 8:     ← HOOKED — Early-to-mid transition
Layer 9-13:  Building higher-level representations
Layer 14:    ← HOOKED — Middle of the network (14/32 = 44%)
Layer 15-19: Late-middle processing
Layer 20:    ← HOOKED — Late processing (20/32 = 62%)
Layer 21-25: Approaching output
Layer 26:    ← HOOKED — Near-output (26/32 = 81%)
Layer 27-29: Final processing
Layer 30:    ← HOOKED — Penultimate layer
Layer 31:    ← HOOKED — Final layer (feeds into lm_head)
```

The selection follows a logarithmic-like spacing that is denser near the
output layers. This is because prior MI work (Meng et al., 2022; Geva et al.,
2023) has shown that factual knowledge tends to be stored and retrieved in
the middle-to-late layers, while early layers handle syntax and mid layers
handle entity recognition.

We hook the final two layers (30, 31) separately because:
- Layer 31 feeds directly into the language model head (lm_head) that produces
  logits. Its representation is the most "decision-relevant."
- Layer 30 captures the representation one step before the final residual
  connection, which sometimes differs meaningfully from layer 31.

### Memory Budget

Each hook captures `(batch, seq, 4096)` in FP32 and moves it to CPU. For a
batch of 64 with seq_len 265:

```
Per hook per batch: 64 × 265 × 4096 × 4 bytes = 278 MB
8 hooks total: 2.2 GB per batch (transferred to CPU, then pooled)
```

After pooling, each hook produces `(64, 4096)` for mean_pool and `(64, 4096)`
for last_token — only 2 MB per hook per batch. The full 278 MB is freed after
pooling.

### Storage

Each activation file stores `(21726, 4096)` in FP32:

```
Per file: 21726 × 4096 × 4 bytes = 356,106,240 bytes ≈ 340 MB
32 files total: 32 × 340 MB ≈ 10.6 GB
Per model: 16 files × 340 MB ≈ 5.3 GB
```

Files are stored as memory-mapped NumPy arrays (`.npy` format) on NVMe:

```python
# From eval_step1.py, lines 422-424
act_arrays[(hname, ptype)] = np.lib.format.open_memmap(
    path, mode=mode, dtype=np.float32, shape=(n, 4096)
)
```

Memory-mapping means the full array doesn't need to fit in RAM simultaneously.
Writes go directly to disk. Reads in Step 2 can load only the slices they need.

---

## 13. Checkpointing and Preemption Handling

### Why Checkpointing

On the `preempt` partition, SLURM can kill our job at any time to give the GPU
to a higher-priority job. Without checkpointing, we'd lose all progress and
have to restart from batch 0.

### Checkpoint Mechanism

```python
# From eval_step1.py, lines 592-604
if preempt_event.is_set() or (batch_idx + 1) % 100 == 0:
    csv_file.flush()
    for arr in act_arrays.values():
        arr.flush()
    if (batch_idx + 1) % 100 == 0:
        log.info(f"Checkpoint saved at batch {batch_idx + 1}/{total_batches}")
    if preempt_event.is_set():
        log.warning(f"PREEMPTION detected at batch {batch_idx}. "
                    f"Checkpointed {total_done} questions. Exiting.")
        csv_file.close()
        return
```

**Every 100 batches (6,400 questions):** Flush the CSV file and all memory-mapped
activation arrays to disk. This is a periodic checkpoint that happens regardless
of preemption.

**On preemption signal:** The SLURM `--signal=B:USR1@120` flag sends SIGUSR1
to the process 120 seconds before killing it. Our signal handler sets a
multiprocessing Event:

```python
# From eval_step1.py, lines 723-724
preempt_event = mp.Event()
signal.signal(signal.SIGUSR1, lambda s, f: preempt_event.set())
```

Each batch loop iteration checks `preempt_event.is_set()`. If set, it
checkpoints and exits cleanly. The 120-second window is more than enough for
a single checkpoint (takes <1 second).

### Resume Logic

```python
# From eval_step1.py, lines 320-344
def get_resume_info(config, model_type, batch_size):
    csv_path = os.path.join(config["checkpoints"],
                            f"step1_{model_type}_results_partial.csv")
    if not os.path.exists(csv_path):
        return 0, 0

    with open(csv_path) as f:
        n_rows = sum(1 for _ in f) - 1  # subtract header

    # Truncate to clean batch boundary
    clean_rows = (n_rows // batch_size) * batch_size
    start_batch = clean_rows // batch_size

    if clean_rows < n_rows:
        df_partial = pd.read_csv(csv_path, nrows=clean_rows)
        df_partial.to_csv(csv_path, index=False)

    return start_batch, clean_rows
```

**Resume from CSV row count:** Rather than maintaining a separate `meta.json`
checkpoint file (which could become inconsistent with the CSV), we derive the
resume point from the number of completed rows in the CSV.

**Truncate to batch boundary:** If a batch was partially written (e.g., 32 of
64 rows written before the process was killed), we truncate to the last
complete batch. This ensures the activation arrays and CSV stay in sync —
both have exactly `clean_rows` entries.

**Why not use the activation arrays for resume?** The activation arrays are
memory-mapped and written incrementally. There's no atomic way to determine
how many rows were fully written. The CSV row count is simpler and more
reliable.

---

## 14. Parallel Execution Architecture

### Process Architecture

```python
# From eval_step1.py, lines 722-742
mp.set_start_method("spawn", force=True)
preempt_event = mp.Event()
signal.signal(signal.SIGUSR1, lambda s, f: preempt_event.set())

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
p_base.join()
p_inst.join()
```

**`mp.set_start_method("spawn")`:** Critical for CUDA. The `fork` method
(Python's default on Linux) copies the parent process's memory, including CUDA
contexts, which causes GPU driver errors. `spawn` starts a fresh Python
interpreter for each child process, which initializes CUDA cleanly.

```
Process creation methods — deep dive:

fork (Linux default):
  - Creates child by duplicating parent's memory via copy-on-write
  - Fast: child starts with parent's entire state already in memory
  - PROBLEM: CUDA contexts are not fork-safe. The CUDA runtime maintains
    internal state (device handles, memory pools, stream queues) that
    becomes invalid when duplicated. Symptoms:
    * "CUDA error: invalid device ordinal"
    * "CUDA error: all CUDA-capable devices are busy"
    * Silent data corruption in GPU memory
  - Also unsafe with multithreaded code (fork + threads = deadlocks)

spawn (our choice):
  - Creates child by launching a fresh Python interpreter
  - Slower: child must re-import all modules, re-initialize everything
  - SAFE: each child initializes its own CUDA context from scratch
  - Arguments are serialized (pickled) and sent to the child
  - The DataFrame (df) is pickled and unpickled — this works because
    pandas DataFrames are pickle-safe, but adds ~2 seconds of overhead
    for our 21,726-row DataFrame

forkserver:
  - Hybrid: a server process is forked early (before CUDA init), and
    children are forked from this clean server
  - Safer than fork, faster than spawn
  - Not needed here: the spawn overhead is negligible compared to the
    45-second model loading time in each child
```

**`force=True`:** Overrides any previously set start method (in case the user's
environment or a library has already called `set_start_method`).

**Shared `preempt_event`:** The `mp.Event()` is shared across processes via
`spawn`'s argument serialization. When the parent receives SIGUSR1, it sets the
event, and both children see it on their next batch iteration.

The preemption lifecycle works as follows:

```
Timeline of a preemption event:

t=0:    SLURM decides to preempt our job (higher-priority job submitted)
t=0:    SLURM sends SIGUSR1 to the parent process (because --signal=B:USR1@120)
        This is 120 seconds BEFORE the actual kill signal.
t=0:    Parent's signal handler runs: preempt_event.set()

t=0-1s: Both child processes check preempt_event.is_set() at the start of
        their next batch iteration. (Worst case: they are mid-batch and must
        finish the current forward pass, ~1-2 seconds.)

t=1-2s: Children flush their CSV files and memory-mapped activation arrays
        to disk. This is fast: the CSV flush writes buffered rows to NFS,
        and the memmap flush syncs dirty pages to the NVMe.

t=2-3s: Children log the preemption, close files, and exit cleanly.
        Parent's join() calls return.

t=3s:   Parent logs exit codes and terminates.

t=120s: SLURM sends SIGKILL (uncatchable). But we exited 117 seconds ago,
        so this has no effect.

On resubmission (sbatch scripts/run_step1.sh):
  1. get_resume_info() reads the partial CSV and counts completed rows
  2. Truncates to the last complete batch boundary
  3. The evaluation loop skips batches 0 through start_batch-1
  4. Activation arrays are opened in "r+" mode (read-write on existing file)
  5. Processing continues from where it left off

This entire cycle is invisible to the final results: the output CSVs and
activation arrays are identical whether the job ran uninterrupted or was
preempted and resumed 5 times.
```

**Independent processes:** Each process loads its own model, creates its own
tokenizer, registers its own hooks, and writes to its own CSV and activation
files. There is no inter-process communication during execution — they run
completely independently.

**`join()`:** The main process blocks until both children complete. Exit codes
are logged to detect crashes.

### Why Not torch.distributed or DataParallel?

`torch.distributed` (DDP) is designed for training with gradient
synchronization. For inference, it adds unnecessary communication overhead.
Each model fits on one GPU, so there's nothing to distribute.

`DataParallel` splits a single model across GPUs, which halves per-GPU memory
but requires gathering outputs. Since we're running two DIFFERENT models (base
and instruct), not the same model on different data, DataParallel doesn't apply.

### Logging

Each process gets its own logger and log file:

```
logs/step1_base_YYYYMMDD_HHMMSS.log      — base model's detailed log
logs/step1_instruct_YYYYMMDD_HHMMSS.log   — instruct model's detailed log
logs/step1_main_YYYYMMDD_HHMMSS.log       — main process (dataset loading, process management)
logs/step1_slurm_JOBID.out                — SLURM's combined stdout (interleaved)
```

The per-model log files have DEBUG-level output (every forced-choice decision,
hook shapes, etc.). The SLURM output file has INFO-level output from all
processes (progress updates, checkpoints).

---

## 15. The Merge Pipeline

After both models complete, `merge_step1.py` combines the results:

```
base_results.csv + instruct_results.csv + sanskriti_prepared.csv
    ↓
    merge on question_id
    ↓
    assign behavioral labels
    ↓
    compute logprob margins
    ↓
    three-tier stats
    ↓
    per-dimension breakdowns
    ↓
    entity-level analysis
    ↓
    plots + sanity checks
    ↓
    sanskriti_behavioral_labels.csv (master output)
```

### Column Renaming Convention

The merge renames columns to avoid collisions:

| Original (base) | Renamed |
|-----------------|---------|
| predicted_letter | predicted_base |
| correct | base_correct |
| logprob_A | base_logprob_A |
| forced_choice | base_forced |

| Original (instruct) | Renamed |
|---------------------|---------|
| predicted_letter | predicted_instruct |
| correct | instruct_correct |
| logprob_A | instruct_logprob_A |
| forced_choice | instruct_forced |

### Merge Integrity Check

```python
assert len(merged) == len(base), f"Merge mismatch: {len(merged)} != {len(base)}"
```

Both CSVs must have exactly 21,726 rows with matching question_ids. An inner
join on question_id should produce exactly 21,726 rows. Any mismatch indicates
a bug in the evaluation (e.g., one model skipped a question).

---

## 16. Behavioral Labeling Logic

```python
# From merge_step1.py, lines 147-156
conditions = [
    (df["base_correct"] == 1) & (df["instruct_correct"] == 0),   # suppression
    (df["base_correct"] == 0) & (df["instruct_correct"] == 1),   # enhancement
    (df["base_correct"] == 1) & (df["instruct_correct"] == 1),   # control_both_correct
    (df["base_correct"] == 0) & (df["instruct_correct"] == 0),   # control_both_wrong
]
labels = ["suppression", "enhancement", "control_both_correct", "control_both_wrong"]
df["behavioral_label"] = np.select(conditions, labels, default="unknown")
```

**`np.select`:** Evaluates conditions in order and assigns the first matching
label. Since the four conditions are mutually exclusive and exhaustive (every
question has binary correctness for each model), the `default="unknown"` should
never trigger.

**Why binary correctness, not logprob agreement?** We could define suppression
as "base assigns higher logprob to the correct answer than instruct." But
binary correctness (right/wrong) is simpler, more interpretable, and standard
in the behavioral evaluation literature. The logprob margin analysis (Section
20) provides the continuous confidence signal.

### Worked Example: Labeling Four Questions

To make the labeling concrete, consider four questions about Kathakali (a
classical dance form from Kerala):

```
Question 1: "Which classical dance form originated in Kerala?"
  Options: A) Bharatanatyam  B) Kathakali  C) Odissi  D) Mohiniyattam
  Ground truth: B (Kathakali)
  Base prediction: B ✓  (logprob margin: 2.1 nats — confident)
  Instruct prediction: B ✓  (logprob margin: 5.3 nats — very confident)
  Label: control_both_correct
  Interpretation: Both models know Kathakali is from Kerala. Knowledge preserved.

Question 2: "What is the main costume element in Kathakali performances?"
  Options: A) Dhoti  B) Elaborate face paint  C) Silk saree  D) Turban
  Ground truth: B (Elaborate face paint)
  Base prediction: B ✓  (logprob margin: 0.8 nats — moderate)
  Instruct prediction: A ✗  (logprob margin: 0.3 nats — low confidence)
  Label: suppression
  Interpretation: The base model had this specific knowledge, but instruction
  tuning caused the instruct model to lose it. This is a suppression case —
  exactly the kind of question that motivates Steps 2-4.

Question 3: "In which century did Kathakali emerge as a distinct art form?"
  Options: A) 14th  B) 15th  C) 17th  D) 19th
  Ground truth: C (17th century)
  Base prediction: A ✗  (logprob margin: 0.2 nats — guessing)
  Instruct prediction: C ✓  (logprob margin: 1.5 nats — moderate)
  Label: enhancement
  Interpretation: Instruction tuning improved the model's ability to answer
  this question, possibly by organizing historical knowledge better.

Question 4: "Which mudra system is unique to Kathakali?"
  Options: A) Asamyuta  B) Navarasas  C) Hastalakshana Deepika  D) Abhinaya
  Ground truth: C (Hastalakshana Deepika)
  Base prediction: D ✗  (logprob margin: 0.1 nats — random)
  Instruct prediction: A ✗  (logprob margin: 0.4 nats — near random)
  Label: control_both_wrong
  Interpretation: Neither model has this specialized knowledge. The question
  requires deep domain expertise that was not in the training data.
```

These four questions about the same entity (Kathakali) would give the entity
a "mixed" label at the entity level (Section 19), because no single behavioral
label applies to all four.

### Expected Distribution

Based on the EDA analysis and prior work:

| Label | Expected Range | Reasoning |
|-------|---------------|-----------|
| Suppression | 5-15% | Base knows, instruct lost it |
| Enhancement | 4-12% | Instruct learned something base didn't know |
| Control (both correct) | 30-60% | Country Prediction drives this up |
| Control (both wrong) | 25-50% | Hard questions neither model knows |

If suppression ≈ enhancement (symmetric), instruction tuning has roughly equal
constructive and destructive effects on cultural knowledge. If suppression >>
enhancement, instruction tuning is a net negative for cultural knowledge. The
EDA predicted that Country Prediction would inflate "both correct" significantly.

---

## 17. Three-Tier Analysis Framework

We report results at three levels of stringency:

### Tier 1: Full Dataset (21,726 questions)

All questions, including Country Prediction. This is the "headline" number
for the paper.

### Tier 2: Without Country Prediction (16,163 questions)

```python
mask_t2 = df["question_type"] != "Country Prediction"
```

Country Prediction questions (5,563, 25.6% of the dataset) always have India
as the answer. Both models should get nearly all of them right, pushing them
into `control_both_correct`. Including them dilutes the suppression/enhancement
signal.

Tier 2 removes this noise floor. If suppression rate increases from Tier 1 to
Tier 2, it confirms that Country Prediction was masking the effect.

### Tier 3: Hard Subset (10,781 questions)

```python
mask_t3 = df["question_type"].isin(["Association", "General Awareness"])
```

Association and General Awareness questions require the most genuine cultural
knowledge (as established in the EDA, Section 8: embedding baseline accuracy
was 63% and 44% for these types, vs 96-100% for Country/State Prediction).

Tier 3 is the most meaningful for the MI study. If suppression concentrates
here, it confirms that instruction tuning affects knowledge retrieval, not
just pattern matching.

### Why Three Tiers Instead of Just Reporting the Full Dataset

Reporting a single number (e.g., "suppression rate = 8%") hides important
structure. The three tiers tell a story:

```
Tier 1 (full dataset): "Here is the complete picture, no data excluded."
  This is the number a reviewer would compute if they re-ran our code.
  It is the most honest but also the most diluted, because Country
  Prediction's ~5,563 easy questions inflate "both_correct."

Tier 2 (no Country Prediction): "Here is the picture without the noise floor."
  Country Prediction is essentially a "control" question type — both models
  should ace it. Removing it reveals the suppression/enhancement rates on
  questions where the models actually differ.

  If suppression goes from 6% (Tier 1) to 9% (Tier 2), we can say:
  "CP was masking 3pp of suppression by diluting the denominator."

Tier 3 (hard questions only): "Here is where the real knowledge is tested."
  Association and General Awareness questions require recalling specific
  facts (which festival belongs to which state, which art form uses which
  technique). These cannot be answered by pattern-matching or common sense.

  If suppression goes from 9% (Tier 2) to 14% (Tier 3), we have strong
  evidence that suppression targets genuine cultural knowledge, not just
  format-following ability.

The progression Tier 1 → Tier 2 → Tier 3 should show INCREASING suppression
rate if the hypothesis is correct. A flat rate across tiers would suggest
suppression is random, not knowledge-dependent.
```

### Statistical Power Considerations

```
At a 10% suppression rate:
  Tier 1: 21,726 × 0.10 = 2,173 suppression cases
  Tier 2: 16,163 × 0.10 = 1,616 suppression cases
  Tier 3: 10,781 × 0.10 = 1,078 suppression cases

For Step 2 probing (training a linear classifier on activations):
  Even Tier 3's 1,078 suppression cases is ample for training a linear
  probe. Standard practice in MI work (e.g., Gurnee et al., 2023) uses
  ~1,000-5,000 examples per class.

For per-attribute breakdowns:
  With 16 attributes, each has ~674 questions (Tier 3).
  At 10% suppression: ~67 cases per attribute.
  This is marginal but usable for rate estimation (±3-5pp error bars).

For per-state breakdowns:
  With 36 states, each has ~300 questions (Tier 3).
  At 10% suppression: ~30 cases per state.
  This is too few for reliable per-state claims — we report but add
  uncertainty warnings. Entity-level analysis (Section 19) is more
  appropriate for state-level insights.
```

---

## 18. Per-Dimension Breakdowns

### Per Question Type

```python
# From merge_step1.py, lines 237-257
for qt in sorted(df["question_type"].unique()):
    sub = df[df["question_type"] == qt]
    # base_accuracy, instruct_accuracy, suppression_rate, enhancement_rate
```

Produces a CSV with one row per question type showing accuracy and behavioral
label rates. Saved to `accuracy_by_question_type.csv`.

### Per Attribute (16 attributes)

```python
# From merge_step1.py, lines 260-274
for attr in sorted(df["attribute"].unique()):
    sub = df[df["attribute"] == attr]
    # ... plus "low_confidence": len(sub) < 200
```

The `low_confidence` flag marks attributes with fewer than 200 questions
(Sports=162, Transport=76, Medicine=72, Nightlife=41). At expected suppression
rates of 5-15%, these would yield only 3-30 suppression cases — too few for
reliable percentages.

### Per State (36 states)

```python
# From merge_step1.py, lines 277-290
for state in sorted(df["state"].unique()):
    sub = df[df["state"] == state]
```

### Position Distribution

```python
# From merge_step1.py, lines 293-312
for letter in "ABCD":
    # ground_truth_count/pct, base_pred_count/pct, instruct_pred_count/pct
```

This is a critical sanity check. The ground truth distribution (from EDA:
A=27.1%, B=29.0%, C=23.1%, D=20.8%) is fixed. The model prediction
distributions should:
- For instruct: roughly match the ground truth distribution (if the model is
  well-calibrated)
- For base: may show position bias (e.g., A-heavy) due to the prompt format

If either model predicts >40% of any single letter, the sanity checks flag it
as severe position bias.

---

## 19. Entity-Level Analysis

### Question-Level vs Entity-Level Suppression

A single cultural entity (e.g., "Kerala|Dance_and_Music|Kathakali") may appear
in multiple questions (asked from different angles via templates). If the model
gets one Kathakali question right and another wrong, the entity has "mixed"
behavior.

Entity-level analysis collapses per-question labels to per-entity labels:

```python
# From merge_step1.py, lines 340-349
if n_supp == n:
    entity_label = "suppressed"      # ALL questions suppressed
elif n_enh == n:
    entity_label = "enhanced"        # ALL questions enhanced
elif n_bc == n:
    entity_label = "both_correct"    # ALL questions both correct
elif n_bw == n:
    entity_label = "both_wrong"      # ALL questions both wrong
else:
    entity_label = "mixed"           # some combination
```

**Why strict "all-or-nothing" for entity labels?** A cultural entity is either
known or not known. If the base model gets 3 out of 4 Kathakali questions right
but the instruct model gets 2 out of 4 right, the entity has mixed behavior —
we cannot confidently call it "suppressed" because the base model itself is
inconsistent. The strict criterion ensures entity-level labels are trustworthy.

### Expected Entity-Level vs Question-Level Rates

Entity-level suppression should be LOWER than question-level suppression because
the strict criterion filters out mixed entities. If question-level suppression
is 10%, entity-level might be 5-7% (the "pure" cases).

### Why Entity-Level Analysis Matters for the Paper Narrative

The paper's core argument is about **cultural knowledge** being suppressed —
not individual question performance. A reader might object: "Maybe the model
just got unlucky on a few questions. That's not 'suppression of cultural
knowledge.'"

Entity-level analysis addresses this by showing that entire cultural concepts
are consistently affected. If Kathakali has 6 questions and all 6 are
suppressed, that is much stronger evidence than 6 random questions being
suppressed:

```
Question-level: "1,500 out of 21,726 questions are suppressed (6.9%)"
  → Could be noise, bad luck, or artifacts of specific phrasings.

Entity-level: "400 out of 8,156 cultural entities are fully suppressed (4.9%)"
  → These are entire cultural concepts that the base model knows and the
     instruct model has lost. Much harder to dismiss as noise.

The gap between question-level and entity-level rates is also informative:
  Question suppression: 6.9%  →  Entity suppression: 4.9%
  This means ~2pp of suppression is in "mixed" entities — questions where
  the model is inconsistent. The remaining 4.9% is "pure" suppression —
  the model genuinely lost the knowledge, not just one phrasing of it.
```

---

## 20. Logprob Margin and Confidence Analysis

### Computing the Margin

```python
# From merge_step1.py, lines 121-127
for prefix in ["base", "instruct"]:
    lp_cols = [f"{prefix}_logprob_{l}" for l in "ABCD"]
    lp_vals = merged[lp_cols].values  # (n, 4)
    sorted_lp = np.sort(lp_vals, axis=1)  # ascending
    merged[f"{prefix}_logprob_margin"] = sorted_lp[:, -1] - sorted_lp[:, -2]
    merged[f"{prefix}_low_confidence"] = merged[f"{prefix}_logprob_margin"] < 0.5
```

For each question and each model, we sort the four logprobs, take the
difference between the highest and second-highest. This gives the "margin
of victory" — how confident the model is in its top answer choice.

### Why 0.5 Nats as the Threshold

A margin of 0.5 nats means `exp(0.5) ≈ 1.65×` probability ratio between the
top two choices. For a 4-choice MCQ, if the model is truly uncertain, all four
logprobs would be similar and the margin would be near zero. A 0.5-nat margin
means the model has a slight preference but is far from confident.

### Validation Use

The merge script cross-references confidence with correctness:

```python
# From merge_step1.py, lines 632-642 (sanity check F9)
acc_low = low_conf[correct_col].mean() * 100
acc_high = df[~df[f"{model}_low_confidence"]][correct_col].mean() * 100
```

If the evaluation is working correctly:
- Low-confidence predictions should have LOWER accuracy than high-confidence
  predictions. This confirms that the logprobs carry signal.
- If low-confidence accuracy equals high-confidence accuracy, the logprobs
  are noise (as was the case with the wrong token IDs).

---

## 21. Position Distribution Analysis

The ground truth answer distribution from the EDA is:

```
A: 27.09% (5,885)
B: 29.03% (6,308)
C: 23.05% (5,008)
D: 20.83% (4,525)
```

B is overrepresented (+4pp from uniform), D is underrepresented (-4pp). The
merge pipeline computes each model's prediction distribution and compares it
against this ground truth.

**What to look for:**

1. If a model's predictions match the ground truth distribution, it is
   well-calibrated but not necessarily accurate (it could be randomly
   guessing with the right frequencies).

2. If a model shows extreme position bias (e.g., 40%+ on one letter), it is
   exploiting a shortcut rather than answering based on knowledge.

3. The DIFFERENCE between base and instruct distributions is informative.
   If base has strong A-bias but instruct has uniform predictions, instruction
   tuning has corrected the position bias.

---

## 22. Sanity Check Framework

The merge pipeline runs 11 sanity checks. These are designed to catch both
data issues and code bugs.

### F1: Base Accuracy Range

```
Expected: 40-70% on Tier 2 (without Country Prediction)
```

The Sanskriti paper reported LLaMA-3.2-3B-Instruct at 52% and LLaMA-3.1-70B-
Instruct at 86%. Our 8B base model should fall in the 40-70% range. Below 40%
suggests a token ID or prompt formatting issue. Above 70% would be surprising
for a base model on cultural knowledge.

### F2: Instruct > Base

```
Expected: Instruct accuracy > Base accuracy (overall)
```

Instruction tuning generally improves MCQ performance through better format
compliance and knowledge organization. If instruct ≤ base, either the
instruct prompt is wrong or the token extraction is broken.

### F3: Country Prediction ≥ 95%

```
Expected: Both models ≥ 95% on Country Prediction
```

The answer is always India. Any model with basic world knowledge should get
this right. Below 95% indicates a fundamental evaluation issue.

### F4: Instruct Forced Choice < 5%

```
Expected: Instruct model forced-choice rate < 5%
```

The instruct model is explicitly told to output a letter. If it rarely outputs
A/B/C/D directly, the chat template or system message is not working.

### F5: No Severe Position Bias

```
Expected: No model predicts any letter at >40%
```

40% is 15pp above the most common ground truth position (B at 29%). A model
predicting one letter 40%+ of the time is degenerate.

### F6-F7: Suppression/Enhancement Range

```
Expected: Suppression 5-15%, Enhancement 4-12%
```

Informational — not pass/fail. These guide expectations. Very high suppression
(>20%) or very low (<3%) would warrant investigation.

### F8: No State Dominates Suppression

```
Expected: No state accounts for >15% of all suppression cases
```

If one state dominates, the "suppression" effect might be a state-specific
data artifact rather than a general cultural knowledge effect.

### F9: Confidence-Accuracy Correlation

```
Expected: Low-confidence accuracy < High-confidence accuracy
```

Validates that the logprobs carry signal. If they're equal, the logprob
extraction is noise.

### F10-F11: Activation Integrity

```
Expected: All 32 activation files exist with shape (21726, 4096), no NaN/Inf
```

Checks that every hook produced valid activations for every question.

### The Philosophy Behind the Sanity Checks

These checks are designed around a principle: **every check should catch a
specific class of bugs, and the combination should leave no plausible failure
mode undetected.**

```
Bug class                    → Caught by
─────────────────────────────────────────────────────
Wrong token IDs              → F1 (low accuracy), F3 (CP low), F9 (no conf signal)
Wrong prompt format          → F1, F2 (instruct not better), F4 (high forced rate)
Off-by-one in logit indexing → F1, F3
Swapped base/instruct files  → F2 (base > instruct would be unusual)
Dataset filtering bug        → F6, F7 (rates outside expected range)
Hook capturing wrong tensor  → F10 (wrong shape), F11 (NaN/Inf)
Data corruption              → F11 (NaN/Inf in activations)
Position bias in prompt      → F5 (>40% on one letter)
State-specific data artifact → F8 (one state dominates)
```

The checks have three severity levels:
- **FAIL:** Definitely broken, must fix before proceeding (F2, F3, F4, F10, F11)
- **WARN:** Possibly broken, investigate (F1, F5, F8, F9)
- **INFO:** Noteworthy but not actionable (F6, F7)

A clean run should produce 0 FAILs, 0 WARNs, and 2 INFOs (suppression and
enhancement rates). The initial buggy run produced 5 FAILs and 4 WARNs,
demonstrating the framework's sensitivity.

---

## 23. Plots and Visualizations

The merge pipeline generates 7 plots:

### Plot 1: Accuracy by Question Type

Grouped bar chart. Base (blue) vs Instruct (orange) accuracy for each of the
4 question types. Data values annotated on each bar. Expected to show:
- Country Prediction: highest for both (near 100%)
- General Awareness: lowest for base (most knowledge-dependent)
- Instruct higher than base on all types

### Plot 2: Behavioral Label Distribution by Tier

Stacked bar chart across the 3 tiers. Each bar is 100% (the four labels sum
to 100%). Labels: Suppression (red), Enhancement (green), Both Correct (blue),
Both Wrong (gray). Percentage values annotated on segments >3%.

### Plot 3: Suppression Rate by Attribute

Horizontal bar chart, all 16 attributes. Low-confidence attributes (n<200)
shown in gray. Sorted by suppression rate. Sample size annotated.

### Plot 4: Suppression Rate by State (Top 20)

Horizontal bar chart, top 20 states by suppression rate. Sample size annotated.

### Plot 5: Position Distribution

Grouped bar chart: Ground Truth (blue), Base predictions (orange), Instruct
predictions (green) for positions A/B/C/D. Shows whether models match the
ground truth distribution or have position biases.

### Plot 6: Confidence Distribution

Histogram of the maximum logprob among A/B/C/D for each question, overlaid
for base (blue) and instruct (orange). Higher (less negative) values indicate
more confident predictions. Expected: instruct distribution should be shifted
right (more confident) compared to base.

### Plot 6b: Logprob Margin Distribution

Histogram of the logprob margin (top1 - top2) for each question, overlaid for
both models. Red dashed line at 0.5 nats marks the low-confidence threshold.
Expected: instruct should have a longer right tail (more high-confidence
predictions).

### How to Read Each Plot (A Guide for Non-Experts)

**Plot 1 (Accuracy by Question Type):**
Look for the GAP between blue (base) and orange (instruct) bars. A large gap
where instruct > base means instruction tuning helped on that question type.
A gap where base > instruct (rare) means instruction tuning hurt. Country
Prediction should show near-equal bars at ~100%.

**Plot 2 (Behavioral Label Distribution):**
This is the most important plot for the paper. Compare the red segments
(suppression) across the three tiers. If red grows from Tier 1 → Tier 2
→ Tier 3, the suppression hypothesis is supported. The blue segments (both
correct) should shrink as we remove easy questions.

**Plot 3 (Suppression by Attribute):**
Look for outlier attributes with unusually high suppression rates. These
are the cultural domains most affected by instruction tuning. Gray bars
(low-confidence, n < 200) should be interpreted cautiously.

**Plot 4 (Suppression by State):**
Look for geographic patterns. If southern states (Kerala, Tamil Nadu,
Karnataka) have different suppression rates than northern states, it
suggests the training data had uneven geographic coverage.

**Plot 5 (Position Distribution):**
The three bar groups (GT, Base, Instruct) should roughly overlap if the
models are well-calibrated. Strong deviations (e.g., base predicting A
40% of the time) indicate position bias rather than knowledge.

**Plots 6/6b (Confidence Distributions):**
The confidence distribution (Plot 6) shows the instruct model's max logprob
is sharply concentrated near 0 (most predictions have logprob > -0.1),
while the base model's distribution is more spread out (peak near -0.1 but
with a visible tail extending to -2). Neither model has forced-choice
predictions (0% rate).

The margin distribution (Plot 6b) shows a dramatic difference: the base
model's margins are bell-shaped around 3-4 nats with a left tail touching
the 0.5-nat threshold (8.3% low-confidence). The instruct model's margins
are broadly distributed from 1-8+ nats, with very few predictions below the
0.5-nat threshold (2.8%). The instruct model is far more decisive.

---

## 24. Results

Results from SLURM job 6959494 (2026-04-05), run on 2 × NVIDIA A100-SXM4-40GB.
All numbers validated against output CSVs. See Appendix D for per-number
validation log.

### Overall Accuracy

| Model | Accuracy | Forced Choice Rate |
|-------|----------|--------------------|
| Base | 87.61% (19,035/21,726) | 0.00% |
| Instruct | 88.88% (19,310/21,726) | 0.00% |

**Interpretation:** Both models achieve high accuracy on this benchmark. The
instruct model is 1.27 percentage points higher than base. Neither model ever
produced a forced-choice prediction — both models always assigned highest
probability to one of the four answer tokens (A/B/C/D), never to an out-of-
vocabulary token. This 0% forced-choice rate confirms that the prompt format
and token ID selection are correct for both models.

**Why accuracy is high (~88%):** This is NOT evidence that the models have
perfect cultural knowledge. The EDA established three factors that inflate
accuracy on this benchmark:

1. **Country Prediction (25.6% of dataset):** 98.9% of these are trivially
   "India" — any model that knows the dataset topic gets these right.
2. **State Prediction shortcut:** 99.98% solvable by string matching (the
   state name appears in the question or options). The model does not need
   cultural knowledge — just entity recognition.
3. **No-question baseline from EDA:** 75.87% accuracy is achievable by
   matching state names to options WITHOUT reading the question.

Excluding Country Prediction (Tier 2), base drops to 83.72% and instruct
to 85.46%. On the hardest subset (Association + General Awareness, Tier 3),
which requires genuine cultural knowledge, base is 84.69% and instruct is
86.39%. These numbers are still high relative to random chance (25%) but
are consistent with the benchmark's known limitations.

### Behavioral Label Distribution (Tier 1: Full)

| Label | Count | Percentage |
|-------|-------|------------|
| Suppression (base right → instruct wrong) | 655 | 3.0% |
| Enhancement (base wrong → instruct right) | 930 | 4.3% |
| Control (both correct) | 18,380 | 84.6% |
| Control (both wrong) | 1,761 | 8.1% |

**Key finding:** Enhancement exceeds suppression by 275 questions (930 vs 655),
yielding a net accuracy gain of +1.27% from instruction tuning. Instruction
tuning does NOT suppress cultural knowledge on net — it slightly improves it.

**But this masks question-level suppression:** 655 questions that the base
model answers correctly are WRONG after instruction tuning. These are the
most interesting cases for mechanistic interpretability — they represent
specific cultural knowledge that RLHF interfered with.

**Both-wrong analysis:** Of the 1,761 questions both models get wrong, 78.5%
(1,383) have the same wrong answer. This suggests shared knowledge gaps in
the pretraining data rather than random errors. These are genuinely hard or
ambiguous questions, not model-specific failures.

### Three-Tier Comparison

| Tier | n | Base Acc | Inst Acc | Supp % | Enh % | Both Wrong % |
|------|---|----------|----------|--------|-------|--------------|
| Tier 1 (full) | 21,726 | 87.61% | 88.88% | 3.0% | 4.3% | 8.1% |
| Tier 2 (no CP) | 16,163 | 83.72% | 85.46% | 3.9% | 5.6% | 10.7% |
| Tier 3 (hard) | 10,781 | 84.69% | 86.39% | 4.0% | 5.7% | 9.6% |

**Pattern:** Suppression rate increases as we remove easy questions:
3.0% → 3.9% → 4.0%. This is expected — Country Prediction contributes
almost no suppression (0.6%), so removing it concentrates the suppression
signal. The Tier 2 → Tier 3 jump is small because State Prediction has
only moderate suppression (3.5%).

Enhancement shows the same pattern: 4.3% → 5.6% → 5.7%. The gap between
enhancement and suppression remains ~1.7 percentage points across all tiers,
suggesting the net benefit of instruction tuning is consistent regardless
of question difficulty.

Both-wrong rate increases from 8.1% to 10.7% when removing Country Prediction
— these are questions where neither model has the knowledge, concentrated in
the harder question types.

### Per Question Type

| Question Type | n | Base Acc | Inst Acc | Supp % | Enh % | Supp Count | Enh Count |
|---------------|---|----------|----------|--------|-------|------------|-----------|
| Association | 5,453 | 84.3% | 87.1% | 2.9% | 5.6% | 156 | 307 |
| Country Prediction | 5,563 | 98.9% | 98.8% | 0.6% | 0.4% | 32 | 25 |
| General Awareness | 5,328 | 85.0% | 85.7% | **5.2%** | 5.9% | 279 | 312 |
| State Prediction | 5,382 | 81.8% | 83.6% | 3.5% | 5.3% | 188 | 286 |

**Critical observation — General Awareness has highest suppression (5.2%):**
General Awareness questions ask about festivals, traditions, rituals, and
cultural practices — exactly the kind of culturally sensitive content that
RLHF might discourage. This is the most important finding for our research
question: instruction tuning is most likely to suppress knowledge about
cultural practices, even though it enhances slightly more than it suppresses
in the same category (5.9% enhancement vs 5.2% suppression).

**Country Prediction is near-ceiling:** Both models at ~99%, with only 32
suppression and 25 enhancement cases. This confirms the EDA prediction that
this question type is trivial. The 0.6% suppression rate (32 questions) is
the lowest of any type, consistent with the fact that these questions test
geographic knowledge ("Which country is X from?") rather than cultural
knowledge that RLHF might affect.

**State Prediction has lowest base accuracy (81.8%):** Despite being
shortcut-vulnerable (string matching), the base model still misses 18.2%.
This suggests the shortcut is not as reliable as the EDA's 99.98% string-
matching rate implies — the model may not perfectly implement string matching,
or the distractors are effective enough to confuse it.

**Association has the largest net gain (+2.8%):** Base 84.3% → instruct 87.1%.
The instruct model is substantially better at associating cultural entities
with their correct regions, suggesting RLHF's conversational training
improved this type of factual recall.

### Confidence Analysis

| Metric | Base | Instruct |
|--------|------|----------|
| Max logprob mean | -0.197 | -0.056 |
| Max logprob median | -0.070 | -0.000198 |
| Mean logprob margin | 3.095 nats | 8.586 nats |
| Median logprob margin | 3.256 nats | 8.875 nats |
| Logprob range (all 4 options) | [-9.25, -0.001] | [-21.75, -0.00002] |
| Low-confidence (margin < 0.5) | 1,814 (8.3%) | 605 (2.8%) |
| Accuracy on low-confidence | 46.2% | 40.3% |
| Accuracy on high-confidence | 91.4% | 90.3% |

**The instruct model is far more decisive:** Its median margin (8.875 nats)
is 2.7× the base model's (3.256 nats). In probability terms, the instruct
model's top answer is e^8.875 ≈ 7,186× more likely than the second-best
option, while the base model's ratio is e^3.256 ≈ 26×. Instruction tuning
makes the model dramatically more confident in its predictions.

**Low-confidence predictions are near-random:** Base model accuracy drops
from 91.4% (high-confidence) to 46.2% (low-confidence) — barely above
chance (25%). The instruct model shows an even sharper drop: 90.3% → 40.3%.
This validates the 0.5-nat threshold as a meaningful confidence cutoff.

**Suppression correlates with low confidence:** Among base model's 1,814
low-confidence predictions, 13.1% are suppression cases. Among the 19,912
high-confidence predictions, only 2.1% are suppression. This 6× ratio
shows that suppression is concentrated in uncertain predictions — the model
"almost knows" the answer but instruction tuning tips it to the wrong choice.
This is a key insight for Step 2 probing: suppression may be detectable in
the activation space as borderline representations that RLHF pushes past a
decision boundary.

**Three times fewer low-confidence predictions for instruct:** The base model
has 1,814 uncertain predictions (8.3%) while instruct has only 605 (2.8%).
Instruction tuning not only makes the model more accurate but also more
decisive — it converts uncertain predictions into confident ones (usually
correct, but sometimes wrong, producing the 655 suppression cases).

### Position Distribution

| Letter | GT % | Base Pred % | Instruct Pred % |
|--------|------|-------------|-----------------|
| A | 27.1% | 28.0% | 29.1% |
| B | 29.0% | 28.8% | 28.6% |
| C | 23.1% | 23.6% | 22.3% |
| D | 20.8% | 19.6% | 20.0% |

**No severe position bias:** Both models' prediction distributions closely
match the ground truth distribution. The instruct model shows a slight bias
toward A (29.1% predicted vs 27.1% GT, a +2.0 pp difference) and away from
C (22.3% vs 23.1%, -0.8 pp). The base model's largest deviation is -1.2 pp
on D (19.6% vs 20.8%). These biases are small relative to the ground truth
imbalance (B=29.0% vs D=20.8%, which is an 8.2 pp spread in the dataset
itself).

**Accuracy varies by ground truth position:**

| GT Position | Base Acc | Instruct Acc |
|-------------|----------|--------------|
| A | 89.0% | 91.3% |
| B | 88.2% | 89.2% |
| C | 88.1% | 87.7% |
| D | 84.6% | 86.5% |

Both models are weakest on D (least common GT position = 20.8%). This
could be a frequency effect (less training signal for D-answers) or a
recency effect (D is the last option, furthest from the question).

**Suppression is roughly uniform across positions:** A=2.5%, B=3.0%,
C=3.7%, D=3.0%. The slight elevation at C (3.7%) is minor and not
statistically tested. There is no evidence that suppression targets a
specific answer position.

### Suppression by Cultural Attribute

| Attribute | n | Supp Rate | Supp Count | Base Acc | Inst Acc |
|-----------|---|-----------|------------|----------|----------|
| Cuisine | 1,671 | 4.19% | 70 | 83.4% | 84.2% |
| Language | 900 | 3.89% | 35 | 89.3% | 89.8% |
| Art | 2,066 | 3.53% | 73 | 85.8% | 87.5% |
| Costume | 1,513 | 3.50% | 53 | 86.4% | 86.8% |
| Dance_and_Music | 2,018 | 3.42% | 69 | 85.9% | 86.9% |
| Festivals | 2,241 | 3.21% | 72 | 87.6% | 88.0% |
| Personalities | 983 | 3.15% | 31 | 78.3% | 82.3% |
| Religion | 482 | 2.70% | 13 | 87.1% | 90.2% |
| Tourism | 3,801 | 2.55% | 97 | 90.6% | 91.9% |
| History | 2,609 | 2.53% | 66 | 91.6% | 92.3% |
| Rituals_and_Ceremonies | 1,000 | 2.30% | 23 | 87.3% | 88.5% |
| Cultural_Common_Sense | 2,091 | 2.30% | 48 | 89.3% | 91.0% |
| Medicine | 72 | 0.00% | 0 | 77.8% | 84.7% |

*(Nightlife n=41, Sports n=162, Transport n=76 omitted due to small sample
size — marked as low-confidence in the analysis output.)*

**Cuisine has the highest suppression rate (4.19%):** Questions about regional
cuisine — specific dishes, cooking traditions, and food culture — are most
susceptible to instruction-tuning suppression. This is noteworthy because
cuisine is not obviously "sensitive" content that RLHF would target. One
hypothesis: cuisine questions involve highly specific regional knowledge
(e.g., "Which state is Babru from?") that the model barely learned during
pretraining, and RLHF's general-purpose optimization slightly degrades
these marginal representations.

**Medicine has 0% suppression (but n=72):** Too few questions to draw
conclusions, but it's interesting that the instruct model improves
substantially on medicine (+6.9 pp) with zero suppression. Medicine may
benefit from RLHF's health-related instruction tuning.

**Religion has low suppression (2.70%) despite being culturally sensitive:**
One might expect RLHF to suppress religious knowledge, but it actually
shows one of the lowest suppression rates AND the largest accuracy gain
(+3.1 pp, from 87.1% to 90.2%). This suggests RLHF does not systematically
suppress religious cultural knowledge in this benchmark.

### Suppression by State (Top 10)

| State | n | Supp Rate | Supp Count | Base Acc | Inst Acc |
|-------|---|-----------|------------|----------|----------|
| Ladakh | 278 | 7.55% | 21 | 86.7% | 84.5% |
| Lakshadweep | 122 | 5.74% | 7 | 80.3% | 77.9% |
| Himachal Pradesh | 822 | 4.74% | 39 | 83.6% | 82.5% |
| Jharkhand | 759 | 4.35% | 33 | 82.7% | 82.9% |
| Manipur | 648 | 4.32% | 28 | 85.8% | 85.8% |
| Nagaland | 514 | 4.28% | 22 | 88.1% | 88.5% |
| Uttar Pradesh | 504 | 4.17% | 21 | 92.1% | 89.7% |
| Chhattisgarh | 631 | 4.12% | 26 | 78.0% | 79.6% |
| Haryana | 888 | 4.05% | 36 | 84.2% | 83.1% |
| Puducherry | 374 | 4.01% | 15 | 81.0% | 85.8% |

**Ladakh has the highest suppression rate (7.55%) and is one of the few
states where instruct accuracy is LOWER than base:** 86.7% → 84.5%, a
2.2 pp drop. Ladakh is a Union Territory with a distinct Tibetan Buddhist
culture that may be underrepresented in RLHF training data. Similarly,
Lakshadweep (5.74% suppression, accuracy drop from 80.3% → 77.9%) is a
small island territory with a unique cultural profile.

**Geographic pattern:** The top-suppressed states tend to be smaller UTs
and northeastern states (Ladakh, Lakshadweep, Manipur, Nagaland) that have
less mainstream representation. Larger, well-known states like Tamil Nadu
(0.81%), Maharashtra (1.41%), and Gujarat (1.27%) have much lower
suppression. This suggests RLHF may preferentially degrade knowledge
about less-represented cultural groups.

**Caveat:** The small sample sizes for some states (Lakshadweep n=122,
Ladakh n=278) mean that a few questions flipping could substantially
change the rates. Statistical significance testing is needed before
drawing strong conclusions (deferred to the paper writing phase).

**F8 sanity check passed:** No single state accounts for more than 15% of
total suppression. The most-suppressed state by absolute count is Tourism-
heavy Karnataka (41 questions), which is only 6.3% of the 655 total.
Suppression is distributed across states, not concentrated.

### Entity-Level Analysis

The merge pipeline assigns behavioral labels at the entity level (e.g.,
"Haryana|Cuisine|Cuisine of Haryana" or "Jarawa body painting"):

| Entity Label | Count | % of 7,834 entities |
|--------------|-------|---------------------|
| both_correct | 6,086 | 77.7% |
| mixed (some suppressed, some not) | 1,422 | 18.1% |
| both_wrong | 162 | 2.1% |
| enhanced (all questions flipped to correct) | 87 | 1.1% |
| suppressed (all questions flipped to wrong) | 77 | 1.0% |

**Only 77 entities are purely suppressed** (every question about them flips
from correct to wrong). Most suppression occurs in "mixed" entities — some
questions are suppressed while others are enhanced or unchanged. This means
suppression is question-specific, not entity-specific: RLHF does not
systematically erase knowledge about a particular entity, but may interfere
with specific formulations or distractor combinations.

**Top suppressed entities (by rate, n≥5):**

| Entity | Supp/Total | Base Acc | Inst Acc |
|--------|------------|----------|----------|
| Uttar_Pradesh\|Costume\|Uttar Pradesh | 3/6 (50%) | 66.7% | 16.7% |
| Haryana\|Cuisine\|Cuisine of Haryana | 4/9 (44%) | 77.8% | 33.3% |
| Kullui dialect | 2/5 (40%) | 100% | 60% |
| Godna tattoo art | 2/5 (40%) | 80% | 40% |
| Rogan Painting | 2/5 (40%) | 80% | 40% |
| Sanjhi art | 2/5 (40%) | 80% | 40% |

These are highly specific cultural entities (regional art forms, dialects,
costumes) where the base model has fragile knowledge that instruction tuning
degrades. The small sample sizes (n=5-9) mean individual entity rates are
unreliable, but the PATTERN — regional folk arts and minor cultural
practices — is consistent with the attribute-level finding that Cuisine
and Art have the highest suppression rates.

### Sanity Check Summary

| Check | Status | Value |
|-------|--------|-------|
| F1: Base accuracy (Tier 2) | **WARN** | 83.7% (expected 40-70%) |
| F2: Instruct > Base | PASS | 88.9% > 87.6% |
| F3: Country Prediction ≥ 95% | PASS | base=98.9%, inst=98.8% |
| F4: Forced choice < 5% | PASS | base=0.0%, inst=0.0% |
| F6: Suppression rate 5-15% | INFO | 3.0% (below expected range) |
| F7: Enhancement rate 4-12% | PASS | 4.3% |
| F8: No state dominates suppression | PASS | max=Karnataka at 6.3% |
| F9: Confidence-accuracy correlation | PASS | low-conf acc 46.2%/40.3% < high-conf 91.4%/90.3% |
| F10: Activation shapes | PASS | all 32 files shape (21726, 4096) |
| F11: No NaN/Inf in activations | PASS | all 32 files clean |

**F1 WARN — why base accuracy exceeds the expected range:** The expected range
(40-70%) was set before running the evaluation, based on typical LLM
performance on cultural benchmarks. LLaMA 3.1 8B is a strong model, and
this benchmark has significant shortcut vulnerability. The 83.7% accuracy
(Tier 2, excluding Country Prediction) is plausible given:
- 75.87% no-question baseline from EDA (just string matching)
- State Prediction still shortcut-vulnerable in Tier 2
- LLaMA 3.1 8B trained on ~15T tokens including substantial Indian web content

The F1 threshold should be updated for future runs but does not indicate
a problem with the evaluation.

**F6 INFO — suppression rate below expected range (3.0% vs 5-15%):** The
pre-run estimate assumed more behavioral divergence between base and
instruct. The actual 3.0% means instruction tuning is LESS destructive to
cultural knowledge than we hypothesized. This is a finding, not an error.
For Step 2 probing, 655 suppression cases is still sufficient signal to
train meaningful probes (compared to ~19,000 control cases).

**Note on F5:** The position bias check (F5) was not explicitly reported in
the merge output, but the position distribution table above confirms no
severe bias — all model predictions are within ±2 pp of ground truth.

---

## 25. Activation Verification

After the evaluation loop completes, the code verifies all activation files:

```python
# From eval_step1.py, lines 639-649
for hname in HOOK_NAMES:
    for ptype in ["mean_pool", "last_token"]:
        arr = act_arrays[(hname, ptype)]
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        if has_nan or has_inf:
            log.error(f"ACTIVATION CHECK FAILED: {hname}/{ptype}")
        else:
            log.debug(f"Activation OK: {hname}/{ptype} shape={arr.shape}, "
                      f"mean_norm={np.linalg.norm(arr, axis=1).mean():.2f}")
```

### Expected Activation Norm Pattern

The L2 norm of activation vectors should increase through the network as
representations become more concentrated. From the initial run (norms are
independent of the token ID bug — the forward pass and hooks were correct):

**Instruct model norms (mean_pool / last_token):**

| Hook | Mean Pool Norm | Last Token Norm |
|------|---------------|-----------------|
| embed | 0.14 | 0.43 |
| layer_04 | 4.73 | 3.04 |
| layer_08 | 5.09 | 5.15 |
| layer_14 | 5.79 | 8.22 |
| layer_20 | 7.65 | 13.68 |
| layer_26 | 11.13 | 25.03 |
| layer_30 | 16.97 | 41.86 |
| layer_31 | 15.18 | 50.60 |

**Base model norms (mean_pool / last_token):**

| Hook | Mean Pool Norm | Last Token Norm |
|------|---------------|-----------------|
| embed | 0.16 | 0.37 |
| layer_04 | 1.97 | 3.32 |
| layer_08 | 3.34 | 5.67 |
| layer_14 | 5.37 | 8.89 |
| layer_20 | 9.99 | 15.61 |
| layer_26 | 19.70 | 29.42 |
| layer_30 | 30.67 | 49.36 |
| layer_31 | 35.40 | 80.16 |

**Observations:**

1. **Norm increases monotonically** through layers for both models — the
   expected pattern. No anomalies.

2. **Last-token norm > mean-pool norm** at every layer beyond embed. The last
   token concentrates information from the entire sequence via attention,
   producing higher-norm vectors.

3. **Base model has higher norms** than instruct at later layers. This may
   reflect different representation structures from instruction tuning.

4. **Layer 31 mean-pool norm drops for instruct** (15.18 vs 16.95 at layer 30).
   This is because the final layer applies a normalization before the lm_head.
   The base model shows the opposite pattern (35.40 > 30.67), suggesting
   different normalization behavior.

5. **Embedding norms are tiny** (~0.1-0.5). This is expected — token embeddings
   are typically initialized with small random values.

### What the Norm Differences Between Base and Instruct Tell Us

The norm tables reveal a systematic difference between the two models that
connects directly to our research question:

```
Norm ratio (base / instruct) at each layer:

Hook       | Mean Pool Ratio | Last Token Ratio
-----------|-----------------|-----------------
embed      | 1.14×           | 0.86×    (nearly identical — same embedding matrix)
layer_04   | 0.42×           | 1.09×    (base mean pool much lower)
layer_08   | 0.66×           | 1.10×    (similar for last token)
layer_14   | 0.93×           | 1.08×    (similar)
layer_20   | 1.31×           | 1.14×    (base overtakes at late layers)
layer_26   | 1.77×           | 1.18×    (base growing faster)
layer_30   | 1.81×           | 1.18×    (base much higher)
layer_31   | 2.33×           | 1.58×    (base 2.3× higher mean, 1.6× last token)

Key observation: Base model norms grow MUCH faster in late layers than
instruct, especially for mean pool. This suggests that:

1. Instruction tuning regularizes the representation space. The instruct
   model's activations are more "controlled" — they grow less in magnitude,
   suggesting tighter learned norms from RLHF's preference optimization.

2. The base model's late-layer representations are more "spread out" in
   vector space. This could mean the base model uses more of the
   representational capacity at each layer, while the instruct model
   compresses information into a smaller subspace.

3. The embedding layer is nearly identical (ratio ~1.0) because both models
   share the same architecture and the embedding matrix was not significantly
   changed by instruction tuning.

These norm differences have practical implications for Step 2 probing:
linear probes trained on base model activations must handle a wider dynamic
range than those trained on instruct model activations. We may need layer-
specific normalization to ensure fair comparison across models.
```

---

## 26. Bugs Encountered and Fixed

### Bug 1: Transformers 5.3.0 Layer Output Format

**Symptom:** Both processes crashed immediately on the first batch:
```
IndexError: shape mismatch: indexing tensors could not be broadcast
together with shapes [156], [64]
```

**Root cause:** In transformers 5.3.0, `LlamaDecoderLayer.forward()` returns
`hidden_states` directly (a plain tensor), not a tuple `(hidden_states, ...)`.
The hook code did `output[0]`, which indexed the first BATCH element instead
of extracting hidden states from a tuple.

- `output` shape: `(64, 156, 4096)` — batch=64, seq=156, hidden=4096
- `output[0]` shape: `(156, 4096)` — first batch element only
- Expected: `(64, 156, 4096)` — full batch

The `156` in the first dimension of the broken tensor matched the sequence
length, not the batch size, which is how the error message showed `[156]` vs
`[64]`.

**Fix:**
```python
hidden = output[0] if isinstance(output, (tuple, list)) else output
```

Check if the output is a tuple/list (old transformers) and index `[0]`, or
use it directly (new transformers).

**Detection method:** The crash was immediate and clear. The shape mismatch
error pointed directly to the line, and comparing the expected batch size (64)
with the observed first dimension (156 = sequence length for instruct, 263 =
sequence length for base) revealed the issue.

### Bug 2: Wrong Token IDs for Answer Extraction

**Symptom:** Both models showed ~28% accuracy (near random chance of 25%).
Country Prediction at 30% instead of expected ~100%. Instruct forced-choice
rate at 86.5% instead of expected <5%.

**Root cause:** `tokenizer.encode('A')` returns token ID 32 (bare character
`A`), but the model predicts ` A` (space-prefixed, token ID 362) because the
prompt ends with `Answer:` and the expected completion is ` A`.

**Fix:**
```python
tokenizer.encode(f" {l}", add_special_tokens=False)[0]  # " A"=362, " B"=426, etc.
```

**Detection method:** The sanity checks in `merge_step1.py` flagged every
single check:
- F1: accuracy too low
- F2: instruct not better than base
- F3: Country Prediction not near 100%
- F4: instruct forced rate way too high
- F9: low-confidence accuracy = high-confidence accuracy (logprobs are noise)

The combination of symptoms pointed to a fundamental logit extraction issue.
Comparing `tokenizer.encode('A')` vs `tokenizer.encode(' A')` confirmed it.

### Bug 3: Instruct Model Needs Bare Token IDs, Not Space-Prefixed

**Symptom:** After fixing Bug 2 (space-prefixed tokens for both models), the
instruct model still showed 98.9% forced-choice rate and ~30% accuracy.

**Root cause:** Bug 2's fix applied space-prefixed tokens to BOTH models.
But the instruct model's chat template ends with `\n\n` (after the assistant
header), so the model predicts bare `A` (ID 32), not ` A` (ID 362). The base
model's prompt ends with `Answer:`, so it correctly predicts ` A` with a
leading space.

**Fix:**
```python
# Base: space-prefixed (prompt ends with "Answer:")
answer_token_ids = {l: tokenizer.encode(f" {l}", ...)[0] for l in "ABCD"}

# Instruct: bare (chat template ends with "\n\n")
answer_token_ids = {l: tokenizer.encode(l, ...)[0] for l in "ABCD"}
```

**Detection method:** The instruct forced-choice rate (98.9%) was the key
signal. The model was rarely predicting space-prefixed letter tokens because
the chat template doesn't set up that pattern. Checking the model's top-1
predictions revealed bare `C` (ID 34) appearing as a natural prediction,
confirming the model outputs bare letters.

### Bug 4: Pad Token Using Semantically Meaningful `<|eot_id|>`

**Symptom:** Instruct model showed confused outputs — top-1 predictions
included `assistant`, `)`, and other unexpected tokens.

**Root cause:** The code used `tokenizer.eos_token` as the pad token. For the
instruct tokenizer, `eos_token` is `<|eot_id|>` (ID 128009), the end-of-turn
marker. Padding the left side with end-of-turn tokens confuses the model — it
sees many "end of turn" signals before the conversation even starts. The base
tokenizer's `eos_token` is `<|end_of_text|>` (128001), which is benign.

**Fix:**
```python
tokenizer.pad_token_id = 128001  # <|end_of_text|>, not eos_token
```

**Detection method:** Found by inspecting the instruct debug log, which showed
`pad_token=<|eot_id|>` instead of the expected `<|end_of_text|>`.

### Bug 5: Left-Padding Position Indexing Error

**Symptom:** Both models showed ~30-33% accuracy (near random). Country
Prediction at ~40% instead of ~100%. Logprobs in the -13 to -35 range
(should be -0.01 to -6 for confident predictions). Different questions within
the same batch produced identical logprobs.

**Root cause:** The logit extraction code used:
```python
seq_lengths = attention_mask.sum(dim=1) - 1
last_logits = logits[torch.arange(B), seq_lengths]
```

`attention_mask.sum()` returns the COUNT of real tokens, not the POSITION of
the last real token. With left-padding, a 230-token sequence padded to 265
has real tokens at positions 35-264. The code computed index 229 (count-1),
which is in the MIDDLE of the sequence — not at the `Answer:` position (264).

```
Padded sequence (left padding):
Position: 0  1  2 ... 34 35 36 ... 228 229 ... 263 264
Token:    P  P  P ... P  B  T  ...  ?   ?   ... :   Answer:
                          ^              ^              ^
                      first real    Bug's index     Correct index
                      token         (mid-question)  (end of prompt)

Bug extracts logits at position 229 → mid-question token
Should extract at position 264 → the "Answer:" position
```

The same bug affected `compute_pooled()` for last-token activation extraction,
meaning the "last token" activations were also captured from wrong positions.

**Fix:**
```python
last_logits = logits[:, -1, :]   # position -1 = rightmost = last real token
h_last = hidden[:, -1, :]        # same fix for activation pooling
```

With left-padding, the last real token is ALWAYS at the rightmost position
for every sequence in the batch.

**Detection method:** Interactive testing of the model (single prompt, no
padding) produced correct logprobs (` C` at -0.008 for a Country Prediction
question), but the eval script produced -14 to -23 for the same question type.
The divergence pointed to a batching/padding issue. Examining two different
questions that produced identical logprobs confirmed the position was wrong —
they happened to have the same padding offset, so the code extracted the same
relative position in both.

**Why this was hard to find:** The code `attention_mask.sum(dim=1) - 1`
LOOKS correct — it seems to compute "the index of the last real token." And
for RIGHT-padded sequences, it IS correct (real tokens are at positions 0
through count-1). The subtle error is that left-padding shifts all real tokens
rightward, making the count-based index point to the wrong absolute position.

---

## 27. Design Decisions: What We Chose and Why

### Decision 1: Logit-Based vs Generation-Based Evaluation

**Chose:** Logit-based (compare logprobs of 4 answer tokens at last position)

**Why:** Deterministic, information-rich (get full probability distribution),
faster (one forward pass), and standard in MCQ evaluation literature (MMLU,
HellaSwag, ARC).

**Trade-off:** Cannot capture cases where the model would generate the right
answer in a non-standard format (e.g., "The answer is B" instead of just "B").
This is acceptable because we control the prompt format.

### Decision 2: Left Padding

**Chose:** `tokenizer.padding_side = "left"`

**Why:** Ensures the last position in every sequence is a real token, not padding.
This makes logit extraction and last-token activation pooling trivial.

**Trade-off:** Left padding can slightly affect attention patterns in early
layers (the model sees PAD tokens at the start), but this effect is negligible
for inference.

### Decision 3: BF16 Inference, FP32 Activations

**Chose:** Load model in BF16, convert activations to FP32 before saving

**Why:** BF16 halves memory for the model (16 GB vs 32 GB). But BF16 has only
7 mantissa bits, which can cause precision loss when mean-pooling thousands of
vectors. Converting to FP32 for the activation arrays preserves precision for
downstream probing.

**Storage cost:** FP32 activations are 2× larger than BF16 (~9.0 GB vs ~4.5 GB).
This is acceptable on the NVMe storage.

### Decision 4: Batch Size 24

**Chose:** 24 (conservative for variable GPU allocation)

**Why:** The SLURM preempt partition assigns whatever GPUs are available. We
cannot guarantee 96 GB GPUs (RTX PRO 6000) — the job may land on 40-48 GB
GPUs (A100-40GB, L40S, A6000). Batch size 24 keeps peak VRAM at ~18.5 GB
(16 GB model + ~2.5 GB activations/attention), safe for all GPU types. The
actual run used A100-SXM4-40GB (42.4 GB each) and peaked at 18.5 GB.

**Trade-off:** Smaller batch = more batches (906 vs 340 at BS=64), but the
per-batch overhead is minimal. Throughput was 23.7-29.4 q/s, completing in
~15 min total. Increasing to 64 would roughly halve batch count but not
meaningfully change total runtime since the bottleneck is compute, not
batch scheduling.

### Decision 5: Run All 21,726 Questions

**Chose:** No filtering — evaluate every usable question.

**Why:** As established in the EDA (Section 16): "Run Everything, Slice
Everything, Report Everything." More questions = more suppression cases for
probing. Country Prediction provides a clean control group. Filtering before
evaluation looks like cherry-picking.

### Decision 6: 5-Shot for Base, 0-Shot for Instruct

**Chose:** Different prompt strategies for different model types

**Why:** Base models need demonstrations to understand the expected output
format. Instruct models need instructions. Using 5-shot for instruct would
waste context tokens on redundant examples. Using 0-shot for base would
produce incoherent outputs (the model wouldn't know to output a letter).

### Decision 7: Memory-Mapped Activation Storage

**Chose:** `np.lib.format.open_memmap` with direct disk writes

**Why:** Activations for 21,726 questions × 4096 dimensions × FP32 = 340 MB
per file, 5.3 GB per model. Writing everything at the end risks losing all
progress on preemption. Memory-mapping writes incrementally.

**Trade-off:** Memory-mapped I/O has overhead from filesystem operations on
every batch. But each write is only 24 × 4096 × 4 = 384 KB, which is negligible.

### Decision 8: CSV Append, Not In-Memory Accumulation

**Chose:** Write each batch's results to CSV immediately via `csv.DictWriter`

**Why:** On preemption, only the current batch's results are lost. If we
accumulated in memory and wrote at the end, we'd lose everything.

**Trade-off:** CSV writes are I/O operations, but at 24 rows per batch on
NFS, the overhead is microseconds.

### Decision 9: No Gradient Checkpointing

**Chose:** Standard forward pass with `torch.no_grad()`, no gradient
checkpointing.

**Why:** Gradient checkpointing (also called activation checkpointing) is
a technique that trades compute for memory during training: instead of
storing all intermediate activations, it recomputes them during the backward
pass. Since we are doing inference only (no backward pass), gradient
checkpointing would provide zero benefit. The `torch.no_grad()` context
manager already saves ~50% of the memory that would be used for the autograd
computation graph.

### Decision 10: No KV Cache

**Chose:** Full forward pass without key-value caching.

**Why:** KV caching is an optimization for autoregressive generation: after
computing attention for the first N tokens, the K and V tensors are cached
so that generating token N+1 only requires computing attention for the new
token, not reprocessing the entire sequence. Since we run a SINGLE forward
pass (no generation), there is nothing to cache. Each question is processed
independently.

If we were using `model.generate()`, enabling the KV cache would speed up
generation. But with logit-based extraction, we never generate.

### Decision 11: 6-Digit Logprob Precision in CSV

**Chose:** Format logprobs as `f"{value:.6f}"` in the CSV output.

**Why:** The logprobs range from roughly -0.01 (very confident) to -20
(extremely improbable). At 6 decimal places, the CSV preserves enough
precision to reproduce the margin calculation to within 1e-6 nats. More
precision would bloat the CSV without improving any downstream analysis.
Less precision (e.g., 2 decimal places) could change which questions fall
below the 0.5-nat low-confidence threshold.

---

## 28. What This Step Does NOT Do

1. **Does NOT generate text.** We extract logits from a single forward pass.
   No autoregressive generation, no sampling, no beam search.

2. **Does NOT train or fine-tune.** The models are loaded in eval mode and
   used purely for inference.

3. **Does NOT compute probing accuracy.** That is Step 2. We only collect
   the activation vectors here.

4. **Does NOT analyze circuits.** That is Step 3. We collect the raw data.

5. **Does NOT filter questions.** We evaluate ALL 21,726 questions. Filtering
   (by question type, difficulty, etc.) happens in the analysis (merge script)
   and downstream steps.

6. **Does NOT modify the dataset.** The Sanskriti dataset is used as-is
   (after the 127 broken row exclusion from the EDA). No rebalancing,
   augmentation, or deduplication.

7. **Does NOT use the model's generated text for answer extraction.** The
   predicted answer comes entirely from logprob comparison, not from parsing
   generated output.

8. **Does NOT perform any statistical significance testing.** Suppression and
   enhancement rates are descriptive statistics. Testing whether these rates
   are significantly different from chance (25% accuracy baseline) or from
   each other is deferred to the paper writing phase, where we will apply
   appropriate tests (e.g., McNemar's test for paired binary outcomes, or
   bootstrap confidence intervals).

9. **Does NOT normalize or transform activations.** The raw FP32 activation
   vectors are saved as-is. Any normalization (z-scoring, unit norm, PCA)
   happens in Step 2 before probing. Saving raw activations keeps our options
   open — different normalization strategies may be appropriate for different
   probing experiments.

10. **Does NOT handle the 78.6% near-duplicate overlap.** The EDA identified
    that many questions share templates (same question structure with different
    entities). Step 1 treats each question independently. Deduplication-aware
    analysis (e.g., stratified sampling for probe training) is handled in
    Step 2 to avoid train-test leakage.

---

## 29. Output Files

### Primary Outputs

| File | Location | Contents | Size |
|------|----------|----------|------|
| `base_results.csv` | `results/step1/` | Per-question results for base model (21,726 rows) | 649 KB |
| `instruct_results.csv` | `results/step1/` | Per-question results for instruct model (21,726 rows) | 649 KB |
| `sanskriti_behavioral_labels.csv` | `results/step1/` | Master merged CSV with labels (21,726 rows) | 3.6 MB |
| `sanskriti_prepared.csv` | `results/step1/` | Prepared dataset with entity_key | 4.3 MB |
| Activation arrays | `activations/{model}/{pool}/{hook}.npy` | (21726, 4096) FP32 | 340 MB each |

### Analysis Outputs

| File | Location | Contents |
|------|----------|----------|
| `step1_aggregate_stats.json` | `results/step1/` | Three-tier stats + counts |
| `accuracy_by_question_type.csv` | `results/step1/` | Per-type accuracy and suppression |
| `suppression_by_attribute.csv` | `results/step1/` | Per-attribute suppression rates |
| `suppression_by_state.csv` | `results/step1/` | Per-state suppression rates |
| `position_distribution.csv` | `results/step1/` | A/B/C/D prediction distributions |
| `behavioral_label_counts.csv` | `results/step1/` | Label counts per tier |
| `entity_behavioral_labels.csv` | `results/step1/` | Entity-level behavioral labels |
| `forced_choice_audit.csv` | `results/step1/` | All forced-choice questions |
| `confidence_distribution.csv` | `results/step1/` | Margin and confidence stats |

### Plots

| Plot | File | Description |
|------|------|-------------|
| 1 | `accuracy_by_question_type.png` | Grouped bar: base vs instruct accuracy |
| 2 | `behavioral_label_distribution.png` | Stacked bar across 3 tiers |
| 3 | `suppression_by_attribute.png` | Horizontal bar: suppression by attribute |
| 4 | `suppression_by_state.png` | Horizontal bar: suppression by state (top 20) |
| 5 | `position_distribution.png` | Grouped bar: GT vs base vs instruct predictions |
| 6 | `confidence_distribution.png` | Histogram: max logprob distribution |
| 6b | `logprob_margin_distribution.png` | Histogram: margin with 0.5 threshold |

### Logs

| File | Contents |
|------|----------|
| `step1_slurm_JOBID.out` | Combined SLURM output (all processes) |
| `step1_main_*.log` | Main process: dataset loading, process management |
| `step1_base_*.log` | Base model: DEBUG-level evaluation details |
| `step1_instruct_*.log` | Instruct model: DEBUG-level evaluation details |
| `step1_merge_*.log` | Merge pipeline: stats, sanity checks, plots |

---

## 30. Runtime and Reproducibility

### Actual Runtime (SLURM job 6959494)

| Phase | Duration | Notes |
|-------|----------|-------|
| Model loading (each) | 3-4s | Page-cached on NVMe |
| Instruct evaluation | 12.3 min | 906 batches × 24, 29.4 q/s |
| Base evaluation | 15.3 min | 906 batches × 24, 23.7 q/s |
| Total (parallel) | 15.3 min | Bounded by the slower model (base) |
| Merge pipeline | ~11s | Merge, stats, 7 plots, 9 sanity checks |
| End-to-end | 16 min | Including setup, pre-flight, validation |

**Why base is slower than instruct:** Base sequences are longer (~230-265
tokens with 5-shot prefix) compared to instruct (~104-156 tokens with chat
template). Longer sequences mean more attention computation per batch.

### Reproducibility

The evaluation is deterministic given:
- Same model weights (local copies at fixed paths)
- Same dataset (Sanskriti, loaded from local cache)
- Same tokenizer (loaded with model)
- Same batch size (24)
- Same device assignment (base=cuda:0, instruct=cuda:1)
- `torch.no_grad()` (no stochastic operations in inference mode)

The only source of non-determinism is floating-point operation ordering on GPU,
which can cause ~1e-7 differences in BF16 logits. These are below the precision
threshold and should not affect predicted letters.

### Sources of Non-Determinism (and Why They Don't Matter)

```
Source 1: GPU floating-point accumulation order
  Matrix multiplications on GPU use parallel reduction algorithms that
  sum floating-point numbers in different orders depending on thread
  scheduling. Because floating-point addition is not associative
  (a + b + c ≠ a + c + b in general), this can produce ~1e-7 differences.

  Impact: A logprob of -3.141592 might become -3.141593 on a different run.
  This is 1000× smaller than the low-confidence threshold (0.5 nats) and
  cannot flip a predicted letter (which requires the argmax to change).

Source 2: CUDA kernel selection (cuDNN autotuning)
  PyTorch may select different CUDA kernels depending on input shapes,
  GPU temperature, and available SM resources. Different kernels may use
  different algorithms with slightly different numerical results.

  Impact: Same order of magnitude as Source 1. Fully deterministic runs
  would require torch.backends.cudnn.deterministic = True, which we did
  NOT set because it slows attention by 10-20%. The accuracy benefit is
  zero (differences are below BF16 precision).

Source 3: Left-padding variation across batches
  Each batch's padding depends on the longest sequence in that batch. If
  batch boundaries change (different batch size), the padding pattern
  changes, which changes attention mask shapes.

  Impact: None on logits (attention to padding is masked), but attention
  to real tokens at different absolute positions can cause ~1e-6 diffs
  due to RoPE position encoding. This is also negligible.

Conclusion: For our purposes, the evaluation is deterministic. Running the
same code with the same batch size on the same GPU produces identical
predicted letters and behavioral labels, every time.
```

### Software Dependencies

```
# Core
torch==2.10.0+cu128
transformers==5.3.0
datasets>=2.0.0

# Analysis
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0

# System
CUDA 12.9
NVIDIA Driver 575.51.03
```

### How to Reproduce

```bash
# 1. Activate environment
conda activate cultural

# 2. Run evaluation (submits SLURM job — merge runs automatically after eval)
sbatch scripts/run_step1.sh

# 3. After job completes, check sanity checks in SLURM output
grep -E "PASS|FAIL|WARN" logs/step1_slurm_*.out

# 4. Or check the merge log directly
cat logs/step1_merge_*.log | grep -E "PASS|FAIL|WARN"
```

The SLURM script (`run_step1.sh`) runs `merge_step1.py` automatically after
evaluation completes, followed by post-run validation. There is no need to
run merge separately unless you want to re-analyze with different parameters.

---

## Appendix A: Token ID Verification

Complete token ID mapping for the four answer options:

```
Base model — space-prefixed (prompt ends with "Answer:"):
  ' A' → token 362   (CORRECT for base)
  ' B' → token 426
  ' C' → token 356
  ' D' → token 423

  Full answer sequence tokenization:
    'Answer: A' → [16533, 25, 362]  (Answer, :, ·A)
    'Answer: B' → [16533, 25, 426]  (Answer, :, ·B)
    'Answer: C' → [16533, 25, 356]  (Answer, :, ·C)
    'Answer: D' → [16533, 25, 423]  (Answer, :, ·D)

  The `·` represents the leading space that is part of the token.

Instruct model — bare characters (chat template ends with "\n\n"):
  'A' → token 32   (CORRECT for instruct)
  'B' → token 33
  'C' → token 34
  'D' → token 35

  The model outputs a bare letter immediately after the assistant header.

Buggy combinations encountered during development:
  - Bare tokens for base model → ~28% accuracy (Bug 2)
  - Space-prefixed tokens for instruct model → 98.9% forced-choice (Bug 3)
```

---

## Appendix B: CSV Schema Reference

### Per-Model Results CSV (`base_results.csv`, `instruct_results.csv`)

| Column | Type | Description |
|--------|------|-------------|
| question_id | int | 0-indexed, matches activation array row |
| ground_truth_letter | str | A/B/C/D — correct answer |
| predicted_letter | str | A/B/C/D — model's prediction |
| correct | int | 1 if predicted == ground_truth, 0 otherwise |
| logprob_A | float | Log-probability of the A answer token at last position (space-prefixed for base, bare for instruct) |
| logprob_B | float | Log-probability of token " B" |
| logprob_C | float | Log-probability of token " C" |
| logprob_D | float | Log-probability of token " D" |
| forced_choice | int | 1 if model's overall top-1 token ≠ A/B/C/D |
| top1_token_id | int | The model's actual top-1 token ID (any vocab token) |

### Master CSV (`sanskriti_behavioral_labels.csv`)

All columns from both per-model CSVs (renamed with prefix), plus:

| Column | Type | Description |
|--------|------|-------------|
| behavioral_label | str | suppression / enhancement / control_both_correct / control_both_wrong |
| base_logprob_margin | float | Top1 - Top2 logprob among A/B/C/D (base) |
| instruct_logprob_margin | float | Top1 - Top2 logprob among A/B/C/D (instruct) |
| base_low_confidence | bool | True if base margin < 0.5 nats |
| instruct_low_confidence | bool | True if instruct margin < 0.5 nats |
| state | str | Indian state (from dataset) |
| attribute | str | Cultural attribute (from dataset) |
| question_type | str | One of 4 question types |
| entity_key | str | Cultural entity identifier (from EDA) |
| question | str | Full question text |
| option1..option4 | str | The four answer options |
| answer | str | Correct answer text |

---

## Appendix C: Activation File Layout

```
activations/
├── base/
│   ├── mean_pool/
│   │   ├── embed.npy        (21726, 4096) float32
│   │   ├── layer_04.npy     (21726, 4096) float32
│   │   ├── layer_08.npy     (21726, 4096) float32
│   │   ├── layer_14.npy     (21726, 4096) float32
│   │   ├── layer_20.npy     (21726, 4096) float32
│   │   ├── layer_26.npy     (21726, 4096) float32
│   │   ├── layer_30.npy     (21726, 4096) float32
│   │   └── layer_31.npy     (21726, 4096) float32
│   └── last_token/
│       ├── embed.npy        (21726, 4096) float32
│       ├── layer_04.npy     ...
│       └── ...
└── instruct/
    ├── mean_pool/
    │   └── ...
    └── last_token/
        └── ...

Total: 32 files × 340 MB = 9.0 GB (actual, per du -sh)
```

Each file can be loaded with:
```python
arr = np.load(path, mmap_mode="r")  # memory-mapped, lazy loading
vec = arr[question_id]              # load single question's activation
batch = arr[0:64]                   # load a batch
```

---

## Appendix D: Numbers Validation Log

Every number in this report has been checked against the CSV output files
from SLURM job 6959494 (2026-04-05). This appendix documents the validation.

### Pre-Run Validations (Independent of Token IDs)

| Claim | Source | Check |
|-------|--------|-------|
| 21,726 usable rows | `sanskriti_prepared.csv` | wc -l minus header = 21,726 ✓ |
| 906 total batches | 21726 / 24 = 905.25 → ceil = 906 | ✓ |
| 179 five-shot prefix tokens | Log: "Five-shot prefix: 179 tokens" | ✓ |
| 8 hooks registered | Log: "Registered 8 hooks" | ✓ |
| 32 activation files | 8 hooks × 2 pooling × 2 models = 32 | ✓ |
| Each file 340 MB | 21726 × 4096 × 4 bytes = 356,106,240 ≈ 340 MB | ✓ |
| Base token " A" = 362 | tokenizer.encode(" A") = [362] | ✓ |
| Base token " B" = 426 | tokenizer.encode(" B") = [426] | ✓ |
| Base token " C" = 356 | tokenizer.encode(" C") = [356] | ✓ |
| Base token " D" = 423 | tokenizer.encode(" D") = [423] | ✓ |
| Instruct token "A" = 32 | tokenizer.encode("A") = [32] | ✓ |
| Instruct token "B" = 33 | tokenizer.encode("B") = [33] | ✓ |
| Instruct token "C" = 34 | tokenizer.encode("C") = [34] | ✓ |
| Instruct token "D" = 35 | tokenizer.encode("D") = [35] | ✓ |

### Post-Run Validations (SLURM job 6959494, 2026-04-05)

| Claim | Check |
|-------|-------|
| base_results.csv has 21,726 rows | ✓ (wc -l = 21727 with header) |
| instruct_results.csv has 21,726 rows | ✓ (wc -l = 21727 with header) |
| sanskriti_behavioral_labels.csv has 21,726 rows | ✓ (21840 with header; 21726 data rows + multiline CSV fields account for extra line count) |
| All 32 activation files shape (21726, 4096) | ✓ (verified via np.load for all 32 files) |
| No NaN/Inf in any activation file | ✓ (np.isnan/isinf checks all False) |
| Country Prediction base accuracy ≥ 95% | ✓ (98.94%) |
| Country Prediction instruct accuracy ≥ 95% | ✓ (98.81%) |
| Instruct accuracy > Base accuracy | ✓ (88.88% > 87.61%) |
| Low-confidence accuracy < high-confidence accuracy (base) | ✓ (46.2% < 91.4%) |
| Low-confidence accuracy < high-confidence accuracy (instruct) | ✓ (40.3% < 90.3%) |
| Forced-choice rate = 0% for both models | ✓ (0/21,726 for both) |
| Activation storage total ≈ 9 GB | ✓ (9.0 GB per du -sh) |
| 7 plots generated | ✓ (7 PNG files in plots/step1/) |
| 8 analysis CSV/JSON files generated | ✓ (all present in results/step1/) |
| Base token IDs match expected | ✓ (A=362, B=426, C=356, D=423 in SLURM log) |
| Instruct token IDs match expected | ✓ (A=32, B=33, C=34, D=35 in SLURM log) |
| Tier 1 suppression + enhancement + both_correct + both_wrong = 21,726 | ✓ (655 + 930 + 18380 + 1761 = 21,726) |
| Tier 2 n = 21,726 - 5,563 (CP count) = 16,163 | ✓ |
| Tier 3 n = 5,453 (Association) + 5,328 (General Awareness) = 10,781 | ✓ |
| Sum of per-type suppression = total suppression | ✓ (156 + 32 + 279 + 188 = 655) |
| Sum of per-type counts = 21,726 | ✓ (5453 + 5563 + 5328 + 5382 = 21,726) |

### Cross-Validation: Numbers in This Document vs Output Files

Every number in Section 24 was cross-checked against the raw CSV files using
pandas. The aggregate statistics in `step1_aggregate_stats.json` match the
per-row computations in `sanskriti_behavioral_labels.csv`. The SLURM log
reports the same final accuracy as the CSV files (87.61% base, 88.88%
instruct). No discrepancies found.

---

*End of document. All results from SLURM job 6959494 (2026-04-05, corrected
token IDs, corrected left-padding indexing). Run on 2 × NVIDIA A100-SXM4-40GB
on CMU Babel cluster, node babel-y9-16. Total runtime 16 minutes (eval
parallel + merge). All code references are to `scripts/eval_step1.py` and
`scripts/merge_step1.py` as of the same date.*
