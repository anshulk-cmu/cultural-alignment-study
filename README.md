# Mechanistic Interpretability of Cultural Knowledge in Instruction-Tuned LLMs

**Authors:** Anshul Kumar and Pragati Bhattad — Carnegie Mellon University
**Target venue:** EMNLP 2026 Workshops

## Research Question

How does instruction tuning (SFT + RLHF) mechanistically alter cultural knowledge representations inside LLMs? Does behavioral suppression of cultural knowledge reflect **representational erasure** (knowledge destroyed) or **output-level gating** (knowledge preserved but blocked)?

## Approach

Compare **Llama-3.1-8B** (base) vs **Llama-3.1-8B-Instruct** (RLHF-treated) on the **Sanskriti** benchmark — 21,853 multiple-choice questions spanning Indian cultural knowledge across 36 states/UTs and 16 cultural attributes.

Both models share identical architecture (32 layers, 4096 hidden dim, 8B params) and pretraining. The only difference is instruction tuning. This controlled setup isolates the effect of RLHF on internal representations.

## Pipeline

| Step | What | Status |
|------|------|--------|
| EDA | Dataset exploration and validation | **Done** |
| 1 | Behavioral Evaluation — run both models, assign suppression/enhancement/control labels | Planned |
| 2 | Activation Extraction — extract hidden states at selected layers | Not started |
| 3 | Probing & Analysis — linear probing, KL divergence, MDL probing | Not started |
| 4 | Circuit Interpretation — identify specific circuits responsible | Not started |

## EDA Key Findings

Full analysis: [docs/eda_analysis.md](docs/eda_analysis.md)

The dataset was validated and profiled across 8 dimensions. Critical findings that shape Step 1:

| Finding | Number | Impact |
|---------|--------|--------|
| Usable rows (after excluding broken ground truth) | 21,726 of 21,853 | 127 rows excluded (0.58%) |
| Country Prediction answers that are "India" | 100% (5,563 questions) | 25.6% of dataset is trivial |
| No-question baseline accuracy (state name vs options, no question read) | **75.87%** | Fundamental benchmark limitation |
| State Prediction solvable by string matching alone | **99.98%** | Shortcut-vulnerable |
| Near-duplicate question pairs (cosine sim > 0.85) | 77,833 pairs (78.6% of questions involved) | Effective sample size much smaller than 21K |
| Conflicting duplicates (same question, different correct answer) | 351 groups | Data quality issue (not fatal) |
| Answer-in-question leakage | 1,615 (7.4%) | Inflates control_both_correct |
| Ground truth position bias | B=29.0%, D=20.8% (χ²=363.6, p<1e-78) | Must check model position bias |
| Questions following 7 templates | 55.4% | Highly formulaic dataset |
| State-attribute cells with reliable data (n>=125) | 39 of 576 (6.8%) | Cannot do per-cell behavioral analysis |

**Recommendations:** Report metrics with and without Country Prediction. Flag State Prediction as shortcut-vulnerable. Aggregate behavioral labels to state or attribute level, never at the intersection.

## Models

| Role | Model | Params | Precision | Local Path |
|------|-------|--------|-----------|------------|
| Base | `meta-llama/Llama-3.1-8B` | 8.03B | BF16 | `/data/.../models/base/` |
| Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8.03B | BF16 | `/data/.../models/instruct/` |

Both downloaded and verified. Architecture: 32 layers, 4096 hidden, GQA (32 Q-heads, 8 KV-heads), SwiGLU MLP (14336 intermediate). Only differences: instruct has 3 EOS token IDs and a Jinja2 chat template.

## Dataset

**Sanskriti** ([HuggingFace](https://huggingface.co/datasets/13ari/Sanskriti), ACL 2025 Findings) — 21,853 MCQs about Indian culture. 36 states/UTs, 16 attributes, 4 question types.

Downloaded to `/data/user_data/anshulk/cultural-mi/dataset/`.

## Project Structure

```
/home/anshulk/cultural-mi/              # Code, plots, docs (NFS home)
├── scripts/
│   └── eda_pipeline.py                 # Full EDA pipeline (8 sections, 2.7 min)
├── plots/                              # 13 EDA plot PNGs
├── configs/
│   ├── config.yaml                     # All paths and settings
│   └── environment.yml                 # Conda env spec
├── docs/
│   ├── step1_plan.md                   # Validated Step 1 plan
│   └── eda_analysis.md                 # Complete EDA report with all numbers
├── logs/
├── notebooks/
└── old_version/                        # Previous Qwen2-1.5B study

/data/user_data/anshulk/cultural-mi/    # Heavy files (local NVMe)
├── models/{base,instruct}/             # LLaMA 3.1 8B weights (~30GB each with original/)
├── dataset/                            # Sanskriti HF cache
├── analysis/                           # 52 EDA CSV files + embeddings (~183MB)
├── results/{step1..step4}/             # Result CSVs (Step 1 pending)
├── activations/                        # Extracted hidden states (Step 2)
└── checkpoints/                        # Inference checkpoints
```

## Previous Work

The `old_version/` directory contains the first iteration using **Qwen2-1.5B** (base vs instruct), which found ~8% suppression and ~7% enhancement with a net effect of ~1.27%. This new iteration uses a larger model (LLaMA 3.1 8B) with a more rigorous methodology.

## Environment

```bash
conda env create -f configs/environment.yml
conda activate cultural
```

Key packages: torch 2.10, transformers 5.3, datasets 4.8, accelerate 1.13, sentence-transformers 5.3, BERTopic 0.17.

## Cluster

CMU Babel — SLURM with L40S (48GB), A100-80GB, A100-40GB, H200, and A6000 GPUs.
