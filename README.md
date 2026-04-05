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
| 1 | Behavioral Evaluation + Activation Extraction — run both models on all 21,726 questions, assign suppression/enhancement/control labels, extract hidden states at 8 hook points | **Done** |
| 2 | Probing Analysis — train linear probes on activations to predict behavioral labels, find which layers encode cultural knowledge | Not started |
| 3 | Circuit Identification — attention pattern analysis, causal tracing to identify suppression circuits | Not started |
| 4 | Interpretation & Synthesis — coherent narrative of how RLHF affects cultural knowledge | Not started |

## Step 1: Behavioral Evaluation — Results

Full documentation: [docs/step1_behavioral_evaluation.md](docs/step1_behavioral_evaluation.md)

Step 1 runs both models on all 21,726 Sanskriti questions in a single forward pass per question. It extracts logit-based answer predictions (argmax over A/B/C/D token logprobs), assigns behavioral labels (suppression/enhancement/control), and captures hidden-state activations at 8 hook points (embedding + layers 4, 8, 14, 20, 26, 30, 31) using both mean-pool and last-token strategies. Both models run in parallel on separate GPUs via `torch.multiprocessing` with SLURM preemption-safe checkpointing.

### Key Results

| Metric | Base | Instruct |
|--------|------|----------|
| Overall accuracy | 87.61% | 88.88% |
| Accuracy (no Country Prediction) | 83.72% | 85.46% |
| Accuracy (hard questions only) | 84.69% | 86.39% |
| Forced-choice rate | 0.00% | 0.00% |
| Low-confidence predictions | 8.3% | 2.8% |

### Behavioral Labels

| Label | Count | % |
|-------|-------|---|
| Control (both correct) | 18,380 | 84.6% |
| Control (both wrong) | 1,761 | 8.1% |
| Enhancement (base wrong, instruct right) | 930 | 4.3% |
| Suppression (base right, instruct wrong) | 655 | 3.0% |

**Net effect of instruction tuning: +1.27% accuracy** (enhancement > suppression). Instruction tuning does not suppress cultural knowledge on net, but 655 questions show specific cultural knowledge that RLHF interfered with.

**Highest suppression:** General Awareness questions (5.2%), Cuisine attribute (4.19%), and smaller/northeastern states (Ladakh 7.55%, Lakshadweep 5.74%). Suppression concentrates in questions about regional cultural practices and less-represented geographic areas.

**Activation extraction:** 32 files (8 hooks x 2 pooling x 2 models), 9.0 GB total, all verified (no NaN/Inf, correct shapes). Ready for Step 2 probing.

### Key Design Choices
- **Logit-based evaluation** (not generation) — deterministic, single forward pass, full probability distribution preserved
- **5-shot prompting for base**, chat template for instruct — matches MMLU protocol and Sanskriti paper methodology
- **Model-specific answer token IDs** — base uses space-prefixed tokens (` A`=362) because the prompt ends with `Answer:`, instruct uses bare tokens (`A`=32) because the chat template ends with `\n\n`
- **Three-tier analysis** — full dataset, without Country Prediction, hard questions only — to isolate suppression signal from noise floor
- **11 sanity checks** in the merge pipeline to catch token ID bugs, prompt issues, and data artifacts

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
│   ├── eda_pipeline.py                 # Full EDA pipeline (8 sections, 2.7 min)
│   ├── eval_step1.py                   # Step 1: behavioral eval + activation extraction
│   ├── merge_step1.py                  # Step 1: merge results, labels, stats, plots
│   └── run_step1.sh                    # SLURM submission script for Step 1
├── plots/                              # EDA + Step 1 plot PNGs
│   ├── *.png                          # 13 EDA plots
│   └── step1/                         # 7 Step 1 plots (accuracy, labels, suppression, confidence)
├── configs/
│   ├── config.yaml                     # All paths and settings
│   └── environment.yml                 # Conda env spec
├── docs/
│   ├── eda_analysis.md                 # Complete EDA report with all numbers
│   ├── step1_behavioral_evaluation.md  # Step 1 complete analysis doc
│   └── plan.md                         # Project plan
├── logs/
├── notebooks/
└── old_version/                        # Previous Qwen2-1.5B study

/data/user_data/anshulk/cultural-mi/    # Heavy files (local NVMe)
├── models/{base,instruct}/             # LLaMA 3.1 8B weights (~30GB each with original/)
├── dataset/                            # Sanskriti HF cache
├── analysis/                           # 52 EDA CSV files + embeddings (~183MB)
├── results/{step1..step4}/             # Result CSVs + analysis (Step 1: 15 files, 3.6 MB master CSV)
├── activations/{base,instruct}/        # Hidden states: 32 .npy files (8 hooks × 2 pooling × 2 models, 9.0 GB)
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

CMU Babel — SLURM with L40S (48GB), A100-80GB, A100-40GB, H200, RTX PRO 6000 (96GB), and A6000 GPUs. Step 1 ran on 2 × A100-SXM4-40GB on the `preempt` partition (one GPU per model, parallel execution, batch size 24, 16 min total).
