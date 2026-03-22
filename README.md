# Mechanistic Interpretability of Cultural Knowledge in Instruction-Tuned LLMs

**Author:** Anshul Kumar (anshulk@andrew.cmu.edu), Carnegie Mellon University

## Research Question

How does instruction tuning (SFT + RLHF) mechanistically alter cultural knowledge representations inside LLMs? Does behavioral suppression of cultural knowledge reflect **representational erasure** (knowledge destroyed) or **output-level gating** (knowledge preserved but blocked)?

## Approach

Compare **Llama-3.1-8B** (base) vs **Llama-3.1-8B-Instruct** (RLHF-treated) on the **Sanskriti** benchmark — 21,853 multiple-choice questions spanning Indian cultural knowledge across 36 states/UTs and 16 cultural attributes.

Both models share identical architecture (32 layers, 4096 hidden dim, 8B params) and pretraining. The only difference is instruction tuning. This controlled setup isolates the effect of RLHF on internal representations.

## Pipeline

| Step | What | Purpose |
|------|------|---------|
| 1 | Behavioral Evaluation | Run both models on all 21,853 questions, label each as suppression / enhancement / control |
| 2 | Activation Extraction | Extract hidden states at selected layers for behaviorally-labeled questions |
| 3 | Probing & Analysis | Linear probing, KL divergence, MDL probing on extracted activations |
| 4 | Circuit Interpretation | Identify specific circuits responsible for cultural knowledge changes |

## Models

| Role | Model | Params | Precision |
|------|-------|--------|-----------|
| Base | `meta-llama/Llama-3.1-8B` | 8B | BF16 |
| Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | BF16 |

## Dataset

**Sanskriti** ([HuggingFace](https://huggingface.co/datasets/13ari/Sanskriti)) — 21,853 MCQs about Indian culture. 36 states/UTs, 16 attributes (Rituals, History, Cuisine, Dance & Music, Art, etc.), 4 question types (General Awareness, State Prediction, Country Prediction, Association Based).

## Project Structure

```
/home/anshulk/cultural-mi/          # Code, plots, logs (NFS)
├── scripts/                         # Python scripts
├── notebooks/                       # Jupyter notebooks
├── plots/                           # Figures
├── analysis/                        # Analysis outputs
├── logs/                            # Run logs
├── configs/                         # config.yaml, environment.yml
├── docs/                            # Documentation & plans
└── old_version/                     # Previous study (Qwen2-1.5B)

/data/user_data/anshulk/cultural-mi/ # Heavy files (local NVMe)
├── models/{base,instruct}/          # LLaMA weights (~16GB each)
├── dataset/                         # Sanskriti cache
├── results/{step1..step4}/          # Result CSVs and stats
├── activations/                     # Extracted hidden states
└── checkpoints/                     # Inference checkpoints
```

## Previous Work

The `old_version/` directory contains the first iteration of this study using **Qwen2-1.5B** (base vs instruct), which found 42% behavioral suppression but 99.7% representational similarity — suggesting output gating rather than erasure. This new iteration uses a larger model (LLaMA 3.1 8B) with a more rigorous methodology.

## Environment

```bash
conda env create -f configs/environment.yml
conda activate cultural
```

## Cluster

SLURM cluster with L40S (48GB), A100-80GB, and A6000 (48GB) GPUs.
