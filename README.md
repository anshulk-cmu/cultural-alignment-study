# Cultural Alignment Study: RLHF Impact on Cultural Representations

Research project investigating how RLHF reshapes cultural marker representations in language models using Sparse Autoencoders (SAEs).

## Research Questions

**RQ1: Cultural Feature Discovery**  
Can SAEs identify interpretable cultural features showing systematic representation differences between base and post-training aligned models on Indian English/Hindi text?

## Project Structure
```
rq1_cultural_features/
├── configs/          # Configuration files
├── scripts/          # Executable scripts
│   ├── download_updesh.py
│   ├── phase1_data_setup.py
│   └── test_setup.py
├── utils/            # Utility modules
│   ├── data_loader.py
│   └── model_utils.py
├── data/             # Data storage (gitignored)
├── outputs/          # Results (gitignored)
└── notebooks/        # Analysis notebooks
```

## Setup

1. **Environment Setup**
```bash
conda create -n rq1 python=3.10 -y
conda activate rq1
pip install torch transformers datasets sae-lens
```

2. **Download Data**
```bash
python scripts/download_updesh.py
```

3. **Extract Activations (Phase 1)**
```bash
python scripts/phase1_data_setup.py
```

## Models

- **Base**: Qwen/Qwen1.5-1.8B
- **Chat**: Qwen/Qwen1.5-1.8B-Chat
- **Target Layers**: 6, 12, 18 (early, middle, late)

## Datasets

- **Updesh_beta**: 30K samples (15K English + 15K Hindi)
- **DOSA**: 615 community-generated artifacts (validation)
- **SNLI Control**: 5K generic sentences

## Current Status

✅ Phase 1: Data Infrastructure Setup - COMPLETE
- Environment configured on Azure A100 VM
- Updesh_beta dataset downloaded (30K samples)
- Activation extraction pipeline ready

⏳ Next: DOSA & SNLI dataset preparation

## Hardware

- **VM**: Azure A100-Machine-004
- **GPUs**: 4x NVIDIA A100 80GB
- **Storage**: 8TB `/datadrive` + 5TB NFS shared
