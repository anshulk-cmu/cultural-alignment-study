# Cultural Alignment Study: RLHF Impact on Cultural Representations

**Work in Progress** | Research Question 1 (RQ1): Cultural Feature Discovery

A research project investigating how Reinforcement Learning from Human Feedback (RLHF) systematically reshapes cultural marker representations in language models using Sparse Autoencoders (SAEs).

---

## Research Question

**RQ1: Cultural Feature Discovery**  
Can Sparse Autoencoders identify interpretable cultural features showing systematic representation differences between base and post-training aligned models on Indian English/Hindi text?

### Hypothesis
SAEs can achieve reconstruction loss < 0.05 with L0 sparsity > 10× baseline, discovering interpretable features with Cohen's d > 0.8 between base and chat models.

---

## Current Status

### ✅ Completed
- [x] Environment setup on Azure A100 VM
- [x] Updesh_beta dataset downloaded (30K samples)
- [x] SNLI control set downloaded (5K English samples)
- [x] IITB Hindi control set downloaded (5K Hindi samples)
- [x] Total: 40K sentences ready for activation extraction

### 🔄 In Progress
- [ ] Phase 1: Activation extraction from base and chat models
- [ ] DOSA validation dataset preparation

### 📋 Upcoming
- [ ] Phase 2: Triple SAE training (base/chat/delta)
- [ ] Phase 3: Feature validation and analysis

---

## Project Structure

```
rq1_cultural_features/
├── configs/          # Configuration files
│   └── config.py     # Model, dataset, and training configs
├── scripts/          # Executable scripts
│   ├── download_updesh.py
│   ├── download_snli.py
│   ├── download_hindi_control.py
│   ├── phase1_data_setup.py
│   └── test_setup.py
├── utils/            # Utility modules
│   ├── data_loader.py      # Dataset loading and preprocessing
│   └── model_utils.py      # Model and activation extraction
├── data/             # Data storage (on /datadrive, gitignored)
├── outputs/          # Results and activations (gitignored)
└── notebooks/        # Analysis notebooks
```

---

## Models

- **Base Model**: `Qwen/Qwen1.5-1.8B` (pre-training only)
- **Chat Model**: `Qwen/Qwen1.5-1.8B-Chat` (RLHF/SFT aligned)
- **Target Layers**: 6, 12, 18 (early, middle, late representations)

---

## Datasets

### Cultural Content (30K samples)

#### Updesh_beta
- **Size**: 30,000 samples (15K English + 15K Hindi)
- **Content**: Assistant responses from cultural multi-hop reasoning tasks
- **Source**: `microsoft/Updesh_beta` (HuggingFace)
- **Purpose**: Primary dataset for cultural feature discovery

### Control Content (10K samples)

#### SNLI Control Set
- **Size**: 5,000 English sentences
- **Content**: Generic image captions from Flickr30k
- **Source**: `stanfordnlp/snli` (HuggingFace)
- **Purpose**: Non-cultural baseline for comparison

#### IITB Hindi Control Set
- **Size**: 5,000 Hindi sentences
- **Content**: Generic translated sentences (news, web content)
- **Source**: `cfilt/iitb-english-hindi` (HuggingFace)
- **Purpose**: Hindi non-cultural baseline

### Validation Set (Planned)

#### DOSA Dataset
- **Size**: 615 samples
- **Content**: Community-generated social artifacts from 19 Indian geographic subcultures
- **Purpose**: Cross-validation of discovered cultural features
- **Status**: To be integrated

---

## Setup Instructions

### 1. Environment Setup

```bash
# Create conda environment
conda create -n rq1 python=3.10 -y
conda activate rq1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate safetensors huggingface_hub
pip install sae-lens einops
pip install pandas numpy scipy scikit-learn tqdm matplotlib seaborn
```

### 2. Download Datasets

```bash
# Download Updesh_beta (cultural content)
python scripts/download_updesh.py

# Download SNLI control (English)
python scripts/download_snli.py

# Download IITB control (Hindi)
python scripts/download_hindi_control.py
```

### 3. Extract Activations (Phase 1)

```bash
# Extract activations from both models at layers 6, 12, 18
python scripts/phase1_data_setup.py
```

---

## Hardware Configuration

- **VM**: Azure A100-Machine-004 (NeuLab Bridge Compute)
- **GPUs**: 4x NVIDIA A100 80GB PCIe
- **Storage**: 
  - `/datadrive` (8TB) - Data and activations
  - `/mnt/nfs-shared-centralus` (5TB) - Code and environments
- **Location**: `/mnt/nfs-shared-centralus/anshulk/rq1_cultural_features`

---

## Dataset Citations & Acknowledgments

### Updesh_beta
```bibtex
@misc{updesh_beta,
  title={Updesh: A Cultural Multi-hop Reasoning Dataset},
  author={Microsoft Research},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/datasets/microsoft/Updesh_beta}}
}
```

### SNLI
```bibtex
@inproceedings{bowman2015large,
  title={A large annotated corpus for learning natural language inference},
  author={Bowman, Samuel R and Angeli, Gabor and Potts, Christopher and Manning, Christopher D},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2015}
}
```

### IITB English-Hindi Parallel Corpus
```bibtex
@inproceedings{kunchukuttan2018iit,
  title={The IIT Bombay English-Hindi Parallel Corpus},
  author={Kunchukuttan, Anoop and Mehta, Pratik and Bhattacharyya, Pushpak},
  booktitle={Language Resources and Evaluation Conference},
  year={2018}
}
```

### Qwen Models
```bibtex
@article{qwen,
  title={Qwen Technical Report},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2023}
}
```

---

## Acknowledgments

We gratefully acknowledge:

- **Microsoft Research** for the Updesh_beta cultural reasoning dataset
- **Stanford NLP Group** for the SNLI corpus
- **IIT Bombay CFILT** for the English-Hindi parallel corpus
- **Alibaba Cloud** for the Qwen model family
- **NeuLab (CMU)** for providing Azure compute resources

---

## Validation Metrics

Our study will evaluate discovered features against 7 validation criteria:

1. **Reconstruction Loss** (< 0.05): Faithfulness to original activations
2. **L0 Sparsity** (> 10×): Decomposition into interpretable features
3. **SAEBench Score** (≥ 0.70): Predictive validity of interpretations
4. **Cohen's d** (> 0.8): Effect size of RLHF-induced shifts
5. **Algorithmic Coherence** (≥ 0.60): Geometric consistency of features
6. **Human Agreement** (≥ 0.70): Inter-annotator reliability
7. **Cross-validation Stability** (> 0.80): Generalization across splits

---

## License & Usage

This is an ongoing research project. Code and findings are not yet ready for public release. For questions or collaboration inquiries, please contact the research team.

---

## Progress Log

- **2025-10-19**: Project initialized, datasets downloaded (40K samples)
- **Next**: Phase 1 activation extraction from Qwen base and chat models

---

*This research is part of a broader investigation into cultural alignment in large language models.*
