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
- [x] **Phase 1: Activation extraction from base and chat models** *(Completed: 2025-10-19)*
  - Base model activations extracted (40K samples, layers 6/12/18)
  - Chat model activations extracted (40K samples, layers 6/12/18)
  - Delta activations computed (chat - base)
  - Output: 27 activation sets saved to `/datadrive/anshulk/activations/run_20251019_192554/`

### ✅ Phase 2 Complete
- [x] **Phase 2: Triple SAE training (base/chat/delta)** *(Completed: 2025-10-20)*
  - **Run 1** (k=128, dict=16,384): 2/9 SAEs passed validation (22% success)
  - **Run 2** (k=256, dict=8,192): 3/9 SAEs passed validation (33% success)
  - **Run 3** (k=256, dict=8,192 + auxiliary loss): Final implementation
    - Implemented OpenAI's dead neuron revival mechanism
    - Encoder transpose initialization + auxiliary loss
    - Periodic dead neuron monitoring and reset
    - Training on 3 GPUs (0, 1, 2)
    - Results pending analysis

### 🔄 In Progress
- [ ] Phase 2.5: LLM-based feature labeling and interpretation

### 📋 Upcoming
- [ ] DOSA validation dataset integration
- [ ] Phase 3: Feature validation and analysis
  - Algorithmic coherence computation
  - SAEBench evaluation
  - Cohen's d effect size measurement
  - Human annotation (dual-judge consensus)
  - Cross-validation stability testing
  - DOSA cross-validation

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
│   ├── phase1_extract_activations.py     # ✅ Phase 1
│   └── phase2_train_saes.py              # ✅ Phase 2
├── utils/            # Utility modules
│   ├── data_loader.py              # Dataset loading
│   ├── activation_extractor.py     # Activation extraction
│   ├── activation_dataset.py       # Activation data loading
│   ├── sae_model.py                # SAE architecture
│   └── sae_trainer.py              # Multi-GPU training
├── data/             # Data storage (on /datadrive, gitignored)
├── outputs/          # Results and activations (gitignored)
│   ├── activations/  # Phase 1 outputs
│   ├── sae_models/   # Phase 2 outputs
│   └── logs/         # Training logs
└── notebooks/        # Analysis notebooks
```

---

## Models

- **Base Model**: `Qwen/Qwen1.5-1.8B` (pre-training only)
- **Chat Model**: `Qwen/Qwen1.5-1.8B-Chat` (RLHF/SFT aligned)
- **Target Layers**: 6, 12, 18 (early, middle, late representations)
- **SAE Architecture**: 2048 → 8,192 → 2048 (input → dictionary → reconstruction)
  - TopK sparsity (k=256, ~3% active features)
  - Auxiliary loss for dead neuron revival (OpenAI 2024)
  - 31× sparsity ratio (exceeds 10× threshold)

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
python scripts/phase1_extract_activations.py
```

### 4. Train SAEs (Phase 2)

```bash
# Train Triple SAE (base/chat/delta) on 3 GPUs
python scripts/phase2_train_saes.py
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

## Phase 1 Results

### Activation Extraction
- **Duration**: ~1.5 hours total
- **Base Model Processing**: 40K samples across 3 datasets
- **Chat Model Processing**: 40K samples across 3 datasets
- **Delta Computation**: Automated chat - base differences
- **Memory Usage**: Peak 3.5GB/80GB per GPU
- **Output Format**: Sentence-level activations [N, 2048] saved as compressed .npz chunks
- **Storage Location**: `/datadrive/anshulk/activations/run_20251019_192554/`

### Activation Statistics
- **Shape**: [batch_size, 2048] per layer (mean-pooled over sequence length)
- **Datasets Processed**:
  - updesh_beta: 30,000 samples (cultural content)
  - snli_control: 5,000 samples (English control)
  - hindi_control: 5,000 samples (Hindi control)
- **Total Activation Sets**: 27 (3 datasets × 3 model types × 3 layers)

---

## Phase 2 Results

### SAE Training Configuration
- **Architecture**: TopK Sparse Autoencoder (2048 → 8,192 → 2048)
- **Sparsity Method**: TopK (k=256, 3.1% active features per token)
- **Auxiliary Loss**: 0.03 coefficient for dead neuron revival
- **Dead Neuron Monitoring**: Check every 3,000 steps, reset if <0.1% activation
- **Batch Size**: 256 per GPU
- **Learning Rate**: 1e-4 with 1,000 step warmup
- **Epochs**: 100 per SAE (early stopping when no improvement)
- **Hardware**: Multi-GPU DataParallel (GPUs 0, 1, 2)
- **Training Data**: updesh_beta (30K cultural samples)
- **Validation Data**: snli_control + hindi_control (10K control samples)

### Training Evolution

#### Iteration 1: Pure TopK (k=128, dict=16,384)
- **Success Rate**: 2/9 SAEs (22%)
- **Problem**: Excessive sparsity (128× ratio) → 98% dead features
- **Outcome**: Only Delta SAEs passed (matching pattern where difference vectors are lower-dimensional)

#### Iteration 2: Relaxed Sparsity (k=256, dict=8,192)
- **Success Rate**: 3/9 SAEs (33%)
- **Problem**: Still high dead feature percentage (~75%)
- **Outcome**: All Delta SAEs passed, Base/Chat still failing

#### Iteration 3: Auxiliary Loss Implementation (Current)
- **Method**: Encoder transpose initialization + auxiliary loss using top-512 dead latents
- **Theoretical Improvement**: Dead latents reduced from 90% to 7% in OpenAI's GPT-4 experiments
- **Expected Outcome**: All 9 SAEs pass validation
- **Status**: Training in progress

### Implementation Details

Following Gao et al. (2024), we implemented two critical techniques:

1. **Encoder Transpose Initialization**
   - Encoder weights initialized as decoder transpose
   - Prevents early feature death during training
   - Gives all features equal initial activation probability

2. **Auxiliary Loss Mechanism**
   - Models reconstruction error using top-k_aux dead latents
   - Loss coefficient: 0.03 (3% of total gradient signal)
   - Provides training signal to dormant features
   - Total loss: `recon_loss + 0.03 × aux_loss`

3. **Dead Neuron Monitoring**
   - Tracks feature activations over 1,000-step windows
   - Every 3,000 steps: resets features with <0.1% activation frequency
   - Xavier uniform re-initialization + optimizer state reset
   - Prevents permanent feature collapse

---

## SAE Methodology & Dead Feature Mitigation

### The Dead Feature Problem

In large sparse autoencoders, an increasing proportion of features stop activating entirely during training, with up to 90% dead features observed without mitigation. This results in:
- Wasted computational capacity
- Worse reconstruction quality
- Information bottleneck preventing full semantic representation

### Solution: Auxiliary Loss Framework

We implemented OpenAI's k-sparse autoencoder approach with auxiliary loss modifications to address this:

**TopK Sparsity Control**
- Directly enforces exactly k=256 active features per token
- Eliminates need to tune L1 penalty hyperparameter
- Improves reconstruction-sparsity frontier over baseline ReLU autoencoders

**Dead Feature Revival**
- Auxiliary loss models reconstruction error using dead features, reducing dead latents to 7% even in 16 million feature autoencoders
- Encoder transpose initialization prevents early feature death
- Periodic neuron reset prevents permanent dormancy

### Why Delta SAEs Succeed First

Delta activations (chat - base) consistently outperformed raw activations in early training iterations because:
- **Lower effective dimensionality**: Difference vectors capture only RLHF-shifted dimensions
- **Sparser signal**: Only hundreds of dimensions change, not full 2048D space
- **Easier decomposition**: Less semantic information requires fewer dictionary features

This pattern validated our hypothesis that RLHF creates systematic, localizable shifts rather than holistic representation changes.

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

### OpenAI Sparse Autoencoders
```bibtex
@article{gao2024scaling,
  title={Scaling and evaluating sparse autoencoders},
  author={Gao, Leo and Dupr{\'e} la Tour, Tom and Tillman, Henk and Goh, Gabriel and Troll, Rajan and Radford, Alec and Sutskever, Ilya and Leike, Jan and Wu, Jeffrey},
  journal={arXiv preprint arXiv:2406.04093},
  year={2024},
  url={https://arxiv.org/abs/2406.04093}
}
```

**Key contribution**: Demonstrated that auxiliary loss and encoder transpose initialization reduce dead features from 90% to 7%, enabling scaling to 16 million features on GPT-4.

---

## Acknowledgments

We gratefully acknowledge:

- **OpenAI Research** for sparse autoencoder scaling techniques and auxiliary loss methodology
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

This is an ongoing research project. Code and findings are not yet ready for public release. For questions or collaboration inquiries, please contact anshulk@andrew.cmu.edu

---

## Progress Log

- **2025-10-19 (Morning)**: Project initialized, datasets downloaded (40K samples)
- **2025-10-19 (Afternoon)**: Phase 1 activation extraction completed
  - Base model: 40K samples processed
  - Chat model: 40K samples processed
  - Delta activations computed
  - Total processing time: ~1.5 hours
- **2025-10-19 (Evening)**: Phase 2 SAE training initiated
  - Base Layer 6 SAE completed (reconstruction loss: 0.0000092)
  - Remaining 8 SAEs in progress
  - ETA for Phase 2 completion: ~70 minutes
- **2025-10-19 (Late Evening)**: Phase 2 iterations
  - Run 1 (k=128): 22% success rate, identified dead feature problem
  - Run 2 (k=256): 33% success rate, validated Delta SAE hypothesis
- **2025-10-20 (Early Morning)**: Implemented auxiliary loss mechanism
  - Following OpenAI's dead neuron revival techniques
  - Encoder transpose initialization
  - Periodic monitoring and reset of dormant features
  - Final training run in progress

---

*This research is part of a broader investigation into cultural alignment in large language models.*
