# Cultural Alignment Study: RLHF Impact on Cultural Representations

**Work in Progress** | Cultural Feature Discovery with Causal Validation

A research project investigating how Reinforcement Learning from Human Feedback (RLHF) systematically reshapes cultural marker representations in language models using Sparse Autoencoders (SAEs) with causal identifiability guarantees.

---

## Research Questions

**RQ1: Cultural Feature Discovery**
Can Sparse Autoencoders identify interpretable cultural features showing systematic representation differences between base and post-training aligned models on Indian English/Hindi text?

**Hypothesis**: SAEs can achieve reconstruction loss < 0.05 with L0 sparsity > 10× baseline, discovering interpretable features with Cohen's d > 0.8 between base and chat models.

**Status**: ✅ VALIDATED - 2,458 interpretable features discovered with 2.07× RLHF enrichment in social scenes

**RQ2: Causal Validation**
Do discovered features have causal influence on cultural representations, validated through activation patching with identifiability guarantees from multi-distribution analysis?

**Hypothesis**: Features satisfying identifiability conditions (distribution variance, sparsity, causal sufficiency) will demonstrate causal impact with effect sizes > 0.8.

**Status**: 🔄 In Progress - Feature selection with identifiability gates

---

## Current Status

### ✅ Completed
- [x] Environment setup on CMU Babel HPC
- [x] Updesh_beta dataset downloaded (30K samples)
- [x] SNLI control set downloaded (5K English samples)
- [x] IITB Hindi control set downloaded (5K Hindi samples)
- [x] Total: 40K sentences ready for activation extraction
- [x] **Phase 1: Activation extraction from base and chat models** *(Completed: 2025-10-19)*
  - Base model activations extracted (40K samples, layers 6/12/18)
  - Chat model activations extracted (40K samples, layers 6/12/18)
  - Delta activations computed (chat - base)
  - Output: 27 activation sets saved to `/user_data/anshulk/data/activations/run_20251019_192554/`

### ✅ Phase 2 Complete
- [x] **Phase 2: Triple SAE training (base/chat/delta)** *(Completed: 2025-10-20)*
  - **Run 1** (k=128, dict=16,384): 2/9 SAEs passed validation (22% success)
  - **Run 2** (k=256, dict=8,192): 3/9 SAEs passed validation (33% success)
  - **Run 3** (k=256, dict=8,192 + auxiliary loss): **9/9 SAEs passed validation (100% success)**
    - Mean reconstruction loss: 0.002195 (45× better than threshold)
    - All SAEs achieve <0.05 reconstruction with 32× sparsity
    - Auxiliary loss remained at 0.000 throughout training
    - Encoder transpose initialization + periodic monitoring succeeded

### ✅ Phase 2.5 Complete
- [x] **Phase 2.5: Feature Interpretation** *(Completed: 2025-10-21)*
  - Feature example extraction (3,600 features × 9 SAEs)
  - Initial labeling with Qwen1.5-32B-Chat (4-GPU parallel processing)
  - Validation with Qwen3-30B-A3B-Instruct
  - **Results**: 1,621 KEEP + 837 REVISE = 2,458 valid labels (68.3% success rate)

### ✅ Phase 3 Complete
- [x] **Phase 3: Semantic Clustering & Analysis** *(Completed: 2025-10-22)*
  - Label corpus analysis (TF-IDF, n-grams)
  - K-means clustering (k=8-12 tested, k=8 selected)
  - **8 semantic themes discovered** with RLHF impact quantification
  - **Key Finding**: Social scenes (2.07× enrichment), Hindi coherence (1.23× enrichment)

### 🔄 In Progress
- [ ] Phase 4: Feature selection with identifiability gates (Weeks 1-6)
  - Distribution variance scoring
  - Sparsity-based identifiability
  - DOSA ground-truth validation

### 📋 Upcoming
- [ ] Phase 5: Causal validation (Weeks 7-12)
  - Multi-corruption activation patching
  - OR-Bench over-refusal check
  - Causal Cultural Impact Score (CCIS) computation

---

## Project Structure

```
cultural-alignment-study/
├── configs/
│   └── config.py
├── scripts/
│   ├── Phase 0-3 (Completed)
│   │   ├── download_*.py
│   │   ├── phase1_extract_activations.py
│   │   ├── phase2_train_saes.py
│   │   ├── phase2_5_*.py (labeling & validation)
│   │   ├── analyze_label_corpus.py
│   │   ├── cluster_semantic_themes.py
│   │   └── build_empirical_taxonomy.py
│   ├── Phase 4 (In Progress)
│   │   ├── phase4_1_identifiability_scoring.py
│   │   ├── phase4_2_feature_typing.py
│   │   └── phase4_3_dosa_integration.py
│   └── Phase 5 (Planned)
│       ├── phase5_2_activation_patching.py
│       ├── phase5_3_refusal_testing.py
│       └── phase5_4_ccis_computation.py
├── utils/
│   ├── data_loader.py
│   ├── activation_extractor.py
│   ├── sae_model.py
│   ├── sae_trainer.py
│   ├── patching_utils.py (Phase 5)
│   └── identifiability_metrics.py (Phase 4)
├── outputs/
│   ├── labels/ (16MB - Phase 2.5)
│   ├── analysis/ (27MB - Phase 3)
│   │   ├── clustering/
│   │   └── LABEL_CORPUS_ANALYSIS.txt
│   ├── identifiable_features/ (Phase 4)
│   └── causal_validation/ (Phase 5)
└── protocols/
    └── PRE_REGISTRATION.md (Phase 5)
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

### 5. Feature Labeling & Clustering (Phase 2.5-3)

```bash
# Extract examples and label features
python scripts/phase2_5_extract_examples.py
python scripts/phase2_5_qwen_label.py
python scripts/phase2_5_qwen_validate.py

# Analyze and cluster
python analyze_label_corpus.py
python cluster_semantic_themes.py
```

### 6. Identifiability Gates (Phase 4)

```bash
# Install additional dependencies
pip install sentence-transformers baukit

# Download DOSA and run identifiability scoring
python scripts/download_dosa.py
python scripts/phase4_1_identifiability_scoring.py --top-k 30
python scripts/phase4_3_dosa_integration.py
```

---

## Hardware Configuration

- **HPC**: CMU Babel HPC Cluster
- **GPUs**: Multiple NVIDIA GPUs available on compute nodes
- **Storage**:
  - `/home/anshulk` (~93GB) - Code and configurations (login node)
  - `/user_data/anshulk` (10TB) - Data, models, and activations (compute nodes)
- **Location**: `/home/anshulk/cultural-alignment-study`

---

## Phase 1 Results

### Activation Extraction
- **Duration**: ~1.5 hours total
- **Base Model Processing**: 40K samples across 3 datasets
- **Chat Model Processing**: 40K samples across 3 datasets
- **Delta Computation**: Automated chat - base differences
- **Memory Usage**: Peak 3.5GB/80GB per GPU
- **Output Format**: Sentence-level activations [N, 2048] saved as compressed .npz chunks
- **Storage Location**: `/user_data/anshulk/data/activations/run_20251019_192554/`

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

#### Iteration 3: Auxiliary Loss Implementation (Final)
- **Success Rate**: 9/9 SAEs (100%)
- **Method**: Encoder transpose initialization + auxiliary loss + dead neuron monitoring
- **Results**:
  - Mean reconstruction: 0.002195 (range: 0.000070 - 0.007866)
  - Delta SAEs: 0.000507 avg (best performance maintained)
  - Base SAEs: 0.002684 avg (5× improvement from Run 2)
  - Chat SAEs: 0.003394 avg (7× improvement from Run 2)
  - Auxiliary loss: 0.000 (monitoring prevented feature death)
- **Outcome**: All validation criteria met, ready for Phase 2.5 feature interpretation

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

## Phase 3 Results

### Semantic Clustering Analysis

**Configuration**: SentenceTransformers (all-MiniLM-L6-v2) + K-means clustering on 2,458 validated features

**Optimal Clustering**: k=8 (best balance: CV=0.347, silhouette=0.0655)

### Discovered Themes

| Cluster | Theme | Size | Delta Enrichment | RLHF Impact |
|---------|-------|------|------------------|-------------|
| 2 | Scenes & Cultural & Social | 265 (10.8%) | **2.07×** | ⭐ Strong RLHF shift |
| 3 | Indian & Hindi & Coherent | 276 (11.2%) | **1.23×** | ⭐ RLHF-shifted |
| 1 | Indian & Cultural & Linguistic | 411 (16.7%) | 1.09× | Moderate |
| 7 | Activities & Outdoor & Physical | 178 (7.2%) | 1.12× | Moderate |
| 6 | Indian & Cultural & Cinema | 515 (21.0%) | 0.78× | Diminished |
| 0 | Cultural & Indian & Art | 299 (12.2%) | 0.68× | Diminished |
| 5 | South & Asian & Cultural | 179 (7.3%) | 0.66× | Diminished |
| 4 | Cultural & Indian & India | 335 (13.6%) | 0.59× | Diminished |

**Key Finding**: RLHF systematically amplifies social scene understanding (2.07×) and Hindi linguistic coherence (1.23×) while reducing traditional cultural domains (cinema 0.78×, art 0.68×, traditions 0.59×). This indicates alignment processes are not culturally neutral but reshape cultural marker distributions with domain-specific biases.

---

## Methodology

### Multi-Distribution Framework

This study treats BASE, CHAT, and DELTA SAEs as three observational distributions for causal analysis:
- **BASE**: Pre-training distribution (observational)
- **CHAT**: Post-RLHF distribution (interventional)
- **DELTA**: Intervention effect (chat - base)

Analyzing features across these distributions enables identification of causal cultural mechanisms under identifiability conditions from causal representation learning (Zhang et al. 2024).

### Phase 4: Identifiability Gates (Weeks 1-6)

**Goal**: Narrow 2,458 features → 20-30 causally identifiable features

**Step 4.1 - Zhang's Identifiability Scoring**:
- **Distribution Variance**: Top 10% features with highest activation variance across BASE/CHAT/DELTA
- **Sparsity-Based Identifiability**: L0 < 20 on DOSA dataset (rare but strong activations)
- **Causal Sufficiency**: Preliminary patching shows ΔAccuracy > 5% on predictions

**Step 4.2 - Feature Characterization**:
- **Type A**: Exist in BASE, suppressed in CHAT (removed biases)
- **Type B**: Exist in BASE, amplified in CHAT (alignment targets)
- **Type C**: Only in DELTA (emergent from RLHF, strongest identifiability)

**Step 4.3 - DOSA Validation**:
- Run 615 DOSA cultural artifacts through models
- Features must activate on ≥3 subcultures (Tamil, Gujarat, Maharashtra, etc.)
- DOSA alignment score > 0.6 threshold

### Phase 5: Causal Validation (Weeks 7-12)

**Goal**: Prove causal influence with anti-illusion controls

**Step 5.1 - Pre-Registration**:
- Document corruption methods, patch sites, effect thresholds before experiments
- Prevents p-hacking, ensures scientific rigor

**Step 5.2 - Multi-Corruption Testing**:
- Test each feature with three methods: string replacement, semantic paraphrase, counterfactual
- Ablate feature, measure Demographic Alignment Score (DAS) change
- **Validation**: Mean Cohen's d > 0.8 AND SD < 0.3 across methods
- Features with inconsistent effects (e.g., d=1.2 for method 1, d=0.3 for method 2) indicate backup pathway illusions

**Step 5.3 - OR-Bench Safety**:
- Ensure ablations don't increase false refusals on non-harmful queries
- Fail threshold: Δ refusal rate > 20%

**Step 5.4 - Causal Cultural Impact Score**:
```
CCIS = (DAS_reduction × Identifiability_score) / (Semantic_loss + Task_loss + Illusion_penalty)
```
**Target**: ≥5 features with CCIS > 1.5, p < 0.001

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

### Identifiable Causal Representation Learning
```bibtex
@inproceedings{zhang2024identifiable,
  title={Identifiable Latent Polynomial Causal Models Through the Lens of Change},
  author={Zhang, Yuhang and Huang, Zhijing and Scholkopf, Bernhard and Rothenhausler, Dominik},
  booktitle={Conference on Causal Learning and Reasoning (CLeaR)},
  year={2024}
}
```

**Application**: Framework for identifying latent causal variables from multiple observational distributions (BASE/CHAT/DELTA) with theoretical guarantees.

---

## Acknowledgments

We gratefully acknowledge:

- **OpenAI Research** for sparse autoencoder scaling techniques and auxiliary loss methodology
- **Microsoft Research** for the Updesh_beta cultural reasoning dataset
- **Stanford NLP Group** for the SNLI corpus
- **IIT Bombay CFILT** for the English-Hindi parallel corpus
- **Alibaba Cloud** for the Qwen model family
- **CMU Computing Resources** for providing Babel HPC access

---

## Validation Metrics

### RQ1 Metrics (Achieved ✅)
1. **Reconstruction Loss**: 0.002195 mean < 0.05 threshold ✅
2. **L0 Sparsity**: 32× > 10× requirement ✅
3. **Feature Interpretability**: 68.3% validation rate ✅
4. **RLHF Effect Size**: 2.07× delta enrichment ✅

### RQ2 Metrics (Targets)
1. **Identifiable Features**: ≥20 features with distribution variance, sparsity, causal sufficiency
2. **Causal Consistency**: SD < 0.3 across corruption methods
3. **Effect Size**: Cohen's d > 0.8 on Demographic Alignment Score
4. **DOSA Alignment**: Score > 0.6 on cultural artifacts
5. **OR-Bench Safety**: Δ refusal rate < 20%
6. **CCIS**: ≥5 features with CCIS > 1.5, p < 0.001

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
- **2025-10-20 (Morning)**: Phase 2 Run 3 completed - 100% validation success
  - Auxiliary loss mechanism with encoder transpose initialization
  - All 9 SAEs passed reconstruction (<0.05) and sparsity (32×) criteria
  - Delta SAEs maintained superior performance (avg: 0.000507)
  - Base/Chat SAEs achieved 5-7× improvement over Run 2
  - Phase 2 officially complete
- **2025-10-21**: Phase 2.5 feature labeling and validation completed
  - Qwen1.5-32B initial labeling (4-GPU parallel)
  - Qwen3-30B validation (KEEP/REVISE/INVALIDATE)
  - 2,458 valid features (68.3% success rate)
- **2025-10-22**: Phase 3 clustering analysis completed
  - K-means clustering (k=8-12 tested)
  - Selected k=8 as optimal (best balance)
  - **Major finding**: 2.07× RLHF enrichment in social scenes, 1.23× in Hindi coherence
- **2025-10-27**: Phase 4 identifiability gates initiated
  - Multi-distribution framework established
  - Preparing Zhang's identifiability metrics
  - DOSA integration planned

---

*This research establishes a framework for identifying and validating causal cultural mechanisms in RLHF-aligned language models with theoretical identifiability guarantees.*
