# Cultural Representation Learning: Investigating RLHF Effects Using Sparse Autoencoders

**Research Project: Work in Progress** | Mechanistic Analysis of Cultural Features in Aligned Language Models with Causal Validation

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
│       ├── download_*.py
|       ├── phase1_extract_activations.py
│       ├── phase1_5_prepare_texts.py
│       ├── phase2_train_saes.py
|       ├── phase2_analyze_results.py
|       ├── phase2_5_extract_examples.py
│       ├── phase2_5_*.py (labeling & validation)
│       ├── analyze_label_corpus.py
│       ├── cluster_semantic_themes.py
│       └── build_empirical_taxonomy.py
├── utils/
│   ├── data_loader.py
│   ├── activation_extractor.py
│   ├── sae_model.py
│   ├── sae_trainer.py
│   └── activation_dataset.py
└── outputs/
    ├── labels/ (16MB - Phase 2.5)
    ├── analysis/ (27MB - Phase 3)
        ├── clustering/
        └── LABEL_CORPUS_ANALYSIS.txt
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
- **Source**: Microsoft Research India
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

# Analyze label corpus and cluster themes
python scripts/analyze_label_corpus.py
python scripts/cluster_semantic_themes.py
python scripts/build_empirical_taxonomy.py
```

---

## Methodology

### Multi-Distribution Framework

This study treats BASE, CHAT, and DELTA SAEs as three observational distributions for causal analysis:
- **BASE**: Pre-training distribution (observational)
- **CHAT**: Post-RLHF distribution (interventional)
- **DELTA**: Intervention effect (chat - base)

Analyzing features across these distributions enables identification of causal cultural mechanisms under identifiability conditions from causal representation learning (Liu et al. 2024).

### Phase 4: Identifiability Gates (Weeks 1-6)

**Goal**: Narrow 2,458 features → 20-30 causally identifiable features

**Step 4.1 - Identifiability Scoring**:
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

### DOSA Dataset
```bibtex
@inproceedings{seth-etal-2024-dosa,
  title={{DOSA}: A Dataset of Social Artifacts from Different {I}ndian Geographical Subcultures},
  author={Seth, Agrima and Ahuja, Sanchit and Bali, Kalika and Sitaram, Sunayana},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  month=may,
  year={2024},
  address={Torino, Italia},
  publisher={ELRA and ICCL},
  url={https://aclanthology.org/2024.lrec-main.474},
  pages={5323--5337}
}
```

### Qwen Models
```bibtex
@article{bai2023qwen,
  title={Qwen Technical Report},
  author={Bai, Jinze and Bai, Shuai and Chu, Yunfei and Cui, Zeyu and others},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023},
  url={https://arxiv.org/abs/2309.16609}
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
@inproceedings{liu2024identifiable,
  title={Identifiable Latent Polynomial Causal Models Through the Lens of Change},
  author={Liu, Yuhang and Zhang, Zhen and Gong, Dong and Gong, Mingming and Huang, Biwei and van den Hengel, Anton and Zhang, Kun and Shi, Javen Qinfeng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://arxiv.org/abs/2310.15580}
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
- **Microsoft Research** for the DOSA dataset

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

## Limitations & Considerations

### Methodological Limitations
- **Seed Dependence**: SAE features show variation across random initialization seeds; reproducibility requires fixed seeds and multiple runs
- **Reconstruction-Sparsity Tradeoff**: The balance between reconstruction quality and sparsity means some semantic information may be compressed
- **Feature Interpretability**: Reliance on LLM-based labeling introduces subjective assessment and potential biases
- **Single Model Family**: Validation limited to Qwen 1.5 models; generalization to other architectures remains to be tested

### Scope Limitations
- **Cultural Coverage**: Focus on Indian cultural markers; framework may need adaptation for other cultural contexts
- **Dataset Scale**: 40K samples may not capture full diversity of cultural representations
- **Layer Selection**: Analysis of layers 6, 12, 18 provides sampling but not complete coverage of model depth

### Causal Claims
- Phase 5 causal validation is ongoing; current findings represent correlational patterns in learned representations
- Identifiability guarantees depend on assumptions about distribution shifts and feature independence
- Causal sufficiency testing may not detect all confounding pathways

---

## Computational Requirements

### Training Infrastructure
- **Phase 1 (Activation Extraction)**: ~12 hours on 1× A100 GPU (40GB)
- **Phase 2 (SAE Training)**: ~6 hours on 3× A100 GPUs (total 9 SAE models)
- **Phase 2.5 (Feature Labeling)**: ~16 hours on 4× A100 GPUs with Qwen models
- **Storage**: ~500GB for activations, models, and analysis outputs

### Memory Requirements
- Activation caching: 16GB RAM minimum
- SAE training: 40GB GPU memory per SAE
- Feature labeling: 80GB GPU memory for Qwen inference

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
  - Preparing identifiability metrics following Liu et al. (2024)
  - DOSA integration planned

---

*This research explores methodologies for identifying and validating causal cultural mechanisms in RLHF-aligned language models using sparse autoencoders and multi-distribution identifiability analysis.*
