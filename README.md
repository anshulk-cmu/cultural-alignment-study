# Cultural Alignment Study: RLHF Information Suppression via Mechanistic Interpretability

## Objective

This research investigates whether RLHF (Reinforcement Learning from Human Feedback) models suppress cultural information through representational changes or decision-boundary modifications. By comparing Qwen2-1.5B base and instruct models using mechanistic interpretability techniques, we aim to prove information suppression mechanisms in aligned models.

## Dataset

**Source**: 33,522 sentences covering Indian cultural knowledge across:
- **36 States**: Telangana, Uttarakhand, Tamil Nadu, Sikkim, etc.
- **16 Attributes**: Tourism, History, Festivals, Cuisine, Dance & Music, Art, Costume, Rituals, Language, Personalities, Religion, Sports, Transport, Medicine, Nightlife

**Data Structure**: Three experimental groups based on model behavior:
1. **Suppression** (11,934 sentences, 35.6%): Base model correct → Instruct model incorrect
2. **Enhancement** (11,970 sentences, 35.7%): Base model incorrect → Instruct model correct
3. **Control** (9,618 sentences, 28.7%): Both models behave similarly

**Behavioral Divergence**:
- Suppression: 79.9% (base) → 37.5% (instruct) = **-42.5% drop**
- Enhancement: 41.5% (base) → 76.8% (instruct) = **+35.3% gain**
- Control: 80.4% for both models

## Methodology Pipeline

### Phase 1: Activation Extraction
**Script**: `scripts/extract_activations.py`

Extracted mean-pooled hidden state activations from both models across 4 layers (8, 16, 24, 28) using dual-GPU parallel processing:
- **Output**: 1536-dimensional activation vectors per sentence
- **Storage**: `.npy` files per model-layer combination
- **Metadata**: Sentence-level annotations with group type, state, attribute, correctness labels

### Phase 2: Exploratory Data Analysis (EDA)
**Script**: `scripts/eda_12k.py` | **Status**: ✅ Complete

**Key Analyses**:
1. **Text Quality**: Verified sentence length (mean: 46 words), lexical diversity, zero near-duplicates
2. **Semantic Clustering**: HDBSCAN clustering (264 clusters, 19.7% noise)
3. **Activation Geometry**: Measured cosine similarity between base and instruct activations

**Critical Finding**: Despite massive behavioral divergence in the suppression group, internal representations remain **99.7-99.9% similar** across all layers:
- Layer 8: 0.9997 similarity
- Layer 16: 0.9993 similarity
- Layer 24: 0.9990 similarity
- Layer 28: 0.9970 similarity

This suggests RLHF does NOT rewrite internal representations but operates through downstream mechanisms.

### Phase 3: Linear Probing Analysis
**Script**: `scripts/linear_probing_v2.py` | **Status**: 🔄 In Progress

Training logistic regression probes to decode information from activations:

**Probe Types**:
1. **Attribute Probe** (16-class): Classifies cultural attributes from activations
   - Early results: ~80% accuracy on base model layer 8

2. **Correctness Probe** (binary): Predicts whether model will answer correctly
   - Tests if correctness information is encoded in representations

3. **State Probe** (36-class): Classifies Indian state from activations

4. **Cross-Model Transfer** (CRITICAL): Train probe on base activations, test on instruct
   - High transfer rate → Representations remain aligned
   - Low transfer rate → RLHF rewrote representations

5. **Multi-Task Joint Probing**: Tests information entanglement across tasks

**Experimental Design**:
- Train/test split: 75%/25% stratified by group type
- 5-fold cross-validation
- StandardScaler normalization
- Class-balanced logistic regression

## Current Results

The working hypothesis is being tested: **If cross-model transfer rates remain >95%, this proves suppression operates via decision boundaries rather than representational erasure**, providing mechanistic evidence for how RLHF alignment works.

## Repository Structure

```
├── scripts/
│   ├── extract_activations.py      # Phase 1: Activation extraction
│   ├── eda_12k.py                   # Phase 2: EDA analysis
│   └── linear_probing_v2.py         # Phase 3: Linear probing
├── outputs/
│   ├── eda_results/                 # EDA plots, reports, tables
│   └── linear_probing/              # Probing results (in progress)
└── README.md
```

## Key Insight

The 99.9% representational similarity despite 42.5% behavioral divergence is the smoking gun: RLHF instruction-tuning preserves knowledge representations but modifies downstream decision-making layers—a fundamental insight into alignment mechanisms.
