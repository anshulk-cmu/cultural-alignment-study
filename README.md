# Mechanistic Interpretability of RLHF-Induced Information Suppression in Language Models

- **Author**: Anshul Kumar
- **Email**: anshulk@andrew.cmu.edu
- **Institution**: Carnegie Mellon University

## Abstract

This study investigates information suppression mechanisms in RLHF-aligned language models using mechanistic interpretability techniques. We compare Qwen2-1.5B base and instruct models on Indian cultural knowledge to determine whether instruction-tuning suppresses information through representational changes or decision-boundary modifications.

## Dataset Construction

### 1. Initial Knowledge Testing (`sanskriti_knowledge_test.py`)

- **Dataset**: Sanskriti Cultural Knowledge Benchmark
- **Total Questions**: 21,853 multiple-choice questions
- **Coverage**: 36 Indian states × 16 cultural attributes × 4 question types

**Models Evaluated**:
- Qwen2-1.5B (base model)
- Qwen2-1.5B-Instruct (RLHF-aligned model)

**Results**:
- Base model accuracy: 71.23% (15,566/21,853)
- Instruct model accuracy: 69.96% (15,289/21,853)
- Knowledge gap: +1.27% (base outperforms instruct)
- Suppression instances: 1,788 questions (8.18%)
- Enhancement instances: 1,511 questions (6.91%)

**Dimensions**:
- **Question Types**: Association, Country Prediction, General Awareness, State Prediction
- **Cultural Attributes**: Tourism, History, Festivals, Cuisine, Dance & Music, Art, Costume, Rituals & Ceremonies, Language, Personalities, Religion, Sports, Transport, Medicine, Nightlife
- **States**: All 36 Indian states and union territories

### 2. Targeted Dataset Selection (`analyze_combinations_12k.py`)

**Objective**: Select 12,000 questions with maximal behavioral divergence between base and instruct models

**Selection Strategy**:
- Analyzed all state × attribute × question_type combinations
- Selected questions maximizing suppression and enhancement effects
- Ensured balanced representation across groups

**Final Dataset** (11,206 questions):
- **Suppression group**: 4,000 questions (35.7%)
  - Base accuracy: 79.95%
  - Instruct accuracy: 37.65%
  - **Suppression effect: -42.30%**

- **Enhancement group**: 4,000 questions (35.7%)
  - Base accuracy: 41.60%
  - Instruct accuracy: 76.85%
  - **Enhancement effect: +35.25%**

- **Control group**: 3,206 questions (28.6%)
  - Base accuracy: 80.38%
  - Instruct accuracy: 80.41%
  - Gap: -0.03%

### 3. Sentence Generation (`generate_sentences_sanskriti.py`)

**Method**: Claude Sonnet 4.5 batch API with few-shot prompting

**Generation Protocol**:
- 3 sentences per question (40-60 words each)
- Definitional, usage/contextual, and contrast perspectives
- Self-contained, factually consistent statements
- Lexically diverse to avoid template artifacts

**Output**: 33,522 sentences (11,206 questions × 3 sentences)

## Experimental Pipeline

### 4. Activation Extraction (`extract_activations.py`)

**Architecture**: Dual-GPU parallel processing

**Extraction Details**:
- Layers: 8, 16, 24, 28 (out of 28 total layers)
- Pooling: Mean pooling over sequence dimension with attention masking
- Dimensions: 1,536-dimensional hidden states per sentence
- Batch size: 512 sentences
- Max sequence length: 256 tokens

**Output Files**:
- Base model: `base_layer{8,16,24,28}_activations.npy` (33,522 × 1,536 each)
- Instruct model: `instruct_layer{8,16,24,28}_activations.npy` (33,522 × 1,536 each)
- Metadata: `activation_index.csv` with sentence-level annotations

### 5. Exploratory Data Analysis (`eda_12k.py`)

**Analyses Performed**:

**Text Quality Verification**:
- Sentence length: Mean = 46.1 words, SD = 2.5 words
- Near-duplicate detection: 0% duplicates (TF-IDF similarity < 0.8)
- Lexical diversity verified across groups

**Semantic Structure**:
- HDBSCAN clustering: 264 clusters identified
- Noise points: 6,612 (19.7%)
- Baseline attribute classification (on embeddings): 77.5% ± 1.6%
- Group type classification: 39.9% ± 2.5%

**Critical Finding - Activation Geometry**:

Cosine similarity between base and instruct model activations (per-sentence, layer-wise):

| Layer | Overall Similarity | Suppression | Enhancement | Control |
|-------|-------------------|-------------|-------------|---------|
| 8     | 0.9997            | 0.9997±0.0001 | 0.9997±0.0001 | 0.9997±0.0001 |
| 16    | 0.9993            | 0.9993±0.0002 | 0.9992±0.0002 | 0.9993±0.0002 |
| 24    | 0.9990            | 0.9990±0.0002 | 0.9990±0.0002 | 0.9990±0.0002 |
| 28    | 0.9970            | 0.9970±0.0005 | 0.9970±0.0005 | 0.9969±0.0005 |

**Key Observation**: Despite 42.30% behavioral divergence in the suppression group, internal representations remain 99.7-99.9% identical across all layers and groups.

### 6. Linear Probing Analysis (`linear_probing_v2.py`)

**Status**: ✅ Complete

**Method**: Trained logistic regression probes on hidden state activations across 4 layers (8, 16, 24, 28) with 5-fold cross-validation, 75/25 train/test split (25,141/8,381 sentences), stratified by group type.

**Results**:

**Semantic Task Probes (High Performance)**:
- Attribute (16-class): 80.7-84.1% accuracy
- State (36-class): 90.4-96.6% accuracy
- Both models encode semantic information identically with peak performance at deeper layers (24, 28)

**Correctness Probe (Weak Signal)**:
- Binary accuracy: 61.4-62.9% (barely above chance at 50%)
- ROC-AUC: 0.663-0.679
- **Critical insight**: Correctness decisions not strongly encoded in representations

**Cross-Model Transfer (Definitive Evidence)**:
- State transfer rate: 98.8-100.1%
- Attribute transfer rate: 96.2-99.0%
- Correctness transfer rate: 92.0-103.6%
- **>95% transfer rates prove representational isomorphism**: Probes trained on base activations work almost perfectly on instruct activations despite 42% behavioral divergence

**Multi-Task Probing**:
- Joint vs. independent probing shows negligible differences (Δ < 0.002)
- Information independently encoded, not entangled

**Group-Wise Analysis**:
- Suppression group: Base correctness 57.1%, Instruct correctness 59.7%
- Control group: Both models 70.8-71.9%
- Semantic attributes remain 80%+ accurate even in suppression groups

**Mechanistic Interpretation**: The 96-100% cross-model transfer rates with weak correctness encoding (62%) prove RLHF operates via **policy-layer blocking mechanisms**, not representational erasure. Knowledge exists internally but is gated at output layers—textbook decision-boundary suppression.

### 7. MDL Probing Analysis (`mdl_probing_v2.py`)

**Status**: 🔄 In Progress

**Method**: Information-theoretic analysis using Minimum Description Length (MDL) principle to measure compression efficiency and model complexity across representations.

**Experiments**:

**Online Prequential Coding**:
- Sequential prediction with adaptive model updates
- Measures data efficiency across learning trajectory
- Tracks cumulative bits per sample as model observes more data
- Tests: Single-task probes for attribute, state, and correctness

**Variational MDL with Multiple Priors**:
- **L0 Prior**: Concrete dropout for automatic feature selection and sparsity measurement
- **L1 Prior**: Lasso-style regularization for soft sparsity
- **L2 Prior**: Ridge regularization for baseline complexity
- Decomposes total MDL = Data Cost (NLL) + Model Cost (Regularization)
- Measures Fisher Information Matrix for decision boundary sharpness

**Triple-Task Entanglement Test** (Critical Innovation):
- Simultaneous probing of State (36-class) + Attribute (16-class) + Correctness (2-class)
- **Balanced loss weights**: Attribute=1.0, State=2.25, Correctness=0.125 (proportional to task complexity)
- Tests whether models maintain unified representations despite behavioral suppression
- Compression ratio = Triple-task MDL / Sum(Single-task MDLs)
- **Hypothesis**: Low joint compression with high semantic accuracy but low correctness accuracy → policy-layer masking

**Cross-Model Isomorphism Test**:
- Train probe on Base activations, evaluate on Instruct activations
- Measure MDL drift for suppression group
- Complementary to linear probing transfer rates

**Group-Stratified Analysis**:
- Separate MDL measurements for suppression/enhancement/control groups
- Tests if suppression affects compression efficiency differently

**Expected Insights**:
- Sparsity patterns reveal which features are critical vs. redundant
- Fisher information quantifies decision boundary sharpness
- Triple-task compression tests unified vs. fragmented representations
- MDL drift confirms or refutes representational isomorphism from linear probing

### 8. Causal Intervention Analysis (`causal_intervention_v1.py`)

**Status**: 🔄 Planned

**Method**: Layer-wise activation patching to causally identify where suppression occurs in the forward pass.

**Intervention Protocol**:
- Replace Instruct model activations with Base model activations at specific layers
- Measure answer entity probability in top-k predictions (k=10)
- Recovery rate = (Intervention accuracy - Instruct baseline) / (Base baseline - Instruct baseline)
- Dual-GPU parallel processing for efficiency

**Multi-Level Analysis**:

**Group-Level Recovery**:
- Suppression group: Test if Base activations restore suppressed answers
- Enhancement group: Test if Base activations reduce enhanced performance
- Control group: Verify minimal intervention effects

**Attribute-Level Localization**:
- 16 cultural attributes analyzed separately
- Identify which attributes are most sensitive to intervention
- Heatmap visualization across layers

**State-Level Granularity**:
- 36 states analyzed for geographic patterns
- Rank states by suppression strength and recovery rates

**Combined Analysis**:
- Fine-grained State × Attribute × Group interactions
- Identify top-20 combinations with strongest suppression
- Test if recovery varies by specific cultural domain

**Expected Results**:
- **Early layers (8, 16)**: Low recovery → semantic encoding still shared
- **Middle layers (24)**: Moderate recovery → divergence begins
- **Late layers (28)**: High recovery → causal locus of suppression
- Confirms temporal dynamics of when/where suppression emerges

**Complementarity with Probing**:
- Linear probing shows *what* information exists (high transfer rates)
- MDL probing shows *how efficiently* it's encoded (compression ratios)
- Causal intervention shows *where* it's blocked (recovery by layer)

## Repository Structure

```
cultural-alignment-study/
├── scripts/
│   ├── sanskriti_knowledge_test.py        # Initial 21K question evaluation
│   ├── analyze_combinations_12k.py        # Dataset filtering and selection
│   ├── generate_sentences_sanskriti.py    # Claude-based sentence generation
│   ├── extract_activations.py             # Hidden state extraction
│   ├── eda_12k.py                         # Exploratory data analysis
│   ├── linear_probing_v2.py               # Linear probing experiments
│   ├── mdl_probing_v2.py                  # MDL information-theoretic analysis
│   └── causal_intervention_v1.py          # Layer-wise activation patching
├── outputs/
│   ├── sanskriti_test_knowledge/          # Initial evaluation results
│   ├── eda_results/                       # EDA plots and reports
│   │   ├── plots/                         # Visualization outputs
│   │   ├── reports/                       # JSON analysis reports
│   │   └── SUMMARY_REPORT.txt            # Executive summary
│   ├── linear_probing/                    # Linear probing results
│   │   ├── plots/                         # Accuracy curves, transfer rates
│   │   ├── results/                       # JSON metrics files
│   │   └── SUMMARY_REPORT.txt            # Probing analysis summary
│   ├── mdl_probing/                       # MDL analysis outputs
│   │   ├── data/                          # Online coding, variational MDL data
│   │   ├── plots/                         # Compression curves, Fisher info
│   │   └── logs/                          # Execution logs
│   └── causal_intervention/               # Intervention experiment results
│       ├── data/                          # Recovery rates by group/attr/state
│       ├── plots/                         # Layer-wise recovery visualizations
│       └── logs/                          # Intervention logs
└── README.md
```

## Key Findings

### Tripartite Evidence for Policy-Layer Suppression

This study employs three complementary mechanistic interpretability techniques to triangulate how RLHF suppresses cultural knowledge:

**1. Linear Probing (What Information Persists)**
- **Semantic preservation**: 80-96% accuracy on attribute/state classification despite 42% behavioral suppression
- **Cross-model transfer**: 96-100% transfer rates prove representational isomorphism between Base and Instruct models
- **Weak correctness encoding**: 62% accuracy (barely above chance) shows decision information is not strongly represented
- **Interpretation**: Knowledge exists internally but behavioral decisions are weakly encoded in hidden states

**2. MDL Probing (How Efficiently Information Is Encoded)** 🔄
- **Triple-task entanglement**: Tests State + Attribute + Correctness simultaneously with balanced loss weights (2.25:1.0:0.125)
- **Expected pattern**: Low compression ratio with high semantic accuracy but low correctness accuracy
- **Sparsity analysis**: L0 regularization reveals which features are critical vs. redundant
- **Fisher information**: Quantifies decision boundary sharpness differences between models
- **Interpretation**: Compression efficiency validates unified semantic representations despite behavioral gating

**3. Causal Intervention (Where Suppression Occurs)** 🔄
- **Activation patching**: Replace Instruct activations with Base activations layer-by-layer
- **Expected recovery pattern**: Low at layers 8-16 (shared semantics), high at layer 28 (decision divergence)
- **Multi-level localization**: Group/attribute/state-specific recovery rates identify granular suppression patterns
- **Interpretation**: Causal evidence for when/where representations diverge in forward pass

### Convergent Mechanistic Conclusion

**RLHF operates via late-stage policy-layer masking, not representational erasure:**
- Behavioral divergence: 42.30% suppression, 35.25% enhancement
- Representational similarity: 99.7-99.9% across all layers (cosine similarity)
- Information preservation: 96-100% cross-model transfer rates
- Decision-layer blocking: Weak correctness encoding (62%) despite strong semantic encoding (80-96%)

This pattern is inconsistent with information erasure (which would show low transfer rates and representational drift) but consistent with output gating mechanisms that preserve internal knowledge while blocking downstream decisions.

## Current Status

- ✅ **Complete**: Dataset construction, activation extraction, EDA, linear probing
- 🔄 **In Progress**: MDL probing (running with balanced triple-task loss weights)
- 🔄 **Planned**: Causal intervention experiments

## Disclaimer

This is ongoing research. Linear probing experiments are complete and confirm representational isomorphism (98.6% average transfer rate). MDL probing experiments are currently running with task-proportional loss weighting for multi-task analysis. Causal intervention experiments are planned to complete the tripartite evidence architecture.

For detailed results, methodology questions, or collaboration inquiries, please contact Anshul Kumar at anshulk@andrew.cmu.edu.

Full technical report and citations will be added upon completion of all analyses.