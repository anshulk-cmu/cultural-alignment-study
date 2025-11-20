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

## Repository Structure

```
cultural-alignment-study/
├── scripts/
│   ├── sanskriti_knowledge_test.py        # Initial 21K question evaluation
│   ├── analyze_combinations_12k.py        # Dataset filtering and selection
│   ├── generate_sentences_sanskriti.py    # Claude-based sentence generation
│   ├── extract_activations.py             # Hidden state extraction
│   ├── eda_12k.py                         # Exploratory data analysis
│   └── linear_probing_v2.py               # Linear probing experiments (ongoing)
├── outputs/
│   ├── sanskriti_test_knowledge/          # Initial evaluation results
│   ├── eda_results/                       # EDA plots and reports
│   │   ├── plots/                         # Visualization outputs
│   │   ├── reports/                       # JSON analysis reports
│   │   └── SUMMARY_REPORT.txt            # Executive summary
│   └── linear_probing/                    # Probing results (in progress)
└── README.md
```

## Key Findings

1. **Behavioral Divergence**: RLHF instruction-tuning creates massive behavioral differences (42.30% suppression, 35.25% enhancement)

2. **Representational Preservation**: Despite behavioral divergence, internal representations remain 99.7-99.9% similar across all layers

3. **Linear Probing Evidence**: Cross-model transfer rates of 96-100% prove representational isomorphism—probes trained on base model work almost perfectly on instruct model despite behavioral suppression

4. **Decision-Boundary Suppression**: Weak correctness encoding (62%) combined with strong semantic encoding (80-96%) and near-perfect transfer rates demonstrate RLHF operates via policy-layer blocking mechanisms, not representational erasure

## Disclaimer

This is a work in progress. Linear probing experiments are complete. MDL probing experiments are currently running. For detailed results, methodology questions, or collaboration inquiries, please contact Anshul Kumar at anshulk@andrew.cmu.edu.

Citations and full technical report will be added upon completion of all analyses.