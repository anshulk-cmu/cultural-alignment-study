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

**Status**: In Progress

**Probe Types**:

1. **Attribute Probe** (16-class classification)
   - Task: Decode cultural attribute from activations
   - Preliminary result (Base Layer 8): 80.67% accuracy

2. **Correctness Probe** (binary classification)
   - Task: Predict whether model answers correctly
   - Tests if correctness information is encoded in representations

3. **State Probe** (36-class classification)
   - Task: Classify Indian state from activations

4. **Cross-Model Transfer Probe** (Critical test)
   - Train on base model activations → Test on instruct model activations
   - High transfer rate (>95%) → Decision-boundary suppression
   - Low transfer rate (<85%) → Representational suppression

5. **Multi-Task Joint Probe**
   - Tests information entanglement across attribute, correctness, and state

**Experimental Design**:
- Train/test split: 75%/25% (25,141/8,381 sentences)
- Stratified sampling by group type
- 5-fold cross-validation
- Logistic regression with balanced class weights
- StandardScaler normalization

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

3. **Mechanistic Hypothesis**: The extreme similarity suggests RLHF operates primarily through downstream decision boundaries rather than rewriting internal knowledge representations

## Disclaimer

This is a work in progress. Linear probing experiments are currently running. For detailed results, methodology questions, or collaboration inquiries, please contact Anshul Kumar at anshulk@andrew.cmu.edu.

Citations and full technical report will be added upon completion of the probing analysis.