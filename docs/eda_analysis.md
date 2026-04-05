# EDA: Complete Analysis

- **Mechanistic Interpretability of Cultural Knowledge in Instruction-Tuned LLMs**
- **Anshul Kumar and Pragati Bhattad — Carnegie Mellon University, March 2026**
- **Target venue: EMNLP 2026 Workshops**

This document records every decision, every number, and every result from the
Exploratory Data Analysis of the Sanskriti benchmark. Every number has been
cross-checked against the actual CSV output files in
`/data/user_data/anshulk/cultural-mi/analysis/`.

---

## Quick Reference: Key Numbers

All numbers below are validated against CSV outputs. Use this as a cheat sheet.

| Metric | Value | Source CSV |
|--------|-------|-----------|
| Total rows | 21,853 | Raw dataset |
| Usable rows | 21,726 | `sanskriti_usable.csv` |
| Excluded rows | 127 (0.58%) | answer ≠ any option |
| States | 36 | `distribution_states.csv` |
| Attributes | 16 | `distribution_attributes.csv` |
| Question types | 4 | `distribution_qtypes.csv` |
| Unique question texts | 20,092 (92.5%) | `data_quality_summary.csv` |
| Exact duplicate rows | 2,053 in 419 groups | `exact_duplicates.csv` |
| Conflicting duplicates | 351 groups | `conflicting_duplicates.csv` |
| Near-duplicate pairs (sim>0.85) | 77,833 | `near_duplicates.csv` |
| Questions in near-dup pairs | 17,078 (78.6%) | `near_duplicates.csv` |
| Answer-in-question leakage | 1,615 (7.43%) | `answer_in_question_leakage.csv` |
| Country Prediction answers = India | 100.0% (5,563) | `country_prediction_audit.csv` |
| Combined entity keys | 8,156 unique | `cultural_entities_combined.csv` |
| Regex-extracted entities | 4,949 unique | `cultural_entities.csv` |

---

## Glossary: Every Abbreviation, Formula, and Metric Explained

This section defines every technical term, abbreviation, and formula used in this
report. If you encounter something unclear elsewhere in the document, check here
first.

### Abbreviations

| Abbreviation | Full Form | What It Is |
|-------------|-----------|-----------|
| EDA | Exploratory Data Analysis | The process of examining a dataset before modeling — looking at distributions, quality issues, and patterns |
| MI | Mechanistic Interpretability | A subfield of ML that studies what happens inside neural networks at the level of individual neurons, layers, and circuits |
| LLM | Large Language Model | A neural network trained on text that can generate and understand language (e.g., LLaMA, GPT) |
| MCQ | Multiple Choice Question | A question with several answer options, one of which is correct |
| CP | Country Prediction | One of the 4 question types in Sanskriti — asks "which country is this cultural element from?" |
| SP | State Prediction | One of the 4 question types — asks "which state is this cultural element from?" |
| GT | Ground Truth | The correct answer, as labeled in the dataset |
| SFT | Supervised Fine-Tuning | Training a base model on curated instruction-response pairs |
| RLHF | Reinforcement Learning from Human Feedback | Training a model using human preference ratings to align outputs with what humans consider helpful and safe |
| UMAP | Uniform Manifold Approximation and Projection | A dimensionality reduction algorithm that projects high-dimensional data to 2D for visualization |
| TF-IDF | Term Frequency–Inverse Document Frequency | A statistic that measures how important a word is to a document within a collection |
| MTLD | Measure of Textual Lexical Diversity | A metric for vocabulary richness that is robust to text length differences |
| NGD | N-gram Diversity | Ratio of unique n-grams to total n-grams — measures how repetitive text is |
| TTR | Type-Token Ratio | Number of unique words divided by total words — a simple vocabulary diversity measure |
| BF16 | Brain Float 16 | A 16-bit floating point format used for efficient neural network computation |
| GQA | Grouped Query Attention | An attention mechanism where multiple query heads share key-value heads, reducing memory |
| RoPE | Rotary Position Embedding | A method for encoding token positions in transformer models |
| pp | Percentage Points | The arithmetic difference between two percentages (e.g., 29% - 25% = 4pp) |
| OOM | Out Of Memory | When a GPU runs out of VRAM during computation |
| SLURM | Simple Linux Utility for Resource Management | A job scheduler for compute clusters |

### Statistical Formulas

**Chi-squared test (χ²)** — Tests whether an observed distribution differs from
an expected distribution. We use it to check if the ground truth answer positions
(A/B/C/D) are uniformly distributed.

```
Formula: χ² = Σ (O_i - E_i)² / E_i

Where:
  O_i = observed count for category i
  E_i = expected count for category i (under the null hypothesis)
  Σ   = sum over all categories

For our position bias test:
  Categories: A, B, C, D (4 categories)
  O = [5885, 6308, 5008, 4525]  (actual counts from the dataset)
  E = [5431.5, 5431.5, 5431.5, 5431.5]  (21726/4 = 5431.5 per category if uniform)

  χ² = (5885-5431.5)²/5431.5 + (6308-5431.5)²/5431.5 + (5008-5431.5)²/5431.5 + (4525-5431.5)²/5431.5
     = 37.9 + 141.5 + 33.0 + 151.2
     = 363.6

Interpretation: Higher χ² = more deviation from uniform. The p-value tells us
the probability of seeing this much deviation by chance. p ≈ 1.7e-78 means
it is essentially impossible that the observed distribution is uniform — the
position bias is real, not random noise.
```

**Cohen's d** — Measures the practical significance of a difference between two
groups. We use it to check if correct answers are systematically longer/shorter
than incorrect answers.

```
Formula: d = (mean₁ - mean₂) / pooled_std

Where:
  mean₁ = mean of group 1 (correct option word count = 1.471)
  mean₂ = mean of group 2 (incorrect option word count = 1.595)
  pooled_std = √((std₁² + std₂²) / 2)
             = √((1.255² + 1.405²) / 2)
             = √((1.575 + 1.974) / 2)
             = √(1.775)
             = 1.332

  d = (1.471 - 1.595) / 1.332 = -0.094

Interpretation scale:
  |d| < 0.2  = negligible (our case: 0.094)
  |d| 0.2-0.5 = small
  |d| 0.5-0.8 = medium
  |d| > 0.8  = large

Our d = -0.094 means the difference in option length between correct and
incorrect answers is negligible — not a usable shortcut.
```

**t-test** — Tests whether two groups have different means. Even with a negligible
Cohen's d, the t-test p-value can be very small (p ≈ 4e-31 here) because
statistical significance depends on sample size. With 86,904 option instances
(21,726 questions × 4 options), even a tiny difference becomes "statistically
significant" without being practically meaningful. Cohen's d is the better
measure here.

**Cosine similarity** — Measures how similar two vectors are, regardless of their
magnitude. Used throughout the semantic analysis.

```
Formula: cos(A, B) = (A · B) / (|A| × |B|)

Where:
  A · B = dot product of vectors A and B = Σ(A_i × B_i)
  |A|   = magnitude (length) of vector A = √(Σ A_i²)

Since our embeddings are already normalized (|A| = |B| = 1), this simplifies to:
  cos(A, B) = A · B = Σ(A_i × B_i)

Range: -1 to +1
  +1 = identical direction (most similar)
   0 = orthogonal (unrelated)
  -1 = opposite direction (most dissimilar)

In practice, sentence embeddings rarely go below 0, so the effective range is 0 to 1.
```

**MTLD (Measure of Textual Lexical Diversity)** — Measures vocabulary richness.
Unlike simple type-token ratio (TTR = unique words / total words), MTLD is not
biased by text length.

```
How it works:
1. Read through the text word by word
2. Track the running TTR (unique words so far / total words so far)
3. When the running TTR drops below a threshold (we use 0.72), that's one "factor"
4. Reset and start counting again
5. MTLD = total words / number of factors
6. Do this in both forward and reverse directions, average the results

Higher MTLD = more diverse vocabulary before it starts repeating.

Example:
  "the red fox the blue fox the green fox" → TTR drops quickly → few words per factor → low MTLD
  "unique varied colorful diverse separate distinct" → TTR stays high → many words per factor → high MTLD

Our threshold: 0.72 (standard in the literature)
Range in our data: 17.75 (Country Prediction) to 36.36 (Rituals_and_Ceremonies)
```

**N-gram Diversity (NGD)** — Ratio of unique n-grams to total n-grams.

```
Formula: NGD_n = count(unique n-grams) / count(total n-grams)

Where n-gram = a sequence of n consecutive words.
  n=1: unigrams (single words)
  n=2: bigrams (word pairs)
  n=3: trigrams (word triples)
  n=4: four-grams

We average across n=1 to n=4:
  NGD_avg = (NGD_1 + NGD_2 + NGD_3 + NGD_4) / 4

Range: 0 to 1
  0 = every n-gram is repeated (completely formulaic)
  1 = every n-gram is unique (no repetition at all)

Example for Personalities (lowest NGD = 0.220):
  NGD_1 = 0.0584, NGD_2 = 0.1792, NGD_3 = 0.3003, NGD_4 = 0.3404
  NGD_avg = (0.0584 + 0.1792 + 0.3003 + 0.3404) / 4 = 0.220
```

**TF-IDF** — Identifies words that are distinctive to a particular document
(in our case, a particular state's questions) within a collection.

```
TF (Term Frequency) = count of word in document / total words in document
IDF (Inverse Document Frequency) = log(total documents / documents containing word)
TF-IDF = TF × IDF

High TF-IDF means: the word appears often in THIS state's questions but rarely
in OTHER states' questions. Example: "hyderabad" has high TF-IDF for Telangana
because it appears frequently in Telangana questions but rarely elsewhere.
Low TF-IDF means: the word is either rare in this state OR common across all
states (like "which" or "famous").
```

### Embedding-Specific Terms

**Sentence embedding** — A fixed-size numerical vector (in our case, 384 numbers)
that represents the meaning of a text. We use the `all-MiniLM-L6-v2` model from
the `sentence-transformers` library to compute these. Similar texts produce
similar vectors.

**Normalized embedding** — An embedding scaled so its magnitude (length) equals 1.
This makes cosine similarity equivalent to a simple dot product, which is faster
to compute.

**UMAP parameters:**
- `n_neighbors=30`: Each point considers its 30 nearest neighbors when computing
  the low-dimensional layout. Higher = more global structure, lower = more local.
- `min_dist=0.3`: Minimum distance between points in the 2D plot. Higher = more
  spread out, lower = tighter clusters.
- `metric="cosine"`: Use cosine similarity (not Euclidean distance) to measure
  distances in the original 384-dimensional space.
- `random_state=42`: Fixed seed for reproducibility.

### BERTopic

BERTopic is a topic modeling algorithm that groups documents (questions) into
topics using:
1. Pre-computed sentence embeddings (we provide MiniLM embeddings)
2. UMAP for dimensionality reduction
3. HDBSCAN for clustering
4. c-TF-IDF for topic representation (identifying key words per topic)

**Topic -1** is the "outlier" cluster — questions that don't fit well into any
topic. A high outlier count (7,889 = 36.3% in our case) means many questions
are too generic to cluster.

**nr_topics=20** — We asked BERTopic to produce approximately 20 topics.
**min_topic_size=100** — Each topic must contain at least 100 questions.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Quality Assessment](#3-data-quality-assessment)
4. [Distribution Analysis](#4-distribution-analysis)
5. [Answer Position Bias](#5-answer-position-bias)
6. [Country Prediction Audit](#6-country-prediction-audit)
7. [Text and Lexical Analysis](#7-text-and-lexical-analysis)
8. [Semantic Analysis](#8-semantic-analysis)
9. [Distractor Quality](#9-distractor-quality)
10. [Cultural Specificity](#10-cultural-specificity)
11. [Critical Findings for Step 1](#11-critical-findings-for-step-1)
12. [What This Stage Does NOT Do](#12-what-this-stage-does-not-do)
13. [Output Files](#13-output-files)
14. [Runtime and Reproducibility](#14-runtime-and-reproducibility)
15. [Alternative Datasets Considered](#15-alternative-datasets-considered)
16. [Dataset Fitness Assessment](#16-dataset-fitness-assessment)

---

## 1. Purpose of This Stage

We need to understand the Sanskriti dataset before running two 8B-parameter LLMs
on it. Without this understanding, we cannot correctly interpret the behavioral
labels (suppression/enhancement) that Step 1 produces.

**What we need to know:**

1. Are there dataset artifacts that could produce false labels?
2. Which slices of the data have enough samples for reliable analysis?
3. Are there shortcuts that could inflate accuracy without cultural knowledge?
4. What confounds need to be reported as limitations?

### Why This Matters

This project has a 4-step pipeline. Step 1 produces behavioral labels. Steps 2-4
build on those labels — extracting activations, running probes, identifying
circuits. A false suppression label caused by a dataset artifact will propagate
through every downstream step. The EDA exists to prevent this.

### Relationship to the Sanskriti Paper

The Sanskriti paper (Bari et al., ACL 2025 Findings, arXiv:2506.15355) evaluated
10 LLMs including LLaMA-3.1-70B-Instruct (0.86 accuracy) and LLaMA-3.2-3B-Instruct
(0.52 accuracy). Our 8B models should fall between these bounds. The paper reported
model-level accuracies but did not publish per-question predictions or detailed
dataset quality analysis. Our EDA fills this gap.

---

## 2. Dataset Overview

**Source:** `13ari/Sanskriti` on HuggingFace
**Local cache:** `/data/user_data/anshulk/cultural-mi/dataset/`

### Raw Statistics

| Property | Value |
|----------|-------|
| Total rows | 21,853 |
| Usable rows (answer matches an option) | **21,726** |
| Excluded rows (answer matches no option) | 127 (0.58%) |
| Missing values | Zero across all columns |
| Columns | 10 |
| Split | Single `train` split |

**Validation:** `distribution_states.csv`, `distribution_attributes.csv`, and
`distribution_qtypes.csv` all sum to 21,726.

Column names: `state`, `attribute`, `question`, `option1`, `option2`, `option3`,
`option4`, `answer`, `short explaination / source link`, `question_type`.

Note: the column `short explaination / source link` has a typo ("explaination"
instead of "explanation"). Code must match this string exactly.

### The 127 Excluded Rows

For each row, we determine which option (A/B/C/D) is correct:

```python
# From eda_pipeline.py, lines 64-69:
def get_ground_truth(row):
    ans = str(row["answer"]).strip().lower()     # normalize the answer text
    for opt, letter in zip(OPT_KEYS, LETTERS):   # OPT_KEYS = [option1..4], LETTERS = [A..D]
        if str(row[opt]).strip().lower() == ans:  # compare normalized strings
            return letter                         # first match wins
    return None                                   # no match = excluded
```

We compare `answer.strip().lower()` against each of the four option values (also
stripped and lowered). `.strip()` removes leading/trailing whitespace.
`.lower()` converts to lowercase so "India" matches "india". If no option matches,
the row is excluded because we cannot determine the ground truth letter.

| Category | Count | Examples |
|----------|-------|---------|
| Fixable typos (>80% string similarity) | 10 | "Uttarakhand" vs "Uttrakhand", "Dhudni Lake" vs "Dudhni Lake" |
| Substring matches | 6 | "Tarpa festival and Vansda festival" vs option "Tarpa Festival" |
| Truly broken (answer unrelated to options) | 111 | Answer = "Primary ingredient in Dhuska" with options = "rice and lentils", "wheat flour", etc. |

**59 of the 111 broken rows come from Karnataka** (rows ~13580-13886). This looks
like a bad data batch where the answer column contains question fragments instead
of actual answers.

**Decision:** Exclude all 127. Even the 10 fixable typos carry ambiguity — we
cannot verify whether the answer column or the option column has the typo. At
0.58% of the dataset, the impact is negligible.

### Why Not Fuzzy-Match the 10 Fixable Typos?

Three reasons: (a) a fuzzy-matching threshold is a subjective choice that others
cannot reproduce without the exact same logic, (b) 10 rows is 0.046% of the
dataset, (c) the Sanskriti paper itself does not document these mismatches.

---

## 3. Data Quality Assessment

*Source files: `data_quality_summary.csv`, `exact_duplicates.csv`,
`conflicting_duplicates.csv`, `near_duplicates.csv`, `answer_in_question_leakage.csv`*

### Summary Table

| Issue | Count | % of Usable | How We Know |
|-------|-------|-------------|-------------|
| Unique question texts | 20,092 | 92.5% | `data_quality_summary.csv` |
| Exact duplicate rows | 2,053 | 9.45% | `exact_duplicates.csv` (419 groups) |
| **Conflicting duplicates** | **351 groups** | — | `conflicting_duplicates.csv` |
| Near-duplicate pairs (cosine > 0.85) | 77,833 pairs | — | `near_duplicates.csv` |
| Questions involved in near-duplicates | 17,078 | 78.6% | unique idx_a ∪ idx_b |
| Answer text in question text | 1,615 | 7.43% | `answer_in_question_leakage.csv` |
| Answer text in source URL column | 5,082 | 23.4% | `data_quality_summary.csv` |

**Validation check:** 21,726 − 20,092 = 1,634 duplicate question texts (rows
that share a question with at least one other row). But 2,053 rows are involved
in duplication — the difference is because "involved in duplication" counts ALL
copies (including the first occurrence of each group), while "unique question
texts" counts each distinct text once.

**How the code finds duplicates:** Section 8 of the pipeline (lines 1018-1036)
uses `df.duplicated(subset=["question"], keep=False)`. The `keep=False` flag
means ALL copies are marked as duplicates (including the first one). So if a
question appears 3 times, all 3 rows are counted. The `groupby("question")` then
groups these into 419 groups, where each group is a distinct question text that
appears more than once.

### Exact Duplicates: 2,053 Rows in 419 Groups

**What this means:** 419 distinct question texts each appear 2 or more times in
the dataset. In total, 2,053 rows are copies. That is 9.45% of usable rows.

**Why it matters:** A model that knows one instance gets credit multiple times.
This inflates accuracy numbers but does not create false suppression/enhancement
labels (if the model gets the first copy right and the second copy right, both
land in the same behavioral category).

### Conflicting Duplicates: 351 Groups

This is the most serious data quality issue. **351 question groups** have the
exact same question text but different correct answer letters across instances.

| Question (truncated) | Instances | Answer letters |
|----------------------|-----------|----------------|
| "According to you, which ... closely associated to Agartala of Tripura?" | 6 | C, D, B |
| "According to you, which ... closely associated to Agra of Uttar_Pradesh?" | 4 | A, B |
| "According to you, which ... closely associated to Ahmedabad of Gujarat?" | 3 | A, C |

**Why this happens:** The question text is identical, but the four options differ
between instances. Here is a concrete example:

```
Row A: "Which is closely associated to Agartala?"
  Options: [Tea gardens, Ujjayanta Palace, Neermahal, Tripuri dance]
  Answer: Ujjayanta Palace → ground truth letter: B

Row B: "Which is closely associated to Agartala?"
  Options: [Neermahal, Tripuri dance, Tea gardens, Ujjayanta Palace]
  Answer: Ujjayanta Palace → ground truth letter: D
```

Same question, same correct answer text, but different letter because the options
are shuffled. The ground truth letter is correctly matched to each row's own
option positions.

**Is this a problem?** Not for our evaluation. Each row is self-consistent — the
ground truth letter matches the option text in that specific row. But it means
the dataset tests the same surface question multiple times with different
distractors, which contributes to the near-duplicate count.

### Near-Duplicates: 77,833 Pairs

We computed sentence embeddings for all 21,726 questions using `all-MiniLM-L6-v2`
(384 dimensions, normalized), then found all pairs with cosine similarity > 0.85.
(See Glossary for what cosine similarity is and how it is computed.)

**How the code works:** The pipeline (`section_5`, lines 720-738 of
`eda_pipeline.py`) iterates through the embedding matrix in chunks of 2,000 rows.
For each chunk, it computes cosine similarity of every question in the chunk
against ALL 21,726 questions. For each question i, it finds all j > i (to avoid
counting the same pair twice) with similarity above 0.85, and records the pair
with both question texts and whether they share the same state/attribute.

| Threshold | Pairs |
|-----------|-------|
| > 0.85 | 77,833 |
| > 0.90 | 43,089 |
| > 0.95 | 19,122 |
| > 0.99 | 16,700 |

**98.8% of near-duplicate pairs share the same state.** (Validated: 76,926 of
77,833 pairs have `same_state=True`.) This is expected: the templated question
structure means "Which state is famous for X in Karnataka?" and "Which state is
famous for Y in Karnataka?" are semantically near-identical when X and Y are
both Karnataka cultural elements.

**17,078 unique questions (78.6%) are involved in at least one near-duplicate
pair.** This is high. It means the effective information content of the 21,726
questions is substantially less than the raw count suggests. Most questions have
a semantic near-twin elsewhere in the dataset.

**What this means for Step 1:** Behavioral labels will cluster. If a model gets
one question about Bihu wrong, it will likely get the semantically similar Bihu
questions wrong too. When computing suppression rates per state or per attribute,
the effective sample size is smaller than the raw question count.

### Answer-in-Question Leakage: 1,615 Questions (7.43%)

In 1,615 questions, the correct answer text appears verbatim within the question
text itself. The code checks: `if len(answer) > 3 and answer.lower() in
question.lower()`.

**Validated breakdown:**

| Question Type | Leakage Count | % of Type |
|---------------|---------------|-----------|
| Association | 1,194 | 21.9% of 5,453 |
| State Prediction | 292 | 5.4% of 5,382 |
| Country Prediction | 69 | 1.2% of 5,563 |
| General Awareness | 60 | 1.1% of 5,328 |
| **Total** | **1,615** | **7.43% of 21,726** |

**Check:** 1,194 + 292 + 69 + 60 = 1,615. ✓

Association questions dominate because their format is often "Where is the
{entity} famous within {state}?" where the entity name IS the correct answer.
Example: "Where is the Nicobari pig-farming customs famous within
Andaman_and_Nicobar?" Answer: "Nicobari pig-farming customs."

**What this means for Step 1:** These questions can be solved by string matching
— the answer is literally in the question. Both models should get them right,
pushing them into `control_both_correct`. They dilute the suppression/enhancement
signal but do not create false labels.

### Source Column Leakage: 5,082 Questions (23.4%)

The `short explaination / source link` column contains the answer text in 5,082
rows. This is NOT a problem because this column is never shown to the model — it
is metadata for the dataset creators, not part of the question prompt.

### The Compound Effect

These quality issues overlap. A single question can be:
- An exact duplicate (9.45%)
- Involved in near-duplicate pairs (78.6%)
- Leaking the answer in the question text (7.43%)
- From a conflicting duplicate group

The "clean" core of the dataset — questions that are unique, non-leaking, and
non-conflicting — is substantially smaller than 21,726. We estimate the effective
information content is closer to 8,156 unique cultural entity keys (see Section
10), each tested from 1-3 angles via templates.

**For an EMNLP workshop paper, this is manageable.** Sanskriti is the only
large-scale Indian cultural MCQ dataset available. We use it with full awareness
of these issues and report all metrics in ways that account for them.

---

## 4. Distribution Analysis

*Source files: `distribution_states.csv`, `distribution_attributes.csv`,
`distribution_qtypes.csv`, `coverage_state_attribute.csv`, `state_summary.csv`*

### States: 36 Unique, Highly Imbalanced

| Rank | State | Count | % |
|------|-------|-------|---|
| 1 | Telangana | 1,705 | 7.85% |
| 2 | Karnataka | 1,391 | 6.40% |
| 3 | Andhra Pradesh | 1,127 | 5.19% |
| 4 | Delhi | 1,076 | 4.95% |
| 5 | Arunachal Pradesh | 1,023 | 4.71% |
| ... | ... | ... | ... |
| 32 | Maharashtra | 283 | 1.30% |
| 33 | Ladakh | 278 | 1.28% |
| 34 | Meghalaya | 267 | 1.23% |
| 35 | Mizoram | 210 | 0.97% |
| 36 | Lakshadweep | 122 | 0.56% |

**Range:** 14.0x (Telangana 1,705 vs Lakshadweep 122).
**Check:** 1705 / 122 = 13.97 ≈ 14.0x. ✓

The top 5 states account for 29.1% of all questions. The bottom 5 account for
5.3%. Southern and northeastern states are overrepresented. Notably, Maharashtra
(India's second most populous state) has only 283 questions (1.3%), less than
Sikkim.

### Regional Grouping

| Region | States | Questions | % of Dataset |
|--------|--------|-----------|-------------|
| North (incl. Himalayan) | 10 | 6,005 | 27.6% |
| South | 6 | 5,809 | 26.7% |
| Northeast | 8 | 4,475 | 20.6% |
| West + Central | 6 | 2,805 | 12.9% |
| East | 4 | 2,152 | 9.9% |

The Northeast is heavily overrepresented: 20.6% of questions for 8 states that
contain ~4% of India's population. This reflects the Sanskriti paper's goal of
covering underrepresented regions. For our MI study, this is actually helpful —
northeastern states have the most obscure cultural knowledge (less likely in
pretraining data), which is where suppression/enhancement effects should be
most visible.

The East (West Bengal, Bihar, Odisha, Jharkhand) is underrepresented at 9.9%.
Bihar, India's third most populous state, has only 367 questions (1.69%).

**What this means for Step 1:** Per-state suppression rates will have much more
statistical power for South/Northeast states than for East/West. When comparing
states, we must use percentages, not raw counts.

### Attributes: 16 Unique, Extremely Imbalanced

| Attribute | Count | % | Sparse? |
|-----------|-------|---|---------|
| Tourism | 3,801 | 17.50% | No |
| History | 2,609 | 12.01% | No |
| Festivals | 2,241 | 10.31% | No |
| Cultural_Common_Sense | 2,091 | 9.62% | No |
| Art | 2,066 | 9.51% | No |
| Dance_and_Music | 2,018 | 9.29% | No |
| Cuisine | 1,671 | 7.69% | No |
| Costume | 1,513 | 6.96% | No |
| Rituals_and_Ceremonies | 1,000 | 4.60% | No |
| Personalities | 983 | 4.52% | No |
| Language | 900 | 4.14% | No |
| Religion | 482 | 2.22% | No |
| Sports | 162 | 0.75% | **Yes** |
| Transport | 76 | 0.35% | **Yes** |
| Medicine | 72 | 0.33% | **Yes** |
| Nightlife | 41 | 0.19% | **Yes** |

**Range:** 92.7x (Tourism 3,801 vs Nightlife 41).
**Check:** 3801 / 41 = 92.7. ✓

Four attributes have fewer than 200 questions and are flagged as sparse: Sports
(162), Transport (76), Medicine (72), Nightlife (41). At an expected 5-10%
suppression rate, Nightlife would produce 2-4 suppression cases — too few to
draw conclusions.

**Recommendation:** Group these 4 sparse attributes into an "Other" category for
per-attribute behavioral analysis, or report them with explicit uncertainty
warnings.

### Question Types: 4, Well-Balanced

| Question Type | Count | % |
|---------------|-------|---|
| Country Prediction | 5,563 | 25.61% |
| Association | 5,453 | 25.10% |
| State Prediction | 5,382 | 24.77% |
| General Awareness | 5,328 | 24.52% |

All four types are within 1.1 percentage points of a perfect 25% split. This is
the one dimension where the dataset is well-balanced.

**What each type asks (with examples):**

- **Country Prediction:** "Which country is home to {cultural element}?"
  Options are India + 3 foreign countries. Answer is always India.
  Example: "Which country is the home to Bharatanatyam? A) Japan B) India
  C) Brazil D) France" → Answer: B (India)

- **State Prediction:** "Which state is {cultural element} from?"
  Options are 4 Indian state names. Model must identify the correct state.
  Example: "Which state is famous for Bihu dance? A) Tamil_Nadu B) Assam
  C) Punjab D) Karnataka" → Answer: B (Assam)

- **Association:** "Which cultural element is associated with this state/region?"
  Options are cultural entities. Model must match entity to region.
  Example: "Where is the Kathakali famous within Kerala? A) Thrissur
  B) Jaipur C) Varanasi D) Imphal" → Answer: A (Thrissur)

- **General Awareness:** Free-form cultural knowledge questions.
  Most varied format, not always templated.
  Example: "What is the primary breakfast dish of Tamil Nadu? A) Idli
  B) Paratha C) Dhokla D) Poha" → Answer: A (Idli)

### Coverage: State × Attribute Matrix

The 36 × 16 grid has **576 total cells.** Of these:

| Category | Count | % of cells |
|----------|-------|------------|
| Empty (zero questions) | 165 | 28.6% |
| Non-zero but below threshold (<125) | 372 | 64.6% |
| Reliable (≥125 questions) | **39** | **6.8%** |

**Validated:** `coverage_state_attribute.csv` has 576 rows. Empty=165,
below_threshold=372, reliable=39. Sum: 165+372+39 = 576. ✓

**Only 6.8% of state-attribute combinations have enough data for reliable
analysis.** The 165 empty cells mean we cannot say anything about those
combinations. Most states have zero questions for Nightlife, Transport, and
Medicine.

**Why 125 as the threshold?** At an expected suppression rate of 5-10%, a cell
with 125 questions would yield 6-12 suppression cases. Below this, a single
question flipping changes the rate by >1 percentage point, making the rate
unreliable. The threshold is a practical minimum, not a statistical formula.

**What this means practically:** We cannot report per-state-per-attribute
behavioral rates for most cells. We must aggregate: either per-state (across all
attributes) or per-attribute (across all states). The 39 reliable cells are
concentrated in large states × large attributes (e.g., Telangana × Tourism).

**Example of a reliable cell:** Telangana × Tourism = 268 questions. At 8%
suppression, we'd expect ~21 suppression cases — enough to compute a meaningful
percentage.

**Example of an unreliable cell:** Lakshadweep × Nightlife = 0 questions. We
cannot say anything about this combination. Even Lakshadweep × Tourism might
have only ~30 questions, yielding ~2 suppression cases — statistically useless.

---

## 5. Answer Position Bias

*Source files: `position_bias_overall.csv`, `position_bias_by_qtype.csv`,
`position_bias_by_attribute.csv`, `position_bias_by_state.csv`*

**What this section measures:** The ground truth answer distribution across
positions A/B/C/D. If the dataset puts the correct answer at position B more
often than D, a model with a tendency to predict B would get a free accuracy
boost.

### Overall Distribution

| Letter | Count | Observed % | Expected (uniform) | Deviation |
|--------|-------|------------|-------------------|-----------|
| A | 5,885 | 27.09% | 25.00% | +2.09pp |
| B | 6,308 | 29.03% | 25.00% | +4.03pp |
| C | 5,008 | 23.05% | 25.00% | -1.95pp |
| D | 4,525 | 20.83% | 25.00% | -4.17pp |

**Validated:** Sum = 5,885 + 6,308 + 5,008 + 4,525 = 21,726. ✓
**Expected per cell:** 21,726 / 4 = 5,431.5. CSV rounds to 5,431. ✓

Chi-squared test: **χ² = 363.6, p ≈ 1.7e-78.** (See Glossary for the formula.
Manual check: (5885-5431.5)²/5431.5 + (6308-5431.5)²/5431.5 +
(5008-5431.5)²/5431.5 + (4525-5431.5)²/5431.5 = 37.9+141.5+33.0+151.2 = 363.6 ✓)

**What p ≈ 1.7e-78 means:** The probability of seeing this much deviation from
a uniform distribution by pure chance is 0.000...0017 (with 77 zeros). In other
words, the position bias is definitely real, not random noise.

**Practical impact:** B is overrepresented (+4pp), D is underrepresented (-4pp).
If a model tends to predict B, it gains ~4 percentage points of spurious accuracy
over a model that predicts uniformly. "pp" means percentage points — the
arithmetic difference between two percentages (29% - 25% = 4pp).

### Per Question Type

| Question Type | A% | B% | C% | D% | χ² | Note |
|---------------|-----|-----|-----|-----|-----|------|
| Association | 25.14 | 28.13 | 23.34 | 23.38 | 33.1 | Mild B-bias |
| Country Prediction | 26.77 | 29.01 | 22.58 | 21.64 | 80.9 | Moderate B-bias |
| General Awareness | **31.85** | 29.95 | 21.73 | **16.46** | **330.5** | **Severe A/D skew** |
| State Prediction | 24.67 | 29.06 | 24.54 | 21.72 | 59.3 | Moderate B-bias |

**General Awareness has the most extreme skew:** 31.85% A vs 16.46% D — nearly
2:1 ratio. A model that blindly predicts A on General Awareness would get 31.85%
correct (vs 25% random). This is worth watching in Step 1.

### Per State: Extreme Cases

Some states have very skewed position distributions:

| State | Max letter % | Min letter % | Spread |
|-------|-------------|-------------|--------|
| Meghalaya | 37.8% | 11.6% | 26.2pp |
| Madhya Pradesh | 33.5% | 8.6% | 24.9pp |
| Chhattisgarh | 33.6% | 9.2% | 24.4pp |

These states have positional spreads exceeding 22pp. If a model has position
bias, its accuracy on these states will be systematically distorted.

### Position Bias by Attribute

Most attributes follow the global B-heavy pattern. Religion is the exception:

| Attribute | A% | B% | C% | D% |
|-----------|-----|-----|-----|-----|
| Tourism | 25.8 | 27.8 | 23.5 | 23.0 |
| History | 27.4 | 28.8 | 22.2 | 21.6 |
| Religion | **28.0** | **23.9** | 24.3 | 23.9 |

Religion is the only attribute where A is the most common position and B is not
dominant. If a model has strong B-bias, it will underperform on Religion — not
because Religion is harder, but because the position distribution differs.

### What Step 1 Must Check

The Step 1 evaluation script must compute:
1. Each model's prediction distribution (% of A/B/C/D predictions overall)
2. Prediction distribution per question type (especially General Awareness)
3. If either model has >5pp deviation from uniform on any letter: flag it

---

## 6. Country Prediction Audit

*Source files: `country_prediction_audit.csv`, `country_prediction_answers.csv`,
`country_prediction_distractors.csv`, `country_prediction_by_attribute.csv`*

### The Obvious Fact

This is a benchmark about Indian culture. Every question asks about an Indian
cultural element. So the "Country Prediction" question type — "Which country is
this cultural element from?" — always has "India" as the answer.

**Validated:** `country_prediction_answers.csv` has exactly one row: India, 5,563.
That is 100.0% of all 5,563 Country Prediction questions. ✓

This is not a surprising finding. It is how the benchmark was designed. The
Country Prediction type exists to test whether models can identify Indian cultural
elements as Indian (vs foreign). Both our models should score near 100% on this
type.

### Details

**5,563 questions (25.6% of the dataset)** are Country Prediction.

**India's position across the four options:**

| Position | Count | % |
|----------|-------|---|
| A | 1,489 | 26.8% |
| B | 1,614 | 29.0% |
| C | 1,256 | 22.6% |
| D | 1,204 | 21.6% |

This follows roughly the same B-heavy, D-light pattern as the overall ground
truth distribution.

**Distractors:** There are 109 unique distractor countries. The top 9 are G7
countries plus China and Brazil, accounting for 89.2% of all distractor slots:

| Distractor | Count | % |
|------------|-------|---|
| Japan | 1,726 | 10.34% |
| Brazil | 1,712 | 10.26% |
| Italy | 1,671 | 10.01% |
| Canada | 1,668 | 9.99% |
| China | 1,644 | 9.85% |
| France | 1,634 | 9.79% |
| USA | 1,630 | 9.77% |
| UK | 1,621 | 9.71% |
| Germany | 1,576 | 9.44% |

The remaining 10.8% includes geographically proximate countries: Bangladesh
(164), Nepal (145), Pakistan (112), Thailand (99), Sri Lanka (96). These are
marginally more challenging distractors, but they appear in less than 1% of
questions each.

**Per attribute:** Country Prediction questions are uniformly distributed across
attributes (21-32% of each attribute's total questions).

### What This Means for Step 1

Both models should score near 100% on Country Prediction. These 5,563 questions
will overwhelmingly land in `control_both_correct`. They inflate overall accuracy
without testing cultural knowledge in any meaningful way.

**Action:** Report all behavioral metrics **both with and without Country
Prediction questions.** The "without CP" numbers (16,163 questions, 74.4% of
the dataset) will be more informative for understanding suppression/enhancement
of cultural knowledge.

---

## 7. Text and Lexical Analysis

*Source files: `question_length_by_qtype.csv`, `question_templates.csv`,
`word_freq_unigrams_top200.csv`, `word_freq_bigrams_top100.csv`,
`lexical_diversity_by_qtype.csv`, `lexical_diversity_by_attribute.csv`,
`ngram_diversity_by_attribute.csv`, `option_length_correct_vs_incorrect.csv`,
`option_length_by_position.csv`*

### Question Length

| Question Type | Mean words | Median | Std | Min | Max |
|---------------|-----------|--------|-----|-----|-----|
| General Awareness | 12.10 | 13 | 4.65 | 4 | 36 |
| State Prediction | 11.98 | 12 | 3.63 | 5 | 56 |
| Association | 10.73 | 10 | 3.42 | 6 | 40 |
| Country Prediction | 9.58 | 9 | 2.97 | 6 | 26 |

All types fall in the 9-13 word range. General Awareness questions are slightly
longer because they use more varied phrasing. Country Prediction is shortest
because the template is simple ("Which country is the home to {X}?").

### Question Templates

**What we did:** Matched each question against known template patterns using
regex (see `extract_template` function in `eda_pipeline.py`, lines 557-577).

**55.4% of questions match one of 7 named templates:**

| Template | Count | % |
|----------|-------|---|
| _other_ (no template match) | 9,694 | 44.62% |
| Other "which country" pattern | 2,206 | 10.15% |
| Which state is famous for {X}? | 1,611 | 7.42% |
| Which of the given regions is home to the {X}? | 1,607 | 7.40% |
| Where is the {X} famous within {state}? | 1,607 | 7.40% |
| Which country is the home to {X}? | 1,607 | 7.40% |
| The {X} is associated to which country? | 1,607 | 7.40% |
| According to you, which ... closely associated to {region}? | 1,607 | 7.40% |
| Other "which state" pattern | 177 | 0.81% |
| Other "closely associated" pattern | 3 | 0.01% |

**Validated:** 9,694 + 2,206 + 1,611 + 1,607×5 + 177 + 3 = 9,694 + 2,206 +
1,611 + 8,035 + 177 + 3 = 21,726. ✓

**Key observation:** Six templates each appear exactly 1,607 times. This is
not a coincidence — it means the dataset was generated systematically by feeding
each cultural entity through each template. One entity produces 3-4 questions
from different templates, which explains the high near-duplicate rate.

The 44.62% in "_other_" includes General Awareness questions (which use free-form
phrasing) and less common patterns.

### Option Length Bias

**What we checked:** Is the correct answer systematically longer or shorter than
the distractors? If so, a model could exploit length as a shortcut — for example,
always picking the longest option or the shortest option.

Each of the 21,726 questions has 4 options, giving 86,904 total option instances.
We split these into "correct" (21,726 instances) and "incorrect" (65,178
instances) and compare their word counts.

| Category | Mean words | Median | Std |
|----------|-----------|--------|-----|
| Correct option | 1.471 | 1.0 | 1.255 |
| Incorrect option | 1.595 | 1.0 | 1.405 |

Cohen's d = **-0.094** — negligible effect size (see Glossary for the formula
and interpretation scale). The t-test is significant (p ≈ 4.0e-31) only because
of the large sample size (86,904 total option instances = 21,726 questions × 4
options each). With this many samples, even a 0.1-word difference becomes
"statistically significant" — but it is not practically meaningful.

**By position:**

| Position | Mean words |
|----------|-----------|
| A | 1.559 |
| B | 1.575 |
| C | 1.561 |
| D | 1.561 |

All four positions have virtually identical mean option lengths (spread < 0.02
words). There is no length-based position shortcut.

**By question type — a closer look:**

| Question Type | Correct Mean | Incorrect Mean | Note |
|---------------|-------------|---------------|------|
| Country Prediction | 1.000 | 1.009 | All single-word country names |
| State Prediction | 0.773 | 0.778 | All state names |
| Association | 1.829 | **2.444** | Incorrect 34% longer |
| General Awareness | **2.300** | 2.163 | Correct 6% longer |

Country Prediction and State Prediction have no length difference (all options
are proper nouns). Association shows the biggest asymmetry: distractors average
2.44 words vs correct answers at 1.83 words. But the distributions overlap
heavily, so this is a weak signal at best. General Awareness reverses the
pattern. The effects largely cancel across question types.

**Verdict:** Option length is not a usable shortcut for either humans or models.

### Dominant Vocabulary

The top unigrams (excluding stopwords) are overwhelmingly template scaffolding:

| Rank | Word | Count | Source |
|------|------|-------|--------|
| 1 | associated | 6,554 | "closely associated to" template |
| 2 | country | 5,515 | "which country" template |
| 3 | famous | 5,080 | "famous for"/"famous within" templates |
| 4 | home | 3,290 | "home to the" template |
| 5 | states | 3,240 | "which of the states" template |
| ... | ... | ... | ... |
| 14 | **festival** | **1,179** | First genuinely cultural word |
| 15 | **dance** | **921** | Second genuinely cultural word |

The first 13 words are all template scaffolding. "Festival" at rank 14 is the
first word that carries cultural content. This quantifies how template-heavy the
dataset is — the cultural signal is embedded within a thick layer of formulaic
text.

### Lexical Diversity (MTLD)

**What MTLD measures:** Measure of Textual Lexical Diversity (see Glossary for
the algorithm). Higher MTLD = more diverse vocabulary, less formulaic text.

Why not use simple TTR (type-token ratio = unique words / total words)? Because
TTR decreases with text length — a 1,000-word text almost always has a lower TTR
than a 100-word text, even if both are equally diverse. MTLD avoids this problem
by measuring how many words you can read before the vocabulary starts repeating.
This lets us fairly compare attributes with 41 questions (Nightlife) against
attributes with 3,801 questions (Tourism).

**By question type:**

| Question Type | MTLD | Interpretation |
|---------------|------|----------------|
| General Awareness | 31.66 | Most diverse — free-form statements |
| State Prediction | 25.93 | Moderate — templated but varied entities |
| Association | 25.04 | Moderate |
| Country Prediction | **17.75** | **Least diverse — extremely formulaic** |

Country Prediction's MTLD of 17.75 is 44% lower than General Awareness (31.66).
This confirms that Country Prediction questions are mechanically generated text
with minimal vocabulary variation.

**By attribute (extremes):**

| Attribute | MTLD | n |
|-----------|------|---|
| Rituals_and_Ceremonies | 36.36 | 1,000 |
| Language | 29.33 | 900 |
| Cultural_Common_Sense | 29.25 | 2,091 |
| ... | ... | ... |
| Personalities | 19.79 | 983 |
| Sports | 18.59 | 162 |
| Nightlife | 18.48 | 41 |

Rituals_and_Ceremonies has the richest vocabulary (MTLD 36.36), nearly 2x that
of Nightlife (18.48). Sparse attributes (Sports, Nightlife) tend to have low
MTLD, likely because small sample size limits vocabulary breadth. But
Personalities (983 questions) also has low MTLD (19.79), suggesting genuinely
formulaic question construction for that attribute.

### N-gram Diversity

The N-gram Diversity Score (NGD) computes `unique_ngrams / total_ngrams` averaged
over n=1 to n=4 (see Glossary for the formula). An n-gram is a sequence of n
consecutive words. For example, in "the red fox", the bigrams (n=2) are "the red"
and "red fox". Higher NGD = more diverse (fewer repeated phrases).

| Position | Attribute | NGD Average |
|----------|-----------|-------------|
| Top 3 | Rituals_and_Ceremonies | 0.477 |
| | Transport | 0.474 |
| | Language | 0.414 |
| Bottom 3 | Art | 0.266 |
| | Tourism | 0.255 |
| | Personalities | **0.220** |

Personalities has the lowest NGD (0.220), confirming it as the most repetitive
attribute. Tourism has the second-lowest (0.255) — despite being the largest
attribute (3,801 questions), its repeated template phrases drag down diversity.

### Lexical Diversity by State

**Highest MTLD (most diverse questions):**

| State | MTLD | n |
|-------|------|---|
| Chhattisgarh | 35.47 | 631 |
| Andaman_and_Nicobar | 34.33 | 338 |
| Arunachal_Pradesh | 33.40 | 1,023 |
| Uttarakhand | 32.74 | 910 |
| Rajasthan | 31.88 | 483 |

**Lowest MTLD (most formulaic questions):**

| State | MTLD | n |
|-------|------|---|
| Tamil_Nadu | 18.98 | 741 |
| Chandigarh | 19.14 | 323 |
| Delhi | 19.17 | 1,076 |
| Maharashtra | 19.25 | 283 |
| Mizoram | 19.34 | 210 |

The diversity difference is nearly 2:1 (Chhattisgarh 35.47 vs Tamil Nadu 18.98).
Delhi's low MTLD (19.17) is notable because it has 1,076 questions — this is not
a small-sample artifact. Delhi has many well-known landmarks (India Gate, Red
Fort, Qutub Minar) that slot easily into templates without varied phrasing.

---

## 8. Semantic Analysis

*Source files: `umap_coordinates.csv`, `near_duplicates.csv`,
`no_question_baseline.csv`, `no_question_baseline_by_attribute.csv`,
`qa_overlap_by_qtype.csv`, `tfidf_terms_per_state.csv`, `bertopic_topics.csv`,
`bertopic_assignments.csv`*

### UMAP Visualization

All 21,726 questions were embedded using `all-MiniLM-L6-v2` (384 dimensions,
normalized) and projected to 2D via UMAP (n_neighbors=30, min_dist=0.3,
metric=cosine, random_state=42). See `eda_09_umap.png`.

**By question type:** Country Prediction and State Prediction form distinct
clusters. Association and General Awareness overlap significantly. This reflects
structural differences: CP/SP use specific templated formats that are semantically
distinct, while Association and General Awareness share more varied phrasing.

**By attribute:** No clear attribute-based clusters. Tourism, History, and
Festivals are scattered across the same regions. This means attributes are more
of a metadata label than a semantic differentiator — "Which state is famous for
{festival}?" and "Which state is famous for {tourism site}?" look the same to
an embedding model.

**By state:** Top states (Telangana, Karnataka) form weak subclusters within
the larger question-type clusters. Smaller states are diffusely scattered.

### Embedding-Based Baseline: 75.87%

**What this measures:** For each question, we take the state name from the
dataset's `state` column (this is metadata — the model does NOT see this column),
embed it using MiniLM, and compare that embedding against the four option
embeddings using cosine similarity (see Glossary). We pick the option with the
highest similarity to the state name as the predicted answer.

**This is NOT something the model can do.** The model receives a question and
four options. It does not receive the state column. This baseline measures how
much information the option text alone leaks about the answer, IF you already
know which state the question is about.

| Slice | Baseline Accuracy | Random Baseline | What It Tells Us |
|-------|------------------|-----------------|------------------|
| **Overall** | **75.87%** | 25.00% | High option-state correlation |
| State Prediction | **99.98%** | 25.00% | Trivially true — correct answer IS the state name |
| Country Prediction | 95.88% | 25.00% | "India" embedding > all foreign country embeddings for any Indian state |
| Association | 63.27% | 25.00% | Regional entities partially identifiable from state name |
| General Awareness | 43.52% | 25.00% | Most distant from state-name shortcut |

**Why State Prediction is 99.98%:** For State Prediction questions, the four
options are state names. The correct option IS the state that the question is
about. So we are comparing `embed("Tamil_Nadu")` against options like
`["Tamil_Nadu", "Karnataka", "Bihar", "Punjab"]`. Of course the embedding of
"Tamil_Nadu" is closest to itself. This is trivially true and does not tell us
anything about whether the question is easy for an LLM.

**The LLM's actual task is different.** For State Prediction, the LLM sees:
"Which state is famous for Bharatanatyam? A) Tamil Nadu B) Karnataka C) Bihar
D) Punjab." It needs to know that Bharatanatyam is from Tamil Nadu. The
embedding baseline cheats by using the state metadata column.

**Why Country Prediction is 95.88%:** We compare embeddings of Indian state
names (e.g., "Kerala", "Rajasthan") against options like ["India", "Japan",
"Brazil", "France"]. Indian state name embeddings are semantically closer to
"India" than to foreign countries. Again, this is expected and does not tell us
about LLM difficulty.

**Why this baseline still matters:** It tells us about the STRUCTURE of the
options. For Association questions (63.27%), the correct answer often contains
the state/region name or a semantically close term, meaning models with good
geographic associations can partly solve these without deep cultural knowledge.
For General Awareness (43.52%), options are most distant from the state name,
meaning these questions most require genuine cultural knowledge beyond geography.

**By attribute:**

| Attribute | Baseline Accuracy | n |
|-----------|------------------|---|
| Nightlife | 85.37% | 41 |
| Rituals_and_Ceremonies | 80.10% | 1,000 |
| Cuisine | 79.89% | 1,671 |
| ... | ... | ... |
| Religion | 69.29% | 482 |
| Transport | 67.11% | 76 |

All attributes exceed 25%. The range is 67.1% (Transport) to 85.4% (Nightlife),
though sparse attributes (Nightlife n=41) have high variance.

### Question-Answer Semantic Overlap

**What this measures:** For each question, we compute cosine similarity (see
Glossary) between the question text embedding and the correct answer text
embedding. High overlap means the answer text is semantically similar to the
question text — often because the answer is partially restated in the question.

| Question Type | Mean | Median | Std |
|---------------|------|--------|-----|
| Association | 0.3864 | 0.3558 | 0.1793 |
| State Prediction | 0.3257 | 0.3077 | 0.1356 |
| Country Prediction | 0.3126 | 0.3149 | 0.0887 |
| General Awareness | **0.2490** | 0.2151 | 0.1596 |

Association questions have the highest Q-A overlap (mean 0.386) because the
question often contains the answer entity. General Awareness has the lowest
(0.249), meaning these questions require the most external knowledge.

**By attribute:** The range is modest (0.254 for Transport to 0.355 for Sports).
No attribute has dramatically higher or lower Q-A overlap.

### TF-IDF Distinctive Terms Per State

For each state, all questions were concatenated into a single "document." Then
TF-IDF (see Glossary for the formula) was computed across all 36 state-documents
to identify words that are distinctive to each state — words that appear often in
that state's questions but rarely in other states' questions. Examples:

| State | Top TF-IDF terms (excluding state name) |
|-------|----------------------------------------|
| Goa | portuguese, panaji, konkan, feni |
| Telangana | hyderabad, charminar, biryani, warangal |
| Kerala | backwaters, kathakali, onam, ayurveda |
| Lakshadweep | minicoy, kavaratti |
| Assam | majuli, bihu, one-horned rhinoceros |

Most states' top TF-IDF term is the state name itself (because templates include
it). The second-ranked terms are more informative: "portuguese" for Goa, "bastar"
for Chhattisgarh, "majuli" for Assam — these reflect genuine cultural associations.

Lakshadweep has the most culturally specific vocabulary — "minicoy" and "kavaratti"
are island names that appear almost nowhere else in typical pretraining data.
However, Lakshadweep has only 122 questions — the smallest of any state.

### BERTopic: Discovered Topics

BERTopic (see Glossary for how it works) was run using our pre-computed MiniLM
embeddings (so it did not re-compute embeddings), with `nr_topics=20` (target
approximately 20 topics) and `min_topic_size=100` (each topic must have at least
100 questions). It discovered 19 topics plus an outlier cluster (Topic -1).

**How to read the topic table:** Each topic has a count (how many questions were
assigned to it) and key terms (the most representative words, determined by
c-TF-IDF — a variant of TF-IDF applied within each cluster to find its
distinctive vocabulary).

**Validated against `bertopic_topics.csv`:**

| Topic | Count | Key Terms |
|-------|-------|-----------|
| -1 (outlier) | 7,889 | the, which, is, to, of |
| 0 | 7,665 | to, which, the, is, states |
| 1 | 1,206 | dance, festival, the, is |
| 2 | 817 | this, and, culture, often |
| 3 | 668 | cuisine, dishes, dish, rice |
| 4 | 524 | lake, beach, kerala, which |
| 5 | 499 | park, sanctuary, national, wildlife |
| 6 | 336 | silk, bamboo, weaving, sarees |
| 7 | 333 | painting, paintings, art, the |
| 8 | 319 | traditional, this, attire, women |
| 9 | 211 | sikkim, of, which, to |
| 10 | 181 | delhi, belongs, one, closely |
| ... | ... | ... |

**36.3% of questions (7,889) fall into the outlier cluster** — these are too
generic to assign to any topic, mostly consisting of templated question
boilerplate.

**Key observation:** BERTopic topics are organized primarily by **cultural
category** (cuisine=Topic 3, wildlife=Topic 5, textiles=Topic 6, art=Topic 7,
costume=Topic 8) and **geography** (Sikkim=Topic 9, Delhi=Topic 10). The
predefined attributes only partially align with these discovered topics.

### BERTopic-Attribute Alignment

**What this measures:** For each discovered topic, we look at what percentage of
its questions belong to each predefined attribute. If a topic is 80% Cuisine
questions, then the "Cuisine" attribute label aligns well with what BERTopic found
organically from the text. If a topic is spread across 5 attributes, the
predefined labels don't capture a real semantic boundary.

The alignment values are "row-normalized" — for each topic, the percentages across
all attributes sum to 100%. (This is what "normalize='index'" does in the code at
line 844 of `eda_pipeline.py`.)

**Topic 3 (cuisine, 668 questions):** 79.2% of its questions are labeled as the
Cuisine attribute. (Validated: `bertopic_vs_attribute.csv`, topic 3, Cuisine
= 0.792.) This is the best-aligned topic — cuisine vocabulary ("rice", "dishes",
"spices") is distinctive enough to form a clean cluster that matches the
predefined label.

**Topic 0 (generic, 7,665 questions):** This mega-cluster fragments across all
attributes. It contains the most templated questions that share generic vocabulary
("which", "state", "famous") regardless of cultural domain.

**Topic 5 (wildlife/parks, 499 questions):** Primarily Tourism, with overlap into
History and Cultural_Common_Sense. A temple question is a temple question whether
the dataset labels it "Tourism" or "History" — BERTopic does not see the
predefined label, only the text.

**Why this matters for our MI study:** If we find different suppression rates for
Tourism vs History, the difference might reflect how questions that BERTopic would
cluster together are split by an arbitrary attribute boundary, not genuine
differences in how RLHF treats these cultural domains. BERTopic gives us an
independent clustering to cross-validate attribute-level findings.

---

## 9. Distractor Quality

*Source files: `distractor_similarity.csv`, `distractor_similarity_by_qtype.csv`,
`distractor_similarity_by_attribute.csv`, `distractor_quality_summary.csv`,
`answer_in_question_leakage.csv`*

### What This Section Measures

For each question, we compute the cosine similarity (see Glossary) between the
correct answer's embedding and each of the 3 distractor embeddings. We then take
the mean, max, and min of those 3 similarities.

- **Higher mean similarity** = distractors are more plausible (semantically
  similar to the correct answer) = harder question.
- **Lower mean similarity** = distractors are obviously different from the
  correct answer = easier question.

This tells us how "confusable" the wrong options are with the right answer at
the semantic level.

### Per Question Type

| Question Type | Mean Sim | Count | Interpretation |
|---------------|----------|-------|----------------|
| Country Prediction | **0.631** | 5,563 | Country names are inherently similar in embedding space |
| State Prediction | 0.444 | 5,382 | State names are moderately similar |
| Association | 0.352 | 5,453 | Cultural entity names are more varied |
| General Awareness | **0.282** | 5,328 | Most diverse option types |

**Important caveat about Country Prediction:** The 0.631 similarity does NOT mean
CP questions are hard. Here is why: country name embeddings cluster together in
semantic space because they are all the same TYPE of thing (countries). So
`cosine_sim(embed("India"), embed("Japan"))` is high (~0.6) because both are
country names, not because India and Japan are confusable. The LLM is not
comparing embeddings — it processes tokens and knows India is not Japan. This
metric measures semantic similarity of the WORDS, not the difficulty of the
QUESTION for an LLM.

**General Awareness has the lowest similarity (0.282)** — its options span diverse
types (place names, concepts, people), making distractors genuinely dissimilar
from correct answers.

### State Prediction: Distractor Domain

**99.6% of State Prediction distractors (16,086 of 16,146) are state/UT names.**
(Validated: `distractor_quality_summary.csv`.)

**How this was computed:** For each State Prediction question, we look at the 3
incorrect options (distractors). We check if each distractor text, after
normalization (`.strip().lower().replace("_"," ")`), matches any of the 36 state
names in the dataset. 16,086 out of 16,146 total distractors (= 5,382 SP
questions × 3 distractors each) are state names.

The remaining 60 distractors (0.4%) are either: (a) misspelled state names,
(b) city names used instead of state names, or (c) data entry errors.

This is by design — "which state?" questions use states as options. The
distractors are plausible at the domain level (all valid Indian states) but
may vary in geographic plausibility (e.g., Lakshadweep as a distractor for a
Rajasthan question is geographically implausible).

### Per Attribute

| Attribute | Mean Sim | Count |
|-----------|----------|-------|
| Language | 0.444 | 900 |
| Costume | 0.442 | 1,513 |
| Festivals | 0.437 | 2,241 |
| Cuisine | 0.434 | 1,671 |
| ... | ... | ... |
| Tourism | 0.417 | 3,801 |
| Religion | 0.410 | 482 |
| Medicine | 0.392 | 72 |

The range is narrow (0.392 to 0.444). Distractor quality is consistent across
attributes. This means distractor difficulty is unlikely to be a major confound
for per-attribute behavioral analysis.

---

## 10. Cultural Specificity

*Source files: `cultural_entities.csv`, `cultural_entities_detail.csv`,
`cultural_entities_combined.csv`, `entity_extraction_by_qtype.csv`*

### What This Section Does

We try to identify the cultural entity that each question is about. For templated
questions, we extract the entity name using regex. For free-form questions, we
use the answer text as a proxy.

### Entity Extraction: Regex

The pipeline uses regex patterns (lines 103-119 of `eda_pipeline.py`) to match
the 7 identified templates and extract the cultural entity name. A regex
(regular expression) is a pattern-matching language. For example:

```
Pattern: r"famous for (.+?)\?"
Matches: "Which state is famous for Bharatanatyam?"
Extracts: "Bharatanatyam" (the part inside the parentheses)

Pattern: r"home to the (.+?)\?"
Matches: "Which of the given regions is home to the Bihu dance?"
Extracts: "Bihu dance"

Pattern: r"^(.+?)\s+is associated"
Matches: "Hyderabad biryani is associated to which country?"
Extracts: "Hyderabad biryani"
```

When no regex pattern matches (mainly General Awareness questions), the code
falls back to a composite key: `state|attribute|answer` (e.g.,
`Tamil_Nadu|Cuisine|Idli`).



**Validated against `entity_extraction_by_qtype.csv`:**

| Question Type | Extracted | Total | Rate | Missing |
|---------------|-----------|-------|------|---------|
| Association | 4,821 | 5,453 | 88.4% | 632 |
| Country Prediction | 4,859 | 5,563 | 87.3% | 704 |
| State Prediction | 3,270 | 5,382 | 60.8% | 2,112 |
| **General Awareness** | **29** | **5,328** | **0.5%** | **5,299** |

**General Awareness is almost completely missed by regex** because its questions
use free-form phrasing. Examples: "Which dish is most often associated with the
Tamil Nadu breakfast?", "A classical dance form that utilizes elaborate
costumes..." These do not match any template pattern.

**4,949 unique cultural entities** were identified from regex extraction.
(Validated: `cultural_entities.csv` has 4,949 data rows.)

### Fallback: Answer Text as Proxy

For the 8,747 questions (40.3%) where regex fails, we use a combination of
`(state, attribute, answer_text)` as the entity key. This is implemented in
`_build_entity_key()` (line 122-127 of `eda_pipeline.py`).

**Why this works:**
- General Awareness answers ARE the cultural entity ("Puanchei", "Idli", "Odissi")
- The (state, attribute) prefix distinguishes same-name entities from different
  contexts ("temple" in Tamil Nadu vs "temple" in Rajasthan)
- This is NOT perfect — some General Awareness answers are generic ("rice",
  "silk"), but prefixing with state+attribute makes them unique enough

**Example fallback keys:**
```
"Tamil_Nadu|Cuisine|Idli"          ← Tamil breakfast dish
"Rajasthan|Costume|Bandhani"       ← Rajasthani tie-dye fabric
"Nagaland|Festivals|Hornbill"      ← Naga festival
```

### Combined Entity Coverage

| Method | Unique Keys | Questions Covered |
|--------|-------------|-------------------|
| Regex-extracted | 4,949 | 12,979 (59.7%) |
| Answer-text fallback | 3,207 | 8,747 (40.3%) |
| **Combined total** | **8,156** | **21,726 (100%)** |

**Validated:** `cultural_entities_combined.csv` has 8,156 data rows (header +
8,156). ✓

Every question now has an entity key for grouping. This is essential for Step 1:
entity-level suppression rates account for the redundancy in the dataset.

### Entity State Uniqueness

**99.98% of regex-extracted entities are unique to a single state.** "Bihu" only
appears in Assam questions, "Bharatanatyam" only in Tamil Nadu. This is by
construction: the dataset pairs cultural elements with their origin states.

### Most-Asked Entities

| Entity | Questions | State |
|--------|-----------|-------|
| Hyderabad | 70 | Telangana |
| Delhi | 64 | Delhi |
| Haryana | 38 | Haryana |
| Karnataka | 28 | Karnataka |
| Chandigarh | 27 | Chandigarh |

The most-asked "entities" are geographic names (cities, states), not cultural
elements (dances, festivals). This reflects the "According to you, which of the
following is closely associated to {region}?" template.

### Entity Repetition Across Question Types

| Coverage | # Entities |
|----------|-----------|
| 1 question type only | 3,350 |
| 2 question types | 8 |
| 3 question types | 1,591 |
| All 4 question types | 0 |

**1,591 entities appear in 3 question types** — typically Country Prediction,
State Prediction, and Association. A model that knows one fact about a cultural
entity gets credit 3 times. Mean questions per regex entity: 12,979 / 4,949 = 2.6.

### What "8,156 Entity Keys" Means for Our Study

The dataset's 21,726 questions test approximately 8,156 distinct cultural facts.
For behavioral labeling, a suppression event on one question about entity X is
likely to co-occur with suppression on other questions about X. Behavioral labels
cluster by entity, not independently by question.

**Expected suppression at 8%:** ~650 entities suppressed × ~2.6 questions/entity
≈ ~1,700 suppressed questions. But the 650 entities are the independent
observations, not the 1,700 questions.

**Why the distinction matters:** Imagine entity X has 3 questions, all suppressed.
That is 3 suppressed questions but only 1 independent suppression event. If we
report "1,700 questions suppressed," it sounds like 1,700 independent failures.
If we report "650 entities suppressed," it more accurately reflects 650
independent cases where RLHF broke a specific piece of cultural knowledge.

**Per-state power:** A state with 150 entities might have ~12 suppressed — barely
enough for a percentage (12/150 = 8%, but ±5pp uncertainty). Per-attribute: an
attribute with 200 entities might have ~16 suppressed — marginally better.

**What "power" means here:** Statistical power is the ability to detect a real
effect. With only 12 suppressed entities, we cannot distinguish between "8%
suppression" and "4% suppression" with any confidence. We need at least 30-50
events to estimate a percentage reliably. States/attributes with fewer than ~50
entities will have unreliable suppression rate estimates.

**Recommendation:** Report entity-level suppression rates alongside question-level
rates. "X% of cultural entities are suppressed" is a stronger claim than "X% of
questions are suppressed" because it accounts for the redundancy.

---

## 11. Critical Findings for Step 1

### Findings That Require Action

1. **Report metrics with and without Country Prediction.** All CP answers are
   "India" (expected for an Indian culture benchmark). Both models will ace these.
   The interesting behavioral signal comes from the other 16,163 questions.

2. **Check for position bias in model predictions.** Ground truth is non-uniform
   (B=29.0%, D=20.8%). If a model overproduces B, it gets a ~4pp accuracy bonus.
   General Awareness has the most extreme skew (A=31.9%, D=16.5%).

3. **Do not report per-state-attribute behavioral rates.** Only 39 of 576 cells
   (6.8%) have enough data (≥125 questions) for reliable rates. Aggregate to
   state level or attribute level.

4. **Account for question-entity redundancy.** 78.6% of questions are involved
   in near-duplicate pairs. Behavioral labels cluster by entity. Report
   entity-level suppression rates alongside question-level rates.

5. **Group sparse attributes.** Sports (162), Transport (76), Medicine (72),
   Nightlife (41) are too small for reliable per-attribute stats.

### Findings That Are Limitations (Report But Cannot Fix)

6. **351 conflicting duplicate groups.** Same question text, different answer
   positions. Not a bug (each row's ground truth matches its own options), but
   contributes to near-duplicate inflation.

7. **7.4% answer-in-question leakage.** Both models should get these right
   trivially. They will inflate `control_both_correct`.

8. **55.4% of questions follow 7 templates.** The dataset is highly formulaic.
   Template familiarity may be a confound — the instruct model may perform
   better partly because it handles the question format better.

9. **Embedding baseline: 75.87% using state metadata.** This measures
   information leakage from the option text structure, not LLM difficulty.
   For State Prediction (99.98%) and Country Prediction (95.88%), the high
   baseline is trivially expected. For Association (63.27%) and General
   Awareness (43.52%), it shows that options partially encode geographic
   information. This is a structural property of the benchmark, not a
   shortcut the model necessarily uses.

### Findings That Are Expected

10. **Both models should score ~95-100% on Country Prediction.** This is an
    Indian culture benchmark. Identifying Indian cultural elements as Indian is
    the easy case.

11. **Sparse attributes will have noisy per-attribute stats.** Group or flag
    them.

12. **Southern/northeastern state overrepresentation.** Telangana (7.85%) vs
    Maharashtra (1.30%) reflects the Sanskriti paper's annotation priorities.

### Decision Matrix: What to Report in the Paper

| Analysis Level | Report? | Why |
|----------------|---------|-----|
| Overall accuracy (base vs instruct) | Yes, with and without CP | Primary result |
| Per-question-type accuracy | Yes | Shows CP inflation |
| Per-question-type suppression/enhancement | Yes | Core finding |
| Per-attribute suppression/enhancement | Yes (top 12 only) | Sparse attributes unreliable |
| Per-state suppression/enhancement | Yes (top 20 only) | Small states unreliable |
| Per-state-attribute suppression | **No** | Only 6.8% of cells have n≥125 |
| Entity-level suppression rate | Yes | Accounts for redundancy |
| Model position bias check | Yes | Must validate before interpreting labels |
| Suppression by attribute/state/type | Yes | Core analysis dimensions |
| Near-duplicate impact analysis | Appendix | Limitation discussion |
| Embedding baseline comparison | Appendix | Benchmark structural property |

### Behavioral Labels Recap

For each of the 21,726 questions, Step 1 will produce one of 4 labels based on
whether the base model and instruct model get the question right:

```
                          Instruct Correct    Instruct Wrong
                         ┌─────────────────┬─────────────────┐
    Base Correct         │ control_both_    │ suppression     │
                         │ correct          │                 │
                         ├─────────────────┼─────────────────┤
    Base Wrong           │ enhancement     │ control_both_   │
                         │                 │ wrong            │
                         └─────────────────┴─────────────────┘
```

- **Suppression:** Base model got it right, instruct model got it wrong.
  RLHF/SFT training broke knowledge the base model had.
- **Enhancement:** Base model got it wrong, instruct model got it right.
  RLHF/SFT training added or surfaced knowledge the base model lacked.
- **control_both_correct:** Both models got it right. No change in this knowledge.
- **control_both_wrong:** Both models got it wrong. Neither model has this knowledge.

### What Success Looks Like After Step 1

1. **Base accuracy 40-70%** (excluding CP). Below 35% means the prompt is broken.
   Above 75% means something is inflating scores.
2. **Instruct accuracy > base accuracy.** The single strongest signal of correct
   prompt formatting.
3. **CP accuracy > 95% for both models.** Indian culture questions should be
   recognized as Indian.
4. **Null prediction rate < 2%.** "Null" means the model output something other
   than A/B/C/D (e.g., a full sentence explanation). Higher than 2% means the
   prompt needs adjustment.
5. **Suppression 5-12%, enhancement 4-10%.** Based on prior work with
   Qwen2-1.5B. LLaMA 8B may differ, but these are reasonable bounds.

---

## 12. What This Stage Does NOT Do

The EDA is pre-model analysis of the dataset. It does not perform:

- Model inference or accuracy computation (that is Step 1)
- Behavioral labeling (suppression/enhancement/control) (that is Step 1)
- Per-question difficulty estimation from model responses (requires model outputs)
- Item Response Theory analysis (requires response matrix from multiple models)
- Distractor selection frequency analysis (requires model predictions)
- Cross-model agreement analysis (requires multiple model outputs)
- Any causal claims about what models know or don't know

The EDA tells us about the dataset's structure and biases. It does not tell us
how the models will behave on it. That is Step 1's job.

---

## 13. Output Files

### Plots (13 PNG files, ~3.6 MB)

```
/home/anshulk/cultural-mi/plots/
├── eda_01_distributions.png          # State/attribute/type bar charts + coverage summary
├── eda_02_state_attr_heatmap.png     # 36×16 count heatmap (state × attribute)
├── eda_03_state_qtype_heatmap.png    # 36×4 count heatmap (state × question type)
├── eda_04_position_bias.png          # GT letter distribution + per-type + India position
├── eda_05_question_length.png        # Word count histograms by type and attribute
├── eda_06_option_length_bias.png     # Correct vs incorrect option lengths + by type
├── eda_07_word_frequency.png         # Top unigrams, bigrams, trigrams
├── eda_08_lexical_diversity.png      # MTLD by attribute/state/type + NGD
├── eda_09_umap.png                   # 2D UMAP colored by type, attribute, state
├── eda_10_semantic_analysis.png      # Q-A overlap, baseline accuracy, near-dup curve
├── eda_11_bertopic.png               # Discovered topics + attribute alignment heatmap
├── eda_12_distractor_quality.png     # Distractor plausibility by type and attribute
└── eda_13_cultural_specificity.png   # Entity frequency + state uniqueness
```

### What Each Plot Shows

**eda_01_distributions.png** — Four panels: (top-left) horizontal bar chart of
questions per state, sorted by count, with red line at n=125 minimum threshold;
(top-right) horizontal bar chart of questions per attribute, sparse attributes
(<200) colored red; (bottom-left) bar chart of questions per question type with
counts labeled; (bottom-right) text summary of coverage statistics.

**eda_02_state_attr_heatmap.png** — Full 36×16 grid showing the count of
questions at each state-attribute intersection. Annotated with numbers. Dark
red = high count, white/yellow = low or zero. Shows where coverage gaps are.

**eda_03_state_qtype_heatmap.png** — 36×4 grid showing state × question type
counts. All states have roughly equal distribution across the 4 types.

**eda_04_position_bias.png** — Six panels: overall GT distribution, per-type
breakdown (%), per-attribute breakdown (%), chi-squared scores per type, per-state
breakdown (top 10), and position of "India" in Country Prediction questions.

**eda_05_question_length.png** — Three panels: overall word count histogram,
per-type overlaid histograms, and per-attribute boxplots (top 12).

**eda_06_option_length_bias.png** — Three panels: correct vs incorrect option
length distributions, mean option length by position (A/B/C/D), and option length
by type split by correct/incorrect.

**eda_07_word_frequency.png** — Three horizontal bar charts: top 30 unigrams,
top 20 bigrams, top 20 trigrams (all excluding stopwords).

**eda_08_lexical_diversity.png** — Four panels: MTLD by attribute, MTLD by state
(top 10 + bottom 10), MTLD by question type, NGD average by attribute.

**eda_09_umap.png** — Three UMAP scatter plots of all 21,726 questions colored
by: question type, attribute (top 8), and state (top 8).

**eda_10_semantic_analysis.png** — Four panels: Q-A cosine similarity histograms
by type, embedding baseline accuracy by type, near-duplicate pair count by
threshold (log scale), embedding baseline accuracy by attribute.

**eda_11_bertopic.png** — Two panels: topic size bar chart (excluding outlier
count in title), and top-10-topics × attribute heatmap (row-normalized).

**eda_12_distractor_quality.png** — Two panels: distractor similarity histograms
by question type, and mean distractor similarity by attribute.

**eda_13_cultural_specificity.png** — Two panels: top 20 most frequent regex-
extracted entities, and entity state uniqueness distribution (how many states
each entity appears in).

### Analysis CSVs (54 files + 2 numpy caches, ~183 MB)

```
/data/user_data/anshulk/cultural-mi/analysis/
├── sanskriti_usable.csv              # 21,726 rows, master dataset with entity_key
├── distribution_states.csv           # 36 rows
├── distribution_attributes.csv       # 16 rows
├── distribution_qtypes.csv           # 4 rows
├── coverage_state_attribute.csv      # 576 rows (all state-attr cells)
├── cross_tab_state_attribute.csv     # 36×16 count matrix
├── cross_tab_state_qtype.csv         # 36×4 count matrix
├── state_summary.csv                 # 36 rows, per-state breakdown
├── position_bias_overall.csv         # 4 rows
├── position_bias_by_qtype.csv        # 16 rows
├── position_bias_by_attribute.csv    # 64 rows
├── position_bias_by_state.csv        # 144 rows
├── country_prediction_audit.csv      # Summary metrics
├── country_prediction_answers.csv    # 1 row: India, 5563
├── country_prediction_distractors.csv # 109 distractors
├── country_prediction_by_attribute.csv # CP breakdown by attribute
├── question_length_by_qtype.csv      # 4 rows
├── question_length_by_attribute.csv  # 16 rows
├── option_length_correct_vs_incorrect.csv
├── option_length_by_position.csv
├── option_length_by_type_correctness.csv
├── word_freq_unigrams_top200.csv
├── word_freq_bigrams_top100.csv
├── word_freq_trigrams_top100.csv
├── word_freq_by_attribute_top20.csv
├── question_templates.csv            # 10 templates
├── templates_by_qtype.csv
├── lexical_diversity_by_attribute.csv
├── lexical_diversity_by_state.csv
├── lexical_diversity_by_qtype.csv
├── ngram_diversity_by_attribute.csv
├── question_embeddings.npy           # 21,726 × 384 float32 (32 MB)
├── option_embeddings.npz             # 4 × 21,726 × 384 (128 MB)
├── umap_coordinates.csv
├── near_duplicates.csv               # 77,833 pairs
├── no_question_baseline.csv
├── no_question_baseline_by_attribute.csv
├── qa_overlap_by_qtype.csv
├── qa_overlap_by_attribute.csv
├── tfidf_terms_per_state.csv
├── bertopic_topics.csv
├── bertopic_assignments.csv
├── bertopic_vs_attribute.csv
├── bertopic_vs_qtype.csv
├── distractor_similarity.csv         # 21,726 rows
├── distractor_similarity_by_qtype.csv
├── distractor_similarity_by_attribute.csv
├── distractor_quality_summary.csv
├── answer_in_question_leakage.csv    # 1,615 rows
├── cultural_entities.csv             # 4,949 regex-extracted entities
├── cultural_entities_detail.csv
├── cultural_entities_combined.csv    # 8,156 entities (regex + fallback)
├── entity_extraction_by_qtype.csv
├── exact_duplicates.csv              # 419 groups
├── conflicting_duplicates.csv        # 351 groups
└── data_quality_summary.csv          # 1-row summary
```

### Pipeline Script

```
/home/anshulk/cultural-mi/scripts/eda_pipeline.py
```

Single script, 1,109 lines, runs all 8 sections. Usage:
- Full run: `python scripts/eda_pipeline.py`
- Single section: `python scripts/eda_pipeline.py --section 5`

### How the Pipeline Works

The pipeline follows this flow:

```
1. load_data()
   ├── Downloads Sanskriti from HuggingFace (or uses cache)
   ├── Computes ground_truth_letter for each row (match answer to option1-4)
   ├── Filters to 21,726 usable rows (where ground_truth_letter is not null)
   ├── Adds entity_key column (regex extraction + fallback)
   └── Saves sanskriti_usable.csv

2. section_1(df) — Distributions
   ├── Counts per state, attribute, question_type
   ├── Cross-tabulations (state×attribute, state×question_type)
   ├── Coverage analysis (which cells have ≥125 questions)
   └── Saves 6 CSVs + 3 plots

3. section_2(df) — Position Bias
   ├── Counts ground truth letter distribution overall and per slice
   ├── Chi-squared test against uniform distribution
   └── Saves 4 CSVs + 1 plot

4. section_3(df) — Country Prediction Audit
   ├── Checks all CP answers
   ├── Catalogs distractors
   └── Saves 4 CSVs

5. section_4(df) — Text & Lexical Analysis
   ├── Question/option word counts
   ├── Template detection via regex
   ├── Word frequency (unigrams, bigrams, trigrams)
   ├── MTLD computation per group
   ├── N-gram diversity per attribute
   └── Saves 14 CSVs + 4 plots

6. section_5(df) — Semantic Analysis (slowest section, ~2.5 min)
   ├── Computes MiniLM embeddings for all questions and options (or loads cache)
   ├── UMAP projection to 2D
   ├── Near-duplicate detection (pairwise cosine sim > 0.85)
   ├── Embedding baseline (state name vs option similarity)
   ├── Question-answer overlap
   ├── TF-IDF per state
   ├── BERTopic topic modeling
   └── Saves 12 CSVs + 2 numpy caches + 3 plots

7. section_6(df) — Distractor Quality
   ├── Correct-distractor embedding similarity
   ├── Answer-in-question leakage detection
   ├── State Prediction distractor domain check
   └── Saves 5 CSVs + 1 plot

8. section_7(df) — Cultural Specificity
   ├── Regex entity extraction + fallback
   ├── Entity-level aggregation
   └── Saves 4 CSVs + 1 plot

9. section_8(df) — Data Quality
   ├── Exact duplicate detection
   ├── Conflicting duplicate detection
   ├── Source column leakage check
   └── Saves 3 CSVs
```

### Section Mapping

The pipeline sections and report sections are numbered differently. The report
reorders for logical flow (data quality before distributions):

| Pipeline Section | Report Section |
|-----------------|----------------|
| 1: Distributions | 4: Distribution Analysis |
| 2: Position Bias | 5: Answer Position Bias |
| 3: Country Prediction | 6: Country Prediction Audit |
| 4: Text & Lexical | 7: Text and Lexical Analysis |
| 5: Semantic | 8: Semantic Analysis |
| 6: Distractor Quality | 9: Distractor Quality |
| 7: Cultural Specificity | 10: Cultural Specificity |
| 8: Data Quality | 3: Data Quality Assessment |

---

## 14. Runtime and Reproducibility

### Execution Environment

| Property | Value |
|----------|-------|
| Date | 2026-03-25 |
| Machine | CMU Babel cluster, login node |
| CPU | Intel (no GPU used for EDA) |
| Python | 3.11, conda environment `cultural` |
| sentence-transformers | 5.3.0 (all-MiniLM-L6-v2) |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.11 |
| BERTopic | 0.17.4 |
| lexicalrichness | 0.5.1 |
| datasets | 4.8.3 |

### Timing Breakdown

| Section | Time |
|---------|------|
| 1. Distributions | 1.8s |
| 2. Position Bias | 0.9s |
| 3. Country Prediction Audit | 0.2s |
| 4. Text & Lexical Analysis | 3.7s |
| 5. Semantic Analysis (embeddings, UMAP, near-dups, BERTopic) | 146.2s |
| 6. Distractor Quality | 6.0s |
| 7. Cultural Specificity | 0.6s |
| 8. Data Quality | 0.5s |
| **Total** | **159.9s (2.7 minutes)** |

Section 5 dominates (91% of runtime) due to UMAP (n=21,726 points), near-
duplicate detection (pairwise cosine similarity, O(n²)), and BERTopic fitting.

### Reproducibility

**What "reproducible" means:** Running the same code on the same data produces
the same results. This requires controlling all sources of randomness.

All random operations use seed 42:
- UMAP: `random_state=42` (fixes the random initialization of the 2D layout)
- BERTopic: inherits from UMAP seed (the clustering depends on UMAP output)
- Sentence embeddings: deterministic (same model + same input always produces
  the same 384-dimensional vector)
- Chi-squared, cosine similarity, TF-IDF: purely mathematical (no randomness)

The EDA is fully deterministic. Running `python scripts/eda_pipeline.py` on the
same dataset will produce identical CSV values and visually identical plots.

**Caveat:** If the `sentence-transformers` library version changes, embeddings
may differ slightly, which would change near-duplicate counts, UMAP layout, and
BERTopic topics. The CSV values in the current analysis directory were produced
with sentence-transformers 5.3.0.

---

## 15. Alternative Datasets Considered

We surveyed every publicly available Indian cultural knowledge dataset to verify
that Sanskriti is the best fit for this MI study. We need: (1) MCQ format,
(2) large enough for hundreds of suppression cases, (3) English, (4) cultural
knowledge specifically, (5) public with state/attribute metadata.

| Dataset | Year | Size | Format | Why Not |
|---------|------|------|--------|---------|
| **Sanskriti** | 2025 | 21,853 MCQs | MCQ, 4-choice | **Selected** |
| MILU (AI4Bharat) | 2024 | ~80K MCQs | MCQ | Tests exam knowledge, not culture; non-English |
| DIWALI | 2025 | ~8K concepts | Concept inventory | No MCQ format; not QA |
| Indica | 2026 | 515 Qs → 1,630 pairs | Free-form + MCQ | Too small (515 base questions) |
| DRISHTIKON | 2025 | 2,126 MCQs | MCQ + images | Multimodal; too small for MI |
| CulturalBench | 2025 | 1,227 MCQs | MCQ | Only ~100 India-specific questions |
| IndicParam | 2025 | Varies | MCQ | Tests academic knowledge, not cultural |

### DRISHTIKON: The Best Alternative (But Still Insufficient)

DRISHTIKON covers all 28 states and 8 UTs with 2,126 MCQs that were human-
curated with intentionally close distractors. Its question design is better than
Sanskriti's. However: (a) at 2,126 questions, 8% suppression yields ~170 cases —
borderline for activation extraction, (b) it is multimodal (requires images),
(c) no text-only subset has been released.

### DIWALI: Different Purpose, Complementary

DIWALI (EMNLP 2025) catalogs ~8K cultural concepts across 36 sub-regions and 17
facets. It is a concept inventory, not a QA benchmark — it cannot produce
suppression/enhancement labels. But its concept list could be useful in Steps 3-4
for validating whether our probing directions align with known cultural concepts.

---

## 16. Dataset Fitness Assessment

### The Honest Picture

Sanskriti is usable but has real limitations. Here is the unvarnished assessment.

**What works in our favor:**

1. **Size survives filtering.** Even restricting to Association + General
   Awareness (the most knowledge-requiring types), we have ~10,800 questions. At
   5-8% suppression, that's 500-800+ suppression cases — sufficient for
   activation extraction and probing.

2. **The base/instruct comparison is symmetric.** Both models face the same
   dataset. Since we measure the DIFFERENCE between base and instruct behavior,
   many artifacts cancel. If both models exploit the same patterns, those
   questions land in "control" and do not contaminate suppression/enhancement.

3. **16 attributes and 36 states give slicing dimensions.** Even with coverage
   gaps, we can ask: "Does suppression concentrate in Religion vs Tourism?"
   or "In northeastern states vs metropolitan ones?"

**What is genuinely problematic:**

1. **Template structure conflates format recognition with cultural knowledge.**
   55% of questions follow 7 templates. If our probing picks up "the model
   recognized the Country Prediction template" rather than "the model encodes
   knowledge about Bharatanatyam," we are doing MI of template recognition, not
   cultural knowledge. This must be controlled for in Steps 3-4.

2. **Effective sample size is ~1/3 to 1/4 of raw count.** With 78.6% near-
   duplicate involvement and ~8,156 unique entity keys, behavioral labels
   cluster by entity. Per-attribute rates on smaller categories will have
   wide confidence intervals.

3. **Country Prediction is dead weight for suppression/enhancement analysis.**
   25.6% of the dataset where the answer is always "India" contributes nothing
   to understanding cultural knowledge differences. It inflates
   `control_both_correct`.

4. **Option structure partially encodes the answer.** The embedding baseline
   (Section 8) shows that option-state similarity alone can predict the answer
   well above chance, especially for State Prediction and Country Prediction.
   This is a structural property of how the options were designed. For
   Association and General Awareness (the types we care about most), the
   baseline is lower (63% and 44%), meaning these types genuinely require
   knowledge beyond option structure.

### The Decision: Run Everything, Slice Everything, Report Everything

**We run on all 21,726 usable questions.** We do not filter before evaluation.

**Why:**

1. **More data = more suppression cases.** 8% suppression on 21,726 ≈ 1,700
   cases. On 10,800 ≈ 860 cases. We want every sample for probing.

2. **"Easy" questions make a clean control group.** 5,563 Country Prediction
   questions landing in `control_both_correct` give us a massive, clean control
   set for probing. We WANT a large control population.

3. **Filtering looks like cherry-picking.** Running on everything and showing
   the breakdown by question type is more defensible than discarding 50% of the
   benchmark.

4. **Slicing is stronger than filtering.** Show the full-dataset numbers as
   primary results. Then show: "On Association + General Awareness, suppression
   rises from X% to Y%." This framing demonstrates the effect is MORE
   pronounced on knowledge-requiring questions.

### What the Paper Reports

**Primary results (full dataset, 21,726 questions):**
- Overall accuracy for both models
- Overall suppression/enhancement/control rates
- Per-question-type breakdown

**Robustness checks (sliced, not filtered):**
- Without Country Prediction (16,163 questions)
- Hard subset: Association + General Awareness (10,781 questions)
- Per-attribute rates (top 12 attributes only)
- Per-state rates (top 20 states only)
- Entity-level suppression rate (accounting for redundancy)

**Limitations section:**
- Near-duplicate effective sample size reduction
- Template structure conflating format recognition with cultural knowledge
- Position bias in ground truth (B=29.0%, D=20.8%)
- Sanskriti is the best available benchmark; alternatives are too small or wrong
  format

### Scoping Claims

What we **can** claim:
- "RLHF instruction tuning suppresses X% of cultural knowledge questions that
  the base model answers correctly"
- "Suppression concentrates in [specific attributes/states/types]" (if the data
  shows this)
- "The suppression effect is more pronounced on knowledge-requiring questions
  (Association + General Awareness) than on pattern-matching ones (Country
  Prediction, State Prediction)" (the key finding if it holds)

What we **cannot** claim:
- "RLHF suppresses knowledge about Religion more than Tourism" with high
  confidence (Religion has ~150 effective independent samples)
- Anything about per-state-attribute interactions (6.8% reliable cells)
- That suppression reflects "cultural insensitivity" in RLHF training — we can
  only show the behavioral and representational facts

---

## Appendix A: Numbers Validation Log

Every number in this report was checked against the CSV output files. This
appendix documents the validation for key claims.

### Totals

| Claim | Source | Check |
|-------|--------|-------|
| 21,726 usable rows | `sanskriti_usable.csv` | sum of `distribution_states.csv` counts = 21,726 ✓ |
| 127 excluded | 21,853 - 21,726 = 127 | ✓ |
| 36 states | `distribution_states.csv` has 36 rows | ✓ |
| 16 attributes | `distribution_attributes.csv` has 16 rows | ✓ |
| 4 question types | `distribution_qtypes.csv` has 4 rows | ✓ |

### Position Bias

| Claim | Check |
|-------|-------|
| A=5,885, B=6,308, C=5,008, D=4,525 | `position_bias_overall.csv` ✓ |
| Sum = 21,726 | 5885+6308+5008+4525 = 21,726 ✓ |
| χ² = 363.6 | Manual: Σ(O-5431.5)²/5431.5 = 363.6 ✓ |

### Country Prediction

| Claim | Check |
|-------|-------|
| 5,563 CP questions | `distribution_qtypes.csv` ✓ |
| 100% answer = India | `country_prediction_answers.csv`: 1 row, India, 5563 ✓ |
| 109 unique distractors | `country_prediction_distractors.csv` has 109 rows ✓ |
| India positions: A=1489, B=1614, C=1256, D=1204 | `country_prediction_audit.csv` ✓ |
| Sum: 1489+1614+1256+1204 = 5,563 | ✓ |

### Data Quality

| Claim | Check |
|-------|-------|
| 20,092 unique question texts | `data_quality_summary.csv` ✓ |
| 2,053 exact dup rows in 419 groups | `data_quality_summary.csv` ✓ |
| 351 conflicting dup groups | `data_quality_summary.csv` ✓ |
| 77,833 near-dup pairs | `near_duplicates.csv` (proper CSV parsing) ✓ |
| 17,078 questions in near-dups | `data_quality_summary.csv` ✓ |
| 98.8% same-state in near-dups | 76,926/77,833 (proper CSV parsing) ✓ |
| 1,615 answer-in-question leakage | `answer_in_question_leakage.csv` (proper CSV parsing) ✓ |

### Coverage

| Claim | Check |
|-------|-------|
| 576 total cells | `coverage_state_attribute.csv` has 576 rows ✓ |
| 165 empty | sum of empty=True in CSV ✓ |
| 372 below threshold | sum of below_threshold=True ✓ |
| 39 reliable | sum of reliable=True ✓ |
| 165+372+39 = 576 | ✓ |

### Entity Keys

| Claim | Check |
|-------|-------|
| 4,949 regex entities | `cultural_entities.csv` has 4,949 data rows ✓ |
| 8,156 combined keys | `cultural_entities_combined.csv` has 8,156 data rows ✓ |

### Embedding Baseline

| Claim | Check |
|-------|-------|
| Overall: 75.87% | `no_question_baseline.csv` ✓ |
| State Prediction: 99.98% | `no_question_baseline.csv` ✓ |
| Country Prediction: 95.88% | `no_question_baseline.csv` ✓ |
| Association: 63.27% | `no_question_baseline.csv` ✓ |
| General Awareness: 43.52% | `no_question_baseline.csv` ✓ |

---

*End of document. All numbers verified against CSV output files produced by
`scripts/eda_pipeline.py` on 2026-03-25. Dataset landscape survey conducted
2026-03-25. Report revised 2026-04-04 with full validation pass.*
