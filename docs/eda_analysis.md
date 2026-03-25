# EDA: Complete Analysis

- **Mechanistic Interpretability of Cultural Knowledge in Instruction-Tuned LLMs**
- **Anshul Kumar and Pragati Bhattad — Carnegie Mellon University, March 2026**
- **Target venue: EMNLP 2026 Workshops**

This document records every decision, every number, and every result from the
Exploratory Data Analysis of the Sanskriti benchmark. It is the truth document
for this stage. All numbers are validated against the actual CSV output files
in `/data/user_data/anshulk/cultural-mi/analysis/`.

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

The EDA is the reconnaissance pass before running two 8B-parameter LLMs on the
Sanskriti benchmark. The goal is to understand the dataset well enough to:

1. Correctly interpret behavioral labels (suppression/enhancement) after Step 1
2. Catch dataset artifacts that could produce false labels
3. Know which slices of the data have sufficient statistical power
4. Identify confounds that need to be controlled or reported as limitations

The Sanskriti benchmark (ACL 2025 Findings, arXiv:2506.15355) tests cultural
knowledge about India across 36 states, 16 cultural attributes, and 4 question
types. We use it to compare LLaMA-3.1-8B base vs instruct, labeling each question
as suppression (base correct, instruct wrong), enhancement (base wrong, instruct
correct), or control.

The EDA covers 8 sections: distributions, position bias, Country Prediction audit,
text/lexical analysis, semantic analysis, distractor quality, cultural specificity,
and data quality. Each section produces both plots (PNG) and tabular data (CSV).

### Why EDA Matters for This Project Specifically

This project is not a simple model comparison. We are building a 4-step pipeline
where Step 1's behavioral labels propagate into Steps 2-4. A false suppression label
caused by a dataset artifact will lead us to extract activations for that question
(Step 2), probe those activations for suppression geometry (Step 3), and potentially
claim a circuit is responsible for cultural knowledge suppression (Step 4) — when in
reality the artifact caused the misclassification.

The cost of a false label is not just one wrong data point. It is a downstream
chain of wasted GPU hours and potentially false scientific claims. The EDA exists
to prevent this.

### Relationship to the Sanskriti Paper

The Sanskriti paper (Bari et al., ACL 2025 Findings) evaluated 10 LLMs including
LLaMA-3.1-70B-Instruct (0.86 accuracy) and LLaMA-3.2-3B-Instruct (0.52 accuracy).
Our 8B models should fall between these bounds. The paper reported model-level
accuracies but did not publish per-question predictions or detailed dataset quality
analysis. Our EDA fills this gap — we are the first (to our knowledge) to report
the no-question baseline, near-duplicate rates, and conflicting duplicate counts
for Sanskriti.

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

Column names: `state`, `attribute`, `question`, `option1`, `option2`, `option3`,
`option4`, `answer`, `short explaination / source link`, `question_type`.

Note: the column `short explaination / source link` contains a typo in "explaination."
Code must match this string exactly.

### The 127 Excluded Rows

The `answer` column in these rows does not match any of the four options after
`.strip().lower()` normalization. Manual inspection categorized them as:

| Category | Count | Examples |
|----------|-------|---------|
| Fixable typos (>80% string similarity) | 10 | "Uttarakhand" vs "Uttrakhand", "Dhudni Lake" vs "Dudhni Lake" |
| Substring matches | 6 | "Tarpa festival and Vansda festival" vs option "Tarpa Festival" |
| Truly broken (answer unrelated to options) | 111 | Answer = "Primary ingredient in Dhuska" with options = "rice and lentils", "wheat flour", etc. |

Key finding: **59 of the 111 broken rows come from Karnataka** (rows ~13580-13886),
forming a single bad data batch where the answer column appears to contain question
fragments rather than actual answers.

**Decision:** Exclude all 127. Even the 10 fixable typos carry ambiguity risk. At
0.58% of the dataset, the impact is negligible.

### Why Not Fuzzy-Match the 10 Fixable Typos?

We considered using the 10 rows with >80% string similarity (e.g., "Uttarakhand"
vs "Uttrakhand") by mapping them to the closest option. We decided against this
because: (a) introducing a fuzzy-matching threshold creates a subjective decision
that others cannot reproduce without the exact same logic, (b) 10 rows is 0.046%
of the dataset — statistically invisible, (c) the Sanskriti paper itself does not
document these mismatches, so we cannot verify whether the answer column or the
option column contains the typo.

---

## 3. Data Quality Assessment

*Validated against: `data_quality_summary.csv`, `exact_duplicates.csv`,
`conflicting_duplicates.csv`, `near_duplicates.csv`, `answer_in_question_leakage.csv`*

### Summary Table

| Issue | Count | % of Usable | Severity |
|-------|-------|-------------|----------|
| Unique question texts | 20,092 | 92.5% | — |
| Exact duplicate rows | 2,053 | 9.45% | Medium |
| Exact duplicate groups | 419 | — | — |
| **Conflicting duplicates** (same question, different correct answer) | **351 groups** | — | **High** |
| Near-duplicate question pairs (cosine sim > 0.85) | 77,833 pairs | — | Medium |
| Questions involved in near-duplicates | 17,078 | 78.6% | High |
| Answer text appears in question text | 1,615 | 7.43% | Medium |
| Answer text appears in source URL column | 5,082 | 23.4% | Low (not model input) |

### Exact Duplicates: 2,053 Rows in 419 Groups

9.45% of usable rows are exact textual duplicates. These inflate the dataset
without adding information. A model that knows one instance gets credit multiple times.

### Conflicting Duplicates: 351 Groups

This is the most serious data quality issue. **351 question groups** have the
exact same question text but different correct answers across instances. Examples:

| Question (truncated) | Instances | Answer letters |
|----------------------|-----------|----------------|
| "According to you, which ... closely associated to Agartala of Tripura?" | 6 | C, D, B |
| "According to you, which ... closely associated to Agra of Uttar_Pradesh?" | 4 | A, B |
| "According to you, which ... closely associated to Ahmedabad of Gujarat?" | 3 | A, C |

These arise because the "closely associated to {region}" template has different
distractor options across instances, so the position of the correct answer changes.
The question text is identical but the option set differs. **This is not a bug in
the data per se** — the ground truth letter is correctly matched to each row's
own options. However, it means the same surface question tests different knowledge
depending on which distractors are presented.

**Implication for Step 1:** These are not a problem for evaluation (each row has its
own ground truth letter mapped to its own options). But they contribute to the
near-duplicate inflation — the model sees essentially the same question multiple
times with different answer positions.

### Near-Duplicates: 77,833 Pairs

Using sentence embeddings (all-MiniLM-L6-v2, 384-dim), 77,833 question pairs
have cosine similarity > 0.85. The distribution by threshold:

| Threshold | Pairs | Unique questions involved |
|-----------|-------|--------------------------|
| > 0.85 | 77,833 | 17,078 (78.6%) |
| > 0.90 | 43,089 | — |
| > 0.95 | 19,122 | — |
| > 0.99 | 16,700 | — |

**98.8% of near-duplicate pairs are from the same state.** This is expected: the
templated question structure means questions about the same state with different
cultural entities are semantically very similar ("Which state is famous for X?" vs
"Which state is famous for Y?" where both X and Y are from Karnataka).

**78.6% of all questions are involved in at least one near-duplicate pair.** This
is extremely high. It means the effective information content of the 21,726 questions
is substantially less than it appears. A model's "accuracy" is inflated by repeated
testing of similar knowledge.

**Implication for Step 1:** When computing per-state or per-attribute behavioral
label rates, the effective sample size is smaller than the raw question count.
Suppression/enhancement rates on small slices should be interpreted with even
more caution than the raw n suggests.

### Answer-in-Question Leakage: 1,615 Questions (7.43%)

In 1,615 questions, the correct answer text appears verbatim within the question
text itself. Breakdown by question type:

| Question Type | Leakage Count | % of Type |
|---------------|---------------|-----------|
| Association | 1,194 | 21.9% |
| State Prediction | 292 | 5.4% |
| Country Prediction | 69 | 1.2% |
| General Awareness | 60 | 1.1% |

Association questions dominate because their format is "Where is {entity} famous
within {state}?" with the entity name often being the correct answer. Example:
"Where is the Nicobari pig-farming customs famous within Andaman_and_Nicobar?"
Answer: "Nicobari pig-farming customs."

**Implication for Step 1:** These questions are trivially solvable by string
matching. Both models should get them right, pushing them into `control_both_correct`.
They dilute the suppression/enhancement signal but do not create false labels.

### The Compound Effect

These quality issues do not exist in isolation. A single question can be:
- An exact duplicate (9.45%)
- Involved in near-duplicate pairs (78.6%)
- Leaking the answer in the question text (7.43%)
- From a conflicting duplicate group (where the "same" question has a different
  answer elsewhere in the dataset)

The overlap between these categories means the "clean" core of the dataset —
questions that are unique, non-leaking, and non-conflicting — is substantially
smaller than 21,726. We estimate the effective information content is closer to
8,156 unique cultural entity keys (see Section 10), each tested from 1-3
angles via templates.

For an EMNLP workshop paper, this is not a dealbreaker. The Sanskriti benchmark
is the only large-scale Indian cultural MCQ dataset available. We use it with full
awareness of its limitations and report all metrics in ways that account for them.

---

## 4. Distribution Analysis

*Validated against: `distribution_states.csv`, `distribution_attributes.csv`,
`distribution_qtypes.csv`, `coverage_state_attribute.csv`, `state_summary.csv`*

### States: 36 Unique, Highly Imbalanced

| Rank | State | Count | % |
|------|-------|-------|---|
| 1 | Telangana | 1,705 | 7.85% |
| 2 | Karnataka | 1,391 | 6.40% |
| 3 | Andhra_Pradesh | 1,127 | 5.19% |
| 4 | Delhi | 1,076 | 4.95% |
| 5 | Arunachal_Pradesh | 1,023 | 4.71% |
| ... | ... | ... | ... |
| 32 | Maharashtra | 283 | 1.30% |
| 33 | Ladakh | 278 | 1.28% |
| 34 | Meghalaya | 267 | 1.23% |
| 35 | Mizoram | 210 | 0.97% |
| 36 | Lakshadweep | 122 | 0.56% |

Range: **14.0x** (Telangana 1,705 vs Lakshadweep 122).

The top 5 states account for 29.1% of all questions. The bottom 5 account for
5.3%. Southern and northeastern states are overrepresented relative to major
northern states — notably, Maharashtra (India's second most populous state) has
only 283 questions (1.3%), less than Sikkim (683, 3.1%).

### Regional Grouping

Grouping by Indian geographic region reveals the annotation bias more clearly:

| Region | States | Questions | % of Dataset |
|--------|--------|-----------|-------------|
| North (incl. Himalayan) | 10 | 6,005 | 27.6% |
| South | 6 | 5,809 | 26.7% |
| Northeast | 8 | 4,475 | 20.6% |
| West + Central | 6 | 2,805 | 12.9% |
| East | 4 | 2,152 | 9.9% |

The Northeast is heavily overrepresented (20.6% of questions for 8 states that
contain ~4% of India's population). This likely reflects the Sanskriti paper's
explicit goal of covering underrepresented regions. For our MI study, this is
actually beneficial — northeastern states have the most obscure cultural knowledge
(less likely in pretraining data), which is where suppression/enhancement effects
should be most visible.

The East (West Bengal, Bihar, Odisha, Jharkhand) is underrepresented at 9.9%.
Bihar, India's third most populous state, has only 367 questions (1.69%).

**Implication for Step 1:** Per-state suppression rates will have much more
statistical power for South/Northeast states than for East/West. If we want to
claim "suppression correlates with state obscurity," we need to control for
sample size differences.

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

Range: **92.7x** (Tourism 3,801 vs Nightlife 41).

Four attributes have fewer than 200 questions and are flagged as sparse: Sports
(162), Transport (76), Medicine (72), Nightlife (41). Per-attribute behavioral
label rates for these four will be unreliable. At an expected 5-10% suppression
rate, Nightlife would produce 2-4 suppression cases — statistically meaningless.

**Recommendation:** Group these 4 sparse attributes into an "Other" category for
per-attribute behavioral analysis, or report them with explicit uncertainty warnings.

### Question Types: 4, Roughly Balanced

| Question Type | Count | % |
|---------------|-------|---|
| Country Prediction | 5,563 | 25.61% |
| Association | 5,453 | 25.10% |
| State Prediction | 5,382 | 24.77% |
| General Awareness | 5,328 | 24.52% |

All four types are within 1.1 percentage points of a perfect 25% split. This is
the one dimension where the dataset is well-balanced.

### Coverage: State x Attribute Matrix

The 36x16 grid has **576 total cells.** Of these:

| Category | Count | % of cells |
|----------|-------|------------|
| Empty (zero questions) | 165 | 28.6% |
| Non-zero but below threshold (<125 questions) | 372 | 64.6% |
| Reliable (>=125 questions) | **39** | **6.8%** |

**Only 6.8% of state-attribute combinations have enough data for reliable per-cell
behavioral analysis.** The 165 empty cells mean we cannot say anything about those
combinations at all. For example, most states have zero questions for Nightlife,
Transport, and Medicine.

**Implication for Step 1:** Per-state-attribute breakdowns of suppression/enhancement
are infeasible for the vast majority of cells. Report behavioral labels at the
state level (aggregated across attributes) or the attribute level (aggregated across
states), never at the intersection unless the cell count exceeds 125.

### What the Coverage Gap Means Practically

Consider a specific example. Suppose we find that suppression is higher for
"Cuisine" than "Tourism." We can report this at the attribute level (both have
1,671 and 3,801 questions respectively — sufficient). But we cannot say "Cuisine
suppression is concentrated in northeastern states" because the Cuisine × Northeast
cells are mostly below 125 questions. The coverage gap forces us to report
**marginal** effects (per-state OR per-attribute), never **conditional** effects
(per-state AND per-attribute), for most of the dataset.

The 39 reliable cells (>=125 questions) are concentrated in the top-left of the
heatmap: large states × large attributes. For example:
- Telangana × Tourism (268 questions) — reliable
- Karnataka × History (193 questions) — reliable
- Lakshadweep × Nightlife (0 questions) — impossible

---

## 5. Answer Position Bias

*Validated against: `position_bias_overall.csv`, `position_bias_by_qtype.csv`,
`position_bias_by_attribute.csv`, `position_bias_by_state.csv`*

### Overall Distribution

| Letter | Count | Observed % | Expected (uniform) | Deviation |
|--------|-------|------------|-------------------|-----------|
| A | 5,885 | 27.09% | 25.00% | +2.09pp |
| B | 6,308 | 29.03% | 25.00% | +4.03pp |
| C | 5,008 | 23.05% | 25.00% | -1.95pp |
| D | 4,525 | 20.83% | 25.00% | -4.17pp |

Chi-squared test of uniformity: **χ² = 363.6, p = 1.68e-78.** The distribution
is statistically non-uniform — B is overrepresented, D is underrepresented.

The practical significance is moderate. The maximum deviation is 4.17pp from the
expected 25%. If a model has a tendency to predict B (a common LLM position bias),
it would gain ~4 percentage points of spurious accuracy. This is a confound but
not a fatal one — we report it and check for it in Step 1's option distribution
sanity check.

### Per Question Type

| Question Type | A% | B% | C% | D% | χ² | p-value | Verdict |
|---------------|-----|-----|-----|-----|-----|---------|---------|
| Association | 25.14 | 28.13 | 23.34 | 23.38 | 33.1 | 3.0e-07 | Mild B-bias |
| Country Prediction | 26.77 | 29.01 | 22.58 | 21.64 | 80.9 | 2.0e-17 | Moderate B-bias |
| General Awareness | **31.85** | 29.95 | 21.73 | **16.46** | **330.5** | **2.5e-71** | **Severe A/D skew** |
| State Prediction | 24.67 | 29.06 | 24.54 | 21.72 | 59.3 | 8.3e-13 | Moderate B-bias |

**General Awareness has the most extreme skew:** 31.85% A vs 16.46% D — nearly
2:1 ratio. This means a model that blindly predicts A on General Awareness questions
would get 31.85% correct vs the 25% random baseline. The chi-squared value (330.5)
is an order of magnitude larger than the other types.

**Country Prediction's India position:** India appears at A=26.8%, B=29.0%,
C=22.6%, D=21.6%. This mirrors the overall ground truth distribution exactly,
confirming that the position skew is global, not specific to how India is placed.

### Per State: Extreme Cases

Some states have very skewed position distributions:

| State | Max letter % | Min letter % | Spread | n |
|-------|-------------|-------------|--------|---|
| Meghalaya | 37.8% | 11.6% | 26.2pp | 267 |
| Madhya_Pradesh | 33.5% | 8.6% | 24.9pp | 546 |
| Chhattisgarh | 33.6% | 9.2% | 24.4pp | 631 |
| Odisha | 33.5% | 9.7% | 23.9pp | 465 |
| Rajasthan | 32.5% | 10.1% | 22.4pp | 483 |

These states have positional spreads exceeding 22pp — a model with any position
bias will have systematically different accuracy across these states. This is a
confound for per-state behavioral analysis.

**Implication for Step 1:** If the option distribution check (sanity check F3)
reveals significant model position bias, per-state behavioral labels for the
states above should be interpreted with extra caution.

### Position Bias by Attribute

Most attributes follow the global B-heavy pattern, but some deviate:

| Attribute | A% | B% | C% | D% | Pattern |
|-----------|-----|-----|-----|-----|---------|
| Tourism | 25.8 | 27.8 | 23.5 | 23.0 | Mild B-bias (near-uniform) |
| History | 27.4 | 28.8 | 22.2 | 21.6 | Moderate B-bias, low D |
| Cuisine | 27.0 | 28.1 | 24.1 | 20.9 | Moderate, low D |
| Religion | 28.0 | 23.9 | 24.3 | 23.9 | A-heavy, B not dominant |

Religion is the only attribute where A is the most common position and B is
not dominant. If a model has strong B-bias, it will underperform on Religion
specifically — not because Religion knowledge is harder, but because the
position distribution differs. This is a confound that could masquerade as
an attribute-specific suppression effect.

### Interaction Between Position Bias and Our Behavioral Labels

Consider this scenario: the base model has no position bias (predicts A/B/C/D
roughly uniformly) but the instruct model has a B-bias (predicts B 35% of the
time). Since ground truth B is 29.0%, the instruct model would gain ~4pp of
spurious accuracy overall. But on General Awareness (where D is only 16.5%),
the instruct model loses the least from avoiding D. This asymmetry could create
false "enhancement" on General Awareness and false "suppression" on question
types where D is the correct answer more often.

**Step 1 must check both models' prediction distributions before interpreting
behavioral labels.** If either model shows significant position bias, we should
consider a position-bias-corrected accuracy as a robustness check.

Specifically, the Step 1 script must compute:
1. Each model's prediction distribution (% of A/B/C/D predictions overall)
2. Prediction distribution per question type (especially General Awareness)
3. Chi-squared test of each model's predictions against uniform distribution
4. If either model has >5pp deviation from uniform on any letter: flag it and
   compute what the accuracy would be if predictions were redistributed to
   match the ground truth distribution (upper bound on position-bias-free accuracy)

---

## 6. Country Prediction Audit

*Validated against: `country_prediction_audit.csv`, `country_prediction_answers.csv`,
`country_prediction_distractors.csv`, `country_prediction_by_attribute.csv`*

### The Core Finding

**All 5,563 Country Prediction answers (100.0%) are "India."**

This means 25.6% of the entire dataset is a single question pattern: "Is this
cultural element from India?" with the answer always being yes. The distractors
are foreign countries that are obviously wrong:

| Distractor | Count | % of all distractors |
|------------|-------|---------------------|
| Japan | 1,726 | 10.34% |
| Brazil | 1,712 | 10.26% |
| Italy | 1,671 | 10.01% |
| Canada | 1,668 | 9.99% |
| China | 1,644 | 9.85% |
| France | 1,634 | 9.79% |
| USA | 1,630 | 9.77% |
| UK | 1,621 | 9.71% |
| Germany | 1,576 | 9.44% |

The top 9 distractors (all G7 + China + Brazil) account for **89.2% of all
distractor slots.** The remaining 100 distractors (Bangladesh, Nepal, Pakistan,
etc.) each appear in fewer than 1% of questions.

### Attribute Distribution within Country Prediction

Country Prediction questions are uniformly distributed across attributes, with
each attribute contributing 21-32% of its total questions to this type. This
is by design in the Sanskriti benchmark.

### What This Means

Both models should score near 100% on Country Prediction. The question "Is
Bharatanatyam from India or Japan?" does not require cultural knowledge — it
requires only knowing that Bharatanatyam is Indian, which any model trained on
English text will know.

**These questions will almost entirely fall into `control_both_correct`,** inflating
the overall accuracy of both models and diluting the suppression/enhancement signal.

**Recommendation for Step 1:** Report all behavioral metrics **both with and without
Country Prediction questions.** The "without CP" numbers will be more informative
for cultural knowledge suppression/enhancement. Expected impact: removing 5,563
questions (~25.6%) will lower both models' overall accuracy by several percentage
points and increase the relative suppression/enhancement rates.

### Quantifying the Country Prediction Inflation

To understand the inflation effect, consider two scenarios:

**Scenario A (with CP):** If base gets 60% overall and instruct gets 70%, and
both get 98% on CP (5,563 questions), then on non-CP questions (16,163):
- Base non-CP accuracy: (0.60 × 21,726 - 0.98 × 5,563) / 16,163 = ~47%
- Instruct non-CP accuracy: (0.70 × 21,726 - 0.98 × 5,563) / 16,163 = ~60%

The gap goes from 10pp overall to 13pp on the "hard" subset. More importantly,
the suppression/enhancement rates change:
- With CP: suppression ~6% (1,300 questions, many from CP)
- Without CP: suppression rate on 16,163 questions would be higher in percentage
  terms because the denominator shrinks while the number of actual suppression
  events stays roughly the same.

**We will report a three-tier analysis:**
1. Full dataset (21,726 questions) — for comparability with the Sanskriti paper
2. Without Country Prediction (16,163 questions) — for cultural knowledge signal
3. Hard subset only: Association + General Awareness (10,781 questions) — for
   questions that genuinely require cultural knowledge and cannot be solved by
   string matching

### The Distractor Problem in Country Prediction

The 109 unique distractors break down into two tiers:

**Tier 1 (89.2% of slots):** Japan, Brazil, Italy, Canada, China, France, USA,
UK, Germany — each appearing 1,576-1,726 times. These are obviously wrong for
any cultural entity in the dataset. No Indian cultural element is from Japan.

**Tier 2 (10.8% of slots):** Geographically proximate countries — Bangladesh (164),
Nepal (145), Pakistan (112), Thailand (99), Sri Lanka (96). These are marginally
more plausible distractors — a model would need to know that Bharatanatyam is from
India, not Sri Lanka. But they still appear in <1% of questions each, so even if
models occasionally confuse India with Nepal, the impact is negligible.

There are zero South Asian neighbors (Bangladesh, Nepal, Pakistan, Sri Lanka)
among the top 9 distractors. The benchmark does not test whether a model can
distinguish Indian culture from other South Asian cultures — it tests whether
the model can distinguish India from Japan, Brazil, or Germany. This is a
fundamentally different (and much easier) task.

---

## 7. Text and Lexical Analysis

*Validated against: `question_length_by_qtype.csv`, `question_templates.csv`,
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

General Awareness questions are the longest (mean 12.1 words), Country Prediction
the shortest (9.6 words). The difference is modest — all types fall in the 9-13
word range.

### Question Templates

**55.4% of questions follow one of 7 templates:**

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

Six templates each appear exactly **1,607 times** — this is suspiciously uniform
and suggests a systematic generation process. Each cultural entity was fed through
each template to produce the dataset. This explains the high near-duplicate rate:
the same entity appears in 3-4 templates, producing semantically near-identical
questions.

The 44.62% in "_other_" includes the General Awareness questions (which are often
free-form statements rather than templated questions) and less common patterns.

### Option Length Bias

| Category | Mean words | Median | Std |
|----------|-----------|--------|-----|
| Correct option | 1.471 | 1.0 | 1.255 |
| Incorrect option | 1.595 | 1.0 | 1.405 |

Incorrect options are slightly longer than correct ones. Cohen's d = **-0.094** —
negligible effect size. The t-test is significant (p = 4.0e-31) only because of
the large sample size.

Mean option length by position:

| Position | Mean words |
|----------|-----------|
| A | 1.559 |
| B | 1.575 |
| C | 1.561 |
| D | 1.561 |

All four positions have virtually identical mean option lengths (spread < 0.02
words). There is no length-based position shortcut.

**Verdict:** Option length is not a usable shortcut for either humans or models.

### Option Length by Question Type: A Closer Look

While the overall effect is negligible, the breakdown per question type reveals
a structural pattern:

| Question Type | Correct Mean | Incorrect Mean | Interpretation |
|---------------|-------------|---------------|----------------|
| Country Prediction | 1.000 | 1.009 | All options are single-word country names |
| State Prediction | 0.773 | 0.778 | All options are state names (some hyphenated) |
| Association | 1.829 | 2.444 | **Incorrect options are 34% longer** |
| General Awareness | 2.300 | 2.163 | **Correct options are 6% longer** |

Country Prediction and State Prediction have essentially identical option lengths
(all options are single-word proper nouns). The interesting asymmetry is in
Association: incorrect options average 2.44 words vs correct options at 1.83 words.
This suggests that Association distractors include more descriptive phrases while
correct answers are shorter entity names. A model that learns to pick shorter
options on Association questions would gain a small accuracy boost — but this is
a very weak signal (the distributions overlap heavily).

General Awareness reverses the pattern: correct options are slightly longer (2.30
vs 2.16 words). The effects largely cancel across question types, which is why the
overall Cohen's d is only -0.094.

### Dominant Vocabulary

The top 15 unigrams (excluding stopwords) in question text reveal how template-
dominated the dataset is:

| Rank | Word | Count | Source |
|------|------|-------|--------|
| 1 | associated | 6,554 | "closely associated to" template |
| 2 | country | 5,515 | "which country" template |
| 3 | famous | 5,080 | "famous for" / "famous within" templates |
| 4 | home | 3,290 | "home to the" template |
| 5 | states | 3,240 | "which of the states" template |
| 6 | state | 2,819 | "which state" template |
| 7 | region | 1,955 | "which region" template |
| 8 | belongs | 1,662 | "belongs to which" pattern |
| 9 | regions | 1,658 | "given regions" pattern |
| 10 | within | 1,623 | "famous within" template |
| 11 | closely | 1,617 | "closely associated" template |
| 12 | houses | 1,611 | "which of the states houses" pattern |
| 13 | options | 1,610 | "given in the options" pattern |
| 14 | festival | 1,179 | First genuinely cultural word |
| 15 | dance | 921 | Second genuinely cultural word |

The first 13 words are all template scaffolding. "Festival" at rank 14 is the
first word that carries cultural content. "Dance" at rank 15 is the second. This
quantifies just how template-heavy the dataset is — the cultural signal is
embedded within a thick layer of formulaic text.

The top 10 bigrams are even more revealing: every single one is a template fragment.
"state famous" (1,657), "associated country" (1,616), "country famous" (1,614).
No culturally meaningful bigram appears in the top 10.

**Implication for model evaluation:** The LLaMA models will process many tokens of
template boilerplate before reaching the cultural entity name. The base model must
parse this structure through pure next-token prediction; the instruct model was
trained on conversational templates. This structural familiarity may be a confound —
the instruct model may perform better not because it knows more culture, but because
it handles the question format better.

### Lexical Diversity (MTLD)

MTLD (Measure of Textual Lexical Diversity) measures vocabulary richness. Higher
MTLD = more diverse vocabulary, less formulaic text. It is insensitive to text
length, unlike raw type-token ratio.

**By question type:**

| Question Type | MTLD | Interpretation |
|---------------|------|----------------|
| General Awareness | 31.66 | Most diverse — free-form statements |
| State Prediction | 25.93 | Moderate — templated but varied entities |
| Association | 25.04 | Moderate |
| Country Prediction | **17.75** | **Least diverse — extremely formulaic** |

Country Prediction's MTLD of 17.75 is 44% lower than General Awareness (31.66).
This confirms what the template analysis showed: Country Prediction is mechanically
generated text with minimal vocabulary variation.

**By attribute (top 5 / bottom 5):**

| Attribute | MTLD | n |
|-----------|------|---|
| Rituals_and_Ceremonies | 36.36 | 1,000 |
| Language | 29.33 | 900 |
| Cultural_Common_Sense | 29.25 | 2,091 |
| Transport | 25.97 | 76 |
| Costume | 25.72 | 1,513 |
| ... | ... | ... |
| Religion | 20.68 | 482 |
| Personalities | 19.79 | 983 |
| Sports | 18.59 | 162 |
| Nightlife | 18.48 | 41 |

Rituals_and_Ceremonies has the richest vocabulary (MTLD 36.36), nearly 2x that
of Nightlife (18.48). The sparse attributes (Sports, Medicine, Nightlife) tend
to have low MTLD, likely because their small size limits vocabulary breadth. But
Personalities (983 questions) also has low MTLD (19.79), suggesting genuinely
formulaic question construction for that attribute.

### N-gram Diversity

The N-gram Diversity Score (NGD) computes `unique_ngrams / total_ngrams` averaged
over n=1 to n=4. Higher = more diverse.

Top 3: Rituals_and_Ceremonies (0.477), Transport (0.474), Language (0.414)
Bottom 3: Art (0.266), Tourism (0.255), Personalities (0.220)

Personalities has the lowest NGD (0.220), confirming it as the most repetitive
attribute by vocabulary. Tourism, despite being the largest attribute (3,801
questions), has the second-lowest NGD (0.255) — its sheer size makes repeated
phrases even more common.

### Lexical Diversity by State

State-level MTLD reveals which regions have the most varied question phrasing:

**Highest MTLD (most diverse):**

| State | MTLD | n |
|-------|------|---|
| Chhattisgarh | 35.47 | 631 |
| Andaman_and_Nicobar | 34.33 | 338 |
| Arunachal_Pradesh | 33.40 | 1,023 |
| Uttarakhand | 32.74 | 910 |
| Rajasthan | 31.88 | 483 |

**Lowest MTLD (most formulaic):**

| State | MTLD | n |
|-------|------|---|
| Tamil_Nadu | 18.98 | 741 |
| Chandigarh | 19.14 | 323 |
| Delhi | 19.17 | 1,076 |
| Maharashtra | 19.25 | 283 |
| Mizoram | 19.34 | 210 |

The diversity difference is nearly 2:1 (Chhattisgarh at 35.47 vs Tamil Nadu at
18.98). Delhi's low MTLD (19.17) is notable because it has 1,076 questions — this
is not a small-sample effect. The Delhi questions are genuinely formulaic, likely
because Delhi has many well-known landmarks (India Gate, Red Fort, Qutub Minar)
that slot easily into templates without requiring varied phrasing.

**Implication for MI:** States with low MTLD (formulaic questions) may be easier
for the model to pattern-match, regardless of cultural knowledge. If we see
lower suppression rates for Delhi vs Chhattisgarh, it could be a template
familiarity effect rather than a cultural knowledge effect.

---

## 8. Semantic Analysis

*Validated against: `umap_coordinates.csv`, `near_duplicates.csv`,
`no_question_baseline.csv`, `no_question_baseline_by_attribute.csv`,
`qa_overlap_by_qtype.csv`, `tfidf_terms_per_state.csv`, `bertopic_topics.csv`,
`bertopic_assignments.csv`*

### UMAP Visualization

All 21,726 questions were embedded using `all-MiniLM-L6-v2` (384 dimensions,
normalized) and projected to 2D via UMAP (n_neighbors=30, min_dist=0.3,
metric=cosine, random_state=42). See `eda_09_umap.png`.

**By question type:** Country Prediction and State Prediction form distinct
clusters in UMAP space. Association and General Awareness overlap significantly.
This reflects the structural difference: CP/SP use templated question formats
that are semantically distinct, while Association and General Awareness share
more varied phrasing.

**By attribute:** No clear attribute-based clusters. Tourism, History, and
Festivals questions are scattered across the same regions. This suggests
attributes are more of a metadata label than a semantic differentiator — the
model sees "Which state is famous for {festival}?" and "Which state is famous
for {tourism site}?" as structurally identical.

**By state:** Top states (Telangana, Karnataka) form weak subclusters within
the larger question-type clusters. Smaller states are diffusely scattered.

### No-Question Baseline: 75.87% Accuracy

**This is the single most important finding of the EDA.**

The "no-question baseline" measures whether the correct answer can be guessed by
comparing only the state name (embedded) against the four option texts (embedded),
without reading the question at all. For each question, the option with highest
cosine similarity to the state name is selected as the prediction.

| Slice | No-Question Accuracy | Random Baseline |
|-------|---------------------|-----------------|
| **Overall** | **75.87%** | 25.00% |
| State Prediction | **99.98%** | 25.00% |
| Country Prediction | 95.88% | 25.00% |
| Association | 63.27% | 25.00% |
| General Awareness | 43.52% | 25.00% |

**State Prediction is 99.98% solvable without reading the question.** This is
because the correct option IS the state name (or a close variant), and the
distractors are other states. The state name embedding trivially matches itself.

Country Prediction is 95.88% solvable because "India" is semantically distinct
from all distractors (Japan, Brazil, etc.).

Association is 63.27% solvable — the region/district names in the correct option
often embed close to the state name.

General Awareness is the least exploitable at 43.52%, but this is still well
above the 25% random baseline.

**By attribute:** All attributes exceed 25%. The range is 67.1% (Transport) to
85.4% (Nightlife), though sparse attributes have high variance.

| Attribute | No-Question Accuracy | n |
|-----------|---------------------|---|
| Nightlife | 85.37% | 41 |
| Rituals_and_Ceremonies | 80.10% | 1,000 |
| Cuisine | 79.89% | 1,671 |
| Language | 79.33% | 900 |
| Cultural_Common_Sense | 79.24% | 2,091 |
| ... | ... | ... |
| Religion | 69.29% | 482 |
| Transport | 67.11% | 76 |

**Implication for Step 1:** A substantial fraction of "correct" answers from either
model may reflect option-state name matching rather than cultural knowledge.
Suppression/enhancement labels on State Prediction questions are particularly
suspect — if both models use this shortcut, the questions test nothing about
cultural knowledge.

**Recommendation:** Flag State Prediction and Country Prediction questions as
"shortcut-vulnerable" and report behavioral labels separately for the "hard"
types (Association + General Awareness, 10,781 questions).

### Question-Answer Semantic Overlap

Cosine similarity between each question embedding and its correct answer embedding:

| Question Type | Mean | Median | Std |
|---------------|------|--------|-----|
| Association | 0.3864 | 0.3558 | 0.1793 |
| State Prediction | 0.3257 | 0.3077 | 0.1356 |
| Country Prediction | 0.3126 | 0.3149 | 0.0887 |
| General Awareness | 0.2490 | 0.2151 | 0.1596 |

Association questions have the highest Q-A overlap (mean 0.386) because the
question often contains the answer entity. General Awareness has the lowest
(0.249), meaning these questions are the most genuinely "knowledge-requiring."

385 questions have Q-A similarity > 0.7 (answer practically restates the question).
4,704 questions have Q-A similarity < 0.2 (answer requires external knowledge).

**By attribute:** Q-A overlap varies modestly across attributes:

| Attribute | Mean Q-A Sim | Interpretation |
|-----------|-------------|----------------|
| Sports | 0.355 | Highest — sports entity names repeat in Q and A |
| Costume | 0.341 | High — costume names are self-referential |
| Language | 0.339 | |
| Cuisine | 0.336 | |
| ... | ... | |
| Dance_and_Music | 0.302 | |
| Rituals_and_Ceremonies | 0.270 | Low — questions describe rituals indirectly |
| Transport | 0.254 | Lowest — transport questions are most indirect |

The range is modest (0.254 to 0.355). No attribute has dramatically higher or
lower Q-A overlap. The 4,704 low-overlap questions (sim < 0.2) are distributed
across all attributes and represent the genuinely hard subset where the answer
cannot be guessed from the question text alone.

### TF-IDF Distinctive Terms Per State

For each state, all questions were concatenated into a single document, and TF-IDF
identified the most distinctive terms. Examples (top 3 terms per state):

| State | Top TF-IDF terms |
|-------|-----------------|
| Goa | goa, portuguese, region goa |
| Assam | assam, majuli, famous assam |
| Chhattisgarh | chhattisgarh, bastar, district chhattisgarh |
| Delhi | delhi, famous delhi, region delhi |

Most states' top TF-IDF term is simply the state name itself, confirming that
the templated question format ("... famous within {state}?") dominates. The
second-ranked terms are more informative: "portuguese" for Goa, "bastar" for
Chhattisgarh, "majuli" for Assam — these reflect genuine cultural associations.

**Extended TF-IDF analysis for selected states:**

| State | Top distinctive terms (excluding state name) |
|-------|---------------------------------------------|
| Goa | portuguese (0.130), panaji, konkan, feni |
| Telangana | hyderabad (0.255), charminar, biryani, warangal |
| Kerala | backwaters, kathakali, onam, ayurveda |
| Lakshadweep | minicoy (0.214), kavaratti (0.167), lava (0.119) |
| Manipur | thang-ta, ningol, ima market, kangla |
| Assam | majuli (0.169), bihu, one-horned rhinoceros |
| Rajasthan | thar, rajput, havelis, block printing |

Lakshadweep has the most culturally specific vocabulary — "minicoy" and "kavaratti"
are island names that appear almost nowhere else in LLM pretraining data. If any
state's cultural knowledge is suppressed by RLHF, Lakshadweep is the prime
candidate due to its low pretraining data coverage. However, it has only 122
usable questions — the smallest of any state — limiting statistical power.

### BERTopic: Discovered Topics

BERTopic (embedding_model=None, using pre-computed MiniLM embeddings, nr_topics=20,
min_topic_size=100) discovered 19 topics plus an outlier cluster:

| Topic | Count | Key Terms |
|-------|-------|-----------|
| -1 (outlier) | 5,809 | to, which, is, the, of |
| 0 | 4,846 | festival, dance, culture, in |
| 1 | 3,878 | the, which, to, is, states |
| 2 | 1,494 | telangana, andhrapradesh, karnataka |
| 3 | 1,369 | temple, fort, famous |
| 4 | 1,009 | haryana, jharkhand, rajasthan |
| 5 | 608 | cuisine, dishes, rice |
| 6 | 517 | lake, beach, basava |
| 7 | 458 | park, sanctuary, wildlife |
| 8 | 316 | uttarakhand, ladakh, garhwal |
| ... | ... | ... |

**Key observation:** BERTopic's topics are dominated by geography (Topics 2, 4, 8
are state clusters) and tourism entities (Topics 3, 6, 7 are temples/lakes/parks),
not by the 16 predefined attributes. The predefined attribute "Tourism" fragments
across Topics 3, 6, 7, and others. This confirms that the dataset's semantic
structure is organized primarily by **geography**, not by **cultural category.**

**26.7% of questions (5,809) fall into the outlier cluster** — these are too
generic to assign to any topic, mostly consisting of the templated question
boilerplate ("According to you, which of the following...").

### BERTopic-Attribute Alignment

How well do the discovered topics map onto the 16 predefined attributes?

**Topic 5 (cuisine cluster, 608 questions):** 76.5% of its questions are labeled
as the Cuisine attribute. This is the best-aligned topic — the cuisine vocabulary
is distinctive enough to form a clean cluster (rice, dishes, lentils, spices).

**Topic 3 (temples/forts, 1,369 questions):** Fragments across Tourism (34.7%),
History (22.1%), Cultural_Common_Sense (16.2%), and Religion (11.5%). The predefined
attribute boundary between Tourism and History does not correspond to a semantic
boundary in the question text — a temple question is a temple question whether it's
labeled Tourism or History.

This misalignment has implications for our MI study. If we find "Tourism suppression
is 8% but History suppression is 12%," the difference might not reflect different
cultural domains being treated differently by RLHF — it might reflect that the
"temple/fort" questions (which straddle both labels) are being split unevenly by
an arbitrary attribute boundary. BERTopic gives us an independent semantic
clustering to cross-validate attribute-level findings.

### The Three Difficulty Tiers (Pre-Model Estimate)

Combining all semantic and structural signals, we can rank question difficulty
before any model evaluation:

**Tier 1 — Trivial (11,008 questions, 50.7%):**
- All Country Prediction (5,563) — answer is always India
- State Prediction questions where answer appears in question (292)
- Remaining State Prediction with 99.98% no-question baseline (5,090 + 63 overlap)

**Tier 2 — Moderate (5,937 questions, 27.3%):**
- Association questions with answer-in-question leakage (1,194)
- Association questions solvable by no-question baseline but not leakage (~2,260)
- General Awareness questions solvable by no-question baseline (~2,483)

**Tier 3 — Genuinely Hard (4,781 questions, 22.0%):**
- General Awareness questions NOT solvable by no-question baseline (~2,845)
- Association questions NOT solvable by either shortcut (~1,936)

The interesting behavioral labels will concentrate in Tier 3. If suppression/
enhancement is uniformly distributed across tiers, something is likely wrong
with our evaluation (the models are using shortcuts rather than knowledge). If
suppression concentrates in Tier 3 as expected, we have genuine cultural knowledge
effects.

---

## 9. Distractor Quality

*Validated against: `distractor_similarity.csv`, `distractor_similarity_by_qtype.csv`,
`distractor_similarity_by_attribute.csv`, `distractor_quality_summary.csv`,
`answer_in_question_leakage.csv`*

### Correct-Distractor Embedding Similarity

For each question, we compute the cosine similarity between the correct answer's
embedding and each of the 3 distractors' embeddings. Higher similarity = more
plausible distractors = harder question.

| Question Type | Mean Sim | Interpretation |
|---------------|----------|----------------|
| Country Prediction | **0.631** | High — country names are semantically similar to each other |
| State Prediction | 0.444 | Moderate — state names are somewhat similar |
| Association | 0.352 | Lower — region/entity names are more varied |
| General Awareness | **0.282** | Lowest — diverse option types |

**Country Prediction has the highest distractor similarity (0.631)** — paradoxically,
this makes it look "hard" by this metric, when in reality the answer ("India") is
trivially different from distractors like "Japan" or "France." The high similarity
arises because country name embeddings are inherently close in semantic space.
This metric is misleading for Country Prediction.

**General Awareness has the lowest (0.282)** — its options span diverse types
(place names, concepts, people), making distractors genuinely dissimilar from
correct answers. These are structurally the most varied questions.

Overall mean: **0.429**, std: **0.164**.

### State Prediction: Distractor Domain

For State Prediction questions, **99.6% of distractors (16,086 of 16,146) are
state/UT names.** This is by design — the question asks "which state?" and all
options are states. The distractors are plausible at the domain level (all are
valid Indian states) but may vary in geographic plausibility (e.g., Lakshadweep
as a distractor for a Rajasthan question is implausible).

### Distractor Plausibility by Attribute

| Attribute | Mean Sim | Count | Hardness |
|-----------|----------|-------|----------|
| Language | 0.444 | 900 | Hardest distractors |
| Costume | 0.442 | 1,513 | |
| Festivals | 0.437 | 2,241 | |
| Cuisine | 0.434 | 1,671 | |
| Rituals_and_Ceremonies | 0.434 | 1,000 | |
| History | 0.433 | 2,609 | |
| Cultural_Common_Sense | 0.432 | 2,091 | |
| ... | ... | ... | |
| Tourism | 0.417 | 3,801 | |
| Religion | 0.410 | 482 | |
| Medicine | 0.392 | 72 | Easiest distractors |

The range is narrow (0.392 to 0.444) — distractor quality is remarkably consistent
across attributes. Language questions have the highest distractor similarity (0.444),
meaning language-related options are the most plausible/confusable. Medicine has the
lowest (0.392), suggesting its options span more diverse semantic domains.

This consistency means distractor quality is unlikely to be a major confound for
per-attribute behavioral analysis. If we see attribute-level differences in
suppression rates, they are more likely to reflect actual differences in cultural
knowledge representation than differences in question difficulty from distractor
quality.

---

## 10. Cultural Specificity

*Validated against: `cultural_entities.csv`, `cultural_entities_detail.csv`,
`cultural_entities_combined.csv`, `entity_extraction_by_qtype.csv`*

### Entity Extraction

Using regex patterns on the 7 identified templates, we extracted cultural entities
from 12,979 questions (59.7% of usable data). The extraction rate varies
dramatically by question type:

| Question Type | Extracted | Total | Rate |  Missing |
|---------------|-----------|-------|------|----------|
| Association | 4,821 | 5,453 | 88.4% | 632 |
| Country Prediction | 4,859 | 5,563 | 87.3% | 704 |
| State Prediction | 3,270 | 5,382 | 60.8% | 2,112 |
| **General Awareness** | **29** | **5,328** | **0.5%** | **5,299** |

**General Awareness is almost completely missed by regex extraction** because its
questions use free-form phrasing ("Which dish is most often associated with the
Tamil Nadu breakfast?", "A classical dance form that utilizes elaborate costumes...")
rather than the templated slots the regex targets.

**4,949 unique cultural entities** were identified from the regex extraction.

### Entity Extraction Fallback: Answer Text as Proxy

For the 8,747 questions (40.3%) where regex extraction fails, the answer text
itself serves as a proxy entity key. Inspection of General Awareness questions
confirms this: the answer IS the cultural entity — "Puanchei" (a Mizo costume),
"Idli" (Tamil Nadu breakfast), "Odissi" (a classical dance), "Ghoghla Beach"
(a Diu tourism site).

Using a `(state, attribute, answer)` triple as entity key for non-regex questions
produces 2,803 unique keys for the 5,328 General Awareness questions alone. This
is appropriate because the same answer from different states/attributes represents
a different cultural fact (e.g., "temple" in Tamil Nadu vs "temple" in Rajasthan).

**Combined entity coverage:**
- Regex-extracted: 4,949 unique entities covering 12,979 questions (59.7%)
- Answer-text fallback: 3,207 unique keys covering 8,747 questions (40.3%)
- Zero overlap between regex and fallback key spaces (different formats)
- **Total: 8,156 unique entity keys covering 100% of questions**

The fallback has one known issue: for Country Prediction questions that leak into
the no-entity set, "India" appears 729 times as an answer-entity. Since all CP
questions are expected to land in `control_both_correct`, this does not affect
suppression/enhancement entity-level analysis.

**This combined strategy is critical for Step 1.** Without it, entity-level
suppression rates would cover only 60% of questions, missing the most interesting
subset (General Awareness). With it, every question has an entity key for grouping.

### Entity State Uniqueness

| Metric | Count | % |
|--------|-------|---|
| Entities unique to 1 state | 4,948 | 99.98% |
| Entities appearing in 2+ states | 1 | 0.02% |

**99.98% of extracted entities are unique to a single state.** The cultural
entities in Sanskriti are almost entirely state-specific — "Bihu" only appears
in Assam questions, "Bharatanatyam" only in Tamil Nadu. This is by construction:
the dataset pairs cultural elements with their origin states.

### Most-Asked Entities

| Entity | Questions | States |
|--------|-----------|--------|
| Hyderabad | 70 | 1 (Telangana) |
| Delhi | 64 | 1 (Delhi) |
| Haryana | 38 | 1 (Haryana) |
| Karnataka | 28 | 1 (Karnataka) |
| Chandigarh | 27 | 1 (Chandigarh) |
| Silvassa | 22 | 1 (Dadra and Nagar Haveli) |
| Kumaon | 20 | 1 (Uttarakhand) |

The most-asked "entities" are geographic names (Hyderabad, Delhi) rather than
cultural elements (dances, festivals). This reflects the "According to you, which
of the following is closely associated to {region}?" template, which uses
city/region names as the entity.

### Entity Repetition Across Question Types

| Coverage | # Entities |
|----------|-----------|
| Appears in 1 question type only | 3,350 |
| Appears in 2 question types | 8 |
| Appears in 3 question types | 1,591 |
| Appears in all 4 question types | 0 |

**1,591 entities appear in 3 question types** — typically Country Prediction,
State Prediction, and Association (which share the same entity slot in their
templates). A model that knows one fact about a cultural entity gets credit 3
times. This further inflates the effective redundancy: the 4,949 regex-extracted
entities, with 1,591 appearing in 3 question types, explain a large portion of
the 21,726 questions. Mean questions per entity: 12,979 / 4,949 = 2.6.

**Effective information content:** The dataset's 21,726 questions test approximately
5,000 cultural knowledge facts (from regex extraction), each probed from ~2-3 angles. For behavioral
labeling, a suppression event on one template for a given entity is likely to
co-occur with suppression on other templates for the same entity. The behavioral
labels are not independent across questions — they cluster by entity.

### What "5,000 Cultural Knowledge Facts" Means for Our Study

If the dataset tests ~5,000 regex-extractable facts × ~2.6 questions/fact (plus
~3,200 fallback entities from General Awareness), then using combined entity keys
(8,156 total):
- Expected suppression events (at 8%): ~650 entities × ~2.6 questions = ~1,700 questions
- Expected enhancement events (at 7%): ~570 entities × ~2.6 questions = ~1,480 questions

But these are clustered — if entity X is suppressed, all 2-3 questions about X are
likely suppressed. So we effectively have ~650 independent suppression observations,
not 1,700. This matters for statistical tests:
- For overall suppression rate: 650 independent observations is sufficient
- For per-state rates: a state with 150 entities might have ~12 suppressed entities —
  barely enough for a percentage
- For per-attribute rates: an attribute with 200 entities might have ~16 suppressed —
  marginally sufficient

**Recommendation for the paper:** Report entity-level suppression rates alongside
question-level rates. "X% of cultural entities are suppressed" is a stronger claim
than "X% of questions are suppressed" because it accounts for the redundancy.

### The "Obscurity Gradient" Hypothesis

One of the core hypotheses we can test with this data structure is that RLHF
suppresses knowledge about **obscure** cultural entities more than **well-known**
ones. The reasoning: RLHF training data (human preferences) is biased toward
globally known topics. A cultural entity that appears frequently in English-
language web text (e.g., "Taj Mahal", "Diwali") is more likely to have been
reinforced during RLHF. An entity that appears rarely (e.g., "Tope Tenku weaving",
"Ningol Chakouba festival") may have been incidentally suppressed.

We can operationalize "obscurity" using:
1. **State population proxy:** Entities from small states (Lakshadweep, Mizoram)
   are likely more obscure than entities from large states (Delhi, Tamil Nadu)
2. **Entity frequency in pretraining data:** We cannot measure this directly, but
   we can use the state's TF-IDF specificity as a proxy — states with highly
   specific vocabulary (Lakshadweep: "minicoy", "kavaratti") have more obscure
   cultural elements
3. **The no-question baseline score:** Entities where even the no-question baseline
   fails (i.e., General Awareness questions with sim < 0.25) are the most obscure

This gradient — from Taj Mahal (universally known) to Tope Tenku (barely documented
in English) — is the axis along which we expect suppression to concentrate. The EDA
has given us the tools to measure it.

---

## 11. Critical Findings for Step 1

### Findings That Require Action

1. **Report metrics with and without Country Prediction.** 100% of CP answers are
   "India." Both models will ace these. The interesting behavioral signal comes
   from the other 16,163 questions (74.4%).

2. **Flag State Prediction as shortcut-vulnerable.** The no-question baseline solves
   99.98% of SP by matching state names to options. Any "suppression" on SP may
   reflect loss of string-matching ability, not cultural knowledge loss.

3. **Check for position bias in model predictions.** Ground truth is non-uniform
   (B=29.0%, D=20.8%). If a model overproduces B, it gets a ~4pp accuracy bonus.
   General Awareness has the most extreme skew (A=31.9%, D=16.5%).

4. **Do not report per-state-attribute behavioral rates.** Only 39 of 576 cells
   (6.8%) have enough data (>=125 questions) for reliable rates. Aggregate to state
   level or attribute level.

5. **Account for question-entity redundancy.** 78.6% of questions are involved in
   near-duplicate pairs. Behavioral labels cluster by entity, not independently by
   question. Effective sample sizes are smaller than raw counts suggest.

### Findings That Are Limitations (Report But Cannot Fix)

6. **351 conflicting duplicate groups.** Same question text, different answer
   positions. Not a bug (each row's ground truth matches its own options), but
   contributes to near-duplicate inflation.

7. **7.4% answer-in-question leakage.** Both models should get these right trivially.
   They will inflate `control_both_correct`.

8. **75.87% no-question baseline.** Three-quarters of questions are answerable from
   state-option similarity alone. This is a fundamental limitation of the Sanskriti
   benchmark design.

9. **55.4% of questions follow 7 templates.** The dataset is highly formulaic.
   Any model that learns the template structure can exploit it, independent of
   cultural knowledge.

### Findings That Are Expected (Not Bugs)

10. **Both models should score ~95-100% on Country Prediction.** This is correct
    behavior, not a sign that the experiment is working/failing.

11. **Sparse attributes (Sports, Transport, Medicine, Nightlife) will have noisy
    per-attribute stats.** Group or flag them.

12. **Southern/northeastern state overrepresentation.** Telangana (7.85%) vs
    Maharashtra (1.30%) reflects the Sanskriti paper's annotation process, not
    our design choices.

### Decision Matrix: What to Report in the Paper

Based on the EDA findings, here is what the EMNLP workshop paper should include:

| Analysis Level | Report? | Why |
|----------------|---------|-----|
| Overall accuracy (base vs instruct) | Yes, with and without CP | Primary result |
| Per-question-type accuracy | Yes | Shows CP inflation |
| Per-question-type suppression/enhancement | Yes | Core finding |
| Per-attribute suppression/enhancement | Yes (top 12 only) | Sparse attributes unreliable |
| Per-state suppression/enhancement | Yes (top 20 only) | Small states unreliable |
| Per-state-attribute suppression | **No** | Only 6.8% of cells have n>=125 |
| Entity-level suppression rate | Yes | Accounts for redundancy |
| Model position bias check | Yes | Must validate before interpreting labels |
| Option distribution check | Yes | Sanity check per Step 1 plan |
| Suppression by "obscurity gradient" | Yes | Core hypothesis |
| Near-duplicate impact analysis | Appendix | Limitation discussion |
| No-question baseline comparison | Appendix | Benchmark limitation |

### What Success Looks Like After Step 1

The EDA gives us criteria for a successful Step 1 run:

1. **Base accuracy 40-70%** (excluding CP). The Sanskriti paper reports LLaMA-3.2-3B-
   Instruct at 52% and LLaMA-3.1-70B-Instruct at 86% on the full dataset.
   Accounting for CP inflation (~98% on 25.6% of questions), the non-CP range for
   our 8B models should be roughly 43-67%. Below 35% means the prompt is broken.
   Above 75% means something is inflating scores.
2. **Instruct accuracy > base accuracy.** The single strongest signal of correct
   prompt formatting.
3. **CP accuracy > 95% for both models.** If not, the model cannot even identify
   Indian cultural elements as Indian.
4. **Null prediction rate < 2%.** Higher means the model is generating explanations
   instead of letters.
5. **Suppression 5-12%, enhancement 4-10%.** Based on prior work with Qwen2-1.5B.
   Llama 8B may differ, but these are reasonable bounds.
6. **Suppression concentrates in Tier 3 questions** (genuinely hard, not shortcut-
   solvable). If suppression is uniform across difficulty tiers, the effect is
   likely an artifact.

---

## 12. What This Stage Does NOT Do

The EDA is pre-model analysis of the dataset. It does not perform:

- Model inference or accuracy computation (Step 1 execution)
- Behavioral labeling (suppression/enhancement/control) (Step 1 execution)
- Per-question difficulty estimation from model responses (requires model outputs)
- Item Response Theory analysis (requires response matrix from multiple models)
- Classical Test Theory metrics (requires model responses)
- Distractor selection frequency analysis (requires model predictions)
- Cross-model agreement analysis (requires multiple model outputs)
- Any causal claims about what models know or don't know

The EDA tells us about the dataset's structure and biases. It does not tell us
how the models will behave on it. That is Step 1's job.

---

## 13. Output Files

### Plots (lightweight, on home)

```
/home/anshulk/cultural-mi/plots/
├── eda_01_distributions.png          # State/attribute/type distributions
├── eda_02_state_attr_heatmap.png     # 36x16 count heatmap
├── eda_03_state_qtype_heatmap.png    # 36x4 count heatmap
├── eda_04_position_bias.png          # GT letter distribution analysis
├── eda_05_question_length.png        # Word count distributions
├── eda_06_option_length_bias.png     # Correct vs incorrect option lengths
├── eda_07_word_frequency.png         # Top unigrams/bigrams/trigrams
├── eda_08_lexical_diversity.png      # MTLD and NGD by attribute/state/type
├── eda_09_umap.png                   # 2D UMAP colored by type/attribute/state
├── eda_10_semantic_analysis.png      # No-question baseline, Q-A overlap, near-dups
├── eda_11_bertopic.png               # Discovered topics and attribute alignment
├── eda_12_distractor_quality.png     # Distractor plausibility analysis
└── eda_13_cultural_specificity.png   # Entity frequency and state uniqueness

Total: 13 PNG files, ~3.6 MB
```

### Analysis CSVs (heavy, on data volume)

```
/data/user_data/anshulk/cultural-mi/analysis/
├── sanskriti_usable.csv              # 21,726 rows, master dataset with entity_key column
├── distribution_states.csv           # 36 rows
├── distribution_attributes.csv       # 16 rows
├── distribution_qtypes.csv           # 4 rows
├── coverage_state_attribute.csv      # 576 rows (all state-attr cells)
├── cross_tab_state_attribute.csv     # 36x16 count matrix
├── cross_tab_state_qtype.csv         # 36x4 count matrix
├── state_summary.csv                 # 36 rows, per-state breakdown
├── position_bias_overall.csv         # 4 rows
├── position_bias_by_qtype.csv        # 16 rows
├── position_bias_by_attribute.csv    # 64 rows
├── position_bias_by_state.csv        # 144 rows
├── country_prediction_audit.csv      # Summary metrics
├── country_prediction_answers.csv    # Answer distribution (1 row: India)
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
├── question_embeddings.npy           # 21,726 x 384 float32 (32 MB)
├── option_embeddings.npz             # 4 x 21,726 x 384 (128 MB)
├── umap_coordinates.csv
├── near_duplicates.csv               # 77,833 pairs (12 MB)
├── no_question_baseline.csv
├── no_question_baseline_by_attribute.csv
├── qa_overlap_by_qtype.csv
├── qa_overlap_by_attribute.csv
├── tfidf_terms_per_state.csv
├── bertopic_topics.csv
├── bertopic_assignments.csv
├── bertopic_vs_attribute.csv
├── bertopic_vs_qtype.csv
├── distractor_similarity.csv         # 21,726 rows (1.9 MB)
├── distractor_similarity_by_qtype.csv
├── distractor_similarity_by_attribute.csv
├── distractor_quality_summary.csv
├── answer_in_question_leakage.csv    # 1,615 rows
├── cultural_entities.csv             # 4,949 regex-extracted entities
├── cultural_entities_detail.csv
├── cultural_entities_combined.csv   # 8,156 entities (regex + fallback, with extraction_method)
├── entity_extraction_by_qtype.csv   # Regex extraction rate per question type
├── exact_duplicates.csv              # 419 groups
├── conflicting_duplicates.csv        # 351 groups
└── data_quality_summary.csv          # 1-row summary

Total: 54 CSV files + 2 numpy caches, ~183 MB
```

### Pipeline Script

```
/home/anshulk/cultural-mi/scripts/eda_pipeline.py
```

Single script, runs all 8 sections. Usage:
- Full run: `python scripts/eda_pipeline.py`
- Single section: `python scripts/eda_pipeline.py --section 5`

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

Section 5 dominates (91% of runtime) due to UMAP (n=21,726 points), near-duplicate
detection (pairwise cosine similarity, O(n²)), and BERTopic fitting.

### Reproducibility

All random operations use seed 42:
- UMAP: random_state=42
- BERTopic: inherits from UMAP seed
- Sentence embeddings: deterministic (same model, same input → same output)

The EDA is fully deterministic. Running `python scripts/eda_pipeline.py` on the
same dataset will produce identical CSV values and visually identical plots
(minor pixel-level differences possible from matplotlib rendering).

---

## 15. Alternative Datasets Considered

We surveyed every publicly available Indian cultural knowledge dataset to verify
that Sanskriti is the best fit for this MI study. None of the alternatives check
all five boxes we require: (1) MCQ format, (2) large enough for hundreds of
suppression cases, (3) English, (4) cultural knowledge specifically, (5) public
with state/attribute metadata.

| Dataset | Year | Size | Format | Language | Why Not |
|---------|------|------|--------|----------|---------|
| **Sanskriti** | 2025 | 21,853 MCQs | MCQ, 4-choice | English | **Selected** |
| MILU (AI4Bharat) | 2024 | ~80,000 MCQs | MCQ | Multilingual (mostly Indic) | Tests exam knowledge, not culture; non-English |
| DIWALI | 2025 | ~8K concepts | Concept inventory | English | No MCQ format; designed for text adaptation, not QA |
| Indica | 2026 | 515 Qs → 1,630 pairs | Free-form + MCQ | English | Too small (515 base questions); ~40 suppression cases |
| DRISHTIKON | 2025 | 2,126 MCQs | MCQ + images | English | Multimodal (requires images); too small for MI |
| CulturalBench (ICLR) | 2025 | 1,227 MCQs | MCQ | English | Only fraction India-specific; ~100 India questions |
| IndicParam | 2025 | Varies | MCQ | English/Indic | Tests UGC-NET academic knowledge, not cultural |
| IndQA (OpenAI) | 2025 | Varies | QA | 12 languages | Tests cross-lingual reasoning; not public in usable form |

### DRISHTIKON: The Best Alternative (But Still Insufficient)

DRISHTIKON deserves special mention. It covers all 28 states and 8 UTs with 2,126
MCQs that were semi-automatically generated and **human-curated** with intentionally
close distractors. Its question design is objectively better than Sanskriti's —
distractors are semantically proximate (testing fine-grained knowledge) rather than
obviously wrong (Japan vs India). However:

- At 2,126 questions, an 8% suppression rate yields ~170 cases — borderline for
  activation extraction across multiple layers
- It is multimodal (image + text), requiring vision-language models
- No text-only subset has been released

We cite DRISHTIKON and Indica in our related work section. Future work should
replicate our findings on DRISHTIKON's text-only subset if one becomes available.

### DIWALI: Different Purpose, Complementary

DIWALI (EMNLP 2025 Oral) catalogs ~8K cultural concepts across 36 sub-regions and
17 facets (food, dance, festivals, jewellery, etc.). It is a **concept inventory**
for cultural text adaptation, not a QA benchmark. It cannot produce suppression/
enhancement labels because there are no questions to answer right or wrong. However,
its concept list could be useful in Step 3/4 for validating whether our probing
directions align with known cultural concepts.

---

## 16. Dataset Fitness Assessment

### The Honest Picture

Sanskriti is usable but deeply flawed. Here is the unvarnished assessment.

**What works in our favor:**

1. **Size survives filtering.** Even restricting to Association + General Awareness
   (the "hard" subset), we have ~10,800 questions. At 5-8% suppression, that's
   500-800+ suppression cases — sufficient for activation extraction and probing
   across multiple layers.

2. **The base/instruct comparison is symmetric.** Both models face the same dataset
   artifacts (same shortcuts, same position bias, same templates). Since we measure
   the *difference* between base and instruct behavior, many artifacts cancel. If
   both models exploit the state-name shortcut equally, those questions land in
   "control" and do not contaminate suppression/enhancement labels.

3. **The 16 attributes and 36 states give slicing dimensions.** Even with coverage
   gaps, we can ask marginal questions: "does suppression concentrate in Religion
   vs Tourism?" or "in northeastern states vs metropolitan ones?"

**What is genuinely problematic:**

1. **75% of the dataset does not really test cultural knowledge.** The no-question
   baseline of 75.87% means most questions are answerable from surface-level pattern
   matching. For an MI study asking "how is cultural knowledge represented?", we
   need confidence that the model is using cultural knowledge to answer. A
   "suppression" label on a State Prediction question might mean the instruct model
   lost a string-matching ability, not cultural knowledge about Rajasthani cuisine.
   **This is the biggest threat to the study's validity.**

2. **Effective sample size is ~1/3 to 1/4 of raw count.** With 78.6% near-duplicate
   involvement and ~8,156 unique entity keys asked 1-3 times each, behavioral labels
   cluster by entity. If a model doesn't know about Bihu, it gets all 3-4 Bihu
   questions wrong — that's 1 independent suppression event, not 4. Per-attribute
   rates on smaller categories (Religion: 482 questions, maybe ~150 independent)
   will have wide confidence intervals.

3. **The templated structure conflates template recognition with cultural knowledge.**
   55% of questions follow 7 templates. If our probing picks up "this activation
   pattern means the model recognized the Country Prediction template" rather than
   "this pattern encodes knowledge about Bharatanatyam," we are doing MI of template
   recognition, not cultural knowledge. This must be controlled for in Steps 3-4.

4. **Country Prediction is dead weight.** 25.6% of the dataset where the answer is
   always "India" contributes nothing to understanding cultural knowledge suppression.
   Both models will get these right. It inflates control_both_correct.

### The Decision: Run Everything, Slice Everything, Explain Everything

**We run on all 21,726 usable questions.** We do not filter before evaluation.

Rationale:

1. **More data = more suppression cases.** 8% suppression on 21,726 ≈ 1,700 cases.
   On 10,800 ≈ 860 cases. We want every sample for probing across layers and models.

2. **"Easy" questions make a clean control group.** 5,563 Country Prediction questions
   landing in control_both_correct give us a massive, clean control set for probing.
   We *want* a large control population.

3. **Filtering looks like cherry-picking.** A reviewer will ask "why did you throw
   out 50% of the benchmark?" Running on everything and showing the breakdown by
   question type is more defensible.

4. **Slicing is stronger than filtering.** Report the full-dataset numbers as primary
   results. Then show: "on Association + General Awareness, suppression rises from
   X% to Y%." This framing shows the effect is *more* pronounced on genuinely
   knowledge-requiring questions — stronger than claiming you found it on a
   hand-picked subset.

### What the Paper Reports

**Primary results (full dataset, 21,726 questions):**
- Overall accuracy for both models
- Overall suppression/enhancement/control rates
- Per-question-type breakdown

**Robustness checks (sliced, not filtered):**
- Without Country Prediction (16,163 questions)
- Hard subset only: Association + General Awareness (10,781 questions)
- Per-attribute rates (top 12 attributes only)
- Per-state rates (top 20 states only)
- Entity-level suppression rate (accounting for redundancy) — using combined
  entity keys: regex-extracted for templated questions, `(state, attribute,
  answer)` triple for General Awareness. Must be computed in the Step 1 merge
  script, not as a post-hoc analysis. Group questions by entity key, check if
  all questions for entity X share the same behavioral label, report % of
  entities that are uniformly suppressed/enhanced/control

**Limitations section:**
- No-question baseline (75.87%) as benchmark design limitation
- Near-duplicate effective sample size reduction
- Template structure conflating format recognition with cultural knowledge
- Position bias in ground truth (B=29.0%, D=20.8%)
- Note that Sanskriti is the best available benchmark and alternatives are
  either too small (DRISHTIKON, Indica) or wrong format (DIWALI, MILU)

### Scoping Claims for EMNLP

What we **can** claim:
- "RLHF instruction tuning suppresses X% of cultural knowledge questions that
  the base model answers correctly" (primary finding)
- "Suppression concentrates in [specific attributes/states/difficulty tiers]"
  (if the data shows this)
- "Activation probing reveals [specific geometric patterns] that differentiate
  suppressed vs enhanced questions" (Steps 3-4)
- "The suppression effect is more pronounced on genuinely knowledge-requiring
  questions (Association + General Awareness) than on shortcut-vulnerable ones
  (Country Prediction, State Prediction)" (the killer finding if it holds)

What we **cannot** claim:
- "RLHF suppresses knowledge about Religion more than Tourism" with high
  confidence (Religion has ~150 effective independent samples)
- Anything about per-state-attribute interactions (6.8% reliable cells)
- That suppression reflects "cultural insensitivity" in RLHF training — we can
  only show the behavioral and representational facts

---

*End of document. All numbers verified against the output CSV files listed above,
produced by `scripts/eda_pipeline.py` on 2026-03-25. Dataset landscape survey
conducted 2026-03-25.*
