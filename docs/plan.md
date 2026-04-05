# Complete Research Plan: Mechanistic Interpretability of Cultural Knowledge Gating in LLMs

**Authors:** Anshul Kumar, Pragati Bhattad
**Affiliation:** Carnegie Mellon University
**Venue:** BlackboxNLP 2026 — The Ninth Workshop on Analyzing and
Interpreting Neural Networks for NLP (co-located with EMNLP 2026,
Budapest, Hungary, October 28-29, 2026)
**Submission Deadline:** July 17, 2026 (11:59 PM AoE) — Direct via OpenReview
**Notification:** September 8, 2026
**Camera Ready:** September 20, 2026
**Format:** Archival, up to 8 pages + references + optional appendix, ACL
template, double-blind, anonymized
**Timeline:** 14.5 weeks (April 7 — July 17, 2026)
**Models:** LLaMA-3.1-8B (base) vs LLaMA-3.1-8B-Instruct
**Compute:** CMU Babel cluster (A100 GPUs)

---

## 0. How to Read This Document

This document is the complete blueprint for a mechanistic interpretability
study investigating how instruction tuning changes cultural knowledge
access inside large language models. Every design choice, mathematical
formulation, and experimental process is explained in full. No code
appears here. The document is organized chronologically by week, with
each phase building on the outputs of the previous phase.

The central question: when an instruction-tuned model fails on a cultural
knowledge question that its base counterpart answers correctly, does that
failure reflect destroyed knowledge or blocked knowledge? The answer
determines whether fixing cultural bias requires expensive retraining
or cheap inference-time intervention. This distinction matters.

---

## 1. Executive Summary

### 1.1 What We Already Know

From the pilot study on Qwen2-1.5B, we found three things. First,
cross-model linear probe transfer rates between base and instruct
models reach 96-99%, meaning cultural knowledge representations survive
instruction tuning nearly intact. Second, KL divergence between base and
instruct activation distributions stays flat through layers 8-24 but
spikes 2.9x at the final layer (Layer 28), suggesting the behavioral
change happens late. Third, this spike is content-selective (Religion
shows 2.25x more divergence than Nightlife) but behavior-uniform (all
behavioral groups get the same amplification factor).

These findings are promising but have three weaknesses that prevent
publication. The model is too small (1.5B parameters). The evidence is
purely correlational with no causal validation. And the analysis relies
on a single benchmark with known quality issues.

### 1.2 What This Plan Achieves

This plan addresses all three weaknesses. We scale up to LLaMA-3.1-8B
(32 layers, 4096-dim hidden states, 8B parameters), the standard model
for open-source MI research in 2025-2026. We add three causal experiments:
activation patching, tuned lens trajectory analysis, and sparse autoencoder
model-diffing. And we add CulturalBench as a second benchmark to test
whether findings generalize beyond Indian cultural knowledge.

The result will be a paper that makes three contributions: (1) demonstrates
representational preservation of cultural knowledge after instruction tuning
using cross-model probing and information-theoretic measures, (2) provides
causal evidence that late-layer components mediate the gating mechanism via
activation patching, and (3) identifies interpretable features associated
with cultural knowledge suppression using sparse autoencoders.

### 1.3 Why BlackboxNLP

BlackboxNLP is the premier workshop for interpretability of NLP models,
running annually since 2018 at EMNLP. The 2026 edition is confirmed for
October 28-29 in Budapest, co-located with EMNLP 2026.

**Direct submission deadline: July 17, 2026.** Archival papers up to 8
pages + references in ACL format. Double-blind review via OpenReview.
Accepted papers are published in the ACL Anthology. An alternative path
is ARR commitment by August 28, but direct submission gives us more
control over timing.

**Organizer alignment is strong.** Yonatan Belinkov (Technion), whose
probing methodology paper (Belinkov 2022) we cite as foundational, is
a co-organizer. Aaron Mueller (Boston University), a leading MI
researcher who works on causal interpretability and circuit analysis,
is also organizing. These are people who will value rigorous probing
combined with causal validation.

The CFP explicitly lists four topics that our paper addresses directly:
"mechanistic interpretability, reverse engineering approaches,"
"testing if interpretable information can be decoded from internal
representations," "evaluation of techniques for steering LLM output
behavior," and "scaling up analysis methods for large language models."

**The Special Track on Reproducibility** (max 6 pages) is not our target.
Our paper is original research, not a reproduction study. We submit to
the main archival track.

**Dual submission policy:** Archival dual submissions are allowed (check
the other venue's policy). Preprints on arXiv are permitted without
restriction. This means we can post to arXiv before July 17 to
establish priority without affecting our BlackboxNLP submission.

**Anonymization requirement:** Double-blind means no author names, no
institutional affiliations in the paper, no links to personal GitHub
repos or cluster-specific paths (e.g., /home/anshulk/cultural-mi/ must
not appear). Use anonymous links for code and data if needed.

### 1.4 Timeline Overview

The confirmed July 17 deadline gives us 14.5 weeks. The extra 4 weeks
compared to our original 10-week plan go toward: deeper SAE analysis
(Week 8-9), optional second model family for cross-architecture
replication (Week 10), extended writing and revision (Weeks 11-14), and
a comfortable buffer for unexpected issues.

| Week | Dates | Phase | What Happens |
|------|-------|-------|-------------|
| 1 | Apr 7-13 | Setup + Behavioral Eval | Model download, prompt engineering, run both models on Sanskriti + CulturalBench |
| 2 | Apr 14-20 | Behavioral Analysis | Label every question, compute all behavioral metrics, sanity checks |
| 3 | Apr 21-27 | Activation Extraction | Extract hidden states at 8 layers for both models on all questions |
| 4 | Apr 28 - May 4 | Correlational Analysis I | Linear probing, cross-model transfer, MDL probing |
| 5 | May 5-11 | Correlational Analysis II | KL divergence, tuned lens trajectories |
| 6 | May 12-18 | Causal Analysis I | Activation patching layer sweep |
| 7 | May 19-25 | Causal Analysis II | Component-level patching at gate layer |
| 8 | May 26 - Jun 1 | SAE Analysis | SAE model-diffing, gating feature identification |
| 9 | Jun 2-8 | Cross-Benchmark + Optional 2nd Model | CulturalBench validation, optionally begin Gemma-2-9B replication |
| 10 | Jun 9-15 | Gemma Replication / Extra Experiments | Cross-architecture replication OR deepen existing analyses |
| 11 | Jun 16-22 | First Draft | Write complete 8-page paper |
| 12 | Jun 23-29 | Co-Author Iteration | Pragati feedback, revise arguments and figures |
| 13 | Jun 30 - Jul 6 | Second Draft | Address feedback, finalize all figures and tables |
| 14 | Jul 7-13 | Final Polish | Anonymization, ACL formatting, appendix, reproducibility statement |
| 14.5 | Jul 14-17 | Submission | Final proofread, submit to OpenReview by July 17 AoE |

---

## 2. Research Question and Hypotheses

### 2.1 The Core Question

**Does instruction tuning erase cultural knowledge from LLM
representations, or does it install a gating mechanism that blocks
expression of preserved knowledge?**

This question matters because the answer determines the intervention
strategy. If knowledge is erased, you need retraining. If knowledge is
preserved but gated, you can potentially recover it through inference-time
techniques like activation patching or steering vectors. The cost
difference between these two approaches is orders of magnitude.

### 2.2 Three Competing Hypotheses

**H1: Representational Erasure.** Instruction tuning modifies the weight
matrices in ways that destroy or degrade cultural knowledge encoded during
pretraining. The instruct model genuinely does not contain the information
anymore. Predictions: probes trained on base activations will fail when
applied to instruct activations. Cross-model transfer rates will be low.
MDL encoding costs will diverge between models. KL divergence will be
distributed across all layers, not concentrated at any single layer.

**H2: Late-Layer Gating.** Instruction tuning installs a localized mechanism
in late transformer layers that selectively blocks output of cultural
knowledge while preserving underlying representations. The model still
knows the answer but a computational gate prevents it from saying so.
Predictions: probes transfer nearly perfectly. KL divergence concentrates
at a specific late layer. Patching activations at the gate layer causally
restores suppressed outputs. The gating is content-selective, with
identity-marking cultural categories (Religion, Costume, Dance) gated
more than neutral informational categories (Transport, History).

**H3: Distributed Output Modulation.** Instruction tuning applies a
diffuse transformation across multiple layers that changes output
distributions without destroying representations. There is no single
gate layer. Predictions: probes transfer well. KL divergence increases
gradually across layers rather than spiking at one layer. Patching any
single layer has only a partial effect. Behavioral outcomes emerge from
the cumulative effect of many small changes.

### 2.3 Why These Hypotheses Are Mutually Exclusive

H1 predicts probe failure. H2 and H3 both predict probe success but
differ on localization. H2 predicts a sharp spike in divergence at one
layer. H3 predicts a gradual increase. The activation patching experiment
is the definitive test: if patching a single layer causally flips the
behavioral outcome from incorrect to correct, H2 is supported over H3.
If patching any single layer produces only a partial effect, H3 is
supported over H2.

### 2.4 Operationalizing "Cultural Knowledge"

We define cultural knowledge operationally as the information tested by
the Sanskriti and CulturalBench benchmarks. This is not a claim about the
philosophical nature of culture. It is a pragmatic choice: these benchmarks
provide multiple-choice questions with ground truth answers that let us
measure whether a model "knows" a cultural fact.

A question tests cultural knowledge if the correct answer requires
information specific to a culture, region, or community. "What is the
traditional dance of Tamil Nadu?" tests cultural knowledge. "What is the
capital of France?" does not. Both benchmarks were designed to test
cultural knowledge specifically.

---

## 3. Model Selection and Justification

### 3.1 Why LLaMA-3.1-8B

Five reasons drive this choice.

**Reason 1: Pre-trained SAE availability.** Llama Scope (Zhong et al., 2024)
provides 256 TopK sparse autoencoders trained on every layer and sublayer
of LLaMA-3.1-8B-Base, with both 32K and 128K feature widths. These SAEs
are hosted on HuggingFace at fnlp/Llama-Scope. Roughly 90% of extracted
features have been rated human-interpretable. Goodfire has additionally
released SAEs for LLaMA-3.1-8B-Instruct. Having pre-trained SAEs on BOTH
the base and instruct model variants is a massive advantage. Training SAEs
from scratch would take weeks of compute we do not have.

**Reason 2: Clean architecture.** LLaMA-3.1-8B uses a standard decoder-only
transformer with Rotary Position Embeddings (RoPE), Grouped Query Attention
(GQA, 8 KV heads, 32 query heads), SwiGLU MLP activation, and RMSNorm.
There are no architectural complications that would confound interpretability
analysis. Compare this with Gemma-2-9B, which uses interleaved local/global
attention and logit soft-capping, both of which complicate activation
patching experiments.

**Reason 3: Base-instruct alignment.** The base and instruct versions share
identical architecture and parameter count. The only difference is that the
instruct model underwent supervised fine-tuning on instruction-following data
followed by RLHF with human preference data. This makes it a clean controlled
comparison. We can attribute any representational or behavioral difference
solely to the instruction tuning process.

**Reason 4: TransformerLens support.** TransformerLens, the standard library
for mechanistic interpretability research, supports LLaMA-3.1-8B with hook
points at every residual stream position, attention head, and MLP sublayer.
This enables activation patching at arbitrary granularity without custom
engineering.

**Reason 5: Scale credibility.** In 2025-2026, reviewers expect MI papers to
work with models of at least 7-8B parameters unless the research question
specifically requires smaller models. The Qwen2-1.5B pilot was useful for
exploration but would face reviewer pushback as a primary result. LLaMA-3.1-8B
is the sweet spot: large enough for credibility, small enough for a single A100.

### 3.2 Architecture Details

| Property | LLaMA-3.1-8B | Qwen2-1.5B (pilot) |
|----------|--------------|---------------------|
| Parameters | 8.03B | 1.54B |
| Layers | 32 | 28 |
| Hidden dimension | 4096 | 1536 |
| Attention heads | 32 query, 8 KV | 12 query, 2 KV |
| MLP intermediate dim | 14336 | 8960 |
| Vocabulary size | 128256 | 151936 |
| Context length | 131072 | 32768 |
| Normalization | RMSNorm | RMSNorm |
| Position encoding | RoPE | RoPE |
| Activation | SwiGLU | SwiGLU |
| Tokenizer | BPE (tiktoken) | BPE (Qwen) |

### 3.3 Memory and Compute Requirements

LLaMA-3.1-8B in FP16 requires approximately 16 GB of GPU memory for
inference. On a single A100 (80 GB), this leaves 64 GB for activation
caching, batch processing, and SAE inference. For activation extraction
across 21,726 questions at 8 layers for 2 models, we estimate:

- Per-question activation size: 4096 dimensions x 4 bytes (float32) = 16 KB
- Per-question, per-layer, per-model: 16 KB
- Total: 21,726 questions x 8 layers x 2 models x 16 KB = 5.6 GB

This fits comfortably in memory and on disk. The extraction itself should
take approximately 4-6 hours per model on a single A100, accounting for
batch sizes of 32 and sequence lengths up to 256 tokens.

### 3.4 The Instruct Model's Training

LLaMA-3.1-8B-Instruct underwent Meta's standard alignment pipeline:
supervised fine-tuning (SFT) on instruction-following demonstrations,
followed by reinforcement learning from human feedback (RLHF) using
reward models trained on human preference rankings. The exact SFT dataset
composition and RLHF hyperparameters are not fully public, but the
training report confirms the standard pipeline. The instruct model also
adds a system prompt template and date metadata when the chat template
is applied.

**Critical implementation detail:** The instruct model's tokenizer applies
a chat template that wraps user input in special tokens and prepends a
system prompt including the current date. When evaluating both models on
identical prompts, we must use the chat template for the instruct model
and raw text for the base model. Using the chat template for the base
model or omitting it for the instruct model will produce misleading results.

### 3.5 Greedy Decoding Requirement

Both LLaMA-3.1-8B models ship with sampling enabled by default (temperature=0.6
for instruct, temperature=1.0 for base in the generation config). For
behavioral evaluation, we must force greedy decoding explicitly by setting
temperature=0.0 or equivalently do_sample=False and top_k=1. Without this,
results are non-deterministic and harder to interpret.

---

## 4. Dataset Strategy

### 4.1 Primary Benchmark: Sanskriti

The Sanskriti benchmark (Maji et al., ACL 2025 Findings) is the largest
dataset for evaluating Indian cultural knowledge in language models. It
contains 21,853 multiple-choice questions (21,726 usable after excluding
127 with broken ground truth) spanning 28 Indian states, 8 union territories,
and 16 cultural attribute categories.

**Why keep Sanskriti despite its issues:** It is the only public benchmark
satisfying all five of our requirements: MCQ format (needed for behavioral
labeling), sufficient scale (20K+ questions for statistical power), English
language (matching LLaMA's strongest language), cultural specificity (testing
knowledge that differs between base and instruct models), and public
availability with rich metadata (state, attribute, question type labels
enabling stratified analysis).

#### 4.1.1 Known Issues from EDA (Completed March 25, 2026)

The EDA identified eight issues that affect experimental design.

**Issue 1: No-question baseline of 75.87%.** Three-quarters of questions
can be answered correctly by comparing the state name against the four
options using cosine similarity, without reading the question at all. This
means a large fraction of "correct" answers from either model may reflect
pattern matching rather than cultural knowledge. State Prediction questions
are 99.98% solvable this way. Country Prediction is 95.88% solvable.
Association is 63.27%. General Awareness is 43.52%, making it the only
question type where correct answers plausibly require actual knowledge.

**Mitigation:** Report all metrics in three tiers. Tier 1: full dataset
(21,726 questions). Tier 2: without Country Prediction (16,163 questions).
Tier 3: hard subset only, meaning Association + General Awareness questions
(10,903 questions). Tier 3 is the most informative for cultural knowledge
claims. We never filter or exclude data. We slice and report.

**Issue 2: 100% of Country Prediction answers are "India."** This means
25.6% of the dataset tests only one fact: "Is this cultural element from
India?" Both models will score near 100% on this subset. It inflates
overall accuracy and dilutes the suppression signal.

**Mitigation:** Tier 2 reporting removes these. We also use CP accuracy
as a sanity check: if either model scores below 95% on Country Prediction,
something is wrong with the prompt formatting.

**Issue 3: 78.6% of questions involved in near-duplicate pairs.**
Using sentence embeddings (all-MiniLM-L6-v2), 77,833 question pairs have
cosine similarity above 0.85. This means the effective information content
of the dataset is closer to 8,156 unique cultural entity keys than 21,726
independent questions. A model's accuracy is inflated by repeated testing
of similar knowledge.

**Mitigation:** Compute entity-level behavioral labels in addition to
question-level labels. An entity (e.g., "Bharatanatyam") is labeled
suppressed only if ALL questions about it are suppressed. This gives us
the more conservative and honest suppression rate. Report both question-level
and entity-level rates.

**Issue 4: Ground truth position bias.** The correct answer is in position
B 29.0% of the time and position D only 20.8% of the time. If a model
has a preference for certain positions, its accuracy is affected by this
non-uniform distribution.

**Mitigation:** Compute each model's prediction distribution across A/B/C/D
and compare it to the ground truth distribution. Run a chi-squared test.
If either model shows more than 5 percentage points of deviation from
uniform on any letter, flag it and compute position-bias-corrected accuracy
as a robustness check.

**Issue 5: 7.4% answer-in-question leakage.** In 1,615 questions, the
correct answer text appears verbatim in the question text. Both models
should get these right trivially. They inflate the control group.

**Mitigation:** Flag these questions. Verify they land in control_both_correct.
If any appear in the suppression group, investigate individually.

**Issue 6: 55.4% template concentration.** Over half of questions follow
just 7 templates. The dataset is highly formulaic.

**Mitigation:** This is a limitation we report honestly. Template structure
may conflate format recognition with cultural knowledge. The Tier 3 hard
subset partially addresses this because General Awareness questions are
less formulaic than Country Prediction or State Prediction.

**Issue 7: Only 6.8% of state-attribute cells are reliable.** With 36
states and 16 attributes, there are 576 cells, but only 39 have at least
125 questions, the minimum for reliable behavioral rate estimation.

**Mitigation:** Never report per-state-attribute results. Aggregate to
state level or attribute level only. When reporting attribute-level
results, group the four sparse attributes (Sports, Transport, Medicine,
Nightlife) as "Other" or flag them with wide confidence intervals.

**Issue 8: 351 conflicting duplicate groups.** Same question text, different
correct answers across instances. This is because the same template with
different distractor sets produces different answer positions. Each row's
ground truth is correct for its own options, so this does not affect
evaluation. But it contributes to near-duplicate inflation.

**Mitigation:** No action needed beyond reporting.

#### 4.1.2 Question Types and Their Properties

| Type | Count | % of Dataset | No-Q Baseline | Notes |
|------|-------|-------------|---------------|-------|
| Country Prediction | 5,563 | 25.6% | 95.88% | All answers are "India" |
| State Prediction | 5,387 | 24.8% | 99.98% | Solvable by string matching |
| Association | 5,454 | 25.1% | 63.27% | Moderate difficulty |
| General Awareness | 5,449 | 25.1% | 43.52% | Hardest, most informative |

#### 4.1.3 Cultural Attributes (16 Categories)

Art, Costume, Cuisine, Cultural Common Sense, Dance and Music, Festivals,
History, Language, Medicine, Nightlife, Personalities, Religion, Rituals
and Ceremonies, Sports, Tourism, Transport.

The top 5 by question count: Tourism (2,573), Festivals (2,298), History
(2,174), Art (1,891), Cuisine (1,671). The bottom 4: Nightlife (41),
Transport (76), Sports (153), Medicine (234). These four sparse categories
should be treated with caution in per-attribute analysis.

### 4.2 Secondary Benchmark: CulturalBench

CulturalBench (Chiu et al., ICLR 2025) provides 1,696 human-written,
human-verified cultural questions across 45 global regions. Unlike
Sanskriti, which focuses exclusively on Indian culture, CulturalBench
covers a broad geographic range including East Asia, the Middle East,
Africa, Latin America, Europe, and South/Southeast Asia.

**Why add CulturalBench:** It addresses the criticism that findings on
Sanskriti might be India-specific artifacts rather than general cultural
knowledge gating. If the same late-layer divergence pattern appears on
CulturalBench questions about Japanese, Nigerian, and Brazilian culture,
the finding is much stronger.

**Format:** Each question has 4 MCQ options plus a True/False variant. We
use the MCQ variant for consistency with Sanskriti.

**Size:** 1,696 questions is small compared to Sanskriti's 21,726. This
means CulturalBench serves as a validation benchmark, not the primary
analysis target. We compute overall accuracy, suppression rate, and KL
divergence on CulturalBench, but we do not attempt fine-grained per-region
breakdowns with only ~35 questions per region.

**Integration plan:** Run both models on CulturalBench during Week 1
alongside Sanskriti. Label questions as suppression/enhancement/control
using identical logic. Extract activations during Week 3. Compute KL
divergence during Week 5 and compare the layer-wise trajectory to
Sanskriti's trajectory. If both benchmarks show a late-layer spike, this
is strong cross-benchmark validation.

### 4.3 What We Explicitly Do Not Do

We do not pre-filter questions, even the trivially easy Country Prediction
ones. We run the full dataset and slice results afterward. This is
methodologically stronger than pre-filtering because it avoids the
appearance of cherry-picking and lets reviewers verify our decisions by
examining all reporting tiers.

We do not use MILU (85K multilingual Indic questions) because it is not
in English and our models' behavior on Hindi or Tamil prompts would
confound the cultural knowledge signal with multilingual capability
differences.

We do not use BLEnD's short-answer format because our analysis requires
discrete correct/incorrect labels, which MCQ format provides naturally.

---

## 5. Phase 1: Behavioral Evaluation (Weeks 1-2)

### 5.1 Goal

Run both LLaMA-3.1-8B variants (base and instruct) on every question in
Sanskriti (21,726 questions) and CulturalBench (1,696 questions). For each
question, record which model answers correctly. Assign each question a
behavioral label: suppression (base correct, instruct incorrect),
enhancement (base incorrect, instruct correct), control-both-correct
(both correct), or control-both-wrong (both wrong).

These labels are the foundation of everything that follows. A false label
propagates through activation extraction, probing, and causal analysis,
potentially generating false scientific claims. Getting this phase right
is non-negotiable.

### 5.2 Prompt Design

The prompt must satisfy four constraints simultaneously.

**Constraint 1: Force single-letter output.** The model must output exactly
one of A, B, C, D. Any other output (explanations, "I think the answer
is...", refusals) counts as a null prediction and wastes the question.
Target: null prediction rate below 2%.

**Constraint 2: Fair to both models.** The base model does not understand
instruction-following formats. The instruct model expects its chat template.
Using the same raw prompt for both models disadvantages the instruct model
(which was trained to respond to structured conversations) and may
advantage the base model (which was trained on raw text completion).

**Constraint 3: No information leakage.** The prompt must not contain
hints, explanations, or context that could help the model beyond what
the question itself provides.

**Constraint 4: Reproducible.** Greedy decoding with deterministic
settings. Same random seed (42) everywhere.

#### 5.2.1 Base Model Prompt

The base model receives a raw text completion prompt. The format is a
few-shot demonstration followed by the target question:

```
The following are multiple choice questions about Indian culture.

Question: [demo question 1]
A. [option A]
B. [option B]
C. [option C]
D. [option D]
Answer: [correct letter]

Question: [demo question 2]
...
Answer: [correct letter]

Question: [target question]
A. [option A]
B. [option B]
C. [option C]
D. [option D]
Answer:
```

The model completes the text after "Answer:" and we extract the first
token that is A, B, C, or D.

**Few-shot selection:** Use 5 demonstration examples. These must NOT come
from the Sanskriti or CulturalBench datasets. Instead, craft 5 unambiguous
Indian cultural knowledge questions with obvious answers. Example: "What
is the national animal of India?" with options Tiger, Lion, Elephant,
Peacock. These demonstrations teach the base model the expected output
format without leaking test information.

**Why 5-shot, not 0-shot:** Base models without instruction tuning perform
poorly on 0-shot MCQ because they do not know they should output a single
letter. 5-shot is the standard in benchmarking (e.g., MMLU uses 5-shot).

#### 5.2.2 Instruct Model Prompt

The instruct model receives a chat-formatted prompt using the official
LLaMA-3.1 chat template. The system message and user message structure:

System: "You are a helpful assistant. Answer the following multiple choice
question by responding with only the letter (A, B, C, or D) of the
correct answer. Do not provide explanations."

User: "[Question text]\nA. [option A]\nB. [option B]\nC. [option C]\nD. [option D]"

The chat template wraps these in the special tokens that the instruct
model was trained to expect. The model's response is parsed for the first
A/B/C/D character.

**0-shot vs few-shot for instruct:** Instruct models already understand
the task from the system prompt. Few-shot examples are unnecessary and
may actually hurt performance by wasting context window space. Use 0-shot
for the instruct model.

**Date metadata:** The LLaMA-3.1 instruct chat template injects the current
date as metadata. This is fine. It does not affect cultural knowledge
responses.

#### 5.2.3 Answer Extraction Logic

From the model's generated text, extract the predicted answer using this
priority:

1. If the first non-whitespace character is A, B, C, or D, use it.
2. If the response starts with "The answer is [X]" or similar, extract X.
3. If the response contains exactly one of A, B, C, D as a standalone
   character (not part of a word), use it.
4. If none of the above match, label the prediction as null.

Record both the extracted letter and the raw generated text (first 100
characters) for audit purposes. Every null prediction should be manually
inspectable.

### 5.3 Evaluation Metrics

For each model (base, instruct) on each dataset (Sanskriti, CulturalBench):

**Accuracy:** percentage of questions where the predicted letter matches
the ground truth letter. Compute overall, per question type (Sanskriti
only), per attribute, and per state.

**Null rate:** percentage of questions where no valid letter was extracted.
Target: below 2%. Above 5% means the prompt needs redesign.

**Position distribution:** percentage of predictions in each position
A/B/C/D. Compare against both uniform (25% each) and ground truth
distribution using chi-squared test. Flag if any position deviates more
than 5pp from uniform.

### 5.4 Behavioral Labeling

For each question, assign exactly one label based on the pair of
predictions (base_pred, instruct_pred) compared to ground truth:

| Base Correct? | Instruct Correct? | Label |
|:---|:---|:---|
| Yes | No | **Suppression** |
| No | Yes | **Enhancement** |
| Yes | Yes | Control (both correct) |
| No | No | Control (both wrong) |

Questions where either model produced a null prediction are labeled
"null" and excluded from behavioral analysis (but their count is reported).

### 5.5 Expected Outcomes and Sanity Checks

Based on the Sanskriti paper's reported results for LLaMA-3.2-3B-Instruct
(52% accuracy) and LLaMA-3.1-70B-Instruct (86% accuracy), and accounting
for the 8B model sitting between these, we expect:

1. **Base accuracy 40-70% excluding Country Prediction.** Below 35% means
   the prompt is broken. Above 75% means scores are being inflated.

2. **Instruct accuracy higher than base accuracy.** This is the single
   strongest signal that prompt formatting is correct. If the base model
   outperforms the instruct model overall, something is wrong with the
   chat template application.

3. **Country Prediction accuracy above 95% for both models.** Both models
   should know that Bharatanatyam is Indian. If not, the prompt is
   fundamentally broken.

4. **Null prediction rate below 2%.** Higher rates mean the model is
   generating explanations instead of letters.

5. **Suppression rate 5-15%, enhancement rate 4-12%.** Based on the
   Qwen2-1.5B pilot (8.18% suppression, 6.91% enhancement on the full
   dataset). The 8B model may show different rates, but we do not expect
   dramatic deviations from this range.

6. **Both models should have roughly uniform prediction distributions.**
   Neither model should show extreme preference for any single position.

If any of these checks fail, stop and debug before proceeding. False
behavioral labels propagate through the entire pipeline.

### 5.6 Three-Tier Reporting

All behavioral metrics are computed and reported in three tiers:

**Tier 1 (Full dataset):** All 21,726 Sanskriti questions. This is the
most inclusive view. It includes the easy Country Prediction questions
that inflate accuracy.

**Tier 2 (Without Country Prediction):** 16,163 questions. Removes the
trivially easy questions where both models score near 100%.

**Tier 3 (Hard subset):** 10,903 questions. Only Association and General
Awareness, the two question types that actually require cultural knowledge
rather than pattern matching or string comparison.

CulturalBench gets a single tier (all 1,696 questions) since it does not
have the Country Prediction issue.

### 5.7 Entity-Level Analysis

Using the 8,156 unique entity keys identified during EDA, compute entity-level
behavioral labels. An entity is labeled "suppressed" if ALL questions
about that entity are in the suppression group. An entity is labeled
"enhanced" if ALL questions are enhanced. Mixed entities (some questions
suppressed, some not) are labeled "mixed."

Report entity-level suppression rate alongside question-level suppression
rate. The entity-level rate will be lower (more conservative) because a
single correctly-answered question about an entity prevents it from being
labeled "fully suppressed."

---

## 6. Phase 2: Activation Extraction (Week 3)

### 6.1 Goal

Extract hidden state activations from both models at multiple layers for
every question in Sanskriti and CulturalBench. These activations are the
raw material for all subsequent analysis: probing, KL divergence, and
causal experiments.

### 6.2 Which Layers to Extract

For LLaMA-3.1-8B with 32 layers (numbered 0-31), we extract at 8 layers
to provide fine-grained coverage of the computational trajectory:

| Layer | Position | Rationale |
|-------|----------|-----------|
| 0 | Input embeddings | Baseline before any transformer processing |
| 4 | Early | Initial feature extraction |
| 8 | Early-mid | Pattern formation |
| 14 | Middle | Mid-processing |
| 20 | Mid-late | Knowledge integration |
| 26 | Late | Late processing before the critical zone |
| 30 | Pre-final | One layer before the last |
| 31 | Final | The hypothesized gating site |

The pilot study on Qwen2-1.5B (28 layers) found the gating spike at the
final layer (Layer 28). On LLaMA-3.1-8B (32 layers), we hypothesize the
spike occurs at Layer 31 or Layer 30. Extracting both gives us coverage
regardless of exactly where it lands. The dense sampling in the late
layers (26, 30, 31) is specifically designed to capture the transition
from stable to spiking divergence.

### 6.3 Pooling Strategy: Mean Pooling

For each question, the model processes a sequence of tokens. We need to
reduce the sequence of hidden states to a single vector per layer. There
are two standard approaches.

**Mean pooling** averages the hidden state vectors across all non-padding
token positions, weighted by the attention mask. If the input has T tokens
and hidden dimension d, the pooled representation is:

h_pooled = (1/T) * sum_{t=1}^{T} h_t

where h_t is the hidden state at position t.

**Last-token pooling** takes the hidden state at the final token position:

h_pooled = h_T

**Why mean pooling:** For sentence-level representation tasks like ours
(predicting cultural attribute or state from a question's representation),
mean pooling captures information from the entire input. Last-token pooling
is appropriate for next-token prediction analysis (like the tuned lens
experiment in Phase 5), but for probing experiments, mean pooling is more
robust because it does not depend on which token happens to be last.

We use mean pooling for Phases 3-4 (probing and KL divergence) and
last-token extraction for Phase 5 (tuned lens), extracting both during
this phase to avoid re-running extraction later.

### 6.4 Input Formatting

For activation extraction, we feed each model the same question text in
its expected format:

**Base model:** The raw question with options, WITHOUT the few-shot
demonstrations. We want the activations to reflect the model's processing
of this specific question, not the demonstrations. The format is simply:

"Question: [text]\nA. [opt A]\nB. [opt B]\nC. [opt C]\nD. [opt D]"

**Instruct model:** The question wrapped in the chat template with the
same system prompt used during behavioral evaluation.

### 6.5 Storage and Precision

Each activation vector is 4096 dimensions in float32 (16,384 bytes).
Total storage for the complete extraction:

- Questions: 21,726 (Sanskriti) + 1,696 (CulturalBench) = 23,422
- Layers: 8
- Models: 2
- Pooling types: 2 (mean and last-token)
- Per vector: 16 KB

Total: 23,422 x 8 x 2 x 2 x 16 KB = approximately 12 GB

This is manageable on disk and in memory for batch processing.

### 6.6 Batch Processing

Process questions in batches of 32 with maximum sequence length of 256
tokens. For questions that exceed 256 tokens after tokenization (rare for
Sanskriti's concise MCQ format), truncate from the right. Log the number
of truncated questions.

Use PyTorch hooks to intercept activations at each target layer without
modifying the model architecture. Register forward hooks on the output
of each transformer block's layer normalization (post-RMSNorm residual
stream).

### 6.7 Validation

After extraction, run sanity checks:

1. **Shape check:** Every activation file has shape (n_questions, 4096).
2. **No NaN/Inf:** Verify no numerical issues in any activation.
3. **Non-degenerate:** Verify standard deviation across dimensions is
   non-zero (activations are not collapsed to a single point).
4. **Cross-model similarity:** Compute mean cosine similarity between
   base and instruct activations at Layer 0. It should be very high
   (>0.95) since early layers should be minimally affected by instruction
   tuning.

---

## 7. Phase 3: Correlational Analysis I — Probing (Week 4)

### 7.1 Goal

Determine whether cultural knowledge representations are preserved after
instruction tuning by training linear probes on base model activations
and testing their generalization to instruct model activations. This is
the cross-model transfer test: the primary evidence for or against H1
(representational erasure).

### 7.2 Linear Probing Setup

**Probe architecture:** L2-regularized logistic regression (sklearn
LogisticRegression with multinomial loss). This is intentionally simple.
A linear probe can only decode information that is linearly separable in
the representation space. If it succeeds, the information is explicitly
and accessibly encoded. If it fails, the information may still be present
but in a non-linear form. For our purposes, linear probing sufficiency
is the standard.

**Regularization:** L2 penalty with cross-validated regularization strength
(C parameter). Use sklearn's LogisticRegressionCV with 5-fold cross-validation,
Cs=10 (testing 10 logarithmically spaced regularization values), and
balanced class weights to handle multi-class imbalance.

**Feature scaling:** StandardScaler applied WITHIN each cross-validation
fold using sklearn Pipeline. This prevents data leakage from test fold
statistics into training. This is a common methodological error in probing
studies that we avoid.

### 7.3 Probing Tasks

**Task 1: Cultural Attribute Classification (16-class).**
Given an activation vector from a question, predict which of the 16
cultural attributes (Art, Costume, Cuisine, ..., Transport) the question
belongs to. This tests whether the model encodes "what kind of cultural
knowledge this is" in its representations.

Chance level: 1/16 = 6.25%. Majority class baseline: ~12% (Tourism).

**Task 2: State Classification (36-class).**
Predict which Indian state the question is about. This tests geographic
cultural knowledge encoding.

Chance level: 1/36 = 2.78%. Majority class baseline: ~7.85% (Telangana).

**Task 3: Question Type Classification (4-class).**
Predict whether the question is Association, Country Prediction, General
Awareness, or State Prediction. This tests whether the model encodes
the structural format of the question.

Chance level: 25%. Classes are roughly balanced.

### 7.4 The Cross-Model Transfer Protocol

This is the key experiment. For each probing task and each layer:

1. **Train on base, test on base (baseline):** Train a probe on 75% of
   base model activations. Test on the remaining 25% of base activations.
   Record accuracy as Acc(base→base).

2. **Train on base, test on instruct (transfer):** Take the same probe
   (trained on base activations) and evaluate it on the corresponding 25%
   of instruct model activations. Record accuracy as Acc(base→instruct).

3. **Compute transfer rate:** Transfer_rate = Acc(base→instruct) / Acc(base→base).

A transfer rate of 100% means the probe learned on base activations
generalizes perfectly to instruct activations. The two models share
identical representational geometry for that information. A transfer rate
near 0% means the representations are completely different.

The pilot study found transfer rates of 96-99% for attributes and states.
If LLaMA-3.1-8B shows similar rates, this is strong evidence against H1
(representational erasure).

### 7.5 Cross-Model Correctness Probing

Standard probing predicts labels from the same model's activations. This
creates a circularity risk: the probe might learn features correlated
with correctness rather than with cultural knowledge content. To address
this, we implement cross-model correctness probing.

Train a binary probe on base model activations to predict whether the
INSTRUCT model will answer correctly (not whether the base model will
answer correctly). And vice versa.

If this cross-model correctness probe achieves high accuracy, it means
the base model's representations contain information that predicts the
instruct model's behavioral outcome. If it achieves only near-chance
accuracy (around 50-60%), it means behavioral outcomes are orthogonal
to representational content, supporting the claim that suppression
operates through output-level mechanisms.

### 7.6 MDL Probing

Minimum Description Length probing (Voita & Titov, EMNLP 2020) provides
an information-theoretic complement to accuracy-based probing. Instead
of asking "can we decode this information?" it asks "how efficiently
can we encode this information?"

**The key insight:** Two probes might achieve identical accuracy but differ
in how many bits they need to transmit their predictions. A probe that
compresses labels efficiently has found a strong regularity in the
representation space. A probe that achieves the same accuracy but requires
more bits has found a weaker, more distributed pattern.

**Online prequential coding:** Process the data in chunks of increasing
size (10%, 20%, 30%, 50%, 75%, 100% of training data). At each step,
the probe trained on data so far encodes the next chunk using negative
log-likelihood as the code length. The cumulative bits-per-sample measures
how quickly the probe learns to compress labels.

If base and instruct models produce identical learning curves, their
representations encode cultural knowledge with identical efficiency.
This is a stricter test than probe accuracy alone.

**Isomorphism ratio:** Train an MDL probe on base activations. Compute
the total description length on base activations (MDL_base) and on
instruct activations (MDL_instruct). The ratio MDL_instruct / MDL_base
should be close to 1.0 if representations are isomorphic. We define
isomorphism as a ratio within 1.0 ± 0.10.

### 7.7 Baselines and Statistical Rigor

For every probing result, compute three baselines:

1. **Chance level:** 1/k where k is the number of classes.
2. **Majority class:** Always predict the most common class.
3. **Random features:** Train the same probe on shuffled activations
   (destroying the correspondence between activations and labels). This
   measures probe capacity independent of representation quality.

Statistical tests: Paired t-tests across cross-validation folds with
Bonferroni correction for multiple comparisons (we have 8 layers x 3
tasks x 2 models = 48 primary tests). Report Cohen's d effect sizes
with 95% bootstrap confidence intervals (400 bootstrap samples).

---

## 8. Phase 4: Correlational Analysis II — Distributional Divergence (Week 5)

### 8.1 KL Divergence Between Base and Instruct Activations

KL divergence directly quantifies how different the base and instruct
models' activation distributions are at each layer. While probing measures
what information is decodable, KL divergence measures how much the
distributions themselves have shifted.

**Mathematical setup:** At each layer L, we have two sets of activation
vectors: one from the base model and one from the instruct model. Model
each set as a multivariate Gaussian with mean vector mu and covariance
matrix Sigma. The KL divergence from base (P) to instruct (Q) is:

KL(P||Q) = 0.5 * [tr(Sigma_Q^{-1} Sigma_P) + (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P) - d + ln(|Sigma_Q| / |Sigma_P|)]

where d = 4096 is the hidden dimension.

**Why Gaussian assumption:** With 21,726 samples in 4096 dimensions, the
sample-to-dimension ratio is about 5.3. This is sufficient for Gaussian
estimation with shrinkage but would be insufficient for non-parametric
density estimation. The Gaussian assumption is standard in MI work and
provides closed-form computation.

**Ledoit-Wolf shrinkage:** The empirical covariance matrix of a 4096-dim
vector with ~21K samples is ill-conditioned. Ledoit-Wolf shrinkage
computes an optimal convex combination of the empirical covariance and
a scaled identity matrix, minimizing expected squared error. This
ensures numerically stable covariance matrices. Add small diagonal
regularization (epsilon = 1e-6) as an extra safety margin.

**Jensen-Shannon divergence:** As a symmetric complement to KL, compute
JS divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where
M is the mixture distribution with mean (mu_P + mu_Q)/2 and covariance
(Sigma_P + Sigma_Q)/2. JS is bounded and symmetric, providing a
complementary perspective on distributional distance.

### 8.2 Stratified Analysis

Compute KL divergence at multiple granularities:

**Overall:** All questions together. This gives the global layer-wise
trajectory. The pilot study found stable KL through early/middle layers
with a 2.9x spike at the final layer.

**By behavioral group:** Separate the suppression, enhancement, and
control groups. If all three groups show the same amplification factor
at the final layer (as the pilot found: 2.78x for all three groups),
this supports H2/H3's prediction that the representational transformation
is universal, not behavior-specific.

**By cultural attribute:** Compare KL divergence across the 16 attributes.
If identity-marking attributes (Religion, Dance, Costume) show higher
divergence than neutral informational attributes (History, Transport),
this reveals content selectivity. The pilot found Religion (806 nats)
vs Nightlife (358 nats), a 2.25x difference.

**By state:** Compare KL divergence across the 36 states. If culturally
distinctive regions (Northeast tribal states) show higher divergence than
cosmopolitan regions (Delhi, Mumbai), this reveals geographic selectivity.

**Minimum sample thresholds:** Require at least 50 questions per slice for
KL estimation. For slices below this threshold, aggregate upward to the
nearest parent category (e.g., small states aggregate to region).

### 8.3 Tuned Lens Analysis

The tuned lens (Belrose et al., 2023) provides a layer-by-layer view of
what the model is "about to predict" at each processing stage. For each
layer, a learned affine transformation maps the residual stream to the
vocabulary space, producing a distribution over tokens. By examining
this distribution layer by layer, we can see the prediction "forming"
through the network.

**Why this matters for our study:** If the base model's tuned lens shows
the correct cultural answer emerging at Layer 20 and staying stable
through Layer 31, while the instruct model shows the same correct answer
emerging at Layer 20 but then abruptly shifting to a different token at
Layer 30 or 31, this directly VISUALIZES the gating mechanism. The model
"knows" the answer (visible in intermediate layers) but "chooses" to
say something else (visible at the output layer).

**Implementation:** Use the tuned-lens Python package or train thin affine
probes at each layer. For a subset of suppressed questions (where base
answers correctly but instruct fails), generate layer-by-layer prediction
trajectories. Plot the probability assigned to the correct answer token
at each layer for both models.

**Expected result under H2:** The probability of the correct answer in
the instruct model rises through intermediate layers (matching the base
model) then drops sharply at Layer 30-31. The probability of an alternative
answer (refusal token, "I don't know", or a wrong answer) rises at exactly
the layer where the correct answer drops.

**Expected result under H1:** The probability of the correct answer in the
instruct model never rises above chance at any layer. Knowledge is absent
throughout the network.

**Expected result under H3:** The probability of the correct answer in the
instruct model drops gradually across many layers, not sharply at one.

This is one of the most visually compelling analyses we can produce.
A single well-chosen figure showing the base model's green line staying
high while the instruct model's red line drops at the final layer could
be the paper's most memorable image.

---

## 9. Phase 5: Causal Analysis I — Activation Patching (Week 6)

### 9.1 Why Causal Evidence Matters

Everything in Phases 3-4 is correlational. We observe that KL divergence
spikes at the final layer. We observe that probes transfer nearly perfectly.
But we have not shown that the final layer CAUSES the behavioral change.
The spike could be an epiphenomenon — a side effect of some other mechanism
that actually controls behavior.

Activation patching provides the causal test. If we replace the instruct
model's activations at Layer 31 with the base model's activations and the
model suddenly starts answering cultural questions correctly again, then
Layer 31 is not just correlated with the behavioral change — it causes it.

### 9.2 How Activation Patching Works

The procedure for a single question:

1. **Clean run (base model):** Run the base model on the question. Record
   the output logits and the activation at each layer. The base model
   answers correctly, producing logit L_correct for the correct answer.

2. **Corrupted run (instruct model):** Run the instruct model on the same
   question. It answers incorrectly, producing logit L_incorrect for the
   correct answer.

3. **Patched run:** Run the instruct model on the question, but at the
   target layer, REPLACE the instruct model's activation with the base
   model's activation (recorded in step 1). Continue the forward pass
   from the patched point through the remaining layers. Record the
   output logits.

4. **Measure effect:** Compare the patched run's logit for the correct
   answer against the corrupted (instruct) run's logit. If the patched
   logit is much closer to the clean (base) logit, patching at this
   layer restores the correct behavior.

### 9.3 The Effect Metric: Logit Difference Recovery

Define the logit difference as:

logit_diff = logit(correct_answer) - max(logit(wrong_answers))

A positive logit_diff means the model would select the correct answer.
A negative logit_diff means it would select a wrong answer.

The patching effect at layer L is:

effect(L) = (logit_diff_patched(L) - logit_diff_corrupted) / (logit_diff_clean - logit_diff_corrupted)

This normalizes the effect to a 0-1 scale. An effect of 0 means patching
at layer L did nothing. An effect of 1 means patching at layer L fully
restored the base model's behavior. An effect greater than 1 means
patching "over-corrected."

### 9.4 Layer Sweep

Patch each layer individually (layers 0, 4, 8, 14, 20, 26, 28, 29, 30,
31) and measure the effect. The dense sampling at layers 28-31 is
important because we expect the critical transition to happen in this
range. If the effect peaks at Layer 31 (analogous to Layer 28 in the
28-layer Qwen model), H2 is supported.

**Additional granularity:** If Layer 31 shows a large effect, also patch
layers 29 and 30 individually to determine whether the effect is
concentrated at a single layer or spread across the final few layers.

### 9.5 Question Selection for Patching

Activation patching is expensive because it requires a separate forward
pass for each layer being patched. We cannot afford to patch all 21,726
questions at all 10 layers.

**Selection strategy:** Use the suppression group from Phase 1. Among
suppressed questions, select 500 questions stratified by cultural attribute
to ensure coverage of the content selectivity dimension. Also select 200
enhancement questions and 200 control questions as controls.

For suppressed questions, the expected result is that patching the late
layer flips the model from incorrect to correct. For enhancement questions,
patching should flip from correct to incorrect (the base model gets it
wrong, so its activations should make the instruct model get it wrong too).
For control questions, patching should have minimal effect (both models
agree, so swapping activations does not change the outcome).

### 9.6 Statistical Analysis

For each layer L, report:

1. **Mean patching effect** across all 500 suppressed questions, with 95%
   confidence interval.
2. **Flip rate:** What fraction of suppressed questions switch from
   incorrect to correct after patching? This is the most interpretable
   metric — "patching Layer 31 recovers 65% of suppressed cultural
   knowledge answers."
3. **Effect by cultural attribute:** Does patching recover Religion
   questions at a different rate than History questions? This tests
   whether the gating mechanism is content-specific even at the causal
   level.

---

## 10. Phase 6: Causal Analysis II — Component-Level and SAE Analysis (Week 7)

### 10.1 Attribution Patching at Layer 31

If the layer sweep confirms Layer 31 as the gating site, the next question
is: WHICH component within Layer 31 implements the gate? Layer 31 contains
multiple components: 32 attention heads, the MLP sublayer, and the residual
connection. Attribution patching efficiently identifies which components
contribute most to the gating effect.

**How attribution patching works:** Instead of running a full forward pass
for each component (which would require 33+ passes per question),
attribution patching uses a first-order Taylor approximation. Compute the
gradient of the logit difference with respect to each component's output,
multiply by the difference between clean (base) and corrupted (instruct)
activations at that component. This gives an approximate effect score for
each component in a single backward pass.

**Mathematically:** For component c at layer L, the attribution score is:

attr(c) = (a_base(c) - a_instruct(c)) dot grad_c(logit_diff)

where grad_c(logit_diff) is the gradient of the logit difference with
respect to component c's output during the instruct model's forward pass.

**Validation:** After identifying the top 10-20 components by attribution
score, validate the top 5 with full activation patching (replacing just
that component's output). Attribution patching is an approximation; full
patching confirms it.

### 10.2 Sparse Autoencoder Model-Diffing

Sparse autoencoders (SAEs) decompose activation vectors into a sparse
sum of interpretable features. By comparing SAE feature activations
between the base and instruct models, we can identify specific features
that are differentially active — features that "turn on" in one model but
not the other.

**Pre-trained SAE availability:** Llama Scope provides SAEs for every layer
of LLaMA-3.1-8B-Base with 32K and 128K feature widths. Goodfire provides
SAEs for LLaMA-3.1-8B-Instruct. We load these pre-trained SAEs rather
than training our own.

**Model-diffing procedure:**

1. For each question, pass the base model's Layer 31 activation through
   the base SAE to get feature activations f_base (a sparse vector of
   32K dimensions, most entries zero).

2. Pass the instruct model's Layer 31 activation through the instruct SAE
   to get feature activations f_instruct.

3. Compute the difference: delta_f = f_instruct - f_base.

4. Aggregate delta_f across all suppressed questions. Features with
   consistently positive delta_f are MORE active in the instruct model.
   Features with consistently negative delta_f are LESS active.

5. The features with the largest |mean(delta_f)| / std(delta_f) ratio
   across suppressed questions are the candidate "gating features."

**Interpretability verification:** For the top 10 gating feature candidates,
examine their max-activating examples across a general text corpus. If a
feature's max-activating examples involve refusal language, safety hedging,
or cultural sensitivity markers, this provides interpretable evidence for
what the gate "is."

**Causal validation:** For the top 5 candidate features, clamp them to
zero during the instruct model's forward pass and measure whether
suppressed questions flip to correct answers. This establishes that the
feature causally controls the gating behavior.

### 10.3 The Cross-Model SAE Alignment Problem

A subtlety: the base SAE and instruct SAE were trained independently.
Their feature dictionaries are not aligned. Feature #1234 in the base SAE
is not the same concept as feature #1234 in the instruct SAE.

**Three approaches to handle this:**

**Approach A: Same-SAE diffing.** Use only the base SAE. Pass both base
and instruct activations through the base SAE. This works because the
base SAE was trained on base activations and can approximately reconstruct
instruct activations (since we have shown the representations are 96-99%
preserved). The residual error from the instruct activations tells us what
the instruct model has added that the base model does not have.

**Approach B: Separate-SAE diffing.** Use each model's own SAE and compare
feature activation distributions. Identify features by their interpretable
descriptions rather than by index.

**Approach C: Matched features.** Use cosine similarity between SAE decoder
weight vectors to find matching features across the two SAEs. Features with
cosine similarity above 0.9 are treated as "the same concept."

Approach A is simplest and most defensible. We use it as the primary method
and report Approach B results in the appendix.

---

## 11. Phase 7: Cross-Benchmark Validation (Week 8)

### 11.1 CulturalBench Integration

Repeat the complete analysis pipeline on CulturalBench. Specifically:

1. **Behavioral evaluation** (already done in Week 1): Accuracy, suppression
   rate, enhancement rate on 1,696 questions.

2. **KL divergence trajectory:** Compute layer-wise KL divergence on
   CulturalBench questions. If the same late-layer spike pattern appears,
   this validates that the finding is not Sanskriti-specific.

3. **Activation patching:** Run the layer sweep on 200 CulturalBench
   suppressed questions (if there are enough). If patching Layer 31
   recovers suppressed answers on CulturalBench too, the causal finding
   generalizes.

4. **Cross-benchmark comparison:** Overlay the Sanskriti and CulturalBench
   KL divergence trajectories on the same plot. Compute the correlation
   between attribute-level divergence patterns (comparing cultural
   categories shared between the two benchmarks).

### 11.2 What Cross-Benchmark Validation Buys Us

If the late-layer gating pattern appears on both benchmarks, we can claim
the finding is about how instruction tuning handles cultural knowledge
in general, not about Indian cultural knowledge specifically or about
the Sanskriti benchmark's particular quirks.

If the pattern appears on Sanskriti but NOT on CulturalBench, we need to
investigate why. Possible explanations: CulturalBench questions are
structured differently. The instruct model may not suppress non-Indian
cultural knowledge as much. The effect may be culture-specific rather
than universal. Any of these would be an interesting finding in itself.

### 11.3 Compiling All Results

By the end of Week 8, we should have:

| Result | Source | Method |
|--------|--------|--------|
| Base vs instruct accuracy on Sanskriti (3 tiers) | Phase 1 | Behavioral eval |
| Suppression/enhancement rates | Phase 1 | Behavioral labels |
| Cross-model probe transfer rates (8 layers) | Phase 3 | Linear probing |
| MDL isomorphism ratios | Phase 3 | MDL probing |
| KL divergence layer trajectory | Phase 4 | Gaussian KL |
| KL divergence by attribute/state/behavioral group | Phase 4 | Stratified KL |
| Tuned lens prediction trajectories | Phase 4 | Tuned lens |
| Activation patching layer sweep | Phase 5 | Causal patching |
| Patching flip rate by layer | Phase 5 | Causal patching |
| Attribution patching component scores | Phase 6 | Attribution |
| SAE gating feature candidates | Phase 6 | SAE model-diffing |
| CulturalBench replication of KL trajectory | Phase 7 | Cross-benchmark |
| CulturalBench patching results | Phase 7 | Cross-benchmark |

---

## 12. Phase 8: Paper Writing (Weeks 9-10)

### 12.1 Paper Structure

BlackboxNLP 2026 accepts 8-page archival papers (excluding references and
appendices) in ACL format using EMNLP 2026 style guidelines. Double-blind
review requires full anonymization. If accepted, the camera-ready version
gets one extra page (9 total) to address reviewer comments.

**Abstract (15 lines).** State the problem (instruction tuning degrades
cultural knowledge), the approach (compare LLaMA-3.1-8B base vs instruct
using probing, KL divergence, activation patching, and SAE analysis on
Sanskriti + CulturalBench), the key finding (cultural knowledge is
preserved but gated at the final layer, causally validated by activation
patching), and the implication (bias may be recoverable without retraining).

**1. Introduction (1.5 pages).** Open with a concrete example of cultural
suppression (e.g., "What is the traditional dance of Tamil Nadu?" — base
answers Bharatanatyam, instruct does not). State the research question.
Present the three hypotheses. Summarize contributions (3 numbered items).

**2. Related Work (0.75 pages).** Three threads: probing methodology
(Belinkov 2022, Voita & Titov 2020), MI of alignment (Lee et al. 2024,
Arditi et al. 2024, Wang et al. 2025), cultural bias in LLMs (Chiu et al.
2024, Naous et al. 2025, Maji et al. 2025). Position our work as the
first to apply causal MI methods to study alignment's effect on cultural
knowledge.

**3. Experimental Setup (1.5 pages).** Models, benchmarks (Sanskriti +
CulturalBench with EDA-informed caveats), behavioral evaluation
methodology, activation extraction, probing setup, KL divergence
methodology, activation patching procedure.

**4. Results: Representational Preservation (1 page).** Cross-model probe
transfer rates. MDL isomorphism. The message: knowledge survives instruction
tuning.

**5. Results: Late-Layer Gating (1.5 pages).** KL divergence trajectory
(the spike). Tuned lens visualization. Activation patching (the causal
validation). Component-level analysis. SAE gating features. The message:
a specific mechanism in the final layer blocks expression of preserved
knowledge.

**6. Results: Content Selectivity (0.5 pages).** Attribute-level and
state-level divergence patterns. Religion vs Nightlife. Tribal vs
metropolitan. CulturalBench cross-validation.

**7. Discussion (0.75 pages).** The two-stage gating model. Implications
for bias mitigation. Limitations (single model family, Gaussian assumption,
benchmark limitations). Future work (multi-model replication, steering
vectors for cultural recovery).

**8. Conclusion (0.25 pages).** One paragraph summary.

**References.** Approximately 25-30 citations.

**Appendix.** Full behavioral metrics tables. Per-attribute and per-state
breakdowns. Additional tuned lens examples. SAE feature descriptions.

### 12.2 Key Figures

The paper needs 5-7 figures. Planned figures:

**Figure 1: KL divergence layer trajectory.** X-axis: layer. Y-axis: KL
divergence (nats). Two lines: Sanskriti and CulturalBench. Both should
show the late-layer spike. This is the paper's signature figure.

**Figure 2: Cross-model transfer rates.** Bar chart. X-axis: layer.
Y-axis: transfer rate. Bars for attribute classification and state
classification. All bars near 100%.

**Figure 3: Activation patching effect.** X-axis: layer patched. Y-axis:
mean patching effect (0-1 scale). Clear peak at Layer 31.

**Figure 4: Tuned lens trajectories.** Two panels (base and instruct)
showing probability of the correct answer token at each layer. Base
stays high. Instruct drops at final layer.

**Figure 5: Group-level KL divergence.** Three lines (suppression,
enhancement, control) showing identical amplification factor at the
final layer.

**Figure 6: Attribute-level KL divergence at Layer 31.** Bar chart
sorted by KL value. Religion at top, Nightlife at bottom.

**Figure 7 (if space permits): SAE gating features.** Top 5 differential
features with their interpretable descriptions and activation patterns.

### 12.3 Writing Style

Follow ACL convention. Active voice where possible. Present tense for
methods and established facts. Past tense for specific experimental
results. Avoid hedging language except in the limitations section.
Numbers should be precise (96.45%, not "approximately 96%").

The paper's narrative arc: "Conventional wisdom says alignment modifies
knowledge. We show it modifies access to knowledge. Here is the mechanism.
Here is causal proof. Here is what to do about it."

---

## 13. Risk Mitigation

### 13.1 What If the Late-Layer Spike Does Not Replicate?

If LLaMA-3.1-8B does NOT show a KL divergence spike at the final layer,
we have a different but equally publishable finding: the gating mechanism
is architecture- or scale-dependent. Report the Qwen pilot result as
context and show that LLaMA-3.1-8B behaves differently. Investigate WHERE
the divergence does occur and what this tells us about different alignment
implementations.

### 13.2 What If Suppression Rate Is Too Low?

If LLaMA-3.1-8B-Instruct outperforms base on nearly everything (very low
suppression rate), the paper shifts from "gating mechanism" to "alignment
tax is minimal at 8B scale." This is still valuable because it contrasts
with the Qwen-1.5B finding and suggests scale-dependent effects.

Mitigation: CulturalBench may show more suppression than Sanskriti on
certain cultural categories, providing a different angle.

### 13.3 What If Activation Patching Shows No Clear Effect?

If patching individual layers does not cleanly flip behavioral outcomes,
this supports H3 (distributed modulation) over H2 (localized gating).
Report this as a finding: "the alignment transformation is distributed
rather than localized, resisting surgical intervention." This has
different but equally important practical implications.

### 13.4 What If We Run Out of Time?

The plan is ordered by priority. With 14.5 weeks and the confirmed
July 17 deadline, time pressure is much lower than the original 10-week
plan. But if unexpected issues consume weeks (debugging model loading,
cluster downtime, a sanity check failure requiring re-runs), here are
the fallback tiers:

**Minimum viable paper (Weeks 1-6 + Weeks 11-14):** Behavioral evaluation
+ probing + KL divergence + tuned lens + activation patching layer sweep.
This provides both correlational and causal evidence for the gating
hypothesis on LLaMA-3.1-8B. Adequate for BlackboxNLP archival track.

**Strong paper (Weeks 1-8 + Weeks 11-14):** Add SAE model-diffing and
cross-benchmark CulturalBench validation. This provides interpretable
features and generalization evidence. Competitive for best paper.

**Best paper candidate (Weeks 1-10 + Weeks 11-14):** Full pipeline
including Gemma-2-9B cross-architecture replication. This preempts the
main reviewer objection and establishes the finding as architecture-general.
This is the target.

---

## 14. Success Criteria

### 14.1 Must-Have Results (Non-Negotiable)

1. Behavioral evaluation on Sanskriti (3 tiers) with all sanity checks
   passing.
2. Cross-model probe transfer rates at 8 layers.
3. KL divergence layer trajectory.
4. At least one causal experiment (activation patching at the candidate
   gate layer).

### 14.2 Should-Have Results (Strongly Desired)

5. MDL isomorphism analysis.
6. Tuned lens prediction trajectories.
7. Activation patching full layer sweep.
8. CulturalBench behavioral evaluation and KL divergence comparison.

### 14.3 Nice-to-Have Results (If Time Permits)

9. Attribution patching at the component level.
10. SAE model-diffing with feature identification.
11. Causal validation of SAE gating features.
12. CulturalBench activation patching replication.

### 14.4 Quantitative Targets

These are predictions, not hard requirements. The experiment will
produce whatever it produces. But to manage expectations:

| Metric | Target Range | Red Flag |
|--------|-------------|----------|
| Base accuracy (Tier 2) | 45-65% | Below 30% or above 80% |
| Instruct accuracy (Tier 2) | 50-75% | Below base accuracy |
| Suppression rate (Tier 2) | 5-15% | Below 2% or above 30% |
| Cross-model transfer rate | 90-100% | Below 80% |
| MDL isomorphism ratio | 0.90-1.10 | Outside 0.70-1.30 |
| KL divergence final-layer amplification | 1.5-4.0x | Below 1.2x |
| Patching effect at candidate layer | 0.3-0.8 | Below 0.1 |
| Patching flip rate at candidate layer | 20-70% | Below 10% |

---

## 15. Week-by-Week Detailed Schedule

### Week 1 (April 7-13): Setup and Behavioral Evaluation

**Monday-Tuesday:** Download both models from HuggingFace. Verify SHA256
checksums. Load each model on a single A100 and confirm inference works.
Generate 10 test completions from each model to verify correct behavior.
Set up the conda environment with transformers, torch, sklearn, and
TransformerLens.

**Wednesday:** Implement prompt templates for both models. Run a pilot
batch of 100 Sanskriti questions through both models. Manually inspect
outputs to verify answer extraction logic. Debug null predictions. Iterate
on prompt wording if null rate exceeds 5%.

**Thursday-Friday:** Run full Sanskriti evaluation (21,726 questions x 2
models). Batch size 32, greedy decoding. Estimated time: 3-4 hours per
model on A100. Run CulturalBench evaluation (1,696 questions x 2 models).

**Weekend:** Begin behavioral analysis. Compute per-model accuracy, null
rates, position distributions.

### Week 2 (April 14-20): Behavioral Analysis

**Monday-Tuesday:** Assign behavioral labels to every question. Compute
suppression, enhancement, control rates at all three tiers. Run all
sanity checks from Section 5.5. If any check fails, debug and re-run.

**Wednesday:** Compute entity-level behavioral labels using the 8,156
entity keys. Cross-tabulate behavioral labels with attributes and states.
Generate the decision about whether to proceed (all sanity checks must
pass).

**Thursday-Friday:** Compute per-attribute and per-state behavioral rates.
Identify the most-suppressed and most-enhanced attributes and states.
Compare with the Qwen pilot to see if patterns replicate.

**Weekend:** Write up Phase 1 results in a working document. Create
preliminary figures (accuracy bar charts, suppression rate by attribute).

### Week 3 (April 21-27): Activation Extraction

**Monday:** Set up activation extraction hooks using PyTorch forward hooks.
Test on 10 questions to verify correct shapes and values. Verify that
mean-pooled and last-token activations produce reasonable values.

**Tuesday-Wednesday:** Extract activations from the base model at all 8
layers for all 23,422 questions (Sanskriti + CulturalBench). Estimated
time: 5-6 hours.

**Thursday-Friday:** Extract activations from the instruct model at all 8
layers for all 23,422 questions. Run validation checks from Section 6.7.

**Weekend:** Organize activation files on disk. Verify storage requirements.
Create an index mapping from question ID to activation file offset for
fast retrieval during probing.

### Week 4 (April 28 - May 4): Probing

**Monday-Tuesday:** Implement and run linear probing for all three tasks
(attribute, state, question type) at all 8 layers for both models. Run
cross-model transfer experiments.

**Wednesday-Thursday:** Implement and run MDL probing with online
prequential coding. Compute isomorphism ratios. Implement cross-model
correctness probing.

**Friday:** Analyze all probing results. Compare with Qwen pilot numbers.
Compute baselines, statistical tests, and confidence intervals.

**Weekend:** Write up probing results. Create transfer rate figures.

### Week 5 (May 5-11): KL Divergence and Tuned Lens

**Monday-Tuesday:** Compute KL divergence at all 8 layers. Compute
stratified KL by behavioral group, attribute, and state. Generate the
layer trajectory plot and compare with the Qwen pilot.

**Wednesday-Thursday:** Run tuned lens analysis on a subset of 200
suppressed questions, 100 enhancement questions, and 100 control questions.
Generate prediction trajectory plots.

**Friday:** Combine probing and KL results to determine which layers are
candidate gate layers. This determines the target for activation patching
in Week 6.

**Weekend:** Write up distributional analysis results.

### Week 6 (May 12-18): Activation Patching

**Monday:** Set up activation patching using TransformerLens hooks or
nnsight deferred execution. Test on 10 questions at one layer to verify
the implementation produces sensible logit differences.

**Tuesday-Wednesday:** Run the full layer sweep (10 layers) on 500
suppressed questions + 200 enhancement + 200 control. Estimated time:
2-3 hours per layer, so 20-30 hours total. Parallelize across GPUs if
multiple A100s are available.

**Thursday-Friday:** Analyze patching results. Compute mean effect, flip
rates, and per-attribute breakdowns. Determine the causally validated
gate layer.

**Weekend:** Write up causal results. Create the patching effect figure.

### Week 7 (May 19-25): Component-Level Patching

**Monday-Tuesday:** Run attribution patching at the confirmed gate layer
to identify specific components (attention heads vs MLP). Score all 32
attention heads + MLP sublayer. Rank by attribution magnitude.

**Wednesday-Thursday:** Validate the top 5 components with full activation
patching. For each component, patch it alone and measure the effect on
the 500 suppressed questions. Determine whether the gate is primarily
in attention heads, the MLP, or distributed.

**Friday:** Analyze component-level results. If a small number of
attention heads dominate, this is a clean "circuit" finding. If the
effect is spread across many components, this is still informative but
tells a different story (distributed gate rather than single gate).

**Weekend:** Write up component-level results.

### Week 8 (May 26 - June 1): SAE Analysis

**Monday:** Download Llama Scope SAEs for LLaMA-3.1-8B-Base at the gate
layer (32K and 128K widths). Download Goodfire SAEs for LLaMA-3.1-8B-
Instruct at the same layer. Verify loading and reconstruction quality
on 100 sample activations.

**Tuesday-Wednesday:** Run SAE model-diffing using Approach A (pass both
models' activations through the base SAE). Compute differential feature
activations across all suppressed questions. Identify top 20 candidate
gating features by |mean(delta_f)| / std(delta_f).

**Thursday:** Examine max-activating examples for top 10 gating features
across a general corpus. Assess interpretability — do these features
correspond to refusal, safety hedging, cultural sensitivity markers, or
something else?

**Friday:** Attempt causal validation: clamp top 5 features to zero during
instruct model forward pass. Measure flip rate on suppressed questions.

**Weekend:** Write up SAE analysis results.

### Week 9 (June 2-8): Cross-Benchmark Validation

**Monday-Tuesday:** Run KL divergence on CulturalBench questions at all 8
layers. Overlay CulturalBench and Sanskriti trajectories on the same plot.
Assess whether the late-layer spike replicates.

**Wednesday:** Run activation patching on CulturalBench suppressed questions
(if enough exist — need at least 100). Compare patching effects with
Sanskriti.

**Thursday-Friday:** Compile cross-benchmark comparison. Write the
cross-validation section.

**Weekend:** Decision point: do we have enough time and compute to add a
second model family? If Weeks 1-9 went smoothly with no major debugging,
proceed to Week 10 (Gemma replication). If we are behind schedule, skip
Week 10 and go directly to writing.

### Week 10 (June 9-15): Optional Gemma-2-9B Replication

This week is optional. It happens only if the LLaMA experiments are
complete and the results are clear.

**If proceeding:** Download Gemma-2-9B and Gemma-2-9B-IT. Run behavioral
evaluation on Sanskriti (21,726 questions x 2 models). Compute behavioral
labels. Extract activations at the gate layer equivalent (Gemma has 42
layers, so the final layer is Layer 41). Compute KL divergence at 4 layers
(early, middle, late, final). Run activation patching at the final layer
on 200 suppressed questions. Load Gemma Scope SAEs and run model-diffing.

**What this buys:** Cross-architecture replication. If the late-layer gating
pattern appears in both LLaMA and Gemma, the finding is architecture-
general, not model-specific. This preempts the single most common reviewer
objection.

**If skipping:** Use this week for additional analysis depth on LLaMA.
Options: more granular patching (every layer from 24-31 instead of just
the selected layers), larger sample sizes for patching experiments,
additional tuned lens examples, or deeper SAE feature analysis.

### Week 11 (June 16-22): First Draft

**Monday-Wednesday:** Write Sections 1-3 (Introduction, Related Work,
Experimental Setup). The introduction should open with the Tamil Nadu /
Bharatanatyam example, state the research question, present the three
hypotheses, and list contributions.

**Thursday-Friday:** Write Sections 4-5 (Representational Preservation,
Late-Layer Gating). These are the results sections. Each result gets its
own subsection with a figure or table, a statistical test, and one
paragraph of interpretation.

**Weekend:** Write Sections 6-8 (Content Selectivity, Discussion,
Conclusion). Write the abstract last.

### Week 12 (June 23-29): Co-Author Iteration

Share the full draft with Pragati by Monday morning. Allow 3-4 days for
reading and feedback.

**Thursday-Friday:** Incorporate feedback. Common revision targets: tighten
the abstract, sharpen the contributions list, strengthen the limitations
section, improve figure captions, fix any logical gaps in the argument.

**Weekend:** Generate publication-quality figures. All figures should be
vector graphics (PDF or SVG) with readable fonts at the ACL column width.
Match color scheme across all figures for visual coherence.

### Week 13 (June 30 - July 6): Second Draft

**Monday-Tuesday:** Complete the second draft incorporating all feedback.
Read the paper fresh from start to finish. Mark every claim that is not
backed by a specific number. Mark every number that does not have a
confidence interval.

**Wednesday-Thursday:** Write the appendix. Include: full behavioral
metrics tables, per-attribute and per-state breakdowns, all probing
baselines, additional tuned lens examples, SAE feature descriptions,
prompt templates for both models, hyperparameter tables.

**Friday:** Write the reproducibility statement and broader impact
statement (optional but recommended for BlackboxNLP).

**Weekend:** Final read-through by both authors independently.

### Week 14 (July 7-13): Anonymization and Final Polish

**Monday:** Anonymization pass. Remove all author names, affiliations,
acknowledgments. Replace CMU-specific paths (/home/anshulk/) with
generic placeholders. Replace cluster names (Babel) with "university
computing cluster." If code is referenced, set up an anonymous repo
(e.g., Anonymous GitHub). Do NOT thank specific advisors or mentors by
name.

**Tuesday-Wednesday:** Format according to ACL template. Verify:
- Paper is within 8 pages (main text) + references + appendix
- All figures render correctly in the ACL two-column layout
- All tables fit within column width
- References follow ACL format (use acl_natbib)
- No orphan lines, no figures without references in text
- Page numbers are present

**Thursday:** Create the OpenReview submission. Upload the PDF. Fill in
metadata (title, abstract, keywords, conflicts of interest). Verify the
submission renders correctly in OpenReview's preview.

**Friday:** Final check. Both authors read the submission PDF one more
time. Verify: no deanonymizing information, all numbers match between
text and tables, abstract fits on one page, all figures are referenced.

### Week 14.5 (July 14-17): Submission Buffer

Three-day buffer for last-minute issues. If no issues, submit on
July 14 rather than waiting until the deadline.

**July 17, 11:59 PM AoE: Submit.**

After submission: celebrate. Then immediately begin thinking about
the poster or oral presentation for October 28-29.

---

## 16. Open Questions Requiring Decisions

These are design choices that should be discussed with Pragati before
experiments begin.

### 16.1 Do We Include the Qwen2-1.5B Pilot Results?

**Option A:** Include Qwen results as a "pilot study" in the introduction
or appendix, then present LLaMA results as the primary contribution.
Advantage: shows the finding replicates across model families.
Disadvantage: adds complexity and the Qwen methodology had known issues.

**Option B:** Omit Qwen results entirely. Present LLaMA as the sole model.
Advantage: cleaner paper, avoids relitigating the Qwen methodology.
Disadvantage: misses the cross-model replication argument.

**Recommendation:** Option A, with Qwen results in a 1-paragraph summary
and a single table in the appendix. The replication across model families
is too valuable to omit.

### 16.2 Should We Include the Sentence Generation Step?

The Qwen pilot generated 33,522 synthetic sentences from MCQ answers
using Claude, then extracted activations from these sentences. This added
a step (LLM-generated text) that reviewers might question.

For the LLaMA study, we extract activations directly from the MCQ
questions as presented to the model. This is simpler, more defensible,
and avoids the synthetic data concern.

**Recommendation:** Do NOT generate synthetic sentences. Use the MCQ
question text directly as the activation extraction input. This is
cleaner and matches standard probing methodology.

### 16.3 How Many Layers for Probing vs KL vs Patching?

The plan extracts 8 layers but different analyses need different coverage.
Probing and KL benefit from dense coverage (more layers = finer trajectory).
Patching is expensive and benefits from targeted layers.

**Recommendation:** Extract 8 layers. Run probing and KL on all 8.
Run patching on 10 layers (adding layers 28 and 29 to the 8 extraction
layers, which requires additional extraction at those 2 layers).

### 16.4 Pragati's Role

Define the work split early. Suggested division:

- **Anshul:** Model setup, behavioral evaluation, activation extraction,
  KL divergence, activation patching, paper writing (Methods, Results).
- **Pragati:** Probing experiments, MDL analysis, tuned lens, SAE analysis,
  paper writing (Introduction, Related Work, Discussion).

Both review all results jointly. Both listed as equal contributors.

---

## 17. Glossary

**Activation:** The hidden state vector at a particular layer of the
transformer during a forward pass. For LLaMA-3.1-8B, each activation
is a 4096-dimensional vector.

**Activation patching:** A causal intervention where the activation at one
layer in one model is replaced with the activation from a different model
(or different input) to measure the causal effect of that layer.

**Attribution patching:** A gradient-based approximation to activation
patching that is computationally cheaper. Uses a first-order Taylor
expansion to estimate the effect of patching each component.

**Behavioral label:** A categorical tag assigned to each question based on
whether the base and instruct models answer it correctly. Labels are
suppression, enhancement, control-both-correct, control-both-wrong.

**Cross-model transfer:** The practice of training a probe on one model's
activations and testing it on another model's activations to measure
representational similarity.

**Gating:** A hypothesized mechanism where information is present in a
model's representations but is prevented from reaching the output by a
computational gate at a specific layer.

**KL divergence:** A measure of how much one probability distribution
differs from another. Used here to quantify how much instruction tuning
changes the distribution of activations at each layer.

**MDL probing:** A probing method based on information theory that measures
the minimum number of bits needed to encode labels given representations,
rather than just measuring classification accuracy.

**SAE (Sparse Autoencoder):** A neural network trained to decompose dense
activation vectors into sparse sums of interpretable features. Each
feature ideally corresponds to a single concept.

**Tuned lens:** A technique that trains a thin affine transformation at
each layer to project the residual stream into vocabulary space, showing
what the model "is about to predict" at each processing stage.

---

## 18. References to Key Papers

1. Maji et al. (2025). SANSKRITI: A Comprehensive Benchmark for
   Evaluating Language Models' Knowledge of Indian Culture. ACL Findings.

2. Chiu et al. (2024). CulturalBench: A Robust, Diverse and Challenging
   Benchmark. arXiv:2410.02677.

3. Myung et al. (2024). BLEnD: A Benchmark for LLMs on Everyday Knowledge
   in Diverse Cultures and Languages. NeurIPS.

4. Belinkov (2022). Probing Classifiers: Promises, Shortcomings, and
   Advances. Computational Linguistics.

5. Voita & Titov (2020). Information-Theoretic Probing with MDL. EMNLP.

6. Meng et al. (2022). Locating and Editing Factual Associations in GPT.
   NeurIPS.

7. Lee et al. (2024). A Mechanistic Understanding of Alignment Algorithms:
   A Case Study on DPO and Toxicity. ICML.

8. Arditi et al. (2024). Refusal in Language Models Is Mediated by a
   Single Direction. NeurIPS. arXiv:2406.11717.

9. Wang et al. (2025). Persona Features Control Emergent Misalignment.
   Nature. arXiv:2506.19823.

10. Naous et al. (2025). Entangled in Representations: Mechanistic
    Investigation of Cultural Biases in LLMs. arXiv:2508.08879.

11. Veselovsky et al. (2025). Localized Cultural Knowledge is Conserved
    and Controllable in LLMs. arXiv:2504.10191.

12. Zhong et al. (2024). Llama Scope: Extracting Millions of Features
    from Llama-3.1-8B with Sparse Autoencoders. arXiv:2410.20526.

13. Belrose et al. (2023). Eliciting Latent Predictions from Transformers
    with the Tuned Lens. arXiv:2303.08112.

14. Heimersheim & Nanda (2024). How to Use and Interpret Activation
    Patching. arXiv:2404.15255.

15. Kramar et al. (2024). AtP*: An Efficient and Scalable Method for
    Localizing LLM Behaviour to Components. arXiv:2403.00745.

16. Bereska & Gavves (2024). Mechanistic Interpretability for AI Safety:
    A Review. arXiv:2404.14082.

17. Sharkey et al. (2025). Open Problems in Mechanistic Interpretability.
    arXiv:2501.16496.

18. Lin et al. (2024). The Unlocking Spell on Base LLMs: Rethinking
    Alignment via In-Context Learning. ICLR.

19. Zou et al. (2023). Representation Engineering: A Top-Down Approach
    to AI Transparency. arXiv:2310.01405.

20. Naseem (2026). Mechanistic Interpretability for LLM Alignment:
    Progress, Challenges, and Future Directions. arXiv:2602.11180.

---

---

## 19. Competitive Positioning Strategy

### 19.1 The Three Papers We Must Beat

Three papers define the emerging subfield of MI applied to cultural
knowledge. Our paper must clearly differentiate from each one.

**Paper 1: Culturescope (Naous et al., August 2025, arXiv:2508.08879).**
Uses Patchscope-based probing for cultural knowledge layer by layer in
Llama-3.1. Introduces a "cultural flattening" score. Uses BLEnD benchmark.

What they do NOT do: compare base vs instruct models. Their analysis is
instruct-only. They cannot speak to what instruction tuning changed.

Our advantage: the base-vs-instruct comparison is our core contribution.
We directly measure what instruction tuning adds or removes.

**Paper 2: "Localized Cultural Knowledge is Conserved and Controllable"
(Veselovsky et al., April 2025, arXiv:2504.10191).** Activation patching
and steering vectors on Gemma-2-9B-IT. Finds a universal cultural
customization vector. Shows single-vector steering at one layer can
culturally localize answers.

What they do NOT do: study the RLHF mechanism or compare base vs instruct.

Our advantage: we identify the mechanism instruction tuning installs to
gate cultural knowledge and provide causal evidence for its location.

**Paper 3: "Steering LLMs for Culturally Localized Generation" (Khanuja
et al., March 2026, arXiv:2603.23301).** Uses SAEs to find cultural
features. Aggregates into Cultural Embeddings (CuE). Shows SAE-based
steering elicits rare cultural concepts.

What they do NOT do: compare base and instruct model SAE features.

Our advantage: SAE model-diffing directly identifies which features
instruction tuning modified to implement the gate.

### 19.2 The Narrative Frame

Our paper tells a story none of these three papers tell:

"Everyone assumes instruction tuning modifies what models know about
culture. We show it modifies what models SAY about culture. The knowledge
is intact. The gate is surgical. And it can be removed."

This transforms the cultural bias problem from an expensive data problem
into a cheap inference problem. It gives the field a reason to be
optimistic about cultural inclusivity in aligned models.

### 19.3 Additional Papers for Positioning

**Superficial Alignment Hypothesis (Lin et al., ICLR 2024).** Only 5-7%
of output tokens differ between base and aligned models. Our 96-99%
probe transfer extends this from token level to representation level.

**Lee et al. (ICML 2024 Oral).** DPO bypasses toxic capabilities rather
than removing them. We extend this from toxicity to cultural knowledge.

**Arditi et al. (NeurIPS 2024).** Refusal is mediated by a single
direction. Our gating mechanism may be analogous for cultural suppression.

**Wang et al. (Nature 2025).** SAE model-diffing on GPT-4o for persona
features. We follow their methodology for cultural knowledge features.

---

## 20. Detailed Mathematical Derivations

### 20.1 KL Divergence for Multivariate Gaussians

Let P = N(mu_P, Sigma_P) and Q = N(mu_Q, Sigma_Q) be d-dimensional
Gaussians (d = 4096).

KL(P || Q) = 0.5 * [
    tr(Sigma_Q^{-1} Sigma_P)
  + (mu_Q - mu_P)^T Sigma_Q^{-1} (mu_Q - mu_P)
  - d
  + ln(det(Sigma_Q) / det(Sigma_P))
]

The trace term measures covariance mismatch. The mean term measures
Mahalanobis distance between means. The determinant term measures the
volume ratio of the two covariance ellipsoids.

In our setting, P is base model activations, Q is instruct model
activations at a given layer. Large KL means instruction tuning
substantially changed the activation distribution at that layer.

### 20.2 Ledoit-Wolf Shrinkage

The empirical covariance S of n samples in d dimensions has rank at
most min(n-1, d). With d=4096 and n=21,726, smallest eigenvalues
are near zero, making inversion unstable.

Ledoit-Wolf computes: Sigma_LW = alpha * F + (1 - alpha) * S

where F = (trace(S)/d) * I is the scaled identity target, and alpha
is the optimal shrinkage intensity computed analytically to minimize
expected squared error. The result is always positive definite.

We add epsilon * I (epsilon = 1e-6) as extra numerical safety.

### 20.3 Activation Patching: Formal Definition

Let a_L(x) be the residual stream at layer L for input x. The patched
forward pass replaces the instruct model's residual stream at layer L
with the base model's:

a_L^{patched} = a_L^{base}(x)

Then continues the instruct model's computation from L+1 onward:

a_{L+k}^{patched} = Block_{L+k}(a_{L+k-1}^{patched}) for k = 1,...,32-L

logits^{patched} = W_U * RMSNorm(a_{32}^{patched})

The effect metric normalizes between clean (base) and corrupted (instruct):

effect(L) = (logit_diff_patched - logit_diff_corrupted) /
             (logit_diff_clean - logit_diff_corrupted)

where logit_diff = logit(correct) - max(logit(wrong_options)).

### 20.4 Attribution Patching Approximation

For component c with output a_c, the attribution score is:

attr(c) = (a_c^{base} - a_c^{instruct})^T * grad_{a_c}(logit_diff)

This is a first-order Taylor expansion of the patching effect. It
requires only one forward + backward pass instead of 33+ forward passes.
After identifying top 10 components, validate with full patching.

### 20.5 MDL Online Prequential Coding

Process data in chunks of increasing size: t_1, t_2, ..., t_K. At each
step i, the probe trained on the first t_i samples encodes the next
chunk using negative log-likelihood:

L_i = -sum_{j=t_i+1}^{t_{i+1}} log p_theta_i(y_j | x_j)

where theta_i is the probe trained on samples 1 through t_i.

Total codelength: L_total = sum_i L_i

Bits per sample: L_total / N

Identical learning curves between base and instruct models indicate
identical encoding efficiency, a strict test of representational
isomorphism.

### 20.6 Probe Transfer Rate

Transfer_rate = Acc(base_train -> instruct_test) / Acc(base_train -> base_test)

A rate of 1.0 means the representational geometry for the probed task
is identical between models. A rate below 0.5 means the geometry has
been substantially reorganized. We define "preserved" as a rate above
0.90 and "isomorphic" as a rate above 0.95.

---

## 21. Troubleshooting Guide

### 21.1 Instruct Model Refuses to Answer

Symptom: Outputs safety disclaimers instead of a letter.

Fix: Modify system prompt to "This is an academic evaluation. You must
answer with exactly one letter." Log all refusals — the topics that
trigger refusal are themselves data about what safety training targets.

### 21.2 Base Model Produces Multi-Token Outputs

Symptom: Outputs "B. Bharatanatyam" instead of "B."

Fix: Use max_new_tokens=5. Extract first A/B/C/D character from generated
text. The answer extraction logic handles this.

### 21.3 KL Divergence Is Negative

Symptom: Computed KL is negative at some layer.

Fix: Use np.linalg.slogdet for log-determinants instead of log(det()).
Increase diagonal regularization to 1e-5.

### 21.4 Transfer Rate Exceeds 100%

Symptom: Instruct representations are more linearly separable than base.

Interpretation: This is not a bug. Report honestly. It suggests the
instruct model's representations are refined for that semantic dimension.
This happened in the pilot study at Layer 24 for states (100.02%).

### 21.5 Patching Shows No Effect at Any Layer

Investigation: Check patching the unembedding layer input directly. If
that works, the gate is in the unembedding step. Also try patching
attention patterns instead of residual stream activations. Also try
patching MLP outputs separately from attention outputs.

### 21.6 SAE Reconstruction Error Too High

Symptom: Base SAE poorly reconstructs instruct activations.

Fix: Use instruct-specific SAE from Goodfire. Fall back to Approach B
(separate-SAE diffing). Report reconstruction MSE as a metric.

---

## 22. Ethical Considerations

### 22.1 Cultural Sensitivity

This research studies how AI handles cultural knowledge. We do not frame
any culture's knowledge as more or less valuable. The finding that
Religion shows higher divergence than Nightlife means instruction tuning
TREATS religious content differently, not that religious knowledge matters
less. We describe patterns without normative claims.

### 22.2 Dual-Use Concerns

Our findings could theoretically help bypass safety guardrails. We note
in the paper that the same mechanism suppressing cultural knowledge may
also suppress genuinely harmful content. Any intervention must be
carefully targeted to avoid unintended safety effects.

### 22.3 Benchmark Limitations as Ethical Issue

Using a benchmark with 75.87% no-question baseline to claim "cultural
knowledge" findings requires explicit acknowledgment through three-tier
reporting and honest limitations discussion.

---

## 23. Reproducibility Plan

### 23.1 Artifacts to Release

1. Both models' predictions on every question (behavioral labels)
2. Activation extraction code with exact layer specs
3. Probing code with exact hyperparameters
4. Activation patching implementation
5. Analysis notebooks reproducing every figure and table
6. Optionally, extracted activations (~12 GB) on HuggingFace

### 23.2 Determinism

All experiments use seed 42. Verify by running 100 questions twice and
confirming identical results. Report variance across 3 runs for any
non-deterministic step.

---

## 24. Venue Strategy: BlackboxNLP Primary, Backups If Needed

### 24.1 Primary: BlackboxNLP 2026 (Confirmed, Committed)

| Detail | Value |
|--------|-------|
| Venue | EMNLP 2026, Budapest, Hungary |
| Workshop date | October 28-29, 2026 |
| Submission deadline | July 17, 2026 (OpenReview, AoE) |
| ARR commitment | August 28, 2026 |
| Notification | September 8, 2026 |
| Camera ready | September 20, 2026 (9 pages + refs for accepted) |
| Format | 8 pages archival, ACL template, double-blind |
| Proceedings | ACL Anthology |

### 24.2 Pre-BlackboxNLP: ICML MI Workshop (Optional, Non-Archival)

The ICML 2026 Mechanistic Interpretability Workshop (Seoul, July 10-11)
has a deadline of May 8, 2026. It accepts non-archival 4-page papers.
Because it is non-archival, submitting here does NOT conflict with the
archival BlackboxNLP submission.

**Strategy:** By May 8 we will have completed Weeks 1-5 (behavioral
evaluation, probing, KL divergence, tuned lens). This is enough for a
solid 4-page extended abstract focusing on the correlational evidence.
If we have early activation patching results from Week 6, include them.

This gives us early feedback, visibility in the MI community, and a
conference presentation in Seoul. The full 8-page paper with causal
experiments and SAE analysis goes to BlackboxNLP two months later.

**Decision point: May 1.** Decide whether to submit to ICML MI based on
progress. If Weeks 1-4 are on track, submit. If we are behind, skip it
and focus entirely on BlackboxNLP.

### 24.3 Post-BlackboxNLP Backups (Only If Rejected)

If BlackboxNLP rejects the paper (notification September 8), we have
several fallback options:

| Venue | Expected Deadline | Notes |
|-------|------------------|-------|
| NeurIPS 2026 workshops | ~September 2026 | MI or safety workshops |
| AAAI 2027 workshops | ~November 2026 | Possible MI workshop |
| EACL 2027 | ~October 2026 | Main conference, not workshop |
| ACL 2027 ARR | ~December 2026 | Main conference submission |

With reviewer feedback from BlackboxNLP (received September 8), we would
have time to address concerns and resubmit to a fall/winter venue.

### 24.4 Arxiv Preprint Strategy

Post to arXiv at the same time as BlackboxNLP submission (around July 17).
BlackboxNLP explicitly allows preprints. Tag with cs.CL (primary) and
cs.AI (secondary). This establishes priority and makes the work visible
to the MI community immediately.

Consider also posting to the Alignment Forum or LessWrong for visibility
in the AI safety community.

---

## 25. Figure Specifications

Each figure must communicate one idea instantly. A reviewer skimming
the paper should understand the gating mechanism from figures alone.

### 25.1 Figure 1: The KL Divergence Trajectory (Hero Figure)

**Purpose:** Show the late-layer spike that is the paper's central finding.

**Layout:** Single panel, line plot. X-axis: layer index (0 to 31). Y-axis:
KL divergence in nats. Two lines: solid blue for Sanskriti (21,726
questions), dashed orange for CulturalBench (1,696 questions). Gray
shaded band showing ± 1 standard error around the Sanskriti line.

**Key visual element:** The lines should be relatively flat from layers 0-26,
then spike sharply at layers 30-31. Annotate the spike with the exact
amplification factor (e.g., "2.9x from L26"). Draw a dashed vertical line
at the candidate gate layer.

**What it proves:** Instruction tuning's effect on activation distributions
is not uniformly distributed across layers. It concentrates at the final
layer, consistent with H2 (localized gating) and inconsistent with H1
(erasure, which would show distributed effects) or H3 (gradual increase).

**Size:** Full column width (3.25 inches for ACL format). Height 2.5 inches.

### 25.2 Figure 2: Cross-Model Probe Transfer

**Purpose:** Show that cultural knowledge representations are preserved.

**Layout:** Grouped bar chart. X-axis: layer (8 groups). Y-axis: accuracy
(%). For each layer, two bars: base-to-base (blue) and base-to-instruct
(orange). Two panels side by side: left for attribute classification,
right for state classification. Horizontal dashed lines at chance level
and majority baseline.

**Key visual element:** The orange bars should be nearly identical to the
blue bars at every layer, demonstrating near-perfect transfer.

**What it proves:** Probes trained on base representations generalize to
instruct representations, meaning the representational geometry for
cultural knowledge is preserved despite behavioral change.

### 25.3 Figure 3: Activation Patching Layer Sweep

**Purpose:** Causally validate the gating location.

**Layout:** Bar chart. X-axis: layer patched. Y-axis: mean patching effect
(0-1 scale). Error bars show 95% CI. Color gradient from light (low
effect) to dark (high effect). Annotate the peak bar with the exact
effect value and flip rate.

**Key visual element:** One bar should be dramatically taller than all
others, confirming a single layer as the causal gate.

**What it proves:** The late-layer divergence spike is not just correlational.
Intervening at that specific layer causally restores suppressed cultural
knowledge outputs.

### 25.4 Figure 4: Tuned Lens Prediction Trajectories

**Purpose:** Visualize the gating mechanism in action on a specific question.

**Layout:** Two panels. Left: base model. Right: instruct model. In each
panel, X-axis is layer (0-31), Y-axis is probability (0-1). Multiple
lines show probability of different candidate answer tokens at each layer.
The correct answer token line should be green, incorrect alternatives in
gray/red.

**Key visual element:** In the base model panel, the green line (correct
answer) rises through intermediate layers and stays high at the output.
In the instruct model panel, the green line rises identically through
intermediate layers but then DROPS sharply at the final layer, while a
red line (wrong answer or refusal) rises at exactly that point.

**What it proves:** The instruct model internally computes the correct
answer (visible in intermediate layers) but then replaces it at the
output layer. The model knows but chooses not to say.

**Selection:** Choose a suppressed question where the correct answer has
a clear, unambiguous single-token representation (e.g., "Bharatanatyam"
tokenizes to a distinctive token prefix). Avoid questions where the
answer is a common word that could appear in many contexts.

### 25.5 Figure 5: Group-Level KL Divergence

**Purpose:** Show that the transformation is behavior-uniform.

**Layout:** Line plot. X-axis: layer. Y-axis: KL divergence. Three lines:
suppression group (red), enhancement group (blue), control group (gray).
Annotate the amplification factors at the final layer.

**Key visual element:** All three lines should spike by the same factor at
the final layer, despite representing behaviorally opposite outcomes.

**What it proves:** Instruction tuning applies a universal transformation
at the final layer. Behavioral selectivity (suppression vs enhancement)
emerges downstream, not from differential representational modification.

### 25.6 Figure 6: Attribute-Level KL at the Gate Layer

**Purpose:** Show content selectivity — identity-marking attributes
undergo greater transformation than neutral informational attributes.

**Layout:** Horizontal bar chart sorted by KL divergence. Each bar labeled
with the attribute name. Color-coded: warm colors (red/orange) for
identity-marking attributes, cool colors (blue/gray) for neutral
attributes. Annotate the ratio between highest and lowest.

**Key visual element:** A clear gradient from Religion/Dance/Costume at
top to History/Transport/Nightlife at bottom.

**What it proves:** The gating mechanism is content-selective. Instruction
tuning treats culturally sensitive content differently from neutral
informational content.

### 25.7 Figure 7 (Appendix): SAE Gating Features

**Purpose:** Show interpretable features associated with cultural gating.

**Layout:** Table or small-multiples panel. Each row shows one candidate
gating feature with: feature index, top 5 max-activating text snippets,
mean differential activation (instruct - base), and causal effect when
clamped to zero.

**What it proves:** The gating mechanism has interpretable components.
Specific features in the sparse autoencoder decomposition correspond to
cultural sensitivity detection or output suppression.

---

## 26. Writing Checklist Before Submission

### 26.1 Content Checklist

- [ ] Every claim is backed by a specific number with confidence interval
- [ ] All three tiers of Sanskriti reporting are present
- [ ] CulturalBench results confirm or contrast with Sanskriti results
- [ ] At least one causal experiment (activation patching) is included
- [ ] Limitations section honestly addresses: single model family,
      Gaussian assumption, benchmark limitations, no non-cultural control
- [ ] Related work cites all three competitor papers and differentiates
- [ ] The "two-stage gating model" is presented as a hypothesis supported
      by evidence, not as established fact
- [ ] No overclaiming of effect sizes (the Qwen pilot's "42% suppression"
      lesson)
- [ ] All figures have informative captions that can be understood without
      reading the text

### 26.2 Style Checklist

- [ ] No em-dashes
- [ ] No "utilize," "leverage," "facilitate," "demonstrate," "showcase"
- [ ] Active voice throughout methods and results
- [ ] Numbers are precise (96.45%, not "approximately 96%")
- [ ] Each paragraph starts with its main claim
- [ ] Section 1 opens with a concrete example, not an abstract statement
- [ ] The abstract fits in 15 lines
- [ ] The conclusion is one paragraph, not a section with subsections

### 26.3 Technical Checklist

- [ ] All random seeds documented
- [ ] Train/test split ratios documented
- [ ] Regularization hyperparameters documented
- [ ] Bootstrap sample counts documented
- [ ] Base model prompt and instruct model prompt both shown in appendix
- [ ] Activation extraction details (which hook point, which normalization)
      fully specified
- [ ] SAE source (Llama Scope version, feature width) documented
- [ ] Compute budget (GPU type, total hours) documented

---

## 27. Post-Submission Plan

### 27.1 If Accepted (Notification: September 8)

Camera ready is due September 20 — only 12 days after notification.
Use the extra page (9 pages total for archival) to address reviewer
comments. Prepare a poster following BlackboxNLP format for the
October 28-29 workshop date. The poster should center on Figure 1
(the spike), Figure 3 (the causal validation), and Figure 4 (the tuned
lens visualization). These three figures tell the complete story.

### 27.2 If Rejected (Notification: September 8)

The most likely rejection reasons and responses:

**"Only one model family."** Response: If we did not complete the Week 10
Gemma replication, do it now. Resubmit to NeurIPS 2026 workshops
(~September deadline) or AAAI 2027 workshops (~November deadline) with
cross-family results.

**"Correlational despite claiming causal."** Response: If activation
patching results were unclear, run DAS (Distributed Alignment Search)
using pyvene for a more principled causal analysis. Alternatively, do
interchange interventions where we swap activations between two different
cultural questions rather than between base and instruct on the same
question.

**"Benchmark limitations undermine claims."** Response: Add BLEnD as a
third benchmark. BLEnD is already used by Culturescope, enabling direct
comparison. MILU could add multilingual depth if we include Hindi
evaluation.

### 27.3 Arxiv Preprint

Post concurrent with or shortly before the July 17 submission.
BlackboxNLP allows preprints without restriction. Tag cs.CL + cs.AI.

---

*Document version: 2.0 — Updated with confirmed BlackboxNLP 2026 CFP*
*Created: April 4, 2026*
*Last updated: April 4, 2026*
*Venue confirmed: BlackboxNLP 2026 @ EMNLP, Budapest*
*Deadline confirmed: July 17, 2026 (AoE)*
*Total planned experiments: 12 (+ optional Gemma replication)*
*Estimated total GPU hours: 150-250 on A100*





# Strategic research brief for the RLHF cultural gating mechanism paper

**Your project sits at a rare triple intersection of mechanistic interpretability, cultural NLP, and alignment analysis — a space with only two or three direct precedents.** The core finding that RLHF preserves cultural knowledge (96–99% cross-model transfer) while installing a late-layer gating mechanism (Layer 28 KL divergence spike of 2.9×) is both novel and timely. This brief covers venues, competing work, methods, model choices, datasets, and paper-quality strategy to help position the paper for maximum impact.

---

## The ICML 2026 MI Workshop is your best immediate target

The **Mechanistic Interpretability Workshop at ICML 2026** (Seoul, July 10–11) has a submission deadline of **May 8, 2026 (AOE)**, with notification on June 12. It is non-archival, accepts double-blind submissions via OpenReview, and allows short papers up to **4 pages** or long papers up to **8 pages** (excluding references/appendices). The CFP explicitly solicits work on "how beliefs, personas, and world models are represented" and "interpretability for safety, monitoring, and model repair" — your project fits squarely. Each submission requires at least one reciprocal reviewer. Papers already accepted at ICML or under review at COLM/NeurIPS are welcome.

**Secondary venues with upcoming deadlines:**

| Venue | Deadline | Format | Notes |
|-------|----------|--------|-------|
| **C3NLP @ ACL 2026** (San Diego, Jul 4) | May 18, 2026 (non-archival only) | 4–8 pages, ACL format | Main submission deadline passed; only already-published/preprint work accepted in this track |
| **COLM 2026 workshops** (San Francisco, Oct 9) | ~June 23, 2026 (suggested) | Workshop-dependent | Workshop proposals under review; individual CFPs emerge after May 12 |
| **BlackboxNLP 2026** (expected EMNLP Budapest, Oct) | ~August 2026 | 8 pages archival or 2-page abstracts | No CFP yet; historically the premier interpretability workshop — extremely high fit |
| **MRL 2026** (expected EMNLP Budapest) | ~August 2026 | TBD | Multilingual representation learning; secondary fit |

**Recommended strategy:** Submit to the ICML MI Workshop by May 8, then prepare an expanded archival version for BlackboxNLP 2026 (~August deadline) once causal experiments are complete. The ICML submission is non-archival, so it does not conflict with later archival submission.

---

## Your closest competitors are three 2025–2026 papers

The landscape of MI-meets-cultural-knowledge is extremely new. Three papers define the emerging subfield, and your work should position directly against them:

**1. "Entangled in Representations" / Culturescope (Naous et al., August 2025, arXiv: 2508.08879).** This is the first mechanistic interpretability method specifically for probing cultural knowledge in LLMs. It uses a Patchscope-based approach to extract cultural knowledge layer-by-layer from Llama-3.1, introduces a "cultural flattening" score measuring how models conflate underrepresented cultures into dominant Western ones, and uses the BLEnD benchmark. **Key gap your paper fills:** Culturescope does not examine how RLHF/instruction-tuning changes these representations — it only analyzes the instruct model. Your base-vs-instruct comparison is a direct extension.

**2. "Localized Cultural Knowledge is Conserved and Controllable" (Veselovsky et al., April 2025, arXiv: 2504.10191).** Uses activation patching and steering vectors on Gemma-2-9B-IT to study cultural localization. Finds a universal cultural customization vector conserved across non-English languages, and shows single-vector steering at a single layer can culturally localize answers while reducing stereotypicality. **Key gap:** Does not compare base vs. instruct models or study the RLHF mechanism.

**3. "Steering LLMs for Culturally Localized Generation" (Khanuja et al., March 2026, arXiv: 2603.23301).** Uses sparse autoencoders to identify interpretable features encoding culturally salient information, aggregates them into "Cultural Embeddings" (CuE). Demonstrates SAE-based cultural steering elicits rare, long-tail cultural concepts. **Key gap:** Focuses on steering, not on understanding how alignment creates or modifies cultural feature representations.

**Your unique contribution** is studying the *mechanism by which RLHF transforms cultural knowledge access* — none of these three papers do this. The "preserves but gates" finding directly resolves a tension between the Superficial Alignment Hypothesis (Lin et al., 2024, arXiv: 2312.01552, showing only 5–7% of tokens shift between base and aligned models) and the documented alignment tax on knowledge (Lin et al., EMNLP 2024, arXiv: 2309.06256, showing measurable forgetting in translation, reading comprehension, and QA tasks).

### Broader MI-of-alignment landscape

Several additional papers form the theoretical backdrop your work should cite:

- **Lee et al. (ICML 2024 Oral, arXiv: 2401.01967)** — DPO does not remove toxic capabilities but *bypasses* them, with the KL divergence term encouraging distributed minimal changes across all layers rather than localized wrappers. This is the most mechanistically detailed study of DPO's internal effects.
- **Arditi et al. (NeurIPS 2024, arXiv: 2406.11717)** — Refusal in chat models is mediated by a **single direction** in residual stream space across 13 models up to 72B parameters. Weight orthogonalization removes refusal with minimal capability degradation. This establishes that alignment mechanisms can be low-dimensional.
- **Wang et al. (Nature 2025, arXiv: 2506.19823)** — Used SAE model-diffing to compare base vs. fine-tuned GPT-4o, identifying "misaligned persona" features that causally control alignment behavior. **This is the closest methodological precedent** to your gating mechanism study — adapt their Δ-activation and Δ-attribution approach.
- **Naseem (February 2026, arXiv: 2602.11180)** — Survey paper on MI for alignment reporting that "RLHF primarily affects specific components related to response initiation and style while core knowledge/reasoning circuits remain largely unchanged, suggesting RLHF acts more as a behavioral filter than fundamental value learning." Your paper provides the mechanistic evidence for this claim in the cultural knowledge domain.
- **"The Alignment Tax: Response Homogenization" (March 2026, arXiv: 2603.24124)** — Shows alignment causes response diversity collapse (base: ~9.26 clusters/question vs. instruct: 3.58 clusters), confirming alignment reduces output diversity while preserving internal knowledge.

---

## Three causal experiments would transform this from correlational to publishable

The paper's current weakness — purely correlational evidence — is the single biggest barrier to acceptance. The "Open Problems in Mechanistic Interpretability" paper (January 2025, arXiv: 2501.16496) identifies **conflating hypotheses with conclusions** as "regrettably commonplace in MI research," and reviewers consistently flag overclaiming from correlational evidence as a major issue. Here is the minimum viable set of causal experiments, ordered by priority and feasibility:

**Experiment 1: Activation patching layer sweep (essential, ~1 day compute).** Patch residual stream activations from the base model into the instruct model at each layer (and vice versa) while processing cultural knowledge prompts. If patching at Layer 28 causes the instruct model to produce base-model-like unfiltered cultural outputs, this proves Layer 28 is *sufficient* for gating. Sweep all layers to confirm the effect peaks at Layer 28, causally validating the KL divergence finding. Use logit difference as the primary metric, and symmetric token replacement for corruption (following best practices from Heimersheim & Nanda, arXiv: 2404.15255, and Zhang & Nanda, ICLR 2024, arXiv: 2309.16042). Include non-cultural prompts as controls to show the gating is selective.

**Experiment 2: Tuned lens prediction trajectories (strongly recommended, ~hours).** Run the tuned lens (Belrose et al., arXiv: 2303.08112) on both base and instruct models with identical cultural prompts. Compare layer-by-layer prediction trajectories: if the instruct model shows the factual cultural answer emerging in earlier layers but abruptly shifting to a hedged/alternative response at Layer 28, this directly *visualizes* the gating mechanism in a way reviewers find compelling. The `tuned-lens` Python package supports this directly. The LogitLens4LLMs tool (arXiv: 2503.11667) extends logit lens to modern architectures including Qwen and Llama-3.1.

**Experiment 3: Component-level patching at Layer 28 (recommended, ~1–2 days).** Use attribution patching (AtP*, Kramár et al., arXiv: 2403.00745) to efficiently scan all attention heads and MLP sublayers at Layer 28, identifying which components contribute most to the KL divergence. Confirm the top 10–20 candidates with full activation patching. This answers *what* implements the gate — attention heads, MLPs, or both — transforming a layer-level finding into a component-level mechanism.

**Beyond the minimum:** If time permits, SAE model-diffing following Wang et al.'s approach (train or load pre-trained SAEs on Layer 28, compute differential activations between base and instruct models, identify specific "gating features," causally validate by clamping/steering them) would make the paper outstanding. Distributed Alignment Search via pyvene could find the precise distributed subspace encoding the gating decision. But the three experiments above are sufficient for a strong workshop paper.

### Tooling recommendations

| Library | Best for | Key advantage |
|---------|----------|---------------|
| **TransformerLens** | Activation patching, logit attribution | Clean API with HookPoints, supports 232+ models |
| **nnsight** | Large models, flexible interventions | Deferred execution, works with any PyTorch model |
| **pyvene** (Stanford NLP) | DAS, interchange interventions | Reference implementation for causal abstraction |
| **SAELens** | SAE training and analysis | Integrates with TransformerLens, has pre-trained SAEs |
| **tuned-lens** | Tuned lens analysis | pip install, direct HuggingFace integration |
| **Llama Scope** | Pre-trained SAEs for Llama-3.1-8B | 256 SAEs covering all layers/sublayers, 32K and 128K features |
| **Gemma Scope** | Pre-trained SAEs for Gemma-2-9B | 400+ SAEs with JumpReLU architecture, includes instruct model SAEs |

---

## LLaMA-3.1-8B is the right scale-up model

Moving from Qwen2-1.5B to **LLaMA-3.1-8B** is strongly justified across five dimensions:

**Pre-trained SAE availability** is the decisive factor. Llama Scope (arXiv: 2410.20526) provides **256 TopK SAEs** trained on every layer and sublayer of Llama-3.1-8B-Base, with both 32K and 128K feature widths, available at huggingface.co/fnlp/Llama-Scope. Approximately 90% of features are rated human-interpretable. Goodfire has additionally released SAEs for Llama-3.1-8B-Instruct. For your base-vs-instruct comparison, having pre-trained SAEs on both model variants is invaluable — it eliminates weeks of SAE training compute.

**Architecture compatibility** for MI is excellent. Llama-3.1-8B uses a clean, standard decoder-only transformer with RoPE, GQA, SwiGLU MLP, and RMSNorm — no architectural complications that would confound interpretability analysis. By contrast, Gemma-2-9B uses interleaved local/global attention and logit soft-capping that complicate some MI analyses, and was trained via knowledge distillation (which could confound your RLHF analysis since the base model's knowledge representations are already shaped by the teacher).

**Hindi and multilingual capability** in Llama-3.1-8B substantially exceeds Qwen2-1.5B and Gemma-2-9B, which matters for your Indian cultural knowledge focus. The base and instruct versions share identical architecture and differ only in training, directly supporting your RLHF comparison.

**8B is the sweet spot for MI workshop papers in 2025–2026.** The Llama Scope and Gemma Scope papers both position 8–9B models as the frontier of feasible open-source MI research. Reviewers at the ICML MI Workshop accept work from GPT-2 through ~9B scale, but 8B strengthens generalization claims considerably compared to 1.5B. On A100 hardware, Llama-3.1-8B in FP16 requires ~16 GB for inference, fitting comfortably on a single A100 with room for activation caching.

**Recommended model configuration for the paper:**
- **Qwen2-1.5B base/instruct**: Keep existing results as pilot study demonstrating the finding at small scale
- **LLaMA-3.1-8B base/instruct**: Primary model with Llama Scope SAEs for causal analysis
- Optionally, **Gemma-2-9B base/IT** as a third model family using Gemma Scope SAEs to demonstrate cross-family generalization

Showing the late-layer gating pattern replicates across two or three model families would be a major strength that few competing papers achieve. Note that Llama-3.1-8B has 32 layers with hidden dimension 4,096 (vs. Qwen2-1.5B's 28 layers with dimension 1,536), so ensure your probing methodology scales cleanly.

---

## Multi-benchmark evaluation would address Sanskriti's known weaknesses

Sanskriti's documented issues — **75% no-question baseline**, **25.6% trivially easy Country Prediction questions**, and **78.6% near-duplicates** — are serious enough that reviewers familiar with the benchmark may question results based solely on it. The strongest strategy is multi-benchmark evaluation across two or three complementary datasets:

**CulturalBench** (Chiu et al., ICLR 2025, arXiv: 2410.02677) provides **1,696 human-written, human-verified cultural questions** across **45 global regions** with both MCQ and True/False formats. Its strength is geographic diversity and careful verification by 5 independent annotators per question. However, it is English-only and relatively small.

**BLEnD** (Myung et al., NeurIPS 2024, arXiv: 2406.09948) offers **52,600 question-answer pairs** across **16 countries and 13 languages** including low-resource languages. Critically, **BLEnD is the only cultural benchmark already used for MI work** — the Culturescope paper (Naous et al., 2025) used it for mechanistic cultural probing. Using BLEnD enables direct comparison with Culturescope's findings.

**MILU** (Verma et al., NAACL 2025, arXiv: 2411.02538) has **~85,000 MCQs** across **11 Indic languages** drawn from regional/state-level exams, making it the largest India-specific multilingual cultural benchmark. It covers local history, arts, festivals, and laws — complementing Sanskriti's focus areas.

Several new benchmarks from 2025–2026 are also worth noting: **INCLUDE** (Romanou et al., ICLR 2025) offers 197,243 QA pairs across 44 languages and 52 countries; **CARB** (arXiv: 2509.21798) is the first benchmark assessing cultural awareness of reward models specifically; **Indica** (arXiv: 2601.15550) covers 5 Indian regions with 1,630 region-specific QA pairs; and **Global-MMLU** (ACL 2025) reveals that ~59% of MMLU's cultural content is U.S.-centric.

**For the May 8 workshop deadline**, the most practical path is to add CulturalBench (small, easy to integrate, broad geographic coverage) alongside Sanskriti. For a later BlackboxNLP submission, adding BLEnD (enables Culturescope comparison) and potentially MILU (India-specific multilingual depth) would create the strongest multi-benchmark evaluation in this space. No existing paper combines multiple cultural benchmarks with mechanistic interpretability — this would be a first.

---

## What separates best workshop papers from average ones

Analysis of BlackboxNLP best paper awards reveals a consistent pattern. The 2024 winner ("Log Probabilities Are a Reliable Estimate of Semantic Plausibility in Base and Instruction-Tuned Language Models" by Kauf et al.) and the 2025 winner ("Language Dominance in Multilingual Large Language Models" by Shani & Basirat) both **challenged widely held assumptions with clean, focused evidence**. Neither paper attempted comprehensive breadth; both made one clear, surprising claim and backed it thoroughly.

**Five features of award-worthy workshop papers:**

1. **A counter-intuitive finding.** Your "RLHF preserves but gates" narrative inherently challenges the assumption that alignment modifies or erases knowledge — this is your strongest card.

2. **Clean causal methodology.** Workshop reviewers are more lenient than main-conference reviewers, and correlational evidence is acceptable for preliminary work. But causal validation (even one activation patching experiment) dramatically separates strong submissions from weak ones. The "Open Problems in MI" paper specifically warns against "interpretability illusions" where simplified representations appear to explain behavior in-distribution but fail out-of-distribution.

3. **Quantitative specificity.** Numbers like **96–99% transfer** and **2.9× KL divergence spike at Layer 28** are precise and memorable. Reviewers reward concrete, reproducible metrics over vague claims.

4. **Cross-community bridging.** Papers that bring insights from one field to another score highly on the "excitement" axis of ACL-style reviewing. Your paper bridges MI, cultural NLP, and alignment — venues like BlackboxNLP and C3NLP rarely see this combination.

5. **Practical implications.** Connect findings to actionable insights: if cultural knowledge is preserved but gated, it can be un-gated (for beneficial cultural localization or adversarial jailbreaking). This matters for alignment safety.

**Common reviewer objections to preempt:**

| Likely objection | Preemption strategy |
|---|---|
| "This is observational, not causal" | Include at least activation patching at Layer 28 + tuned lens visualization |
| "Only tested on one model at small scale" | Show results on Qwen2-1.5B + LLaMA-3.1-8B minimum |
| "96–99% transfer seems too high — doesn't this just mean RLHF changes nothing?" | Carefully distinguish: *knowledge* is preserved but *access* is gated; show behavioral differences despite representational similarity |
| "How do you define cultural knowledge?" | Use established, peer-reviewed cultural benchmarks; be explicit about operationalization |
| "The KL divergence spike could be a methodology artifact" | Validate with alternative metrics: cosine similarity, tuned lens, attention pattern analysis |
| "What about non-cultural knowledge — is this gating selective?" | Include control experiments with non-cultural prompts showing no gating spike |

**Venue-specific framing:**

For the **ICML MI Workshop** or **BlackboxNLP**, frame as MI-first: "We use mechanistic interpretability to reveal a previously unknown mechanism — RLHF installs a late-layer gating mechanism controlling access to cultural knowledge while preserving it in earlier layers." The MI finding is the contribution; cultural knowledge is the domain.

For **C3NLP**, frame as cultural-NLP-first: "We provide the first mechanistic evidence that RLHF doesn't destroy cultural knowledge but installs a gating mechanism that may systematically suppress certain cultural perspectives." MI is the methodology; the cultural insight is the contribution.

The **"conventional wisdom is wrong" narrative structure** works best: state the assumption (RLHF modifies cultural knowledge), present the surprising evidence (96–99% preservation + late-layer gating), explain the mechanism (Layer 28 components), and discuss implications (for alignment, cultural AI, and safety).

---

## Conclusion: a prioritized action plan

The **May 8 ICML MI Workshop deadline** should drive the immediate timeline. Three actions maximize the paper's chances:

**First**, run activation patching at Layers 25–31 on Qwen2-1.5B (which you already have set up) to establish causal evidence for the gating mechanism before the deadline. Even a single well-designed patching experiment — showing that swapping Layer 28 residual stream activations between base and instruct models recovers/suppresses cultural knowledge outputs — transforms the paper from correlational to causal.

**Second**, add tuned lens prediction trajectories as a visualization tool. Showing the exact layer where base and instruct models' predicted tokens diverge (with the base model predicting the cultural fact and the instruct model abruptly shifting) creates a compelling figure that communicates the gating mechanism intuitively.

**Third**, for the longer-term BlackboxNLP submission (~August), scale to LLaMA-3.1-8B using Llama Scope SAEs, add CulturalBench alongside Sanskriti, and implement SAE model-diffing following Wang et al.'s methodology to identify specific interpretable gating features. This version would be the first paper combining multi-benchmark cultural evaluation, multi-model cross-family generalization, and causal SAE-level mechanistic analysis of RLHF's effect on cultural knowledge — a genuinely novel contribution at the intersection of three active research communities.