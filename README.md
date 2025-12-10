# Output Gating, Not Erasure: How RLHF Modulates Cultural Knowledge in LLMs

- **Author:** Anshul Kumar (anshulk@andrew.cmu.edu)  
- **Institution:** Carnegie Mellon University

## Overview

This project investigates whether RLHF-induced suppression of cultural knowledge reflects **representational erasure** or **output-level gating**. Using mechanistic interpretability techniques on Qwen2-1.5B (base vs. instruct), we analyze 33,522 culturally-grounded sentences derived from the Sanskriti benchmark of Indian cultural knowledge across 36 states and 16 cultural attributes.

**Core Finding:** Despite 42% behavioral suppression, semantic representations remain 99.7% similar with 96-100% cross-model transferability. A 3× KL divergence spike at Layer 28 indicates RLHF operates via late-stage decision gating rather than knowledge erasure—models "know but won't say."

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Behavioral Suppression | 42.30% | Base correct → Instruct incorrect |
| Representational Similarity | 99.7% | Cosine similarity across all layers |
| Cross-Model Transfer | 96-100% | Probes trained on base work on instruct |
| Semantic Probe Accuracy | 84-96% | Attribute and state classification |
| Correctness Probe Accuracy | 62% | Weak encoding of decision information |
| KL Divergence (L8-24) | ~110 nats | Modest distributional shift |
| KL Divergence (L28) | 335 nats | **3× spike at final layer** |
| MDL Isomorphism | <3% drift | Near-identical encoding efficiency |

## Methodology

**Dataset Construction:**
- Initial evaluation: 21,853 MCQ questions from Sanskriti benchmark
- Strategic sampling: 11,206 questions balanced across suppression/enhancement/control groups
- Sentence generation: 33,522 sentences via Claude Sonnet 4.5 (3 per question)

**Four-Method Analysis Pipeline:**
1. **Linear Probing:** Attribute/state/correctness classification with cross-model transfer testing
2. **KL Divergence:** Layer-wise distributional analysis (layers 8, 16, 24, 28)
3. **MDL Probing:** Information-theoretic compression with Fisher information metrics
4. **Activation Geometry:** Per-sentence cosine similarity analysis

## Results Summary

**Convergent Evidence for Output Gating:**

| Method | Evidence | Conclusion |
|--------|----------|------------|
| Linear Probing | 96-100% transfer rates | Representational isomorphism confirmed |
| KL Divergence | 3× spike at L28 (335 vs 110 nats) | Late-stage transformation localized |
| MDL Analysis | <3% drift, 5.5× compression ratio | Distributed orthogonal encoding |
| Activation Geometry | 99.7% cosine similarity | No representational erasure |

**Mechanistic Interpretation:** Knowledge persists in distributed subspaces throughout the network. RLHF installs a gating mechanism at the final layer that selectively suppresses cultural content without erasing underlying representations.

## Repository Structure

```
cultural-alignment-study/
├── scripts/
│   ├── sanskriti_knowledge_test.py    # Initial 21K evaluation
│   ├── analyze_combinations_12k.py    # Strategic sampling
│   ├── generate_sentences_sanskriti.py # Sentence generation
│   ├── extract_activations.py         # Hidden state extraction
│   ├── linear_probing_v2.py           # Probing experiments
│   ├── kl_divergence.py               # Distributional analysis
│   └── mdl_probing_v2.py              # Information-theoretic analysis
├── outputs/
│   ├── sanskriti_test_knowledge/      # Evaluation results and breakdowns
│   ├── linear_probing/                # Probing metrics and visualizations
│   ├── kl_divergence/                 # Layer-wise KL analysis
│   └── mdl_probing/                   # MDL results and Fisher information
└── README.md
```

## Reproducing Results

```bash
# 1. Evaluate models on Sanskriti benchmark
python scripts/sanskriti_knowledge_test.py

# 2. Select balanced dataset maximizing behavioral divergence
python scripts/analyze_combinations_12k.py

# 3. Generate culturally-grounded sentences
python scripts/generate_sentences_sanskriti.py

# 4. Extract activations (dual-GPU recommended)
python scripts/extract_activations.py

# 5. Run analysis pipeline
python scripts/linear_probing_v2.py
python scripts/kl_divergence.py
python scripts/mdl_probing_v2.py
```

## Limitations

- **Correlational evidence:** No causal intervention (activation patching) performed to confirm Layer 28 as the mechanistic source
- **Single model family:** Qwen2-1.5B only; generalization to Llama, Mistral, or other architectures untested
- **Generated inputs:** Probing uses Claude-generated sentences rather than original MCQ prompts, introducing potential label leakage via explicit state/attribute mentions
- **Layer sampling:** Only 4 layers analyzed (8, 16, 24, 28); finer granularity around L27-28 would strengthen spike claims

## Citation

```bibtex
@misc{kumar2025outputgating,
  author = {Kumar, Anshul},
  title = {Output Gating, Not Erasure: How RLHF Modulates Cultural Knowledge in LLMs},
  year = {2025},
  institution = {Carnegie Mellon University}
}
```

## Contact

Questions or collaboration inquiries: anshulk@andrew.cmu.edu
