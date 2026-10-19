# Changelog

## 0.1.0 (unreleased)

First version.

- One interface (`assess`, `SimilarityAssessor`) for string-based, lexical, topic-model,
  static/contextual embedding, alignment-based, supervised (cross-encoder, LoRA) and
  prompt-based (Ollama, OpenAI-compatible, System One) similarity methods.
- Binary or gradient output; coded data used for fine-tuning, thresholds and held-out
  evaluation with leakage-free (response-grouped) splits.
- Dissimilar-pair baselines (coded negatives, cross-group and random re-pairings),
  baseline-quantile / optimised / reference-based (variants a-d) thresholds and
  baseline-relative score scaling.
- Ending-perturbation augmentation and robustness (flip-rate) evaluation for
  supervised methods.
- Hardware detection and memory/runtime estimates with warnings before expensive steps.
- `compare_methods`, explanations (token alignments), `describe()` for methods sections,
  save/load without pickle.
