# Changelog

## 0.1.0 (unreleased)

First version.

- One interface (`assess`, `SimilarityAssessor`) for string-based, lexical, topic-model,
  static/contextual embedding, alignment-based, supervised (cross-encoder, LoRA) and
  prompt-based (Ollama, System One) similarity methods.
- Binary or gradient output; coded data used for fine-tuning, thresholds and held-out
  evaluation with leakage-free (response-grouped) splits.
- Dissimilar-pair baselines (coded negatives, cross-group and random re-pairings),
  baseline-quantile / optimised / reference-based (variants a-d) thresholds and
  baseline-relative score scaling.
- Ending-perturbation augmentation and robustness (flip-rate) evaluation for
  supervised methods.
- `lora_classifier` works with vision-language backbones such as Qwen3.5: labels reach
  models whose `forward` takes them only via `**kwargs`, the padding token is also set in
  the nested text config (needed once a saved model is loaded), and the vision encoder
  gets no LoRA adapters (needs peft >= 0.14).
- Provenance records the name, version and size of every model used
  (`provenance_["models"]`, also named in `describe()`): Hugging Face commit and parameter
  count from the local cache, Ollama digest, parameter count and quantisation, spaCy
  pipeline version and word-vector table, System One release date and served checkpoint.
  Methods report their models with `model_details()`, which replaces `model_ids()`.
- Hardware detection and memory/runtime estimates with warnings before expensive steps.
  Cached Hugging Face models are also found when their snapshot lacks files skipped on
  purpose (e.g. ONNX), which huggingface_hub >= 1.0 otherwise rejects offline.
- `compare_methods`, explanations (token alignments), `describe()` for methods sections,
  save/load without pickle.
