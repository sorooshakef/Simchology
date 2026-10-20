# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Simchology is a high-level Python library that lets psychologists score text similarity
(e.g. "is this reference info contained in the participant's text?") with many NLP
methods behind one interface: a DataFrame with two text columns goes in, the same
DataFrame with a `predicted_similarity` column (binary or gradient) comes out. It mostly
wraps other libraries (scikit-learn, rapidfuzz, spaCy, sentence-transformers,
transformers/peft, POT, ollama, …). It is not on PyPI; it is installed from this folder.

The method choices follow an accompanying paper. Reference implementations that were
ported into the library live outside the repo in the parent folder:
`../Fine-Tuning BERT-perturbed.ipynb` (cross-encoder fine-tuning, perturbation
augmentation, flip-rate robustness) and `../Reimplementing Coding Approaches.ipynb`
(string/lexical/semantic methods, max-over-sentences scoring, reference thresholds a–d).

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # all extras + pytest + ruff
.venv/bin/python -m pytest                                    # default: skips slow/network/gpu/ollama/systemone
.venv/bin/python -m pytest -m "not network and not gpu and not ollama and not systemone"  # incl. slow (tiny-model training)
.venv/bin/python -m pytest tests/unit/test_core_methods.py::test_jaccard_values           # single test
.venv/bin/ruff check src tests
.venv/bin/python scripts/verify_end_to_end.py --level core    # or: --level standard --finetune --ollama-model M --systemone-url U
```

CI (`.github/workflows/tests.yml`) also runs the suite with **no optional extras**
(`pip install -e ".[test]"`) on Ubuntu/macOS/Windows × Python 3.10/3.12. Reproduce that
locally in a separate venv before pushing changes that touch optional dependencies.

`tests/regression/test_notebook_parity.py` only runs when `SIMCHOLOGY_CODED_CSV`,
`SIMCHOLOGY_UNCODED_CSV` and `SIMCHOLOGY_INFOS_CSV` point at the private study data (and
`de_core_news_lg` is installed); it compares F1 with the notebook's printed values.

## Architecture

**Flow.** `api.assess()` builds a `SimilarityAssessor` (`assessor.py`), calls `fit()`
then `predict()`. `fit()`:
1. splits coded data (`data/splitting.py`; `split_by="auto"` keeps all rows of a response
   in one split — unlike the original notebooks, which `split_by="row"` reproduces);
2. `_setup()`: language detection, `Preprocessor`, `Segmenter`, `Runtime`, and the
   `ScoringPipeline`, all bound to the method;
3. fits the method: unlabelled corpus units for vocabulary/topic methods, labelled
   train/dev frames (perturbation-augmented when needed) for supervised and few-shot methods;
4. `_fit_decision_rule()`: a `Threshold` (binary output) or `GradientScaler` (gradient
   output); whatever it `needs` ("baseline" | "labels" | "references") is computed from
   the training pairs (the dev pairs for supervised methods);
5. evaluates on the held-out test split → `report_`; records provenance.

**Methods** (`methods/`). Every method subclasses `SimilarityMethod`, declares a
`MethodInfo`, and is registered with `@register_method`; new method modules must be
added to `registry._MODULES`. `MethodInfo` fields drive the assessor's behaviour:
`requires_corpus_fit`, `requires_labels`, `native_output` ("probability" → 0.5
threshold, "score" → baseline quantile), `segment_default`, `preprocessing` ("full" =
normalise/tokenise/stop words via `Preprocessor`; "light" = whitespace only, for neural
and prompt methods), `cost_tier` ("heavy" methods are skipped by `compare_methods`
unless `include_expensive=True`), `requires_modules`/`extras`. Two base styles:
`RepresentationMethod` (`encode` unique texts once, `compare` row-aligned) and
`PairMethod` (`_score_unique_pairs`, for cross-encoders and LLM/System One judges).
**Higher score always means more similar**; distances are converted inside methods.

**Scoring pipeline** (`scoring/pipeline.py`): deduplicates rows, splits responses into
sentences, scores each unique (segment, reference) pair once, and aggregates per row
(`max` by default). Methods therefore receive segments, not whole responses.

**Internal frames** use the column names in `config.py` (`left`, `right`, `label`,
`group`); `fit(train=..., dev=...)` receives these, not the user's column names.
Column order is (response, reference): asymmetric measures report how much of the
reference is covered by the response.

**Baselines/thresholds** (`baseline/`): dissimilar pairs = coded negatives +
cross-group re-pairings (never uncoded within-group pairs); decision rule is
`score > threshold`; `ReferenceThreshold` ports notebook variants a–d.

**Hardware** (`hardware/`): methods implement `resource_request()`;
`SimilarityMethod.check_hardware()` estimates memory/time and warns via `emit()`.
Supervised methods check inside `fit()` after `max_length` is known; the assessor
checks inference once for non-light methods. `configure_environment()` runs on
`import simchology` (e.g. `PYTORCH_ENABLE_MPS_FALLBACK` must be set before torch loads).

**Prompt-based** (`llm/`, `methods/prompt.py`, `methods/systemone.py`): `llm` prompts a
model through Ollama's native API (the only LLM runtime); `systemone` speaks Jev's
System One protocol and defaults to a local Kev server. `llm/netpolicy.py` blocks non-local hosts unless `allow_remote=True`.
Passing a `client` object in `method_params` replaces the client library (tests use fakes).

**Persistence** (`io.py`): an assessor is a directory (JSON + npz + method artefacts),
written and read without pickle; thresholds, scalers, baselines and reports implement
`to_dict`/`from_dict` (or arrays), and methods implement `save`/`load`.

## Conventions that are easy to get wrong

- Never import optional heavy packages (torch, transformers, sentence_transformers,
  spacy, ot, gensim, peft, ollama, httpx, wtpsplit) at module level; call
  `_optional.require(module, feature=..., extra=...)` inside functions. A subprocess
  test asserts `import simchology` does not load torch.
- New dependencies go into the right extra in `pyproject.toml`, and the import name into
  `_optional.EXTRA_FOR` (plus `PIP_NAME` if the PyPI name differs). Install hints are
  read from the installed package metadata, so re-run `pip install -e .` after editing
  `pyproject.toml`.
- Emit warnings with `_utils.warn.warn(message, Category)` using the `Simchology*Warning`
  classes (it points the warning at the user's code); user-facing progress messages go
  through `_utils.logging.note(message, verbose)`.
- Tests run offline (`HF_HUB_OFFLINE=1` in `tests/conftest.py`) with tiny models built
  in session fixtures (`tiny_bert_dir`, `tiny_st_dir`, `tiny_qwen3_dir`,
  `spacy_vectors_nlp`); unit tests must not download models. Tests that need an
  optional package must skip without it (`from conftest import requires`,
  `@requires("torch", ...)`, or `pytest.importorskip`). Mark training-heavy tests
  `@pytest.mark.slow`; live-server tests `ollama`/`systemone`.
- Support pandas 2 and 3, transformers 4.48+ and 5.x, peft ≥ 0.14 (`exclude_modules`),
  Python ≥ 3.10. The dev venv has the newest versions (pandas 3 string dtype,
  transformers 5 `dtype=` instead of `torch_dtype=`, no `logging_dir`); e.g. save string
  arrays with `np.asarray(values, dtype=str)`, since object arrays cannot be loaded
  without pickle.
- Default models and server URLs live in `config.py` (`DEFAULT_MODELS`, `OLLAMA_HOST`,
  `SYSTEMONE_URL`).
- Design decisions made with the project owner: local-first (remote APIs only with
  `allow_remote=True`), baseline-quantile thresholds by default, no ensemble/combination
  rules. An `openai_compatible` method was removed on purpose: OpenAI-compatible
  endpoints cannot switch off reasoning, so thinking models (e.g. `qwen3.5:4b` via
  Ollama's `/v1`) took 20–65 s per judgment versus ~0.5 s through `llm` (`think=False`),
  and for local models it duplicated `llm`.
