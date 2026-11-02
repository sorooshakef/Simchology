# Simchology

**Text similarity for psychologists — many NLP methods, one interface.**

Simchology scores how similar two texts are, for example whether a participant's recall
contains a piece of information, how close a paraphrase is to its source, or whether two
free responses express the same idea. It wraps string measures, bag-of-words and topic
models, word and sentence embeddings, BERTScore, fine-tuned cross-encoders and prompted
language models behind a single function, so you can switch methods and compare them seamlessly.

```python
import simchology as sim

result = sim.assess(
    data,                       # a pandas DataFrame, one row per pair of texts
    columns=("text", "info"),   # (response, reference)
    method="sbert",             # or "jaccard", "tfidf", "bertscore", "cross_encoder", "llm", ...
    output="binary",            # or "gradient"
    coded=coded_data,           # optional: human codes for training, thresholds and evaluation
    label_col="included",
    group_col="topic",
)
result["predicted_similarity"]
```

The result is your DataFrame with one new column, `predicted_similarity`: 0/1 for
`output="binary"`, a continuous score for `output="gradient"`.

---

## Installation

Simchology is installed from this folder (it is not yet published on PyPI). All
dependencies are declared in `pyproject.toml`, so one command installs Simchology
together with everything it needs. Python 3.10 or newer is required.

```bash
cd Simchology                      # the folder containing pyproject.toml
python -m venv .venv               # a separate environment (recommended)
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # recommended set: spaCy, sentence embeddings, BERTScore, WMD
```

`requirements.txt` is a shortcut for `pip install -e ".[recommended]"`. To install
less or more:

```bash
pip install -e .                   # core only: string, lexical and topic-model methods
pip install -e ".[all]"            # everything, incl. fine-tuning and local LLMs
pip install -e ".[nlp,finetune]"   # any combination of the groups below
```

Dependency groups: `nlp` (spaCy), `sat` (robust sentence splitting), `embeddings`
(torch, transformers, sentence-transformers), `static` (gensim word vectors), `wmd`,
`finetune` (cross-encoders, LoRA), `llm` (Ollama), `systemone` (Jev-style zero-shot
classifiers). Keep the quotes around `".[...]"` (zsh, the default shell on macOS,
otherwise misreads the brackets). If you use a method
whose dependencies are missing, Simchology tells you the exact command to install them.

For lemmatisation and word vectors, install the spaCy pipeline of your language once:

```python
sim.download_resources(language="de")     # or: python -m spacy download de_core_news_lg
```

Simchology runs on Windows, macOS and Linux, on CPUs, NVIDIA GPUs (CUDA) and Apple
silicon (MPS). `print(sim.hardware_report())` shows what it found. On Windows, `pip`
installs a CPU-only torch; for an NVIDIA GPU install torch from
<https://pytorch.org/get-started/locally/> first (the hardware report tells you if
this happened).

## Data format

Long format: one row per pair to compare. Typical recall study:

| participant | topic    | text (response)                         | info (reference)                         | included |
|-------------|----------|-----------------------------------------|------------------------------------------|----------|
| P001        | Giraffes | Giraffes sleep only two hours. They ... | Giraffes sleep only about two hours a day. | 1        |
| P001        | Giraffes | Giraffes sleep only two hours. They ... | Giraffes mainly eat acacia leaves.         | 0        |

`sim.load_demo("en")` / `sim.load_demo("de")` returns a synthetic dataset in this format.

**Column order matters**: `columns=(response, reference)`. Asymmetric measures (e.g.
BERTScore recall, BM25, Jaccard containment, cross-encoders, LLM prompts) ask how much
of the *reference* is found in the *response*.

## Choosing a method

```python
sim.list_methods()        # all methods, with what they need and whether they are installed
```

| family | methods | needs coded data | cost | good for |
|---|---|---|---|---|
| string | `jaccard`, `levenshtein`, `osa`, `damerau_levenshtein` | no | light | verbatim recall, spelling, exact wording |
| lexical | `bow`, `tfidf`, `bm25` | no | light | shared content words |
| topic | `lsa`, `lda` | no | light | longer texts, large corpora |
| static embeddings | `word_vectors`, `wmd` | no | light–moderate | synonyms (no word order / word sense) |
| contextual embeddings | `sbert`, `hf_embedding`, `llm_embedding` | no | moderate | meaning, paraphrases |
| alignment | `bertscore` | no | moderate | paraphrases, with token-level explanations |
| supervised | `cross_encoder`, `lora_classifier`, `pretrained_cross_encoder` | yes (except pretrained) | heavy | *your* operationalisation of similarity |
| prompt-based | `llm` (Ollama), `systemone` (Jev/Kev) | optional (few-shot) | heavy | complex coding manuals, no training data |

Not sure? Compare them on your coded data:

```python
table = sim.compare_methods(coded, "standard", label_col="included", group_col="topic")
```

Every method is evaluated on the *same* held-out pairs; expensive methods only run with
`include_expensive=True`.

## Coded data: training, thresholds and honest evaluation

When you pass `coded=...`, Simchology splits the coded pairs into training and
test pairs (default 80/20, stratified by label and `group_col`). Training pairs are used
to fine-tune models and set thresholds; test pairs are used **only** to report agreement
(precision, recall, F1, Cohen's κ, …, or correlations for gradient output):

```python
result, assessor = sim.assess(data, method="tfidf", output="binary", coded=coded,
                              label_col="included", group_col="topic", return_assessor=True)
print(assessor.report_)                     # agreement on held-out pairs
assessor.evaluate(ci=0.95)                  # with bootstrap confidence intervals
```

**No leakage by default.** In long-format data the same response appears in many rows
(once per reference). With `split_by="auto"`, all rows of a response stay in the same
split, so the test set measures performance on *new* responses. Use `split_by="row"` to
reproduce analyses that split rows, or `split_by="participant"` (any column).

Labels may be binary (0/1, True/False, two categories with `positive_label=`) or
gradient (e.g. 1–5 ratings). For gradient labels with binary output, say which values
count as similar: `positive_threshold=4`.

## Binary decisions: baselines and thresholds

Raw similarity values are not comparable across methods — sentence embeddings give even
unrelated texts cosine similarities well above 0. Simchology therefore builds a
**baseline**: the scores of pairs known to be dissimilar,

* pairs your coders judged dissimilar, and
* responses re-paired with references from **other groups** (other topics/stimuli), which
  are safely unrelated (pass `group_col=`),

and by default calls a pair similar when its score exceeds the **95th percentile** of the
baseline (`baseline_quantile=0.95`).

```python
b = sim.baseline(coded, method="sbert", label_col="included", group_col="topic")
b.summary()           # what scores do unrelated pairs get with this method?
```

Other decision rules (`threshold=`):

| `threshold=` | rule |
|---|---|
| `"auto"` (default) | 0.5 for probabilities (supervised, prompt-based), else the baseline quantile |
| `"baseline"` / `"baseline_group"` | baseline quantile, overall / per group |
| `"optimize"`, `"optimize_youden"`, … | the cut maximising F1 (Youden's J, …) on the coded *training* pairs |
| `"reference_a"` … `"reference_d"` | from similarities between distinct references: overall (a), fixed group-wise (b), group-wise (c), item-wise (d) |
| a number | fixed threshold |

Gradient output can also be expressed relative to the baseline:
`gradient_scale="baseline_percentile"` (0.5 = typical unrelated pair),
`"baseline_z"`, `"baseline_rescaled"` (0 = median unrelated pair, 1 = identical), or
`"calibrated"` (mapped onto your coding scale with the coded training pairs).

## Preprocessing and sentences

For string, lexical, topic and word-vector methods, texts are normalised (e.g. `50%` →
`50 Prozent`), tokenised, lemmatised (with spaCy), and stop words and punctuation are
removed — following common recommendations for these methods. Customise with
`preprocess={"unit": "surface", "remove_stopwords": False, "replacements": {"10m": "10 Meter"}}`.
Neural, supervised and prompt-based methods receive the texts as written.

Responses are split into sentences and the best-matching sentence counts
(`aggregate="max"`; or `"mean"`). If your participants do not use reliable punctuation,
use `segment="sat"` (neural sentence segmentation; install with `pip install -e ".[sat]"`).

## Supervised pair scoring (fine-tuning)

```python
result, assessor = sim.assess(
    data, method="cross_encoder", output="binary", coded=coded, label_col="included",
    method_params={"backbone": "jhu-clsp/mmBERT-base", "epochs": 10}, return_assessor=True,
)
assessor.evaluate(robustness=True)    # how often do truncated endings flip a judgment?
assessor.save("my_coder.simchology")  # reuse later with sim.load(...)
```

Training data are augmented by default when responses lack final punctuation: every
training pair gets a twin whose response lost its final period or had its last word cut,
which makes the model robust to such artefacts.

`method="lora_classifier"` fine-tunes a small decoder LLM with LoRA instead (default
`Qwen/Qwen3-0.6B`). Other backbones, including newer ones, are set with
`method_params={"backbone": ...}`, e.g. `Qwen/Qwen3.5-0.8B` to compare prompting and
fine-tuning of the same model (`method="llm"` with the Ollama model `qwen3.5:0.8b`).
On Apple silicon, Qwen3.5 trains about 2.5 times more slowly than Qwen3-0.6B, because
the fast kernels for its linear-attention layers exist only for CUDA GPUs.

## Prompt-based inference (local by default)

**Local LLMs with Ollama.** Install [Ollama](https://ollama.com), pull a model
(`ollama pull gemma3:4b`), then:

```python
sim.assess(data, method="llm", output="binary",
           method_params={"model": "gemma3:4b",
                          "instructions": "An info counts as recalled if ... (your coding manual)"})
```

Pass `coded=` and `method_params={"mode": "few_shot"}` to include coded examples in the
prompt (examples come only from the training split).

**Zero-shot "System One" classifiers (Jev-style).** Jev itself is a hosted service; the
`systemone` method speaks its protocol and by default talks to a **local** open
re-implementation such as [Kev](https://github.com/jaredpalmer/kev), which you start with
e.g. `python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009`.

**Privacy.** Participant texts never leave your computer unless you explicitly pass
`allow_remote=True` (needed for hosted Jev via `systemone`, or an Ollama server running on
another computer).

## Hardware checks

Before fine-tuning or running large models Simchology estimates the memory and time
needed and warns (`SimchologyHardwareWarning`) when your machine is probably too small or
too slow, with suggestions (e.g. LoRA, shorter `max_length`). Use `check_hardware="error"`
to stop instead, or `"off"`.

## Explanations and reporting

```python
assessor.explain(response_text, reference_text)   # shared words, token alignments, best sentence
sim.explain(response_text, reference_text, method="bertscore")
print(assessor.describe())                         # a paragraph for your methods section
```

`assessor.provenance_["models"]` records every model that was used (including the spaCy
pipeline for preprocessing and sentence splitting) with its name, version and size: the
commit and number of parameters of Hugging Face models, the digest, number of parameters and
quantisation of Ollama models, the version and word-vector table of spaCy pipelines, and the
release date and served checkpoint of System One models.

Saved assessors (`assessor.save(path)`) contain the configuration, thresholds, baseline,
evaluation and provenance (package and model versions), and are loaded without pickle:
`sim.load(path)` (pass `device="cpu"` to run a saved neural model on the CPU).

## Checking an installation

```bash
python scripts/verify_end_to_end.py --level core        # or: --level standard --finetune
python -m pytest                                        # fast offline test suite
```

## Reproducibility

All random steps are seeded (`seed=42`). Results of neural models can still differ
slightly across hardware; prompt-based methods with hosted models can change when the
provider updates or retires a model — prefer local models for archived analyses.

## License

MIT; see [LICENSE](LICENSE).
