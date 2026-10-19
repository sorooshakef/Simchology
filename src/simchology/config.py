"""Library-wide defaults. Change default models here, in one place."""

from __future__ import annotations

# Default models per method. Multilingual where possible, since many psychology
# studies are not run in English.
DEFAULT_MODELS: dict[str, str] = {
    "sbert": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # The model used by the "sentence" method of the original R/Python coding scripts
    # (a multilingual analogue of the Universal Sentence Encoder).
    "use_like": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "llm_embedding": "Qwen/Qwen3-Embedding-0.6B",
    "hf_embedding": "jhu-clsp/mmBERT-base",
    "bertscore": "bert-base-multilingual-cased",
    "cross_encoder": "jhu-clsp/mmBERT-base",
    "pretrained_cross_encoder": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "lora_classifier": "Qwen/Qwen3-0.6B",
    "ollama": "gemma3:4b",
    "sat": "sat-3l-sm",
}

# spaCy pipelines per language. The "lg" pipelines ship static word vectors, which the
# word-vector and WMD methods need; the others only need tokens/lemmas/sentences.
SPACY_MODELS: dict[str, str] = {
    "de": "de_core_news_lg",
    "en": "en_core_web_lg",
    "fr": "fr_core_news_lg",
    "es": "es_core_news_lg",
    "it": "it_core_news_lg",
    "nl": "nl_core_news_lg",
    "pt": "pt_core_news_lg",
}

OLLAMA_HOST = "http://localhost:11434"
# Default address of a locally running System One server such as Kev
# (``python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009``).
SYSTEMONE_URL = "http://localhost:8009"

DEFAULT_SEED = 42
DEFAULT_OUTPUT_COL = "predicted_similarity"
SCORE_COL = "similarity_score"
THRESHOLD_COL = "similarity_threshold"

# Column names of Simchology's internal, standardised pair frame.
LEFT, RIGHT, LABEL, GROUP = "left", "right", "label", "group"

# Column names that :func:`simchology.data.labels.guess_label_col` tries, in order.
LABEL_COL_CANDIDATES = ("label", "included", "similar", "similarity", "coded", "rating", "score")
