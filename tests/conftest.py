"""Shared fixtures. Unit tests run offline on the CPU with tiny, locally built models."""

from __future__ import annotations

import os
import re
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import simchology as sim  # noqa: E402
from simchology._optional import is_available  # noqa: E402


def requires(*modules: str):
    """Skip a test unless all optional ``modules`` are installed."""
    missing = [m for m in modules if not is_available(m)]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)}")


@pytest.fixture(autouse=True)
def _quiet_simchology_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        yield


@pytest.fixture(scope="session")
def demo_de() -> pd.DataFrame:
    return sim.load_demo("de", seed=0)


@pytest.fixture(scope="session")
def demo_en() -> pd.DataFrame:
    return sim.load_demo("en", seed=0)


@pytest.fixture(scope="session")
def small_pairs() -> pd.DataFrame:
    """A handful of hand-written pairs with obvious answers."""
    return pd.DataFrame({
        "text": [
            "Giraffes are the tallest land animals. They sleep two hours a day.",
            "Giraffes are the tallest land animals. They sleep two hours a day.",
            "The Romans built aqueducts to carry water.",
            "I forgot everything.",
        ],
        "info": [
            "Giraffes sleep about two hours a day.",
            "The moai are stone statues on Easter Island.",
            "The Romans built aqueducts to carry water into the cities.",
            "Giraffes sleep about two hours a day.",
        ],
        "label": [1, 0, 1, 0],
    })


# ----------------------------------------------------------------------------- tiny models
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


@pytest.fixture(scope="session")
def tiny_vocab_file(tmp_path_factory, demo_de, demo_en) -> str:
    words: set[str] = set()
    for df in (demo_de, demo_en):
        for col in ("text", "info"):
            for t in df[col].unique():
                words.update(re.findall(r"\w+", t.lower()))
    extra = ("giraffes giraffe tall tallest land animals sleep sleeps hours day romans built "
             "aqueducts carry water cities moai stone statues easter island forgot everything "
             "they a about two the to on into i").split()
    words.update(extra)
    vocab = _SPECIALS + sorted(words) + list(".,!?;:-'\"()%")
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory, tiny_vocab_file) -> str:
    """A 2-layer, 32-dimensional BERT with a WordPiece tokenizer over the demo vocabulary."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch

    torch.manual_seed(0)
    tok = transformers.BertTokenizerFast(vocab_file=tiny_vocab_file, do_lower_case=True)
    cfg = transformers.BertConfig(
        vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=37, max_position_embeddings=256,
    )
    out = tmp_path_factory.mktemp("tiny_bert")
    transformers.BertModel(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="session")
def tiny_st_dir(tmp_path_factory, tiny_bert_dir) -> str:
    """A SentenceTransformer (tiny BERT + mean pooling)."""
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer, models

    model = SentenceTransformer(modules=[models.Transformer(tiny_bert_dir, max_seq_length=128),
                                         models.Pooling(32)], device="cpu")
    out = tmp_path_factory.mktemp("tiny_st")
    model.save(str(out))
    return str(out)


@pytest.fixture(scope="session")
def tiny_qwen3_dir(tmp_path_factory, tiny_vocab_file) -> str:
    """A tiny decoder (Qwen3 architecture) for LoRA sequence-classification tests."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch

    torch.manual_seed(0)
    tok = transformers.BertTokenizerFast(vocab_file=tiny_vocab_file, do_lower_case=True)
    cfg = transformers.Qwen3Config(
        vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=64, head_dim=16, max_position_embeddings=256,
        pad_token_id=tok.pad_token_id,
    )
    out = tmp_path_factory.mktemp("tiny_qwen3")
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="session")
def spacy_vectors_nlp(demo_en):
    """A blank English spaCy pipeline with a sentencizer and random 16-d word vectors."""
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    rng = np.random.default_rng(0)
    words: set[str] = set()
    for col in ("text", "info"):
        for t in demo_en[col].unique():
            words.update(re.findall(r"\w+", t.lower()))
    for w in sorted(words):
        nlp.vocab.set_vector(w, rng.normal(size=16).astype("float32"))
    return nlp
