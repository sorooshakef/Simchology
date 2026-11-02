"""Model records in the provenance: name, version and size of every model a method used."""

from __future__ import annotations

import json
import struct

import pytest
from conftest import requires

import simchology as sim
from simchology._utils.provenance import format_model, hf_model_record, spacy_model_record
from simchology.text.segment import Segmenter


def _write_safetensors(path, shapes):
    """A safetensors file with only a header (enough to count parameters)."""
    header = {name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


@pytest.mark.parametrize("complete_snapshot", [True, False])
def test_hub_model_version_and_size_from_the_cache(tmp_path, monkeypatch, complete_snapshot):
    hub = pytest.importorskip("huggingface_hub")
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    if not complete_snapshot:  # huggingface_hub >= 1.0 refuses snapshots with skipped files offline
        def refuse(*args, **kwargs):
            raise OSError("The cached snapshot is incomplete")

        monkeypatch.setattr(hub, "snapshot_download", refuse)
    sha = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "models--org--tiny-st"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    snapshot = repo / "snapshots" / sha
    _write_safetensors(snapshot / "model.safetensors", {"embeddings": (10, 4), "bias": (4,)})
    _write_safetensors(snapshot / "2_Dense" / "model.safetensors", {"linear": (4, 2)})
    (snapshot / "modules.json").write_text(json.dumps([{"idx": 0, "path": ""}, {"idx": 1, "path": "2_Dense"}]))
    (snapshot / "config.json").write_text("{}")

    assert hf_model_record("org/tiny-st") == {"name": "org/tiny-st", "source": "huggingface",
                                              "version": sha, "parameters": 52}
    # not cached: only the requested revision is known, and nothing is downloaded
    assert hf_model_record("org/missing", "v1.0") == {"name": "org/missing", "source": "huggingface",
                                                      "version": "v1.0"}


@requires("torch", "transformers")
def test_local_model_folder(tiny_bert_dir):
    import transformers

    n_params = sum(p.numel() for p in transformers.BertModel.from_pretrained(tiny_bert_dir).parameters())
    assert hf_model_record(tiny_bert_dir) == {"name": tiny_bert_dir, "source": "local", "version": None,
                                              "parameters": n_params}


def test_spacy_pipeline_record(spacy_vectors_nlp):
    record = spacy_model_record(spacy_vectors_nlp, "en_test_vectors")
    assert record == {"name": "en_test_vectors", "source": "spacy", "version": spacy_vectors_nlp.meta["version"],
                      "vectors": spacy_vectors_nlp.vocab.vectors.shape[0], "vector_dimensions": 16}
    assert spacy_model_record(spacy_vectors_nlp)["name"] == "en_pipeline"


def test_format_model():
    assert format_model({"name": "Qwen/Qwen3-0.6B", "source": "huggingface",
                         "version": "c1899de289a04d12100db370d81485cdf75e47ca", "parameters": 596_049_920}
                        ) == "Qwen/Qwen3-0.6B (revision c1899de, 596M parameters)"
    assert format_model({"name": "qwen3:0.6b", "source": "ollama", "version": "7df6b6e09427a769808717c0",
                         "parameters": 751_632_384, "quantization": "Q4_K_M"}
                        ) == "qwen3:0.6b (Ollama, digest 7df6b6e09427, 752M parameters, Q4_K_M)"
    assert format_model({"name": "de_core_news_lg", "source": "spacy", "version": "3.8.0",
                         "vectors": 500_000, "vector_dimensions": 300}
                        ) == "de_core_news_lg (version 3.8.0, 500,000 word vectors)"
    assert format_model({"name": "kev-latest", "source": "systemone", "url": "http://localhost:8009",
                         "version": "2026-05-01", "run": "jaredpalmer/kev-4b"}
                        ) == "kev-latest @ http://localhost:8009 (release 2026-05-01, run jaredpalmer/kev-4b)"
    assert format_model({"name": "/data/model", "source": "local", "version": None}) == "/data/model"
    assert format_model("gemma3:4b (ollama)") == "gemma3:4b (ollama)"  # assessors saved before records


def test_rule_based_segmentation_uses_no_model():
    assert Segmenter("regex", "en").model_details() is None
    assert Segmenter("none", "en").model_details() is None


def test_assessor_records_models_at_fit_and_keeps_them(tmp_path, demo_en, spacy_vectors_nlp):
    a = sim.SimilarityAssessor("word_vectors", "binary", group_col="topic", language="en",
                               method_params={"vectors": spacy_vectors_nlp}, verbose=False)
    a.fit(demo_en, label_col="label")
    models = a.provenance_["models"]
    assert models["word_vectors"] == spacy_model_record(spacy_vectors_nlp)
    assert set(models) <= {"word_vectors", "preprocessing", "segmentation"}
    assert "Models: en_pipeline (version" in a.describe()
    json.dumps(models)  # saved in simchology.json
    loaded = sim.SimilarityAssessor.load(a.save(tmp_path / "saved"))
    assert loaded.provenance_["models"] == models
