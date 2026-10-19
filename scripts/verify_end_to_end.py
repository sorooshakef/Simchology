"""End-to-end verification of a Simchology installation.

    python scripts/verify_end_to_end.py --level core       # dependency-free methods
    python scripts/verify_end_to_end.py --level standard   # + installed neural methods
    python scripts/verify_end_to_end.py --level full --finetune --ollama-model gemma3:4b \
        --systemone-url http://localhost:8009

Prints one PASS / FAIL / SKIP line per check and exits non-zero if any check failed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

import simchology as sim
from simchology._optional import is_available

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, needs: tuple[str, ...] = ()):
    def deco(fn):
        def run(*args, **kwargs):
            missing = [m for m in needs if not is_available(m)]
            if missing:
                RESULTS.append(("SKIP", name, f"needs {', '.join(missing)}"))
                return None
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    detail = fn(*args, **kwargs) or ""
                RESULTS.append(("PASS", name, f"{detail} ({time.perf_counter() - t0:.1f}s)"))
            except _Skip as exc:
                RESULTS.append(("SKIP", name, str(exc)))
            except Exception as exc:  # noqa: BLE001
                RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
                if ARGS.verbose:
                    traceback.print_exc()
        return run
    return deco


class _Skip(Exception):
    pass


def _assess(data, method, output="binary", **kw):
    return sim.assess(data, method=method, output=output, coded=data, label_col="label",
                      group_col="topic", verbose=False, return_assessor=True, device=ARGS.device, **kw)


@check("hardware report")
def c_hardware():
    report = sim.hardware_report()
    print(report)
    return f"default device {report.best.torch_device}"


@check("binary tfidf with coded data")
def c_binary(data):
    out, a = _assess(data, "tfidf")
    assert str(out["predicted_similarity"].dtype) == "Int8"
    assert out.index.equals(data.index)
    f1 = a.report_.metrics["f1"]
    assert f1 > 0.7, f"F1 {f1:.3f} unexpectedly low"
    return f"F1 {f1:.3f}"


@check("gradient output, baseline percentile")
def c_gradient(data):
    out, a = _assess(data, "jaccard", output="gradient", gradient_scale="baseline_percentile")
    v = out["predicted_similarity"]
    assert v.between(0, 1).all()
    return f"Spearman {a.report_.metrics['spearman_rho']:.3f}"


@check("all threshold strategies")
def c_thresholds(data):
    f1 = {}
    for t in ("baseline", "baseline_group", "optimize", "reference_a", "reference_b",
              "reference_c", "reference_d"):
        _, a = _assess(data, "tfidf", threshold=t)
        f1[t] = a.report_.metrics["f1"]
    return ", ".join(f"{k} {v:.2f}" for k, v in f1.items())


@check("compare_methods")
def c_compare(data):
    preset = "fast" if ARGS.level == "core" else "standard"
    table = sim.compare_methods(data, preset, label_col="label", group_col="topic", verbose=False,
                                include_expensive=ARGS.level == "full")
    print(table[[c for c in ("method", "status", "f1", "cohen_kappa", "seconds") if c in table]]
          .to_string(index=False))
    failed = table[table["status"].str.startswith("failed")]
    assert failed.empty, f"failed: {', '.join(failed['method'])}"
    return f"{(table['status'] == 'ok').sum()} methods ok"


@check("explanations")
def c_explain(data):
    row = data.iloc[0]
    done = []
    methods = ("jaccard", "tfidf") if ARGS.level == "core" else ("jaccard", "tfidf", "bertscore", "wmd")
    for method in methods:
        try:
            exp = sim.explain(row["text"], row["info"], method=method, language=ARGS.language)
            assert exp is not None
            done.append(method)
        except sim.MissingDependencyError:
            continue
        except sim.ModelNotAvailableError:
            continue
    return "explained with " + ", ".join(done)


@check("save/load round trip")
def c_roundtrip(data):
    out, a = _assess(data, "tfidf", keep_scores=True)
    with tempfile.TemporaryDirectory() as tmp:
        b = sim.load(a.save(Path(tmp) / "a"))
        again = b.predict(data, keep_scores=True)
    assert np.allclose(again["similarity_score"], out["similarity_score"])
    assert again["predicted_similarity"].equals(out["predicted_similarity"])
    return "identical predictions"


@check("neural embeddings (sbert)", needs=("torch", "sentence_transformers"))
def c_sbert(data):
    if ARGS.level == "core":
        raise _Skip("use --level standard")
    _, a = _assess(data, "sbert")
    return f"F1 {a.report_.metrics['f1']:.3f}"


@check("fine-tuning (cross-encoder, 1 epoch)", needs=("torch", "transformers", "accelerate"))
def c_finetune(data):
    if not ARGS.finetune:
        raise _Skip("use --finetune")
    small = data.groupby("topic").head(40)
    _, a = _assess(small, "cross_encoder", method_params={"epochs": 1})
    return f"F1 {a.report_.metrics['f1']:.3f}"


@check("Ollama judgments", needs=("ollama",))
def c_ollama(data):
    if not ARGS.ollama_model:
        raise _Skip("use --ollama-model NAME")
    sample = data.head(20)
    params = {"model": ARGS.ollama_model}
    first = sim.assess(sample, method="llm", output="binary", method_params=params, verbose=False)
    second = sim.assess(sample, method="llm", output="binary", method_params=params, verbose=False)
    same = (first["predicted_similarity"] == second["predicted_similarity"]).mean()
    return f"{same:.0%} identical on repetition"


@check("System One judgments", needs=("httpx",))
def c_systemone(data):
    if not ARGS.systemone_url:
        raise _Skip("use --systemone-url URL")
    sample = data.head(20)
    out = {}
    for endpoint in ("/v1/systemone", "/v1/systemone/separate"):
        out[endpoint] = sim.assess(sample, method="systemone", output="gradient", verbose=False,
                                   method_params={"base_url": ARGS.systemone_url, "endpoint": endpoint})
    diff = np.abs(out["/v1/systemone"]["predicted_similarity"] - out["/v1/systemone/separate"]["predicted_similarity"])
    return f"mean |grouped - separate| = {diff.mean():.3f}"


def main() -> int:
    data = sim.load_demo(ARGS.language)
    c_hardware()
    for fn in (c_binary, c_gradient, c_thresholds, c_compare, c_explain, c_roundtrip, c_sbert,
               c_finetune, c_ollama, c_systemone):
        fn(data)
    print()
    for status, name, detail in RESULTS:
        print(f"{status:4s}  {name:<40s} {detail}")
    return 1 if any(s == "FAIL" for s, _, _ in RESULTS) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=("core", "standard", "full"), default="core")
    parser.add_argument("--language", choices=("de", "en"), default="en")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--ollama-model")
    parser.add_argument("--systemone-url")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    ARGS = parser.parse_args()
    sys.exit(main())
