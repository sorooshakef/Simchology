"""OllamaJudge with a fake Ollama client (no server needed)."""

import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import simchology as sim
from simchology.data.labels import LabelInfo
from simchology.exceptions import (
    BackendUnavailableError,
    ModelNotAvailableError,
    RemoteNotAllowedError,
    SimchologyWarning,
)
from simchology.methods.base import Runtime
from simchology.methods.prompt import OllamaJudge

PAIR_RE = re.compile(r'"""\n(.*?)\n"""', re.S)


class FakeOllama:
    """Mimics ollama.Client: list/show/pull/chat. Answers from a lookup of true labels."""

    def __init__(self, truth=None, *, models=("gemma3:4b",), logprobs=True, garbage=0,
                 context_length=8192, connection_error=False, accept_logprobs=True):
        self.truth = truth or {}
        self.models_ = list(models)
        self.logprobs = logprobs
        self.garbage = garbage
        self.context_length = context_length
        self.connection_error = connection_error
        self.accept_logprobs = accept_logprobs
        self.calls = []
        self.pulled = []

    def list(self):
        if self.connection_error:
            raise ConnectionError("Failed to connect to Ollama.")
        return {"models": [{"model": m, "digest": "sha256:0123456789abcdef", "size": 3_300_000_000,
                            "details": {"parameter_size": "4.3B"}} for m in self.models_]}

    def show(self, model):
        return {"modelinfo": {"gemma3.context_length": self.context_length,
                              "gemma3.block_count": 34, "gemma3.embedding_length": 2560,
                              "general.parameter_count": 4_299_915_632},
                "details": {"parameter_size": "4.3B", "quantization_level": "Q4_K_M"}}

    def pull(self, model):
        self.pulled.append(model)
        self.models_.append(model)
        return {"status": "success"}

    def chat(self, model, messages, format=None, options=None, keep_alive=None, think=None,
             **kwargs):
        if kwargs and not self.accept_logprobs:
            raise TypeError("chat() got an unexpected keyword argument 'logprobs'")
        self.calls.append({"model": model, "messages": messages, "format": format,
                           "options": options, "think": think, **kwargs})
        if self.garbage > 0:
            self.garbage -= 1
            return {"message": {"role": "assistant", "content": "I am not sure."}}
        reference, response = PAIR_RE.findall(messages[-1]["content"])[-2:]
        positive = bool(self.truth.get((response, reference), 0))
        if "rating" in format["properties"]:
            lo, hi = format["properties"]["rating"]["minimum"], format["properties"]["rating"]["maximum"]
            value = hi if positive else lo
            content = f'{{"rating": {value}}}'
            lp = [{"token": '{"rating": ', "logprob": 0.0},
                  {"token": str(value), "logprob": math.log(0.8),
                   "top_logprobs": [{"token": str(value), "logprob": math.log(0.8)},
                                    {"token": str(lo + 1), "logprob": math.log(0.2)}]}]
        else:
            word, other = ("yes", "no") if positive else ("no", "yes")
            content = f'{{"answer": "{word}"}}'
            lp = [{"token": '{"answer": "', "logprob": 0.0},
                  {"token": word, "logprob": math.log(0.9),
                   "top_logprobs": [{"token": word, "logprob": math.log(0.9)},
                                    {"token": other, "logprob": math.log(0.1)}]},
                  {"token": '"}', "logprob": 0.0}]
        out = {"message": {"role": "assistant", "content": content}}
        if self.logprobs and kwargs.get("logprobs"):
            out["logprobs"] = lp
        return out


@pytest.fixture()
def demo():
    return sim.load_demo("en", n_per_topic=8, seed=1)


@pytest.fixture()
def truth(demo):
    return {(t, i): y for t, i, y in zip(demo.text, demo["info"], demo.label)}


def make_judge(client, output="binary", **params):
    judge = OllamaJudge(client=client, **params)
    judge.bind(None, Runtime(output=output, language="en", check_hardware="off", verbose=False))
    return judge


def test_binary_end_to_end_with_assess(demo, truth):
    fake = FakeOllama(truth)
    out, a = sim.assess(demo, method="llm", method_params={"client": fake}, output="binary",
                        coded=demo, label_col="label", group_col="topic", check_hardware="off",
                        verbose=False, return_assessor=True)
    assert str(out["predicted_similarity"].dtype) == "Int8"
    assert (out["predicted_similarity"].astype(int) == demo["label"]).all()
    assert a.report_.metrics["f1"] == pytest.approx(1.0)
    assert a.threshold_.kind == "probability"
    call = fake.calls[0]
    assert call["format"]["properties"]["answer"]["enum"] == ["yes", "no"]
    assert call["options"]["temperature"] == 0.0 and call["options"]["seed"] == 42
    assert call["think"] is False and call["logprobs"] is True
    assert a.provenance_["models"]["llm"] == {
        "name": "gemma3:4b", "source": "ollama", "version": "sha256:0123456789abcdef",
        "parameters": 4_299_915_632, "quantization": "Q4_K_M"}
    assert "gemma3:4b (Ollama, digest 0123456789ab, 4.3B parameters, Q4_K_M)" in a.describe()


def test_unique_pairs_asked_once(demo, truth):
    fake = FakeOllama(truth)
    judge = make_judge(fake)
    left = list(demo.text) * 2
    right = list(demo["info"]) * 2
    scores = judge.score_pairs(left, right)
    assert len(fake.calls) == len(set(zip(demo.text, demo["info"])))
    assert np.allclose(scores[: len(demo)], scores[len(demo):])
    assert set(np.round(scores, 3)) <= {0.1, 0.9}  # P(yes) from the token probabilities


def test_gradient_uses_probability_when_logprobs_available(demo, truth):
    judge = make_judge(FakeOllama(truth), output="gradient")
    s = judge.score_pairs(list(demo.text[:10]), list(demo["info"][:10]))
    assert judge._kind == "probability" and judge.native_output() == "probability"
    assert np.all((s > 0) & (s < 1))


def test_gradient_falls_back_to_rating_without_logprobs(demo, truth):
    fake = FakeOllama(truth, logprobs=False)
    judge = make_judge(fake, output="gradient", scale=(1, 5))
    s = judge.score_pairs(list(demo.text[:10]), list(demo["info"][:10]))
    assert judge._kind == "rating" and judge.native_output() == "score"
    expected = np.array([1.0 if truth[(t, i)] else 0.0 for t, i in zip(demo.text[:10], demo["info"][:10])])
    assert np.allclose(s, expected)
    assert any("rating" in c["format"]["properties"] for c in fake.calls)


def test_rating_uses_expected_value_from_logprobs(demo, truth):
    judge = make_judge(FakeOllama(truth), output="gradient", response="rating", scale=(1, 5))
    s = judge.score_pairs([demo.text[0]], [demo["info"][0]])
    positive = truth[(demo.text[0], demo["info"][0])]
    expected = (0.8 * 5 + 0.2 * 2 - 1) / 4 if positive else (0.8 * 1 + 0.2 * 2 - 1) / 4
    assert s[0] == pytest.approx(expected)


def test_retry_then_success_adds_reminder(demo, truth):
    fake = FakeOllama(truth, garbage=1)
    judge = make_judge(fake)
    s = judge.score_pairs([demo.text[0]], [demo["info"][0]])
    assert not np.isnan(s[0])
    assert len(fake.calls) == 2
    assert fake.calls[1]["messages"][-1]["content"].endswith("Answer only with the JSON object.")


def test_unreadable_answers_become_nan_with_one_warning(demo, truth):
    fake = FakeOllama(truth, garbage=100)
    judge = make_judge(fake, max_retries=1)
    with pytest.warns(SimchologyWarning, match="3 of 3 judgments"):
        s = judge.score_pairs(list(demo.text[:3]), list(demo["info"][:3]))
    assert np.isnan(s).all()
    assert len(fake.calls) == 6  # 1 try + 1 retry each


def test_cache_resume(tmp_path, demo, truth):
    first = FakeOllama(truth)
    make_judge(first, cache_dir=tmp_path).score_pairs(list(demo.text[:12]), list(demo["info"][:12]))
    assert len(first.calls) > 0
    second = FakeOllama(truth)
    s = make_judge(second, cache_dir=tmp_path).score_pairs(list(demo.text[:12]), list(demo["info"][:12]))
    assert second.calls == []
    assert not np.isnan(s).any()


def test_few_shot_examples_balanced_and_without_leakage(demo, truth):
    fake = FakeOllama(truth)
    splits = sim.split_coded(demo, label_col="label", group_col="topic", seed=0, verbose=False)
    train = splits.train.rename(columns={"text": "left", "info": "right"})[["left", "right", "label"]]
    judge = make_judge(fake, mode="few_shot", n_examples=4)
    judge.fit(train=train)
    query = splits.test.text.iloc[0]
    examples = judge.select_examples(query, "binary")
    assert len(examples) == 4
    answers = [e[2] for e in examples]
    assert answers.count('{"answer": "yes"}') == 2 and answers.count('{"answer": "no"}') == 2
    assert all(e[0] in set(splits.train.text) for e in examples)
    # an example never shares the query's response
    train_query = train.left.iloc[0]
    assert all(e[0] != train_query for e in judge.select_examples(train_query, "binary"))
    judge.score_pairs([query], [splits.test["info"].iloc[0]])
    roles = [m["role"] for m in fake.calls[0]["messages"]]
    assert roles == ["system"] + ["user", "assistant"] * 4 + ["user"]


def test_few_shot_examples_only_from_training_split(demo, truth):
    fake = FakeOllama(truth)
    out, a = sim.assess(demo, method="llm", method_params={"client": fake, "mode": "few_shot"},
                        output="binary", coded=demo, label_col="label", check_hardware="off",
                        verbose=False, return_assessor=True)
    test_texts = set(a.splits_.test.text)
    for call in fake.calls:
        for m in call["messages"][1:-1]:
            if m["role"] == "user":
                assert PAIR_RE.findall(m["content"])[-1] not in test_texts


def test_few_shot_needs_coded_data():
    judge = make_judge(FakeOllama(), mode="few_shot")
    with pytest.raises(sim.ConfigurationError, match="few_shot"):
        judge.fit(train=None)


def test_gradient_labels_for_examples():
    judge = make_judge(FakeOllama(), output="gradient", response="rating", scale=(1, 5))
    judge.runtime.label_info = LabelInfo("gradient", 0.0, 4.0, 10)
    assert judge._example_answer(4.0, "rating") == '{"rating": 5}'
    assert judge._example_answer(2.0, "rating") == '{"rating": 3}'
    assert judge._example_answer(1.0, "binary") == '{"answer": "no"}'


def test_server_not_running():
    judge = make_judge(FakeOllama(connection_error=True))
    with pytest.raises(BackendUnavailableError, match="ollama.com/download"):
        judge.score_pairs(["a"], ["b"])


def test_model_details_before_scoring_and_without_server():
    fake = FakeOllama()
    record = make_judge(fake).model_details()["llm"]  # asks the server, as nothing was scored yet
    assert record["version"] == "sha256:0123456789abcdef" and record["parameters"] == 4_299_915_632
    assert not fake.calls and not fake.pulled
    for client in (FakeOllama(connection_error=True), FakeOllama(models=())):
        assert make_judge(client).model_details() == {
            "llm": {"name": "gemma3:4b", "source": "ollama", "version": None}}


def test_model_not_pulled_and_auto_pull():
    judge = make_judge(FakeOllama(models=()), model="qwen3:4b")
    with pytest.raises(ModelNotAvailableError, match="ollama pull qwen3:4b"):
        judge.score_pairs(["a"], ["b"])
    fake = FakeOllama(models=())
    make_judge(fake, model="qwen3:4b", auto_pull=True).score_pairs(["a"], ["b"])
    assert fake.pulled == ["qwen3:4b"]


def test_untagged_model_name_matches_latest():
    fake = FakeOllama(models=("mistral:latest",))
    make_judge(fake, model="mistral").score_pairs(["a"], ["b"])
    assert fake.calls


def test_remote_host_needs_opt_in():
    judge = make_judge(FakeOllama(), host="http://gpu-server.example.org:11434")
    with pytest.raises(RemoteNotAllowedError):
        judge.score_pairs(["a"], ["b"])


def test_context_window_sized_and_truncation_warned(demo, truth):
    fake = FakeOllama(truth)
    make_judge(fake).score_pairs([demo.text[0]], [demo["info"][0]])
    assert fake.calls[0]["options"]["num_ctx"] == 2048
    small = FakeOllama(truth, context_length=1024)
    long_text = " ".join([demo.text[0]] * 40)
    with pytest.warns(SimchologyWarning, match="context length"):
        make_judge(small).score_pairs([long_text], [demo["info"][0]])
    assert small.calls[0]["options"]["num_ctx"] == 1024


def test_old_client_without_logprobs_argument(demo, truth):
    fake = FakeOllama(truth, accept_logprobs=False)
    judge = make_judge(fake)
    s = judge.score_pairs([demo.text[0]], [demo["info"][0]])
    assert s[0] in (0.0, 1.0)  # plain yes/no answer
    assert judge.backend.supports_logprobs is False


def test_hardware_check_called_once(monkeypatch, demo, truth):
    import simchology.hardware.checks as checks

    calls = []
    monkeypatch.setattr(checks, "check_ollama_model", lambda *a, **k: calls.append((a, k)))
    judge = OllamaJudge(client=FakeOllama(truth))
    judge.bind(None, Runtime(output="binary", language="en", check_hardware="warn", verbose=False))
    judge.score_pairs([demo.text[0]], [demo["info"][0]])
    judge.score_pairs([demo.text[1]], [demo["info"][1]])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("gemma3:4b", 3_300_000_000) and kwargs["n_layers"] == 34


def test_german_prompts_follow_runtime_language(demo, truth):
    fake = FakeOllama()
    judge = OllamaJudge(client=fake)
    judge.bind(None, Runtime(output="binary", language="de", check_hardware="off", verbose=False))
    judge.score_pairs(["Giraffen schlafen kaum."], ["Giraffen schlafen zwei Stunden."])
    assert "REFERENZ" in fake.calls[0]["messages"][-1]["content"]
    judge_en = OllamaJudge(client=FakeOllama(), language="en")
    judge_en.bind(None, Runtime(output="binary", language="de", check_hardware="off", verbose=False))
    judge_en.score_pairs(["x"], ["y"])
    assert "REFERENCE" in judge_en.backend.client.calls[0]["messages"][-1]["content"]


def test_invalid_params():
    with pytest.raises(sim.ConfigurationError):
        OllamaJudge(mode="many_shot")
    with pytest.raises(sim.ConfigurationError):
        OllamaJudge(task="paraphrase")
    with pytest.raises(sim.ConfigurationError):
        OllamaJudge(scale=(3, 3))


def test_save_and_load_keeps_working(tmp_path, demo, truth):
    fake = FakeOllama(truth)
    out, a = sim.assess(demo, method="llm", method_params={"client": fake}, output="binary",
                        coded=demo, label_col="label", check_hardware="off", verbose=False,
                        return_assessor=True)
    a.save(tmp_path / "judge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = sim.load(tmp_path / "judge")
    loaded.method_.params["client"] = FakeOllama(truth)
    again = loaded.predict(demo.head(10))
    assert (again["predicted_similarity"].astype(int) == demo.label.head(10)).all()
    assert isinstance(again, pd.DataFrame)
