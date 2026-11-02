"""SystemOneJudge against a mocked System One server (httpx.MockTransport)."""

import json
import re

import numpy as np
import pytest

import simchology as sim
from simchology.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    ModelNotAvailableError,
    RemoteNotAllowedError,
    SimchologyPrivacyWarning,
)
from simchology.llm import netpolicy
from simchology.methods.base import Runtime
from simchology.methods.systemone import SystemOneJudge

httpx = pytest.importorskip("httpx")
REF_RE = re.compile(r'"([^"]*)"')


class FakeServer:
    """A System One server that answers from a lookup of true labels."""

    def __init__(self, truth=None, *, max_state=None, status=None, connect_error=False):
        self.truth = truth or {}
        self.max_state = max_state
        self.status = status
        self.connect_error = connect_error
        self.posts = []
        self.gets = []

    def __call__(self, request):
        if self.connect_error:
            raise httpx.ConnectError("connection refused", request=request)
        if request.method == "GET":
            self.gets.append(request)
            return httpx.Response(200, json={"models": [{
                "name": "kev-latest", "description": "Kev pointer head", "release_date": "2026-05-01",
                "run": "jaredpalmer/kev-0.8b", "base": "Qwen/Qwen3.5-0.8B"}]})
        body = json.loads(request.content)
        self.posts.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        if self.status:
            return httpx.Response(self.status, text="model 'kev-x' not found")
        if self.max_state and len(body["state"]) > self.max_state:
            return httpx.Response(422, json={"detail": "state too long"})
        answers = {}
        for name, q in body["questions"].items():
            reference = REF_RE.findall(q["instructions"])[-1]
            positive = self._positive(body["state"], reference)
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9 if positive else 0.1}
            else:
                top = len(q["criteria"]) - 1
                answers[name] = {"type": "score", "score": float(top if positive else 0),
                                 "legend": {}, "probabilities": {}, "confidence": 0.8}
        return httpx.Response(200, json={"model": body["model"], "answers": answers,
                                         "usage": {"input_tokens": 1, "output_tokens": 0}})

    def _positive(self, state, reference):
        if (state, reference) in self.truth:
            return bool(self.truth[(state, reference)])
        # sentence-level fallback for segment retries
        return any(bool(v) and reference == r and state in t for (t, r), v in self.truth.items())


def client_for(server):
    return httpx.Client(transport=httpx.MockTransport(server))


def make_judge(server, output="binary", allow_remote=False, **params):
    judge = SystemOneJudge(client=client_for(server), **params)
    judge.bind(None, Runtime(output=output, language="en", allow_remote=allow_remote, verbose=False))
    return judge


@pytest.fixture()
def demo():
    return sim.load_demo("en", n_per_topic=6, seed=2)


@pytest.fixture()
def truth(demo):
    return {(t, i): y for t, i, y in zip(demo.text, demo["info"], demo.label)}


def test_grouped_requests_payload_and_values(demo, truth):
    server = FakeServer(truth)
    judge = make_judge(server)
    s = judge.score_pairs(list(demo.text), list(demo["info"]))
    assert len(server.posts) == demo.text.nunique()  # one request per response
    assert len(server.gets) == 1  # health check
    body = server.posts[0]["body"]
    assert isinstance(body["state"], str) and body["model"] == "kev-latest"
    q = body["questions"]["q0"]
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert len(body["questions"]) == 5
    assert server.posts[0]["url"] == "http://localhost:8009/v1/systemone"
    assert np.allclose(s, np.where(demo.label == 1, 0.9, 0.1))
    assert judge.native_output() == "probability"


def test_ungrouped_and_separate_endpoint(demo, truth):
    server = FakeServer(truth)
    judge = make_judge(server, group_by_response=False, endpoint="/v1/systemone/separate")
    judge.score_pairs(list(demo.text[:7]), list(demo["info"][:7]))
    assert len(server.posts) == 7
    assert all(len(p["body"]["questions"]) == 1 for p in server.posts)
    assert server.posts[0]["url"].endswith("/v1/systemone/separate")


def test_max_questions_splits_requests(demo, truth):
    server = FakeServer(truth)
    make_judge(server, max_questions=2).score_pairs(list(demo.text[:5]), list(demo["info"][:5]))
    assert [len(p["body"]["questions"]) for p in server.posts] == [2, 2, 1]


def test_gradient_uses_score_levels(demo, truth):
    server = FakeServer(truth)
    judge = make_judge(server, output="gradient")
    s = judge.score_pairs(list(demo.text[:5]), list(demo["info"][:5]))
    q = server.posts[0]["body"]["questions"]["q0"]
    assert q["type"] == "score" and len(q["criteria"]) == 3
    assert np.allclose(s, demo.label[:5].to_numpy(dtype=float))
    assert judge.native_output() == "score"

    custom = make_judge(FakeServer(truth), output="gradient", levels=["no", "a bit", "mostly", "fully"])
    s = custom.score_pairs(list(demo.text[:5]), list(demo["info"][:5]))
    assert np.allclose(s, demo.label[:5].to_numpy(dtype=float))


def test_auth_header_from_param_and_env(monkeypatch, demo, truth):
    server = FakeServer(truth)
    make_judge(server, api_key="secret").score_pairs([demo.text[0]], [demo["info"][0]])
    assert server.posts[0]["headers"]["authorization"] == "Bearer secret"
    monkeypatch.setenv("KEV_API_KEY", "from-env")
    server = FakeServer(truth)
    make_judge(server).score_pairs([demo.text[0]], [demo["info"][0]])
    assert server.posts[0]["headers"]["authorization"] == "Bearer from-env"
    monkeypatch.delenv("KEV_API_KEY")
    server = FakeServer(truth)
    make_judge(server).score_pairs([demo.text[0]], [demo["info"][0]])
    assert "authorization" not in server.posts[0]["headers"]


def test_base_url_env_override(monkeypatch, demo, truth):
    monkeypatch.setenv("SIMCHOLOGY_SYSTEMONE_URL", "http://127.0.0.1:9999/")
    server = FakeServer(truth)
    make_judge(server).score_pairs([demo.text[0]], [demo["info"][0]])
    assert server.posts[0]["url"] == "http://127.0.0.1:9999/v1/systemone"
    server = FakeServer(truth)
    make_judge(server, base_url="http://localhost:11434").score_pairs([demo.text[0]], [demo["info"][0]])
    assert server.posts[0]["url"] == "http://localhost:11434/v1/systemone"


def test_state_too_long_falls_back_to_sentences():
    text = "Giraffes sleep two hours a day. The moai are stone statues on Easter Island."
    truth = {("Giraffes sleep two hours a day.", "Giraffes sleep about two hours a day."): 1}
    server = FakeServer(truth, max_state=40)
    s = make_judge(server).score_pairs([text], ["Giraffes sleep about two hours a day."])
    assert s[0] == pytest.approx(0.9)
    assert [len(p["body"]["state"]) > 40 for p in server.posts] == [True, False, True]


def test_state_too_long_everywhere_is_an_error():
    server = FakeServer(max_state=5)
    with pytest.raises(ConfigurationError, match="sentence by sentence"):
        make_judge(server).score_pairs(["First sentence here. Second sentence here."], ["ref"])


def test_state_too_long_single_sentence_is_an_error():
    server = FakeServer(max_state=5)
    with pytest.raises(ConfigurationError, match="422"):
        make_judge(server).score_pairs(["One long sentence without a break"], ["ref"])


def test_model_details_before_scoring_and_without_server():
    server = FakeServer()
    assert make_judge(server).model_details()["systemone"]["version"] == "2026-05-01"
    assert len(server.gets) == 1 and not server.posts
    assert make_judge(FakeServer(connect_error=True)).model_details() == {
        "systemone": {"name": "kev-latest", "source": "systemone", "url": "http://localhost:8009",
                      "version": None}}


def test_server_not_running_message():
    judge = make_judge(FakeServer(connect_error=True))
    with pytest.raises(BackendUnavailableError) as info:
        judge.score_pairs(["a"], ["b"])
    msg = str(info.value)
    assert "kev.serve" in msg and "github.com/jaredpalmer/kev" in msg and "allow_remote=True" in msg
    assert "jaredpalmer/kev-" in msg


def test_http_errors():
    with pytest.raises(ModelNotAvailableError):
        make_judge(FakeServer(status=404), model="kev-x").score_pairs(["a"], ["b"])
    with pytest.raises(ConfigurationError, match="API key"):
        make_judge(FakeServer(status=401)).score_pairs(["a"], ["b"])


def test_remote_server_needs_opt_in(monkeypatch, demo, truth):
    monkeypatch.setattr(netpolicy, "_WARNED_HOSTS", set())
    with pytest.raises(RemoteNotAllowedError):
        make_judge(FakeServer(truth), base_url="https://api.typesafe.example").score_pairs(["a"], ["b"])
    judge = make_judge(FakeServer(truth), base_url="https://api.typesafe.example", allow_remote=True,
                       api_key="k")
    with pytest.warns(SimchologyPrivacyWarning):
        judge.score_pairs([demo.text[0]], [demo["info"][0]])


def test_cache_resume(tmp_path, demo, truth):
    first = FakeServer(truth)
    make_judge(first, cache_dir=tmp_path).score_pairs(list(demo.text[:10]), list(demo["info"][:10]))
    assert first.posts
    second = FakeServer(truth)
    s = make_judge(second, cache_dir=tmp_path).score_pairs(list(demo.text[:10]), list(demo["info"][:10]))
    assert second.posts == []
    assert not np.isnan(s).any()


def test_custom_instructions_and_german(demo, truth):
    server = FakeServer()
    judge = SystemOneJudge(client=client_for(server), instructions='Nennt der Text "{reference}"?')
    judge.bind(None, Runtime(output="binary", language="de", verbose=False))
    judge.score_pairs(["Giraffen schlafen."], ["Giraffen schlafen zwei Stunden."])
    q = server.posts[0]["body"]["questions"]["q0"]
    assert q["instructions"] == 'Nennt der Text "Giraffen schlafen zwei Stunden."?'
    assert q["criteria"]["true"].startswith("Der Text")


def test_end_to_end_assess(demo, truth):
    server = FakeServer(truth)
    out, a = sim.assess(demo, method="systemone", method_params={"client": client_for(server)},
                        output="binary", coded=demo, label_col="label", group_col="topic",
                        verbose=False, return_assessor=True)
    assert (out["predicted_similarity"].astype(int) == demo.label).all()
    assert a.report_.metrics["f1"] == pytest.approx(1.0)
    assert a.provenance_["models"]["systemone"] == {
        "name": "kev-latest", "source": "systemone", "url": "http://localhost:8009",
        "version": "2026-05-01", "run": "jaredpalmer/kev-0.8b", "base": "Qwen/Qwen3.5-0.8B"}
    assert "System One" in a.describe() or "kev-latest" in a.describe()


def test_aliases_resolve():
    for alias in ("jev", "kev", "system_one", "zero_shot_classifier"):
        assert sim.methods.resolve_name(alias) == "systemone"


def test_invalid_params():
    with pytest.raises(ConfigurationError):
        SystemOneJudge(question="choice")
    with pytest.raises(ConfigurationError):
        SystemOneJudge(levels=["only"])
