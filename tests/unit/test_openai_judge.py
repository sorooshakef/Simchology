"""OpenAICompatibleJudge with fake clients / a mocked HTTP transport (no network)."""

import json
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires

import simchology as sim
from simchology.exceptions import (
    ConfigurationError,
    RemoteNotAllowedError,
    SimchologyPrivacyWarning,
)
from simchology.llm import netpolicy
from simchology.methods.base import Runtime
from simchology.methods.prompt import OpenAICompatibleJudge

PAIR_RE = re.compile(r'"""\n(.*?)\n"""', re.S)


def _completion(content, positive, with_logprobs=True):
    word, other = ("yes", "no") if positive else ("no", "yes")
    choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
    if with_logprobs:
        choice["logprobs"] = {"content": [
            {"token": '{"answer":"', "logprob": 0.0, "bytes": None, "top_logprobs": []},
            {"token": word, "logprob": math.log(0.8), "bytes": None, "top_logprobs": [
                {"token": word, "logprob": math.log(0.8), "bytes": None},
                {"token": other, "logprob": math.log(0.2), "bytes": None}]},
        ]}
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}


def _http_lib():
    """openai>=3 uses httpx2 for its http_client; older versions use httpx."""
    openai = pytest.importorskip("openai")  # the mock transport goes through the real client

    major = int(openai.__version__.split(".")[0])
    if major >= 3:
        return pytest.importorskip("httpx2")
    return pytest.importorskip("httpx")


@pytest.fixture()
def mock_server():
    http = _http_lib()
    state = {"requests": [], "reject": []}

    def handler(request):
        body = json.loads(request.content)
        state["requests"].append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        fmt = body.get("response_format", {}).get("type")
        if fmt and fmt in state["reject"]:
            return http.Response(400, json={"error": {"message": f"response_format {fmt} is not supported",
                                                      "type": "invalid_request_error"}})
        reference, response = PAIR_RE.findall(body["messages"][-1]["content"])[-2:]
        positive = "sleep" in response.lower() and "sleep" in reference.lower()
        content = json.dumps({"answer": "yes" if positive else "no"})
        return http.Response(200, json=_completion(content, positive, with_logprobs=body.get("logprobs", False)))

    import openai

    def client(base_url="http://localhost:1234/v1"):
        return openai.OpenAI(api_key="test-key", base_url=base_url, max_retries=0,
                             http_client=http.Client(transport=http.MockTransport(handler)))

    state["client"] = client
    return state


def make_judge(client, output="binary", allow_remote=False, **params):
    params.setdefault("model", "local-model")
    judge = OpenAICompatibleJudge(client=client, **params)
    judge.bind(None, Runtime(output=output, language="en", allow_remote=allow_remote, verbose=False))
    return judge


def test_request_payload_and_probability(mock_server):
    judge = make_judge(mock_server["client"](), base_url="http://localhost:1234/v1")
    s = judge.score_pairs(["Giraffes sleep two hours.", "Moai are statues."],
                          ["Giraffes sleep about two hours.", "Giraffes sleep about two hours."])
    assert s == pytest.approx([0.8, 0.2])
    body = mock_server["requests"][0]["body"]
    assert body["model"] == "local-model"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"] == ["yes", "no"]
    assert body["temperature"] == 0.0 and body["seed"] == 42
    assert body["logprobs"] is True and body["top_logprobs"] == 5
    assert mock_server["requests"][0]["url"] == "http://localhost:1234/v1/chat/completions"


def test_falls_back_when_json_schema_unsupported(mock_server):
    mock_server["reject"] = ["json_schema"]
    judge = make_judge(mock_server["client"](), base_url="http://localhost:1234/v1")
    s = judge.score_pairs(["Giraffes sleep."], ["Giraffes sleep two hours."])
    assert s[0] == pytest.approx(0.8)
    kinds = [r["body"].get("response_format", {}).get("type") for r in mock_server["requests"]]
    assert kinds == ["json_schema", "json_object"]
    assert judge.backend.features["json_schema"] is False


def test_hosted_api_needs_opt_in_and_warns(mock_server, monkeypatch):
    monkeypatch.setattr(netpolicy, "_WARNED_HOSTS", set())
    with pytest.raises(RemoteNotAllowedError):
        make_judge(mock_server["client"](), base_url=None).score_pairs(["a"], ["b"])
    judge = make_judge(mock_server["client"](), base_url="https://api.openai.com/v1", allow_remote=True)
    with pytest.warns(SimchologyPrivacyWarning, match="api.openai.com"):
        judge.score_pairs(["a"], ["b"])


def test_model_is_required():
    judge = OpenAICompatibleJudge(base_url="http://localhost:1234/v1")
    judge.bind(None, Runtime(output="binary", verbose=False))
    with pytest.raises(ConfigurationError, match="model"):
        judge.score_pairs(["a"], ["b"])


@requires("openai")  # the API key is only checked when the real openai client is built
def test_missing_api_key_for_hosted_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = OpenAICompatibleJudge(model="gpt-x")
    judge.bind(None, Runtime(output="binary", allow_remote=True, verbose=False))
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"), pytest.warns(SimchologyPrivacyWarning):
        judge.score_pairs(["a"], ["b"])


class DuckClient:
    """Minimal object with client.chat.completions.create (no openai package needed)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        content = self.answers.pop(0)
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, logprobs=None)])


def test_rating_without_logprobs_and_end_to_end():
    df = sim.load_demo("en", n_per_topic=4, seed=3)
    client = DuckClient(['{"answer": "no"}'] + ['{"rating": 4}'] * 1000)
    out = sim.assess(df, method="openai_compatible",
                     method_params={"client": client, "model": "m", "base_url": "http://127.0.0.1:8080/v1"},
                     output="gradient", verbose=False, check_hardware="off")
    assert np.allclose(out["predicted_similarity"], 0.75)
    assert "rating" in client.kwargs[-1]["response_format"]["json_schema"]["schema"]["properties"]


def test_describe_and_model_ids():
    judge = make_judge(DuckClient([]), base_url="http://localhost:1234/v1", mode="few_shot")
    assert "few-shot" in judge.describe()
    assert judge.model_ids()["llm"] == "local-model @ http://localhost:1234/v1"
