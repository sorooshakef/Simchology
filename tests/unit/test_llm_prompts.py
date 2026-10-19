import json
import math
from types import SimpleNamespace

import pytest

from simchology.exceptions import ConfigurationError
from simchology.llm.cache import ResponseCache
from simchology.llm.prompts import (
    PromptBuilder,
    default_levels,
    systemone_question,
    texts_for,
    validate_scale,
)
from simchology.llm.schema import (
    answer_json,
    binary_schema,
    expected_rating_from_logprobs,
    parse_binary,
    parse_rating,
    prob_yes_from_logprobs,
    rating_schema,
    rating_to_unit,
)


# --------------------------------------------------------------------------- prompts
def test_language_fallback_and_german():
    assert texts_for("fr") is texts_for("en")
    assert texts_for("de-AT") is texts_for("de")
    assert "REFERENZ" in PromptBuilder("de").user_message("a", "b", "binary")


def test_binary_and_rating_questions():
    b = PromptBuilder("en", task="containment", scale=(0, 4))
    assert '"answer": "yes"' in b.question("binary")
    q = b.question("rating")
    assert "0 (not at all)" in q and "4 (completely)" in q and '{"rating"' in q
    assert "similar in meaning" in PromptBuilder("en", task="similarity").question("binary")


def test_user_instructions_replace_task_text():
    b = PromptBuilder("en", instructions="Code 'mentions price' only if a number is given.")
    msg = b.user_message("resp", "ref", "binary")
    assert msg.startswith("Code 'mentions price'")
    assert "contains the information" not in msg
    assert '"answer"' in msg  # the answer format stays fixed


def test_messages_with_examples_and_reminder():
    b = PromptBuilder("en")
    msgs = b.messages("query resp", "query ref", "binary",
                      [("ex resp", "ex ref", answer_json("binary", True))], reminder=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "ex resp" in msgs[1]["content"]
    assert json.loads(msgs[2]["content"]) == {"answer": "yes"}
    assert "query resp" in msgs[3]["content"] and msgs[3]["content"].endswith("JSON object.")


def test_validate_scale():
    assert validate_scale((1, 7)) == (1, 7)
    with pytest.raises(ConfigurationError):
        validate_scale((5, 1))
    with pytest.raises(ConfigurationError):
        validate_scale("x")


# ---------------------------------------------------------------------------- schema
def test_schemas_are_strict_objects():
    assert binary_schema()["properties"]["answer"]["enum"] == ["yes", "no"]
    r = rating_schema(1, 5)["properties"]["rating"]
    assert (r["minimum"], r["maximum"]) == (1, 5)
    assert binary_schema()["additionalProperties"] is False


@pytest.mark.parametrize("text,expected", [
    ('{"answer": "yes"}', 1.0), ('{"answer":"no"}', 0.0), ('```json\n{"answer": "Yes"}\n```', 1.0),
    ("Ja.", 1.0), ("nein", 0.0), ('{"answer": true}', 1.0), ("The answer is no.", 0.0),
    ("yes or no?", None), ("", None), ("maybe", None),
])
def test_parse_binary(text, expected):
    assert parse_binary(text) == expected


@pytest.mark.parametrize("text,expected", [
    ('{"rating": 4}', 4.0), ('{"rating": 9}', None), ("Rating: 3", 3.0), ("2 or 3", None),
    ('{"score": 1}', 1.0), ("none", None),
])
def test_parse_rating(text, expected):
    assert parse_rating(text, 1, 5) == expected


def test_rating_to_unit():
    assert rating_to_unit(1, 1, 5) == 0.0 and rating_to_unit(5, 1, 5) == 1.0
    assert rating_to_unit(3, 1, 5) == 0.5


def _lp(token, p, tops=()):
    return {"token": token, "logprob": math.log(p),
            "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in tops]}


def test_prob_yes_from_dict_logprobs():
    lps = [_lp('{"', 1.0), _lp("answer", 1.0), _lp('":"', 1.0),
           _lp("yes", 0.6, [("yes", 0.6), ("no", 0.2), ("Yes", 0.1), ("maybe", 0.1)])]
    assert prob_yes_from_logprobs(lps) == pytest.approx(0.7 / 0.9)


def test_prob_yes_from_objects_and_openai_container():
    tops = [SimpleNamespace(token=" no", logprob=math.log(0.75)),
            SimpleNamespace(token=" yes", logprob=math.log(0.25))]
    pos = SimpleNamespace(token=" no", logprob=math.log(0.75), top_logprobs=tops)
    assert prob_yes_from_logprobs([pos]) == pytest.approx(0.25)
    assert prob_yes_from_logprobs(SimpleNamespace(content=[pos])) == pytest.approx(0.25)


def test_prob_yes_none_without_answer_token():
    assert prob_yes_from_logprobs([_lp("{", 1.0), _lp("}", 1.0)]) is None
    assert prob_yes_from_logprobs(None) is None


def test_expected_rating():
    lps = [_lp('{"rating":', 1.0), _lp("4", 0.5, [("4", 0.5), ("5", 0.25), ("3", 0.25), ("x", 0.1)])]
    assert expected_rating_from_logprobs(lps, 1, 5) == pytest.approx(4.0)
    assert expected_rating_from_logprobs([_lp("9", 1.0)], 1, 5) is None


# ----------------------------------------------------------------------------- cache
def test_cache_memory_and_disk(tmp_path):
    mem = ResponseCache()
    k = ResponseCache.key(model="m", prompt="p")
    assert mem.get(k) is None
    mem.put(k, {"content": "x"})
    assert mem.get(k) == {"content": "x"} and k in mem and len(mem) == 1

    disk = ResponseCache(tmp_path, name="t")
    disk.put(k, {"content": "ü"})
    with open(disk.path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "broken"')  # partially written last line
    again = ResponseCache(tmp_path, name="t")
    assert again.get(k) == {"content": "ü"}
    assert ResponseCache.key(model="m", prompt="p") == k
    assert ResponseCache.key(model="m", prompt="q") != k


# ------------------------------------------------------------------------ System One
def test_systemone_noul_question():
    q = systemone_question("Giraffes sleep two hours.", question="noul")
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert "Giraffes sleep two hours." in q["instructions"]


def test_systemone_score_question_levels():
    q = systemone_question("x", question="score", language="de")
    assert q["type"] == "score" and q["criteria"] == default_levels("de")
    assert len(q["criteria"]) == 3
    custom = systemone_question("x", question="score", levels=["no", "yes"])
    assert custom["criteria"] == ["no", "yes"]
    with pytest.raises(ConfigurationError):
        systemone_question("x", question="score", levels=["only one"])


def test_systemone_custom_instructions():
    q = systemone_question("REF", question="noul", instructions="Is {reference} mentioned?")
    assert q["instructions"] == "Is REF mentioned?"
    q = systemone_question("REF", question="noul", instructions="Is it mentioned?")
    assert q["instructions"].startswith("Is it mentioned?") and '"REF"' in q["instructions"]
