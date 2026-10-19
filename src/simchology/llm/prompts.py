"""Prompt templates for LLM judges and System One questions (German and English).

Two tasks are supported:

* ``"containment"`` (default) - does the response contain the information of the
  reference? This is the typical coding question of recall, comprehension and
  information-transmission studies.
* ``"similarity"`` - how similar in meaning are the two texts?

Users can replace the task description with their own construct definition or coding
manual via ``instructions=...``; the answer format stays fixed so it can be parsed.
Languages other than German and English fall back to English prompts (the texts
themselves can be in any language the model understands).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..exceptions import ConfigurationError

TASKS = ("containment", "similarity")

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "system": (
            "You are a careful and consistent coder of participant responses in a psychology "
            "study. Follow the coding instructions exactly and answer only with the requested "
            "JSON object."
        ),
        "task_containment": (
            "Decide whether the RESPONSE contains the information given in the REFERENCE. The "
            "information counts as contained if the response states it, even in other words or "
            "as a paraphrase. It does not count if it is missing, only mentioned vaguely without "
            "the actual claim, or contradicted."
        ),
        "task_similarity": (
            "Judge how similar the RESPONSE and the REFERENCE are in meaning. Focus on what is "
            "said, not on how it is worded: paraphrases with the same meaning are similar, while "
            "texts that share words but differ in meaning (for example through negation) are not."
        ),
        "binary_containment": (
            'Does the RESPONSE contain the information of the REFERENCE? Answer with '
            '{"answer": "yes"} or {"answer": "no"}.'
        ),
        "binary_similarity": (
            'Are the RESPONSE and the REFERENCE similar in meaning? Answer with '
            '{"answer": "yes"} or {"answer": "no"}.'
        ),
        "rating_containment": (
            "How fully does the RESPONSE contain the information of the REFERENCE, on a scale "
            "from {lo} (not at all) to {hi} (completely)? Answer with "
            '{{"rating": <a whole number from {lo} to {hi}>}}.'
        ),
        "rating_similarity": (
            "How similar in meaning are the RESPONSE and the REFERENCE, on a scale from {lo} "
            "(completely different) to {hi} (identical in meaning)? Answer with "
            '{{"rating": <a whole number from {lo} to {hi}>}}.'
        ),
        "pair": 'REFERENCE:\n"""\n{reference}\n"""\n\nRESPONSE:\n"""\n{response}\n"""',
        "reminder": "Answer only with the JSON object.",
        # System One (typed decision) questions; the state is the response text.
        "so_noul_containment": 'Does the text contain the following information (possibly in other words)? Information: "{reference}"',
        "so_noul_similarity": 'Does the text have the same meaning as the following reference text? Reference: "{reference}"',
        "so_score_containment": 'How fully does the text contain the following information? Information: "{reference}"',
        "so_score_similarity": 'How similar in meaning is the text to the following reference text? Reference: "{reference}"',
        "so_true_containment": "The text states this information, possibly paraphrased.",
        "so_false_containment": "The information is missing, only vaguely mentioned, or contradicted.",
        "so_true_similarity": "The text has the same or a very similar meaning.",
        "so_false_similarity": "The text has a different meaning.",
        "so_levels_containment": "The information is not contained.|The information is partially or vaguely contained.|The information is fully contained.",
        "so_levels_similarity": "Different in meaning.|Partly similar in meaning.|Same meaning.",
        "so_reference": 'Reference: "{reference}"',
    },
    "de": {
        "system": (
            "Du kodierst sorgfältig und einheitlich Antworten von Versuchspersonen in einer "
            "psychologischen Studie. Halte dich genau an die Kodieranweisung und antworte nur mit "
            "dem verlangten JSON-Objekt."
        ),
        "task_containment": (
            "Entscheide, ob die ANTWORT die Information aus der REFERENZ enthält. Die Information "
            "gilt als enthalten, wenn die Antwort sie wiedergibt, auch mit anderen Worten oder als "
            "Umschreibung. Sie gilt nicht als enthalten, wenn sie fehlt, nur vage ohne die "
            "eigentliche Aussage erwähnt wird oder ihr widersprochen wird."
        ),
        "task_similarity": (
            "Beurteile, wie ähnlich sich ANTWORT und REFERENZ in ihrer Bedeutung sind. Achte auf "
            "den Inhalt, nicht auf die Formulierung: Umschreibungen mit gleicher Bedeutung sind "
            "ähnlich, Texte mit gleichen Wörtern, aber anderer Bedeutung (z. B. durch Verneinung) "
            "nicht."
        ),
        "binary_containment": (
            'Enthält die ANTWORT die Information der REFERENZ? Antworte mit {"answer": "yes"} '
            '(ja) oder {"answer": "no"} (nein).'
        ),
        "binary_similarity": (
            'Sind ANTWORT und REFERENZ in ihrer Bedeutung ähnlich? Antworte mit '
            '{"answer": "yes"} (ja) oder {"answer": "no"} (nein).'
        ),
        "rating_containment": (
            "Wie vollständig enthält die ANTWORT die Information der REFERENZ, auf einer Skala "
            "von {lo} (gar nicht) bis {hi} (vollständig)? Antworte mit "
            '{{"rating": <ganze Zahl von {lo} bis {hi}>}}.'
        ),
        "rating_similarity": (
            "Wie ähnlich sind sich ANTWORT und REFERENZ in ihrer Bedeutung, auf einer Skala von "
            "{lo} (völlig verschieden) bis {hi} (bedeutungsgleich)? Antworte mit "
            '{{"rating": <ganze Zahl von {lo} bis {hi}>}}.'
        ),
        "pair": 'REFERENZ:\n"""\n{reference}\n"""\n\nANTWORT:\n"""\n{response}\n"""',
        "reminder": "Antworte nur mit dem JSON-Objekt.",
        "so_noul_containment": 'Enthält der Text die folgende Information (eventuell mit anderen Worten)? Information: "{reference}"',
        "so_noul_similarity": 'Hat der Text dieselbe Bedeutung wie der folgende Referenztext? Referenz: "{reference}"',
        "so_score_containment": 'Wie vollständig enthält der Text die folgende Information? Information: "{reference}"',
        "so_score_similarity": 'Wie ähnlich ist der Text in seiner Bedeutung dem folgenden Referenztext? Referenz: "{reference}"',
        "so_true_containment": "Der Text gibt diese Information wieder, eventuell umschrieben.",
        "so_false_containment": "Die Information fehlt, wird nur vage erwähnt oder ihr wird widersprochen.",
        "so_true_similarity": "Der Text hat dieselbe oder eine sehr ähnliche Bedeutung.",
        "so_false_similarity": "Der Text hat eine andere Bedeutung.",
        "so_levels_containment": "Die Information ist nicht enthalten.|Die Information ist teilweise oder vage enthalten.|Die Information ist vollständig enthalten.",
        "so_levels_similarity": "Andere Bedeutung.|Teilweise ähnliche Bedeutung.|Dieselbe Bedeutung.",
        "so_reference": 'Referenz: "{reference}"',
    },
}


def texts_for(language: str | None) -> dict[str, str]:
    """Templates for ``language``, falling back to English."""
    return TEXTS.get((language or "en").split("-")[0].lower(), TEXTS["en"])


def validate_task(task: str) -> str:
    if task not in TASKS:
        raise ConfigurationError(f"task must be one of {TASKS}; got {task!r}")
    return task


def validate_scale(scale: Sequence[int]) -> tuple[int, int]:
    try:
        lo, hi = (int(scale[0]), int(scale[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ConfigurationError("scale must be two whole numbers, e.g. (1, 5)") from exc
    if hi <= lo:
        raise ConfigurationError("scale must be (low, high) with high > low, e.g. (1, 5)")
    return lo, hi


@dataclass(frozen=True)
class PromptBuilder:
    """Builds chat messages for one (response, reference) judgment.

    ``kind`` is ``"binary"`` (yes/no) or ``"rating"`` (whole number on ``scale``).
    Few-shot examples are added as earlier user/assistant turns, so the same prefix is
    shared by all queries (which lets local servers reuse their prompt cache).
    """

    language: str = "en"
    task: str = "containment"
    instructions: str | None = None
    scale: tuple[int, int] = (1, 5)

    @property
    def t(self) -> dict[str, str]:
        return texts_for(self.language)

    def task_text(self) -> str:
        if self.instructions:
            return self.instructions.strip()
        return self.t[f"task_{self.task}"]

    def question(self, kind: str) -> str:
        lo, hi = self.scale
        if kind == "rating":
            return self.t[f"rating_{self.task}"].format(lo=lo, hi=hi)
        return self.t[f"binary_{self.task}"]

    def user_message(self, response: str, reference: str, kind: str) -> str:
        pair = self.t["pair"].format(response=response, reference=reference)
        return f"{self.task_text()}\n\n{pair}\n\n{self.question(kind)}"

    def messages(
        self,
        response: str,
        reference: str,
        kind: str,
        examples: Sequence[tuple[str, str, str]] = (),
        *,
        reminder: bool = False,
    ) -> list[dict[str, str]]:
        """Chat messages; ``examples`` are (response, reference, answer JSON) triples."""
        msgs = [{"role": "system", "content": self.t["system"]}]
        for ex_resp, ex_ref, ex_answer in examples:
            msgs.append({"role": "user", "content": self.user_message(ex_resp, ex_ref, kind)})
            msgs.append({"role": "assistant", "content": ex_answer})
        content = self.user_message(response, reference, kind)
        if reminder:
            content += "\n\n" + self.t["reminder"]
        msgs.append({"role": "user", "content": content})
        return msgs


def systemone_question(
    reference: str,
    *,
    question: str,
    language: str = "en",
    task: str = "containment",
    instructions: str | None = None,
    criteria: dict | None = None,
    levels: Sequence[str] | None = None,
) -> dict:
    """One System One question about ``reference`` (``question`` is "noul" or "score")."""
    t = texts_for(language)
    if instructions:
        text = instructions.strip()
        if "{reference}" in text:
            text = text.replace("{reference}", reference)
        else:
            text = f"{text}\n{t['so_reference'].format(reference=reference)}"
    else:
        text = t[f"so_{question}_{task}"].format(reference=reference)
    if question == "noul":
        crit = dict(criteria) if criteria else {
            "true": t[f"so_true_{task}"], "false": t[f"so_false_{task}"]}
        return {"type": "noul", "instructions": text, "criteria": crit}
    lv = list(levels) if levels else t[f"so_levels_{task}"].split("|")
    if len(lv) < 2:
        raise ConfigurationError("levels must list at least two level descriptions, lowest first.")
    return {"type": "score", "instructions": text, "criteria": lv}


def default_levels(language: str = "en", task: str = "containment") -> list[str]:
    return texts_for(language)[f"so_levels_{task}"].split("|")
