import pytest

from simchology.exceptions import MissingDependencyError
from simchology.text.normalize import Normalizer, light_normalize
from simchology.text.preprocess import PreprocessConfig, Preprocessor
from simchology.text.segment import Segmenter, punctuation_reliability, regex_sentences
from simchology.text.tokenize import detect_language


def test_normalizer_language_rules_and_replacements():
    n = Normalizer("de", {"10m": "10 Meter"})
    assert n("Etwa 50% sind 10m hoch") == "Etwa 50 Prozent sind 10 Meter hoch"
    assert Normalizer("r_compat")("Rapa Nui (ō) 5%") == "Rapa Nui (o) 5 Prozent"
    assert light_normalize("  a \n b ") == "a b"


def test_regex_sentences_respects_ordinals_and_abbreviations():
    text = "Ab dem 7. Jahrhundert wurde Griechisch gesprochen. Das ist z.B. bekannt. Neu!"
    assert regex_sentences(text) == [
        "Ab dem 7. Jahrhundert wurde Griechisch gesprochen.", "Das ist z.B. bekannt.", "Neu!"]
    assert regex_sentences("ohne punkt am ende") == ["ohne punkt am ende"]
    assert regex_sentences("") == []


def test_punctuation_reliability():
    assert punctuation_reliability(["Ein Satz.", "Noch einer!", "ohne"]) == pytest.approx(2 / 3)


def test_segmenter_none_and_regex_and_cache():
    seg = Segmenter("regex", "de", verbose=False)
    assert seg.split(["Ein Satz. Noch einer.", ""]) == [["Ein Satz.", "Noch einer."], [""]]
    assert seg.split(["J. Smith kam."]) == [["J. Smith kam."]]  # initials do not end sentences
    assert Segmenter("none", "de").split(["Ein Satz. Noch einer."]) == [["Ein Satz. Noch einer."]]


def test_detect_language():
    assert detect_language(["Die Giraffe ist das höchste Tier, das an Land lebt und nicht schläft."]) == "de"
    assert detect_language(["The giraffe is the tallest animal that lives on land and it is tall."]) == "en"


def test_regex_preprocessing_filters_stopwords_and_punct():
    p = Preprocessor(PreprocessConfig.from_options("de", {"backend": "regex"}), verbose=False)
    toks = p.tokens(["Die Giraffen sind die höchsten Landtiere der Welt!"])[0]
    assert toks == ("giraffen", "höchsten", "landtiere", "welt")
    assert p.unit == "surface"
    with pytest.raises(MissingDependencyError):
        p.tokens(["x"], unit="lemma")


def test_preprocess_options_validated():
    with pytest.raises(Exception, match="Unknown preprocessing"):
        PreprocessConfig.from_options("de", {"stopwords": False})
    cfg = PreprocessConfig.from_options("de", {"remove_stopwords": False, "replacements": {"a": "b"}})
    assert PreprocessConfig.from_dict(cfg.to_dict()) == cfg


def test_spacy_backend_with_blank_pipeline(monkeypatch):
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    p = Preprocessor(PreprocessConfig.from_options("en", {"backend": "spacy", "unit": "surface"}),
                     verbose=False)
    monkeypatch.setattr(p, "nlp", lambda: nlp)
    toks = p.tokens(["The giraffes are tall, aren't they?"])[0]
    assert "giraffes" in toks and "the" not in toks and "," not in toks
