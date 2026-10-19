import subprocess
import sys

import pytest

import simchology as sim
from simchology._optional import install_hint, install_message, is_available, require
from simchology.exceptions import ConfigurationError, MissingDependencyError
from simchology.methods.registry import get_method_class, make_method, resolve_name


def test_require_missing_module_gives_install_hint():
    with pytest.raises(MissingDependencyError) as err:
        require("definitely_not_a_module_xyz", feature="the test feature")
    assert "pip install" in str(err.value)
    assert "The test feature" in str(err.value)


def test_install_hint_names_the_packages_of_the_extra():
    assert install_hint("ot") == "pip install POT"
    assert install_hint("spacy") == "pip install spacy"
    # read from pyproject.toml via the package metadata, including nested extras
    finetune = install_hint("peft").split()[2:]
    assert {"torch", "transformers", "accelerate", "peft"} <= set(finetune)
    assert install_hint("definitely_not_a_module_xyz") == "pip install definitely_not_a_module_xyz"
    assert 'pip install -e ".[wmd]"' in install_message("ot")
    assert is_available("numpy")


def test_import_does_not_load_heavy_packages():
    code = ("import sys, simchology; heavy = [m for m in ('torch', 'transformers', 'spacy', 'ot', "
            "'sentence_transformers', 'ollama', 'openai') if m in sys.modules]; print(heavy); "
            "assert not heavy, heavy; simchology.list_methods()")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_aliases_resolve():
    assert resolve_name("lv") == "levenshtein"
    assert resolve_name("DL") == "damerau_levenshtein"
    assert resolve_name("cosine") == "tfidf"
    assert resolve_name("Bag-of-Words") == "bow"


def test_unknown_method_suggests_alternatives():
    with pytest.raises(ConfigurationError, match="Did you mean"):
        resolve_name("jacard")


def test_unknown_parameter_is_reported():
    with pytest.raises(ConfigurationError, match="Unknown parameter"):
        make_method("jaccard", {"unti": "lemma"})


def test_list_methods_table():
    table = sim.list_methods()
    assert {"method", "family", "installed", "install"} <= set(table.columns)
    assert {"jaccard", "tfidf", "bm25", "lsa", "lda"} <= set(table["method"])
    assert get_method_class("tfidf").info.requires_corpus_fit
