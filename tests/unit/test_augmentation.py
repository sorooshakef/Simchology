"""The perturbation self-tests of the original fine-tuning notebook, plus integrity checks."""

import pandas as pd

import simchology as sim
from simchology.data.augmentation import (
    ORIG_ID_COL,
    VARIANT_COL,
    assign_variants,
    check_augmentation_integrity,
    perturb_cut_midword,
    perturb_no_period,
)


def test_notebook_perturbation_examples():
    assert perturb_no_period("Ein Satz. Noch einer.") == "Ein Satz. Noch einer"
    assert perturb_no_period("Bereits ohne Punkt") == "Bereits ohne Punkt"
    assert perturb_no_period("Mehrfach!!!") == "Mehrfach"
    assert perturb_cut_midword("Er war hier.") == "Er war hi"
    assert perturb_cut_midword("Das Kaiserreichs.") == "Das Kaiser"
    assert perturb_cut_midword("Frage endet hier?") == "Frage endet hi"
    assert perturb_cut_midword("Ein Satz, Wort.") == "Ein Satz, Wo"
    assert perturb_cut_midword("Tiere.") == "Tie"
    assert perturb_cut_midword("Ende a.") == "Ende"


def test_assign_variants_is_balanced_and_forces_noop_rows():
    texts = ["Satz eins."] * 7 + ["ohne Punkt"] * 3
    kinds = assign_variants(texts, seed=0)
    assert (kinds == "cut_midword").sum() == 5
    assert all(k == "cut_midword" for k in kinds[7:])


def test_augment_keeps_twins_together_and_labels_untouched(demo_de):
    df = demo_de.head(50).set_index("ID")
    aug = sim.augment(df, text_col="text", seed=42)
    assert len(aug) == 100
    assert list(aug[VARIANT_COL].iloc[:2]) == ["original", aug[VARIANT_COL].iloc[1]]
    assert aug[VARIANT_COL].iloc[1] != "original"
    noop = check_augmentation_integrity(aug, text_col="text", ref_col="info", label_col="label")
    assert noop == 0
    assert (aug.groupby(ORIG_ID_COL)["info"].nunique() == 1).all()


def test_augment_handles_duplicate_index():
    df = pd.DataFrame({"text": ["A b.", "C d."], "info": ["x", "y"], "label": [1, 0]}, index=[0, 0])
    aug = sim.augment(df)
    assert len(aug) == 4
