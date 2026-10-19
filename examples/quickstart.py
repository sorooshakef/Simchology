"""Simchology quick start: score how much of each reference "info" a participant recalled.

Run with:  python examples/quickstart.py
Only the core install is needed (pip install simchology); neural methods are tried if
their extras are installed.
"""

import simchology as sim

# 1. Long-format data: one row per (participant text, reference info) pair.
#    `label` is the human coding (1 = info contained), available for all rows here.
data = sim.load_demo("en")
print(data[["participant", "topic", "text", "info", "label"]].head(), "\n")

# 2. One call: fit on the coded rows (split into training and held-out test pairs),
#    build a baseline of dissimilar pairs, set the threshold, evaluate, and predict.
result, assessor = sim.assess(
    data,
    columns=("text", "info"),  # (response, reference)
    method="tfidf",
    output="binary",
    coded=data,
    label_col="label",
    group_col="topic",  # lets the baseline re-pair texts with infos of other topics
    return_assessor=True,
)
print(result[["text", "info", "label", "predicted_similarity"]].head(), "\n")

# 3. What does an unrelated pair score? How was the threshold chosen?
print(assessor.baseline_.summary(), "\n")
print(assessor.threshold_.describe(), "\n")

# 4. Why was a pair judged similar?
row = data.iloc[0]
print(assessor.explain(row["text"], row["info"]), "\n")

# 5. Compare several methods on the same held-out pairs.
print(sim.compare_methods(data, "fast", label_col="label", group_col="topic", verbose=False)
      [["method", "status", "precision", "recall", "f1", "cohen_kappa"]], "\n")

# 6. A paragraph for the methods section, and saving the fitted assessor.
print(assessor.describe(), "\n")
assessor.save("quickstart_assessor.simchology")
reloaded = sim.load("quickstart_assessor.simchology")
print(reloaded.predict(data.head(3))[["info", "predicted_similarity"]])
