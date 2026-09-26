import pandas as pd
import numpy as np
import joblib

# ============================================================
# FILES
# ============================================================

FEATURE_FILE = "dataset/processed/training_features_hard.tsv"
PAIR_FILE = "dataset/processed/training_pairs_hard.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"


# ============================================================
# LOAD
# ============================================================

print("Loading data...")

features = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t",
    dtype=str
)

model = joblib.load(MODEL_FILE)

print("Rows:", len(features))


# ============================================================
# CHECK
# ============================================================

assert len(features) == len(pairs)

feature_columns = [
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "address_ratio",
    "address_token_set",
    "address_token_sort",
    "country_same",
    "name_exact",
    "address_exact",
    "name_length_diff",
    "address_length_diff",
    "name_best",
    "address_best",
    "combined_score"
]

X = features[feature_columns]

y = features["label"].astype(int)

# ============================================================
# MODEL PROBABILITY
# ============================================================

print("Calculating probabilities...")

prob = model.predict_proba(X)[:, 1]

df = pairs[
    [
        "source1_entity_id",
        "candidate_entity_id"
    ]
].copy()

df["label"] = y.values
df["prob"] = prob


# ============================================================
# ENTITY-LEVEL F0.5
# ============================================================

# Ground-truth positives for each S1
truth = (
    df[df["label"] == 1]
    .groupby("source1_entity_id")
    ["candidate_entity_id"]
    .apply(set)
    .to_dict()
)


def calc_f05(threshold):

    predictions = (
        df[df["prob"] >= threshold]
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    scores = []

    # Every S1 entity
    for s1_id in truth.keys():

        true_set = truth.get(
            s1_id,
            set()
        )

        pred_set = predictions.get(
            s1_id,
            set()
        )

        # Empty prediction
        if len(pred_set) == 0:

            if len(true_set) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            continue

        # No true matches
        if len(true_set) == 0:
            scores.append(0.0)
            continue

        tp = len(
            true_set & pred_set
        )

        precision = (
            tp / len(pred_set)
        )

        recall = (
            tp / len(true_set)
        )

        if precision == 0 and recall == 0:
            score = 0.0
        else:
            score = (
                1.25 * precision * recall
                /
                (0.25 * precision + recall)
            )

        scores.append(score)

    return np.mean(scores)


# ============================================================
# SEARCH
# ============================================================

print()
print("==============================================")
print("ENTITY-LEVEL F0.5 THRESHOLD SEARCH")
print("==============================================")

best_threshold = None
best_score = -1

thresholds = np.arange(
    0.70,
    0.991,
    0.01
)

for t in thresholds:

    score = calc_f05(t)

    print(
        f"Threshold {t:.2f} "
        f"-> Entity F0.5 = {score:.4f}"
    )

    if score > best_score:

        best_score = score
        best_threshold = t


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("BEST ENTITY-LEVEL RESULT")
print("==============================================")

print(
    "Best threshold:",
    f"{best_threshold:.2f}"
)

print(
    "Entity-level F0.5:",
    f"{best_score:.4f}"
)

print("==============================================")