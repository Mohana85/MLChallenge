import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD
# ============================================================

FEATURE_FILE = "dataset/processed/training_features_hard.tsv"
PAIR_FILE = "dataset/processed/training_pairs_hard.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"

print("Loading...")

features = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

model = joblib.load(MODEL_FILE)

assert len(features) == len(pairs)

# ============================================================
# MODEL PROBABILITY
# ============================================================

cols = [
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

print("Calculating probabilities...")

pairs["prob"] = model.predict_proba(
    features[cols]
)[:, 1]

pairs["name_best"] = features["name_best"].values
pairs["address_best"] = features["address_best"].values
pairs["combined_score"] = features["combined_score"].values
pairs["name_exact"] = features["name_exact"].values
pairs["address_exact"] = features["address_exact"].values


# ============================================================
# TRUE MATCHES
# ============================================================

ids = pairs["source1_entity_id"].unique()

truth = {
    x: set()
    for x in ids
}

for row in gt[
    gt["source1_entity_id"].isin(ids)
].itertuples(index=False):

    value = str(row.matched_entity_ids).strip()

    if value:
        truth[row.source1_entity_id] = set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )


# ============================================================
# FAST SCORE FUNCTION
# ============================================================

def score_rule(
    rf,
    combined,
    name,
    address
):

    keep = (
        (pairs["prob"] >= rf)
        &
        (
            (pairs["name_exact"] == 1)
            |
            (
                (pairs["name_best"] >= name)
                &
                (pairs["address_best"] >= address)
            )
            |
            (
                (pairs["combined_score"] >= combined)
                &
                (pairs["name_best"] >= name)
            )
            |
            (
                (pairs["address_exact"] == 1)
                &
                (pairs["name_best"] >= name)
            )
        )
    )

    selected = pairs.loc[
        keep,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ]

    pred = (
        selected
        .groupby("source1_entity_id")
        ["candidate_entity_id"]
        .agg(set)
        .to_dict()
    )

    total_score = 0.0

    for sid in ids:

        true_set = truth.get(
            sid,
            set()
        )

        pred_set = pred.get(
            sid,
            set()
        )

        # Correct empty prediction
        if not true_set:

            total_score += (
                1.0 if not pred_set else 0.0
            )

            continue

        # Missed everything
        if not pred_set:
            continue

        tp = len(
            true_set & pred_set
        )

        if tp == 0:
            continue

        precision = tp / len(pred_set)
        recall = tp / len(true_set)

        total_score += (
            1.25 * precision * recall
            /
            (0.25 * precision + recall)
        )

    return total_score / len(ids)


# ============================================================
# SMALL SEARCH
# ============================================================

configs = [
    # rf, combined, name, address
    (0.80, 0.70, 0.75, 0.45),
    (0.825, 0.70, 0.75, 0.45),
    (0.85, 0.70, 0.75, 0.45),
    (0.875, 0.70, 0.75, 0.45),
    (0.90, 0.70, 0.75, 0.45),

    (0.85, 0.75, 0.80, 0.50),
    (0.875, 0.75, 0.80, 0.50),
    (0.90, 0.75, 0.80, 0.50),

    (0.875, 0.80, 0.85, 0.55),
    (0.90, 0.80, 0.85, 0.55),

    (0.90, 0.85, 0.90, 0.60),
    (0.925, 0.85, 0.90, 0.60),

    (0.925, 0.90, 0.90, 0.65),
    (0.95, 0.90, 0.90, 0.65),

    (0.95, 0.95, 0.95, 0.75),
]


# ============================================================
# RUN
# ============================================================

best_score = -1
best_config = None

print()
print("==============================================")
print("FAST ENSEMBLE TUNING")
print("==============================================")

for rf, combined, name, address in configs:

    score = score_rule(
        rf,
        combined,
        name,
        address
    )

    print(
        f"RF={rf:.3f} "
        f"COMBINED={combined:.2f} "
        f"NAME={name:.2f} "
        f"ADDRESS={address:.2f} "
        f"F0.5={score:.4f}"
    )

    if score > best_score:

        best_score = score

        best_config = (
            rf,
            combined,
            name,
            address
        )


# ============================================================
# RESULT
# ============================================================

print()
print("==============================================")
print("BEST FAST CONFIG")
print("==============================================")

print(
    "RF:",
    best_config[0]
)

print(
    "Combined:",
    best_config[1]
)

print(
    "Name:",
    best_config[2]
)

print(
    "Address:",
    best_config[3]
)

print(
    "Entity F0.5:",
    f"{best_score:.4f}"
)

print("==============================================")