import pandas as pd
import numpy as np
import joblib

# ============================================================
# FILES
# ============================================================

FEATURE_FILE = "dataset/processed/training_features_hard.tsv"
PAIR_FILE = "dataset/processed/training_pairs_hard.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"


# ============================================================
# LOAD
# ============================================================

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

print("Feature rows:", len(features))


# ============================================================
# ENTITY IDS PRESENT IN THIS TRAINING EXPERIMENT
# ============================================================

s1_ids = pairs[
    "source1_entity_id"
].unique()

s1_set = set(s1_ids)

gt = gt[
    gt["source1_entity_id"].isin(s1_set)
]


# ============================================================
# TRUE MATCH SETS
# ============================================================

truth = {}

for row in gt.itertuples(index=False):

    sid = row.source1_entity_id
    value = str(
        row.matched_entity_ids
    ).strip()

    if value:
        truth[sid] = set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )
    else:
        truth[sid] = set()


# ============================================================
# MODEL PROBABILITY
# ============================================================

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

print("Calculating RF probabilities...")

prob = model.predict_proba(X)[:, 1]

pairs["prob"] = prob

# ============================================================
# MULTI-ALGORITHM FEATURES
# ============================================================

pairs["name_best"] = features[
    "name_best"
].values

pairs["address_best"] = features[
    "address_best"
].values

pairs["combined_score"] = features[
    "combined_score"
].values

pairs["name_exact"] = features[
    "name_exact"
].values

pairs["address_exact"] = features[
    "address_exact"
].values


# ============================================================
# ENTITY-LEVEL F0.5
# ============================================================

def calculate_score(
    probability_threshold,
    combined_threshold,
    name_threshold,
    address_threshold
):

    # --------------------------------------------------------
    # Final ensemble rule
    #
    # Model must agree with similarity signals.
    #
    # --------------------------------------------------------

    strong_name = (
        pairs["name_best"] >= name_threshold
    )

    strong_address = (
        pairs["address_best"] >= address_threshold
    )

    strong_combined = (
        pairs["combined_score"]
        >= combined_threshold
    )

    exact_name = (
        pairs["name_exact"] == 1
    )

    exact_address = (
        pairs["address_exact"] == 1
    )

    model_ok = (
        pairs["prob"] >= probability_threshold
    )

    keep = model_ok & (
        exact_name
        |
        (strong_name & strong_address)
        |
        (strong_combined & strong_name)
        |
        (exact_address & strong_name)
    )

    selected = pairs.loc[
        keep,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ]

    predictions = (
        selected
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------------
    # ALL S1 ENTITIES
    # --------------------------------------------------------

    scores = []

    for sid in s1_ids:

        true_set = truth.get(
            sid,
            set()
        )

        pred_set = predictions.get(
            sid,
            set()
        )

        # ----------------------------------------------------
        # Singleton / empty prediction
        # ----------------------------------------------------

        if len(true_set) == 0:

            if len(pred_set) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            continue

        # ----------------------------------------------------
        # True matches exist
        # ----------------------------------------------------

        if len(pred_set) == 0:

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

        if precision == 0:

            scores.append(0.0)

        else:

            f05 = (
                1.25
                * precision
                * recall
                /
                (
                    0.25 * precision
                    + recall
                )
            )

            scores.append(f05)

    return np.mean(scores)


# ============================================================
# SEARCH
# ============================================================

print()
print("==============================================")
print("ENSEMBLE ENTITY-LEVEL SEARCH")
print("==============================================")

best_score = -1
best_config = None


# Keep search small and practical
probabilities = [
    0.775,
    0.80,
    0.825,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95
]

combined_values = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

name_values = [
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

address_values = [
    0.45,
    0.55,
    0.65,
    0.75,
    0.85
]


for pt in probabilities:

    for ct in combined_values:

        for nt in name_values:

            for at in address_values:

                score = calculate_score(
                    pt,
                    ct,
                    nt,
                    at
                )

                if score > best_score:

                    best_score = score

                    best_config = (
                        pt,
                        ct,
                        nt,
                        at
                    )

                    print(
                        f"BEST -> "
                        f"RF={pt:.3f}, "
                        f"COMBINED={ct:.2f}, "
                        f"NAME={nt:.2f}, "
                        f"ADDRESS={at:.2f}, "
                        f"F0.5={score:.4f}"
                    )


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("BEST ENSEMBLE CONFIGURATION")
print("==============================================")

pt, ct, nt, at = best_config

print(
    "RF threshold:",
    f"{pt:.3f}"
)

print(
    "Combined score:",
    f"{ct:.2f}"
)

print(
    "Name score:",
    f"{nt:.2f}"
)

print(
    "Address score:",
    f"{at:.2f}"
)

print(
    "Entity-level F0.5:",
    f"{best_score:.4f}"
)

print("==============================================")