import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score
from sklearn.base import clone

import joblib
import os


# ============================================================
# FILES
# ============================================================

FEATURE_FILE = "dataset/processed/training_features_hard.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"
THRESHOLD_FILE = "model/threshold_hard.txt"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading hard training features...")

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

print("Rows:", len(df))

# ============================================================
# FEATURES
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

X = df[feature_columns]
y = df["label"].astype(int)

# Group by Source-1 entity
# Prevent same S1 entity appearing in both train and validation
groups = df.index

# The feature file itself currently does not contain source1 IDs,
# so create a stable group using row order blocks based on the
# original hard-pair file.
pairs = pd.read_csv(
    "dataset/processed/training_pairs_hard.tsv",
    sep="\t",
    dtype=str
)

groups = pairs["source1_entity_id"].values

assert len(groups) == len(df)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("Creating entity-level train/validation split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_val)
)

print(
    "Training positive:",
    int(y_train.sum())
)

print(
    "Training negative:",
    int((y_train == 0).sum())
)

print(
    "Validation positive:",
    int(y_val.sum())
)

print(
    "Validation negative:",
    int((y_val == 0).sum())
)


# ============================================================
# MODEL
# ============================================================

print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# VALIDATION PROBABILITY
# ============================================================

print()
print("Calculating validation probabilities...")

val_prob = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print()
print("Threshold search...")

best_threshold = None
best_score = -1

for threshold in np.arange(
    0.50,
    0.951,
    0.025
):

    pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    score = f05(
        precision,
        recall
    )

    print(
        f"{threshold:.3f} "
        f"precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"F0.5={score:.4f}"
    )

    if score > best_score:

        best_score = score
        best_threshold = threshold


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

with open(
    THRESHOLD_FILE,
    "w"
) as f:
    f.write(
        str(best_threshold)
    )


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("HARD MODEL TRAINING COMPLETED")
print("==============================================")

print(
    "Best threshold:",
    best_threshold
)

print(
    "Validation F0.5:",
    f"{best_score:.4f}"
)

print(
    "Model saved:",
    MODEL_FILE
)

print(
    "Threshold saved:",
    THRESHOLD_FILE
)

print("==============================================")