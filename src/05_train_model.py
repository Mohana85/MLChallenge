import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

FEATURE_FILE = DATA_DIR / "training_features.tsv"

MODEL_FILE = MODEL_DIR / "entity_match_model.pkl"
THRESHOLD_FILE = MODEL_DIR / "threshold.txt"


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading training features...", flush=True)

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

print(
    "Total training rows:",
    len(df),
    flush=True
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "country_same"
]

X = df[feature_columns]

y = df["label"]


print(
    "Positive:",
    (y == 1).sum(),
    flush=True
)

print(
    "Negative:",
    (y == 0).sum(),
    flush=True
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("Splitting data...", flush=True)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_val)
)


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("Training Random Forest...", flush=True)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Model training completed!",
    flush=True
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print(
    "Generating validation probabilities...",
    flush=True
)

probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print()
print("Searching best threshold...", flush=True)

best_threshold = 0.50
best_f05 = 0.0

results = []

for threshold in np.arange(
    0.50,
    0.96,
    0.05
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    results.append(
        [
            threshold,
            precision,
            recall,
            f05
        ]
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F0.5={f05:.4f}",
        flush=True
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("Saving model...", flush=True)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

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
print("==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")

print(
    "Best threshold:",
    best_threshold
)

print(
    "Validation F0.5:",
    round(best_f05, 4)
)

print(
    "Model:",
    MODEL_FILE
)

print(
    "Threshold:",
    THRESHOLD_FILE
)

print("==========================================")