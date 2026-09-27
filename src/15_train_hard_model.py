import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "model"
FEATURE_FILE = "dataset/processed/training_features_hard.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("Loading features...")

features = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

print("\nFeature columns:")
print(list(features.columns))

print("\nFeature shape:", features.shape)


# ============================================================
# 2. FIND LABEL
# ============================================================

if "label" not in features.columns:
    raise ValueError(
        "label column not found in feature file."
    )

y = (
    pd.to_numeric(
        features["label"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# 3. FIND TRAINING PAIRS FILE
# ============================================================

possible_pair_files = [
    "training_pairs_hard.tsv",
    "output/training_pairs_hard.tsv",
    "dataset/processed/training_pairs_hard.tsv",
]

pair_file = None

for p in possible_pair_files:
    if os.path.exists(p):
        pair_file = p
        break

if pair_file is None:

    found = glob.glob(
        "**/training_pairs_hard.tsv",
        recursive=True
    )

    if found:
        pair_file = found[0]

if pair_file is None:

    raise FileNotFoundError(
        "\ntraining_pairs_hard.tsv not found.\n"
        "Expected one of:\n"
        + "\n".join(possible_pair_files)
    )

print("\nUsing pair file:", pair_file)


# ============================================================
# 4. LOAD PAIRS
# ============================================================

pairs = pd.read_csv(
    pair_file,
    sep="\t"
)

print("\nPair columns:")
print(list(pairs.columns))

print("\nPair shape:", pairs.shape)


# ============================================================
# 5. ROW ALIGNMENT CHECK
# ============================================================

if len(features) != len(pairs):

    raise ValueError(
        "\nFeature and pair files are NOT row aligned.\n"
        f"Features rows = {len(features)}\n"
        f"Pair rows     = {len(pairs)}\n"
        "Cannot safely train using pair IDs."
    )

print("\nRow alignment: OK")


# ============================================================
# 6. AUTOMATICALLY FIND ID COLUMN USING VALUES
# ============================================================

def detect_id_column(df, prefix):

    best_col = None
    best_count = 0

    for col in df.columns:

        sample = (
            df[col]
            .dropna()
            .astype(str)
            .head(10000)
        )

        if len(sample) == 0:
            continue

        count = sample.str.startswith(prefix).sum()

        if count > best_count:
            best_count = count
            best_col = col

    return best_col, best_count


# Source 1
s1_col, s1_count = detect_id_column(
    pairs,
    "S1-"
)

# Source 2
s2_col, s2_count = detect_id_column(
    pairs,
    "S2-"
)

# Source 3
s3_col, s3_count = detect_id_column(
    pairs,
    "S3-"
)


print("\nDetected IDs:")
print("S1:", s1_col, s1_count)
print("S2:", s2_col, s2_count)
print("S3:", s3_col, s3_count)


if s1_col is None or s1_count == 0:

    raise ValueError(
        "\nCould not detect S1 ID column from training_pairs_hard.tsv.\n"
        f"Available columns: {list(pairs.columns)}"
    )


# Candidate column
candidate_col = None

if s2_col is not None and s2_count > 0:
    candidate_col = s2_col

if s3_col is not None and s3_count > 0:

    # If same column contains S2/S3, use that
    if candidate_col is None:
        candidate_col = s3_col


if candidate_col is None:

    raise ValueError(
        "\nCould not detect candidate entity ID column.\n"
        f"Available columns: {list(pairs.columns)}"
    )


print("Candidate column:", candidate_col)


# ============================================================
# 7. ADD IDs TO FEATURE DATAFRAME
# ============================================================

features["__s1_id"] = (
    pairs[s1_col]
    .astype(str)
    .values
)

features["__candidate_id"] = (
    pairs[candidate_col]
    .astype(str)
    .values
)


# ============================================================
# 8. NUMERIC FEATURES
# ============================================================

exclude = {
    "label",
    "__s1_id",
    "__candidate_id"
}

feature_cols = []

for col in features.columns:

    if col in exclude:
        continue

    if pd.api.types.is_numeric_dtype(
        features[col]
    ):
        feature_cols.append(col)


print("\nFeatures used:")
for col in feature_cols:
    print(" ", col)


if len(feature_cols) < 5:

    raise ValueError(
        f"Only {len(feature_cols)} usable numeric features found."
    )


X = (
    features[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


groups = features["__s1_id"].astype(str)


# ============================================================
# 9. TRAIN / VALIDATION ENTITY SPLIT
# ============================================================

print("\nCreating entity-level validation split...")

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
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

val_data = features.iloc[val_idx].copy()
val_data = val_data.reset_index(drop=True)

print("Train pairs:", len(X_train))
print("Validation pairs:", len(X_val))
print(
    "Validation S1 entities:",
    val_data["__s1_id"].nunique()
)


# ============================================================
# 10. PRECISION-HEAVY WEIGHTS
# ============================================================

sample_weights = np.where(
    y_train.values == 0,
    1.5,
    1.0
)


# ============================================================
# 11. XGBOOST
# ============================================================

print("\nTraining XGBoost 3.2.0...")

model = XGBClassifier(
    n_estimators=650,
    max_depth=8,
    learning_rate=0.045,
    min_child_weight=8,
    subsample=0.90,
    colsample_bytree=0.90,
    gamma=0.05,
    reg_alpha=0.25,
    reg_lambda=6.0,

    objective="binary:logistic",
    eval_metric="aucpr",

    tree_method="hist",

    n_jobs=max(
        1,
        (os.cpu_count() or 2) - 1
    ),

    random_state=42
)


model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False
)

print("Training completed.")


# ============================================================
# 12. VALIDATION PROBABILITY
# ============================================================

print("\nCalculating validation probabilities...")

val_probability = model.predict_proba(
    X_val
)[:, 1]

val_data["probability"] = val_probability


# ============================================================
# 13. LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t"
)

truth = {}

for _, row in gt.iterrows():

    sid = str(
        row["source1_entity_id"]
    )

    value = row["matched_entity_ids"]

    if pd.isna(value):

        truth[sid] = set()

    elif str(value).strip() == "":

        truth[sid] = set()

    else:

        truth[sid] = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }


# ============================================================
# 14. F0.5
# ============================================================

def f05_score(predicted, actual):

    tp = len(
        predicted & actual
    )

    fp = len(
        predicted - actual
    )

    fn = len(
        actual - predicted
    )


    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)


    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)


    if precision == 0 and recall == 0:
        return 0.0


    return (
        1.25 * precision * recall
        /
        (
            0.25 * precision
            + recall
        )
    )


# ============================================================
# 15. THRESHOLD SEARCH
# ============================================================

print("\nSearching threshold...")

validation_sids = sorted(
    val_data["__s1_id"].unique()
)

best_threshold = 0.50
best_f05 = -1.0
best_precision = 0.0
best_recall = 0.0


for threshold in np.arange(
    0.50,
    0.951,
    0.01
):

    selected = val_data[
        val_data["probability"] >= threshold
    ]


    predictions = {}


    for _, row in selected.iterrows():

        sid = str(
            row["__s1_id"]
        )

        cid = str(
            row["__candidate_id"]
        )

        predictions.setdefault(
            sid,
            set()
        ).add(cid)


    scores = []
    precisions = []
    recalls = []


    for sid in validation_sids:

        predicted = predictions.get(
            sid,
            set()
        )

        actual = truth.get(
            sid,
            set()
        )


        tp = len(
            predicted & actual
        )

        fp = len(
            predicted - actual
        )

        fn = len(
            actual - predicted
        )


        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 1.0
        )


        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 1.0
        )


        score = f05_score(
            predicted,
            actual
        )


        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)


    mean_f05 = float(
        np.mean(scores)
    )

    mean_precision = float(
        np.mean(precisions)
    )

    mean_recall = float(
        np.mean(recalls)
    )


    print(
        f"Threshold={threshold:.2f} "
        f"F0.5={mean_f05:.4f} "
        f"P={mean_precision:.4f} "
        f"R={mean_recall:.4f}"
    )


    if mean_f05 > best_f05:

        best_f05 = mean_f05
        best_threshold = float(
            threshold
        )

        best_precision = mean_precision
        best_recall = mean_recall


# ============================================================
# 16. SAVE MODEL
# ============================================================

model_path = (
    "model/entity_match_best.pkl"
)

threshold_path = (
    "model/best_threshold.txt"
)


joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,

        # Save ID information too
        "s1_col": s1_col,
        "candidate_col": candidate_col
    },
    model_path
)


with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n")
print("=" * 65)
print("XGBOOST ENTITY MATCHING MODEL")
print("=" * 65)

print(
    "Feature file       :",
    FEATURE_FILE
)

print(
    "Pair file           :",
    pair_file
)

print(
    "S1 ID column        :",
    s1_col
)

print(
    "Candidate ID column :",
    candidate_col
)

print(
    "Features            :",
    len(feature_cols)
)

print(
    "Training pairs      :",
    len(X_train)
)

print(
    "Validation pairs    :",
    len(X_val)
)

print(
    "Best threshold      :",
    best_threshold
)

print(
    "Validation F0.5     :",
    round(best_f05, 4)
)

print(
    "Validation precision:",
    round(best_precision, 4)
)

print(
    "Validation recall   :",
    round(best_recall, 4)
)

print(
    "Model saved         :",
    model_path
)

print(
    "Threshold saved     :",
    threshold_path
)

print("=" * 65)