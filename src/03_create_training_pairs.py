import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed"
TRAIN_DIR = BASE_DIR / "dataset" / "train"

OUTPUT_PATH = DATA_DIR / "training_pairs.tsv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...", flush=True)

# ------------------------------------------------------------
# Source 1
# ------------------------------------------------------------

s1 = pd.read_csv(
    DATA_DIR / "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "country_clean"
    ],
    nrows=50000
)

print(
    "Source 1 loaded:",
    len(s1),
    flush=True
)


# ------------------------------------------------------------
# Source 2
# ------------------------------------------------------------

s2 = pd.read_csv(
    DATA_DIR / "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "country_clean"
    ]
)

print(
    "Source 2 loaded:",
    len(s2),
    flush=True
)


# ------------------------------------------------------------
# Source 3
# ------------------------------------------------------------

s3 = pd.read_csv(
    DATA_DIR / "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "country_clean"
    ]
)

print(
    "Source 3 loaded:",
    len(s3),
    flush=True
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Loading ground truth...", flush=True)

gt = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


# ============================================================
# CREATE SOURCE 1 ID SET
# ============================================================

s1_ids = set(
    s1["entity_id"]
)

print(
    "Source 1 IDs prepared.",
    flush=True
)


# ============================================================
# POSITIVE PAIRS
# ============================================================

print(
    "Creating positive pairs...",
    flush=True
)

# IMPORTANT:
# Filter the 2.2M ground-truth rows using pandas isin()
# instead of checking the 50k IDs inside every loop.

gt_small = gt[
    gt["source1_entity_id"].isin(s1_ids)
].copy()

print(
    "Relevant ground-truth rows:",
    len(gt_small),
    flush=True
)


positive_rows = []


for _, row in gt_small.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = row["matched_entity_ids"]

    # No match
    if not matched_ids:
        continue

    # Multiple matched IDs are comma-separated
    for candidate_id in str(matched_ids).split(","):

        candidate_id = candidate_id.strip()

        if candidate_id:

            positive_rows.append(
                [
                    source1_id,
                    candidate_id,
                    1
                ]
            )


positive = pd.DataFrame(
    positive_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label"
    ]
)


print(
    "Positive pairs:",
    len(positive),
    flush=True
)


# ============================================================
# CREATE CANDIDATE POOL
# ============================================================

print(
    "Preparing candidate pool...",
    flush=True
)

other = pd.concat(
    [
        s2[
            [
                "entity_id",
                "country_clean"
            ]
        ],

        s3[
            [
                "entity_id",
                "country_clean"
            ]
        ]
    ],
    ignore_index=True
)


# We don't need every candidate for this quick training stage.
# Keeping 100k makes the process much faster.

other = other.head(100000).copy()


print(
    "Candidate pool:",
    len(other),
    flush=True
)


# ============================================================
# COUNTRY → ENTITY IDS
# ============================================================

country_candidates = (
    other
    .groupby("country_clean")["entity_id"]
    .apply(list)
    .to_dict()
)


print(
    "Country candidate groups created.",
    flush=True
)


# ============================================================
# POSITIVE LOOKUP
# ============================================================

positive_set = set(
    zip(
        positive["source1_entity_id"],
        positive["candidate_entity_id"]
    )
)


# ============================================================
# NEGATIVE PAIRS
# ============================================================

print(
    "Creating negative pairs...",
    flush=True
)


negative_rows = []


# Only 10k Source1 records for negative generation
s1_negative = s1.head(10000)


for _, row in s1_negative.iterrows():

    source1_id = row["entity_id"]

    country = row["country_clean"]

    candidates = country_candidates.get(
        country,
        []
    )

    count = 0


    for candidate_id in candidates:

        # Don't accidentally label a true match as negative
        if (
            source1_id,
            candidate_id
        ) in positive_set:

            continue


        negative_rows.append(
            [
                source1_id,
                candidate_id,
                0
            ]
        )


        count += 1


        # Only 2 negatives per Source1
        if count >= 2:
            break


print(
    "Negative pairs:",
    len(negative_rows),
    flush=True
)


# ============================================================
# CONVERT NEGATIVES TO DATAFRAME
# ============================================================

negative = pd.DataFrame(
    negative_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label"
    ]
)


# ============================================================
# COMBINE POSITIVE + NEGATIVE
# ============================================================

print(
    "Combining training pairs...",
    flush=True
)


pairs = pd.concat(
    [
        positive,
        negative
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

pairs = pairs.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

pairs.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("==========================================")
print("TRAINING PAIRS CREATED SUCCESSFULLY")
print("==========================================")

print(
    "Total pairs :",
    len(pairs)
)

print(
    "Positive    :",
    (pairs["label"] == 1).sum()
)

print(
    "Negative    :",
    (pairs["label"] == 0).sum()
)

print(
    "Saved file  :",
    OUTPUT_PATH
)

print("==========================================")