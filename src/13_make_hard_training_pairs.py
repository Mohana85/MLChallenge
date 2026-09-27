import pandas as pd
import re
import os
import gc

from collections import defaultdict


# ============================================================
# FILES
# ============================================================

S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

OUT_FILE = "dataset/processed/training_pairs_hard.tsv"

S1_SAMPLE = 50000
CHUNK_SIZE = 250000

# Same idea as test candidate generation
MAX_PER_BLOCK = 15

# Hard negatives per Source-1
NEG_PER_S1 = 8


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_WORDS = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "company": "co",
    "private": "pvt",
}


def normalize_name(x):

    if pd.isna(x):
        return ""

    x = str(x).lower()

    for old, new in LEGAL_WORDS.items():
        x = x.replace(old, new)

    x = re.sub(
        r"[^a-z0-9\s]",
        " ",
        x
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x


def compact_name(x):
    return normalize_name(x).replace(
        " ",
        ""
    )


def first_token(x):

    x = normalize_name(x)

    if not x:
        return ""

    return x.split()[0]


def address_digits(x):

    if pd.isna(x):
        return ""

    nums = re.findall(
        r"\d+",
        str(x)
    )

    return "".join(nums)


def exact_name_key(
    country,
    name
):

    return (
        str(country).lower().strip()
        + "|"
        + compact_name(name)
    )


def combo_key(
    country,
    address,
    name
):

    digits = address_digits(address)
    token = first_token(name)

    if len(digits) < 2 or not token:
        return ""

    return (
        str(country).lower().strip()
        + "|"
        + digits
        + "|"
        + token
    )


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("Loading Source 1 sample...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    nrows=S1_SAMPLE,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

s1_ids = set(
    s1["entity_id"]
)

print(
    "S1 sample:",
    len(s1)
)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(
        s1_ids
    )
]


# ============================================================
# TRUE MATCHES
# ============================================================

true_matches = defaultdict(set)

positive_rows = []

for row in gt.itertuples(index=False):

    sid = row.source1_entity_id

    value = str(
        row.matched_entity_ids
    ).strip()

    if not value:
        continue

    for cid in value.split(","):

        cid = cid.strip()

        if cid:

            true_matches[sid].add(cid)

            positive_rows.append(
                (
                    sid,
                    cid,
                    1
                )
            )


print(
    "Positive pairs:",
    len(positive_rows)
)


# ============================================================
# BUILD TEST-STYLE BLOCK INDEX
# ============================================================

name_index = defaultdict(list)
combo_index = defaultdict(list)


def add_to_index(
    index,
    key,
    entity_id
):

    if not key:
        return

    values = index[key]

    # Prevent huge blocks
    if len(values) < MAX_PER_BLOCK:

        values.append(
            entity_id
        )


def scan_source(
    path,
    source_name
):

    print()
    print(
        "Indexing",
        source_name
    )

    count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):

        chunk = chunk.fillna("")

        for row in chunk.itertuples(
            index=False
        ):

            eid = row.entity_id

            nk = exact_name_key(
                row.country,
                row.business_name
            )

            ck = combo_key(
                row.country,
                row.business_address,
                row.business_name
            )

            add_to_index(
                name_index,
                nk,
                eid
            )

            add_to_index(
                combo_index,
                ck,
                eid
            )

            count += 1

        if count % 1000000 < CHUNK_SIZE:

            print(
                "  indexed:",
                f"{count:,}"
            )


scan_source(
    S2_FILE,
    "Source 2"
)

scan_source(
    S3_FILE,
    "Source 3"
)


# ============================================================
# CREATE TRAINING PAIRS
# ============================================================

print()
print("Creating aligned hard negatives...")


all_rows = list(
    positive_rows
)

negative_count = 0

for row in s1.itertuples(
    index=False
):

    sid = row.entity_id

    true_set = true_matches.get(
        sid,
        set()
    )

    # --------------------------------------------------------
    # EXACT NAME BLOCK
    # --------------------------------------------------------

    nk = exact_name_key(
        row.country,
        row.business_name
    )

    # --------------------------------------------------------
    # NAME + ADDRESS-DIGITS BLOCK
    # --------------------------------------------------------

    ck = combo_key(
        row.country,
        row.business_address,
        row.business_name
    )

    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    candidate_ids = []

    candidate_ids.extend(
        name_index.get(
            nk,
            []
        )
    )

    candidate_ids.extend(
        combo_index.get(
            ck,
            []
        )
    )

    # Deduplicate
    seen = set()

    unique_candidates = []

    for cid in candidate_ids:

        if cid not in seen:

            seen.add(cid)
            unique_candidates.append(cid)

    # --------------------------------------------------------
    # HARD NEGATIVES
    # --------------------------------------------------------

    added = 0

    for cid in unique_candidates:

        # Never label a true match negative
        if cid in true_set:
            continue

        all_rows.append(
            (
                sid,
                cid,
                0
            )
        )

        negative_count += 1
        added += 1

        if added >= NEG_PER_S1:
            break


# ============================================================
# DATAFRAME
# ============================================================

pairs = pd.DataFrame(
    all_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label"
    ]
)

pairs = pairs.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

# Shuffle
pairs = pairs.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "dataset/processed",
    exist_ok=True
)

pairs.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("ALIGNED TRAINING PAIRS COMPLETED")
print("==============================================")

print(
    "Total pairs:",
    len(pairs)
)

print(
    "Positive:",
    int(
        (pairs["label"] == 1).sum()
    )
)

print(
    "Negative:",
    int(
        (pairs["label"] == 0).sum()
    )
)

print(
    "Positive ratio:",
    round(
        (
            pairs["label"] == 1
        ).mean(),
        4
    )
)

print(
    "Saved:",
    OUT_FILE
)

print("==============================================")