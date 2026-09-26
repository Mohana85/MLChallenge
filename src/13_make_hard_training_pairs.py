import pandas as pd
import re
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

OUT_FILE = "dataset/processed/training_pairs_hard.tsv"

S1_SAMPLE = 50000
CHUNK_SIZE = 250000

# Maximum hard negatives per S1
NEG_PER_S1 = 4


# ============================================================
# HELPERS
# ============================================================

def compact(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def first_token(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    return x.split()[0] if x else ""


def address_digits(x):
    if pd.isna(x):
        return ""
    return "".join(re.findall(r"\d+", str(x)))


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("Loading Source 1...")

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

s1["country"] = (
    s1["country"]
    .str.lower()
    .str.strip()
)

s1["name_key"] = (
    s1["country"]
    + "|"
    + s1["business_name"].map(first_token)
)

s1["addr_key"] = (
    s1["country"]
    + "|"
    + s1["business_address"].map(address_digits)
)

s1_id_set = set(s1["entity_id"])

print("S1 sample:", len(s1))


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
    gt["source1_entity_id"].isin(s1_id_set)
]

true_ids = defaultdict(set)

positive_rows = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    value = str(row.matched_entity_ids).strip()

    if not value:
        continue

    for cid in value.split(","):

        cid = cid.strip()

        if cid:
            true_ids[s1_id].add(cid)

            positive_rows.append(
                (s1_id, cid, 1)
            )

print(
    "Positive pairs:",
    len(positive_rows)
)


# ============================================================
# REQUIRED BLOCK KEYS
# ============================================================

needed_name_keys = set(
    s1["name_key"]
)

needed_addr_keys = set(
    s1["addr_key"]
)

# Remove empty keys
needed_name_keys.discard("|")
needed_addr_keys.discard("|")


# ============================================================
# BUILD CANDIDATE POOLS
# ============================================================

name_pool = defaultdict(list)
addr_pool = defaultdict(list)

print()
print("Scanning Source 2 + Source 3 for hard negatives...")


def scan_source(file_path, source_name):

    print()
    print("Scanning", source_name)

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ),
        start=1
    ):

        chunk = chunk.fillna("")

        country = (
            chunk["country"]
            .str.lower()
            .str.strip()
        )

        name_keys = (
            country
            + "|"
            + chunk["business_name"].map(first_token)
        )

        addr_keys = (
            country
            + "|"
            + chunk["business_address"].map(address_digits)
        )

        # ----------------------------------------------------
        # NAME BLOCK
        # ----------------------------------------------------

        name_mask = name_keys.isin(
            needed_name_keys
        )

        if name_mask.any():

            selected = chunk.loc[
                name_mask,
                ["entity_id"]
            ]

            selected_keys = name_keys[
                name_mask
            ]

            for key, entity_id in zip(
                selected_keys,
                selected["entity_id"]
            ):

                if len(name_pool[key]) < 10:
                    name_pool[key].append(
                        entity_id
                    )

        # ----------------------------------------------------
        # ADDRESS BLOCK
        # ----------------------------------------------------

        addr_mask = addr_keys.isin(
            needed_addr_keys
        )

        if addr_mask.any():

            selected = chunk.loc[
                addr_mask,
                ["entity_id"]
            ]

            selected_keys = addr_keys[
                addr_mask
            ]

            for key, entity_id in zip(
                selected_keys,
                selected["entity_id"]
            ):

                if len(addr_pool[key]) < 10:
                    addr_pool[key].append(
                        entity_id
                    )

        if chunk_no % 10 == 0:
            print(
                "  chunks processed:",
                chunk_no
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
# CREATE HARD NEGATIVES
# ============================================================

print()
print("Creating hard negatives...")

hard_negative_rows = []

for row in s1.itertuples(index=False):

    s1_id = row.entity_id

    true_set = true_ids.get(
        s1_id,
        set()
    )

    candidates = []

    # First-token candidates
    candidates.extend(
        name_pool.get(
            row.name_key,
            []
        )
    )

    # Address-digit candidates
    candidates.extend(
        addr_pool.get(
            row.addr_key,
            []
        )
    )

    seen = set()

    for cid in candidates:

        if cid in seen:
            continue

        seen.add(cid)

        # Never use a true positive as a negative
        if cid in true_set:
            continue

        hard_negative_rows.append(
            (
                s1_id,
                cid,
                0
            )
        )

        if (
            len([
                x for x in hard_negative_rows
                if x[0] == s1_id
            ])
            >= NEG_PER_S1
        ):
            break


# ============================================================
# DATAFRAME
# ============================================================

print(
    "Hard negative pairs:",
    len(hard_negative_rows)
)


all_rows = (
    positive_rows
    + hard_negative_rows
)

pairs = pd.DataFrame(
    all_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label"
    ]
)

# Remove duplicates
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
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

pairs.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

print()
print("==============================================")
print("HARD TRAINING PAIRS COMPLETED")
print("==============================================")

print(
    "Total pairs:",
    len(pairs)
)

print(
    "Positive:",
    (pairs["label"] == 1).sum()
)

print(
    "Negative:",
    (pairs["label"] == 0).sum()
)

print(
    "Saved:",
    OUT_FILE
)

print("==============================================")