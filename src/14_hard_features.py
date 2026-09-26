import pandas as pd
from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
    token_sort_ratio
)
import os

# ============================================================
# FILES
# ============================================================

PAIR_FILE = "dataset/processed/training_pairs_hard.tsv"
S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"

OUT_FILE = "dataset/processed/training_features_hard.tsv"

# ============================================================
# LOAD PAIRS
# ============================================================

print("Loading hard training pairs...")

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t",
    dtype=str
).fillna("")

print("Pairs:", len(pairs))

# ============================================================
# LOAD SOURCES
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

print("Loading Source 2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")

print("Loading Source 3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
).fillna("")


# ============================================================
# LOOKUPS
# ============================================================

print("Creating lookups...")

s1_lookup = s1.set_index("entity_id")
s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")


# ============================================================
# HELPER
# ============================================================

def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return ratio(str(a), str(b)) / 100.0


def safe_token_set(a, b):
    if not a or not b:
        return 0.0
    return token_set_ratio(
        str(a),
        str(b)
    ) / 100.0


def safe_token_sort(a, b):
    if not a or not b:
        return 0.0
    return token_sort_ratio(
        str(a),
        str(b)
    ) / 100.0


# ============================================================
# FEATURE CALCULATION
# ============================================================

print("Calculating features...")

features = []

for i, row in enumerate(
    pairs.itertuples(index=False),
    start=1
):

    s1_id = row.source1_entity_id
    cid = row.candidate_entity_id
    label = int(row.label)

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    a = s1_lookup.loc[s1_id]

    # --------------------------------------------------------
    # S2 or S3
    # --------------------------------------------------------

    if cid.startswith("S2-"):
        b = s2_lookup.loc[cid]
    else:
        b = s3_lookup.loc[cid]

    name1 = str(a["business_name"])
    name2 = str(b["business_name"])

    addr1 = str(a["business_address"])
    addr2 = str(b["business_address"])

    country1 = str(a["country"]).lower().strip()
    country2 = str(b["country"]).lower().strip()

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = safe_ratio(
        name1,
        name2
    )

    name_token_set = safe_token_set(
        name1,
        name2
    )

    name_token_sort = safe_token_sort(
        name1,
        name2
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_ratio = safe_ratio(
        addr1,
        addr2
    )

    address_token_set = safe_token_set(
        addr1,
        addr2
    )

    address_token_sort = safe_token_sort(
        addr1,
        addr2
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_same = int(
        country1 == country2
    )

    # --------------------------------------------------------
    # EXACT FEATURES
    # --------------------------------------------------------

    name_exact = int(
        name1 == name2
        and name1 != ""
    )

    address_exact = int(
        addr1 == addr2
        and addr1 != ""
    )

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    name_length_diff = abs(
        len(name1) - len(name2)
    )

    address_length_diff = abs(
        len(addr1) - len(addr2)
    )

    # --------------------------------------------------------
    # COMBINED SCORES
    # --------------------------------------------------------

    name_best = max(
        name_ratio,
        name_token_set,
        name_token_sort
    )

    address_best = max(
        address_ratio,
        address_token_set,
        address_token_sort
    )

    combined_score = (
        0.55 * name_best
        +
        0.45 * address_best
    )

    features.append(
        [
            name_ratio,
            name_token_set,
            name_token_sort,
            address_ratio,
            address_token_set,
            address_token_sort,
            country_same,
            name_exact,
            address_exact,
            name_length_diff,
            address_length_diff,
            name_best,
            address_best,
            combined_score,
            label
        ]
    )

    if i % 25000 == 0:
        print(
            "Processed:",
            i,
            "/",
            len(pairs)
        )


# ============================================================
# DATAFRAME
# ============================================================

columns = [
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
    "combined_score",
    "label"
]

df = pd.DataFrame(
    features,
    columns=columns
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "dataset/processed",
    exist_ok=True
)

df.to_csv(
    OUT_FILE,
    sep="\t",
    index=False
)

print()
print("======================================")
print("HARD FEATURES COMPLETED")
print("======================================")

print(
    "Total rows:",
    len(df)
)

print(
    "Positive:",
    (df["label"] == 1).sum()
)

print(
    "Negative:",
    (df["label"] == 0).sum()
)

print()
print(
    "Features:",
    list(df.columns)
)

print()
print("Saved:", OUT_FILE)

print("======================================")