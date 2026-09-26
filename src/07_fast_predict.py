import pandas as pd
import os

# ============================================================
# PATHS
# ============================================================

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"

CANDIDATE_FILE = "output/candidate_pairs.tsv"
MATCHING_FILE = "output/matching_results.tsv"

# ============================================================
# READ SOURCE 1
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
)

s1 = s1.fillna("")

# Important: normalized files already contain cleaned values
s1["name_key"] = (
    s1["country"].str.lower().str.strip()
    + "|"
    + s1["business_name"].str.lower().str.strip()
)

print("Source 1 rows:", len(s1))

# Store only ID + key + address for matching
s1_small = s1[
    ["entity_id", "name_key", "business_address"]
].copy()

# ============================================================
# FUNCTION TO PROCESS ONE SOURCE
# ============================================================

def process_source(source_file, source_name):

    print()
    print("======================================")
    print("Processing", source_name)
    print("======================================")

    src = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ).fillna("")

    print(source_name, "rows:", len(src))

    src["name_key"] = (
        src["country"].str.lower().str.strip()
        + "|"
        + src["business_name"].str.lower().str.strip()
    )

    # --------------------------------------------------------
    # REMOVE EMPTY NAMES
    # --------------------------------------------------------

    src_valid = src[
        src["business_name"].str.strip() != ""
    ].copy()

    print("Valid names:", len(src_valid))

    # --------------------------------------------------------
    # FIND UNIQUE NAME KEYS
    # --------------------------------------------------------

    print("Finding unique business names...")

    counts = (
        src_valid
        .groupby("name_key")
        .size()
        .reset_index(name="count")
    )

    unique_keys = counts[
        counts["count"] == 1
    ][["name_key"]]

    unique_src = src_valid.merge(
        unique_keys,
        on="name_key",
        how="inner"
    )

    print("Unique name records:", len(unique_src))

    # --------------------------------------------------------
    # UNIQUE NAME MATCHES
    # --------------------------------------------------------

    print("Finding exact unique-name matches...")

    matches_unique = s1_small.merge(
        unique_src[
            ["entity_id", "name_key"]
        ],
        on="name_key",
        how="inner",
        suffixes=("_s1", "_src")
    )

    matches_unique = matches_unique[
        ["entity_id_s1", "entity_id_src"]
    ]

    matches_unique.columns = [
        "source1_entity_id",
        "candidate_entity_id"
    ]

    print(
        "Unique-name matches:",
        len(matches_unique)
    )

    # --------------------------------------------------------
    # EXACT NAME + ADDRESS MATCHES
    # --------------------------------------------------------

    print("Finding exact name + address matches...")

    # Only use non-empty addresses
    s1_addr = s1_small[
        (s1_small["name_key"] != "|")
        & (s1_small["business_address"].str.strip() != "")
    ].copy()

    src_addr = src_valid[
        src_valid["business_address"].str.strip() != ""
    ].copy()

    matches_address = s1_addr.merge(
        src_addr[
            [
                "entity_id",
                "name_key",
                "business_address"
            ]
        ],
        on=["name_key", "business_address"],
        how="inner",
        suffixes=("_s1", "_src")
    )

    matches_address = matches_address[
        [
            "entity_id_s1",
            "entity_id_src"
        ]
    ]

    matches_address.columns = [
        "source1_entity_id",
        "candidate_entity_id"
    ]

    print(
        "Name + address matches:",
        len(matches_address)
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    matches = pd.concat(
        [
            matches_unique,
            matches_address
        ],
        ignore_index=True
    )

    matches = matches.drop_duplicates()

    print(
        "Final matches from",
        source_name,
        ":",
        len(matches)
    )

    del src
    del src_valid
    del counts
    del unique_keys
    del unique_src
    del matches_unique
    del matches_address
    del s1_addr
    del src_addr

    return matches


# ============================================================
# PROCESS S2
# ============================================================

matches_s2 = process_source(
    S2_FILE,
    "Source 2"
)

# ============================================================
# PROCESS S3
# ============================================================

matches_s3 = process_source(
    S3_FILE,
    "Source 3"
)

# ============================================================
# COMBINE ALL MATCHES
# ============================================================

print()
print("Combining Source 2 + Source 3...")

all_matches = pd.concat(
    [
        matches_s2,
        matches_s3
    ],
    ignore_index=True
)

all_matches = all_matches.drop_duplicates()

print(
    "Total matched pairs:",
    len(all_matches)
)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)

# ============================================================
# CANDIDATE PAIRS
# ============================================================

print("Writing candidate_pairs.tsv...")

candidate_pairs = all_matches[
    [
        "source1_entity_id",
        "candidate_entity_id"
    ]
].copy()

candidate_pairs.to_csv(
    CANDIDATE_FILE,
    sep="\t",
    index=False
)

# ============================================================
# MATCHING RESULTS
# ============================================================

print("Creating matching_results.tsv...")

grouped = (
    all_matches
    .groupby("source1_entity_id")["candidate_entity_id"]
    .agg(",".join)
    .reset_index()
)

grouped.columns = [
    "source1_entity_id",
    "matched_entity_ids"
]

# ------------------------------------------------------------
# IMPORTANT:
# EVERY Source 1 entity must appear
# ------------------------------------------------------------

result = s1[
    ["entity_id"]
].rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)

result = result.merge(
    grouped,
    on="source1_entity_id",
    how="left"
)

result["matched_entity_ids"] = (
    result["matched_entity_ids"]
    .fillna("")
)

# ------------------------------------------------------------
# REMOVE DUPLICATE IDs INSIDE MATCH LIST
# ------------------------------------------------------------

def clean_ids(x):
    if not x:
        return ""

    ids = x.split(",")

    # preserve order
    seen = set()
    result_ids = []

    for value in ids:
        if value not in seen:
            seen.add(value)
            result_ids.append(value)

    return ",".join(result_ids)


result["matched_entity_ids"] = (
    result["matched_entity_ids"]
    .map(clean_ids)
)

# ============================================================
# SAVE
# ============================================================

result.to_csv(
    MATCHING_FILE,
    sep="\t",
    index=False
)

# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("======================================")
print("FAST PREDICTION COMPLETED")
print("======================================")

print(
    "Source 1 entities:",
    len(result)
)

print(
    "Matched Source 1 entities:",
    (result["matched_entity_ids"] != "").sum()
)

print(
    "Singleton Source 1 entities:",
    (result["matched_entity_ids"] == "").sum()
)

print(
    "Candidate pairs:",
    len(candidate_pairs)
)

print()
print("Created:")
print(CANDIDATE_FILE)
print(MATCHING_FILE)

print("======================================")