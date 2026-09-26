import pandas as pd
import numpy as np
import os
import gc
import joblib

from rapidfuzz import process, fuzz


# ============================================================
# FILES
# ============================================================

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"

CANDIDATE_FILE = "output/candidate_pairs_v3.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"
THRESHOLD_FILE = "model/threshold_hard.txt"

MATCHING_OUT = "output/matching_results_v3.tsv"
CANDIDATE_OUT = "output/candidate_pairs_v3_final.tsv"

CHUNK_SIZE = 100_000


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_FILE)

with open(THRESHOLD_FILE, "r") as f:
    threshold = float(f.read().strip())

print("Model loaded")
print("Threshold:", threshold)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print()
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

s1 = s1.set_index("entity_id")

print("Source 1:", len(s1))


# ============================================================
# LOAD SOURCE 2
# ============================================================

print()
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

print("Source 2:", len(s2))


# ============================================================
# LOAD SOURCE 3
# ============================================================

print()
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

print("Source 3:", len(s3))


# ============================================================
# COMBINE S2 + S3
# ============================================================

print()
print("Combining Source 2 + Source 3...")

src = pd.concat(
    [s2, s3],
    ignore_index=True
)

src = src.set_index("entity_id")

print("Total candidate-source records:", len(src))

del s2
del s3
gc.collect()


# ============================================================
# RESULT STORAGE
# ============================================================

matches = {}

processed = 0
accepted = 0


# ============================================================
# FUZZY DISTANCE
# ============================================================

def cpdist_feature(a, b, scorer):

    return (
        np.asarray(
            process.cpdist(
                a.tolist(),
                b.tolist(),
                scorer=scorer,
                workers=-1,
                dtype=np.float32
            )
        ) / 100.0
    )


# ============================================================
# PROCESS CANDIDATE FILE
# ============================================================

print()
print("==============================================")
print("STARTING STREAMING PREDICTION")
print("==============================================")

reader = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)


for chunk_no, chunk in enumerate(
    reader,
    start=1
):

    chunk = chunk.fillna("")

    # --------------------------------------------------------
    # LOOK UP S1
    # --------------------------------------------------------

    chunk["name1"] = chunk[
        "source1_entity_id"
    ].map(
        s1["business_name"]
    )

    chunk["addr1"] = chunk[
        "source1_entity_id"
    ].map(
        s1["business_address"]
    )

    chunk["country1"] = chunk[
        "source1_entity_id"
    ].map(
        s1["country"]
    )

    # --------------------------------------------------------
    # LOOK UP S2/S3
    # --------------------------------------------------------

    chunk["name2"] = chunk[
        "candidate_entity_id"
    ].map(
        src["business_name"]
    )

    chunk["addr2"] = chunk[
        "candidate_entity_id"
    ].map(
        src["business_address"]
    )

    chunk["country2"] = chunk[
        "candidate_entity_id"
    ].map(
        src["country"]
    )

    chunk = chunk.fillna("")

    # --------------------------------------------------------
    # FUZZY NAME
    # --------------------------------------------------------

    name1 = chunk["name1"]
    name2 = chunk["name2"]

    name_ratio = cpdist_feature(
        name1,
        name2,
        fuzz.ratio
    )

    name_token_set = cpdist_feature(
        name1,
        name2,
        fuzz.token_set_ratio
    )

    name_token_sort = cpdist_feature(
        name1,
        name2,
        fuzz.token_sort_ratio
    )

    # --------------------------------------------------------
    # FUZZY ADDRESS
    # --------------------------------------------------------

    addr1 = chunk["addr1"]
    addr2 = chunk["addr2"]

    address_ratio = cpdist_feature(
        addr1,
        addr2,
        fuzz.ratio
    )

    address_token_set = cpdist_feature(
        addr1,
        addr2,
        fuzz.token_set_ratio
    )

    address_token_sort = cpdist_feature(
        addr1,
        addr2,
        fuzz.token_sort_ratio
    )

    # --------------------------------------------------------
    # SIMPLE FEATURES
    # --------------------------------------------------------

    country_same = (
        chunk["country1"].str.lower().str.strip()
        ==
        chunk["country2"].str.lower().str.strip()
    ).astype(np.float32).to_numpy()

    name_exact = (
        (name1 == name2)
        &
        (name1.str.strip() != "")
    ).astype(np.float32).to_numpy()

    address_exact = (
        (addr1 == addr2)
        &
        (addr1.str.strip() != "")
    ).astype(np.float32).to_numpy()

    name_length_diff = (
        name1.str.len()
        -
        name2.str.len()
    ).abs().to_numpy(
        dtype=np.float32
    )

    address_length_diff = (
        addr1.str.len()
        -
        addr2.str.len()
    ).abs().to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # BEST SCORES
    # --------------------------------------------------------

    name_best = np.maximum(
        name_ratio,
        np.maximum(
            name_token_set,
            name_token_sort
        )
    )

    address_best = np.maximum(
        address_ratio,
        np.maximum(
            address_token_set,
            address_token_sort
        )
    )

    combined_score = (
        0.55 * name_best
        +
        0.45 * address_best
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X = np.column_stack([
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
        combined_score
    ])

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    keep = probabilities >= threshold

    selected = chunk.loc[
        keep,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ]

    # --------------------------------------------------------
    # STORE MATCHES
    # --------------------------------------------------------

    for s1_id, candidate_id in zip(
        selected["source1_entity_id"],
        selected["candidate_entity_id"]
    ):

        if s1_id not in matches:
            matches[s1_id] = []

        matches[s1_id].append(
            candidate_id
        )

    processed += len(chunk)
    accepted += len(selected)

    if chunk_no % 5 == 0:

        print(
            f"Processed: {processed:,} | "
            f"Accepted: {accepted:,}"
        )

    # --------------------------------------------------------
    # FREE MEMORY
    # --------------------------------------------------------

    del chunk
    del X
    del probabilities
    del selected

    gc.collect()


# ============================================================
# CREATE FINAL RESULT
# ============================================================

print()
print("Creating final matching results...")


result = pd.DataFrame({
    "source1_entity_id":
        s1.index.to_numpy()
})

result["matched_entity_ids"] = ""


# ------------------------------------------------------------
# INSERT MATCHES
# ------------------------------------------------------------

for s1_id, ids in matches.items():

    unique_ids = list(
        dict.fromkeys(ids)
    )

    value = ",".join(unique_ids)

    result.loc[
        result["source1_entity_id"] == s1_id,
        "matched_entity_ids"
    ] = value


# ============================================================
# SAVE MATCHING
# ============================================================

result.to_csv(
    MATCHING_OUT,
    sep="\t",
    index=False
)


# ============================================================
# CREATE VALIDATOR-COMPATIBLE CANDIDATE FILE
# ============================================================

candidate_result = result.rename(
    columns={
        "matched_entity_ids":
            "candidate_entity_ids"
    }
)

candidate_result.to_csv(
    CANDIDATE_OUT,
    sep="\t",
    index=False
)


# ============================================================
# STATS
# ============================================================

nonempty = (
    result["matched_entity_ids"]
    .str.strip()
    != ""
).sum()

empty = len(result) - nonempty

print()
print("==============================================")
print("V3 PREDICTION COMPLETED")
print("==============================================")

print(
    "Total candidates processed:",
    f"{processed:,}"
)

print(
    "Accepted matches:",
    f"{accepted:,}"
)

print(
    "Source 1 entities:",
    f"{len(result):,}"
)

print(
    "Non-empty:",
    f"{nonempty:,}"
)

print(
    "Empty:",
    f"{empty:,}"
)

print()
print(
    "Matching:",
    MATCHING_OUT
)

print(
    "Candidates:",
    CANDIDATE_OUT
)

print("==============================================")