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

ACCEPTED_FILE = "output/accepted_matches_v3.tsv"

MATCHING_OUT = "output/matching_results_v3.tsv"
CANDIDATE_OUT = "output/candidate_pairs_v3_final.tsv"

CHUNK_SIZE = 100_000


# ============================================================
# CLEAN OLD TEMP OUTPUT
# ============================================================

if os.path.exists(ACCEPTED_FILE):
    os.remove(ACCEPTED_FILE)

if os.path.exists(MATCHING_OUT):
    os.remove(MATCHING_OUT)

if os.path.exists(CANDIDATE_OUT):
    os.remove(CANDIDATE_OUT)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_FILE)

with open(THRESHOLD_FILE, "r") as f:
    threshold = float(f.read().strip())

print("Threshold:", threshold)


# ============================================================
# LOAD DATA
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
).fillna("").set_index("entity_id")

print("Source 1:", len(s1))


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


print("Combining Source 2 + Source 3...")

src = pd.concat(
    [s2, s3],
    ignore_index=True
).set_index("entity_id")

del s2
del s3
gc.collect()

print("Candidate source records:", len(src))


# ============================================================
# FEATURE FUNCTION
# ============================================================

def feature(a, b, scorer):

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
# OPEN ACCEPTED FILE
# ============================================================

first_write = True

processed = 0
accepted = 0


# ============================================================
# READ CANDIDATES
# ============================================================

print()
print("==============================================")
print("STARTING SAFE PREDICTION")
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
    # LOOKUP S1
    # --------------------------------------------------------

    chunk["name1"] = chunk[
        "source1_entity_id"
    ].map(s1["business_name"])

    chunk["addr1"] = chunk[
        "source1_entity_id"
    ].map(s1["business_address"])

    chunk["country1"] = chunk[
        "source1_entity_id"
    ].map(s1["country"])


    # --------------------------------------------------------
    # LOOKUP S2/S3
    # --------------------------------------------------------

    chunk["name2"] = chunk[
        "candidate_entity_id"
    ].map(src["business_name"])

    chunk["addr2"] = chunk[
        "candidate_entity_id"
    ].map(src["business_address"])

    chunk["country2"] = chunk[
        "candidate_entity_id"
    ].map(src["country"])

    chunk = chunk.fillna("")


    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name1 = chunk["name1"]
    name2 = chunk["name2"]

    name_ratio = feature(
        name1,
        name2,
        fuzz.ratio
    )

    name_token_set = feature(
        name1,
        name2,
        fuzz.token_set_ratio
    )

    name_token_sort = feature(
        name1,
        name2,
        fuzz.token_sort_ratio
    )


    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    addr1 = chunk["addr1"]
    addr2 = chunk["addr2"]

    address_ratio = feature(
        addr1,
        addr2,
        fuzz.ratio
    )

    address_token_set = feature(
        addr1,
        addr2,
        fuzz.token_set_ratio
    )

    address_token_sort = feature(
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
    ).astype(
        np.float32
    ).to_numpy()

    name_exact = (
        (name1 == name2)
        &
        (name1.str.strip() != "")
    ).astype(
        np.float32
    ).to_numpy()

    address_exact = (
        (addr1 == addr2)
        &
        (addr1.str.strip() != "")
    ).astype(
        np.float32
    ).to_numpy()

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
    # MODEL MATRIX
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
    # PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(X)[:, 1]

    keep = probability >= threshold

    selected = chunk.loc[
        keep,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ].drop_duplicates()


    # --------------------------------------------------------
    # WRITE ACCEPTED MATCHES IMMEDIATELY
    # --------------------------------------------------------

    if len(selected) > 0:

        selected.to_csv(
            ACCEPTED_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        accepted += len(selected)


    processed += len(chunk)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

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
    del probability
    del selected

    gc.collect()


# ============================================================
# PREDICTION COMPLETE
# ============================================================

print()
print("==============================================")
print("PREDICTION COMPLETE")
print("==============================================")

print(
    "Candidates processed:",
    f"{processed:,}"
)

print(
    "Accepted pairs:",
    f"{accepted:,}"
)

print(
    "Saved temporary matches:",
    ACCEPTED_FILE
)


# ============================================================
# BUILD FINAL RESULTS
# ============================================================

print()
print("Building final matching results...")


# Load accepted matches only
# This should be much smaller than 26M.

if os.path.exists(ACCEPTED_FILE):

    accepted_df = pd.read_csv(
        ACCEPTED_FILE,
        sep="\t",
        dtype=str
    ).fillna("")

else:

    accepted_df = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )


# ------------------------------------------------------------
# Group matches efficiently
# ------------------------------------------------------------

if len(accepted_df) > 0:

    grouped = (
        accepted_df
        .drop_duplicates()
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .agg(",".join)
        .reset_index()
    )

else:

    grouped = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )


if len(grouped) > 0:

    grouped.columns = [
        "source1_entity_id",
        "matched_entity_ids"
    ]


# ------------------------------------------------------------
# EVERY S1 ENTITY
# ------------------------------------------------------------

result = pd.DataFrame({
    "source1_entity_id":
        s1.index.to_numpy()
})

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
# Remove duplicates within lists
# ------------------------------------------------------------

def clean_ids(x):

    if not x:
        return ""

    values = x.split(",")

    seen = set()
    output = []

    for value in values:

        value = value.strip()

        if value and value not in seen:

            seen.add(value)
            output.append(value)

    return ",".join(output)


result["matched_entity_ids"] = (
    result["matched_entity_ids"]
    .map(clean_ids)
)


# ============================================================
# SAVE FINAL MATCHING
# ============================================================

result.to_csv(
    MATCHING_OUT,
    sep="\t",
    index=False
)


# ============================================================
# CANDIDATE FILE
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
# FINAL STATS
# ============================================================

nonempty = (
    result["matched_entity_ids"]
    .str.strip()
    != ""
).sum()

empty = len(result) - nonempty


print()
print("==============================================")
print("FINAL V3 RESULT CREATED")
print("==============================================")

print(
    "Source 1:",
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
print("Matching:")
print(MATCHING_OUT)

print()
print("Candidate:")
print(CANDIDATE_OUT)

print("==============================================")