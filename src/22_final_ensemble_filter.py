import pandas as pd
import numpy as np
import os
import gc
import joblib

from rapidfuzz import process, fuzz


# ============================================================
# FILES
# ============================================================

ACCEPTED_FILE = "output/accepted_matches_v3.tsv"

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"

MODEL_FILE = "model/entity_match_model_hard.pkl"

FINAL_MATCHING = "output/matching_results_final.tsv"
FINAL_CANDIDATES = "output/candidate_pairs_final.tsv"

CHUNK_SIZE = 100_000


# ============================================================
# THRESHOLDS
# ============================================================

RF_THRESHOLD = 0.825
COMBINED_THRESHOLD = 0.70
NAME_THRESHOLD = 0.75
ADDRESS_THRESHOLD = 0.45


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(ACCEPTED_FILE):
    print("ERROR: accepted_matches_v3.tsv not found")
    print("Expected:", ACCEPTED_FILE)
    raise SystemExit(1)

if not os.path.exists(MODEL_FILE):
    print("ERROR: model not found")
    print("Expected:", MODEL_FILE)
    raise SystemExit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded")


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

s2 = s2.set_index("entity_id")

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

s3 = s3.set_index("entity_id")

print("Source 3:", len(s3))


# ============================================================
# CREATE LOOKUPS
# ============================================================

print()
print("Creating source lookup...")

src_name = pd.concat(
    [
        s2["business_name"],
        s3["business_name"]
    ]
)

src_addr = pd.concat(
    [
        s2["business_address"],
        s3["business_address"]
    ]
)

src_country = pd.concat(
    [
        s2["country"],
        s3["country"]
    ]
)

del s2
del s3

gc.collect()

print(
    "Source lookup records:",
    len(src_name)
)


# ============================================================
# FUZZY FEATURE FUNCTION
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
        )
        / 100.0
    )


# ============================================================
# REMOVE OLD OUTPUTS
# ============================================================

if os.path.exists(FINAL_MATCHING):
    os.remove(FINAL_MATCHING)

if os.path.exists(FINAL_CANDIDATES):
    os.remove(FINAL_CANDIDATES)


# ============================================================
# COUNTERS
# ============================================================

processed = 0
kept = 0


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("STARTING FINAL ENSEMBLE FILTER")
print("==============================================")

print(
    "Input accepted pairs:",
    ACCEPTED_FILE
)

print(
    "RF threshold:",
    RF_THRESHOLD
)

print(
    "Combined threshold:",
    COMBINED_THRESHOLD
)

print(
    "Name threshold:",
    NAME_THRESHOLD
)

print(
    "Address threshold:",
    ADDRESS_THRESHOLD
)


# ============================================================
# READ ACCEPTED PAIRS IN CHUNKS
# ============================================================

reader = pd.read_csv(
    ACCEPTED_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)


# ============================================================
# PROCESS
# ============================================================

for chunk_no, chunk in enumerate(
    reader,
    start=1
):

    chunk = chunk.fillna("")

    # --------------------------------------------------------
    # LOOKUP SOURCE 1
    # --------------------------------------------------------

    chunk["name1"] = (
        chunk["source1_entity_id"]
        .map(s1["business_name"])
        .fillna("")
    )

    chunk["addr1"] = (
        chunk["source1_entity_id"]
        .map(s1["business_address"])
        .fillna("")
    )

    chunk["country1"] = (
        chunk["source1_entity_id"]
        .map(s1["country"])
        .fillna("")
    )


    # --------------------------------------------------------
    # LOOKUP SOURCE 2 / SOURCE 3
    # --------------------------------------------------------

    chunk["name2"] = (
        chunk["candidate_entity_id"]
        .map(src_name)
        .fillna("")
    )

    chunk["addr2"] = (
        chunk["candidate_entity_id"]
        .map(src_addr)
        .fillna("")
    )

    chunk["country2"] = (
        chunk["candidate_entity_id"]
        .map(src_country)
        .fillna("")
    )

    chunk = chunk.fillna("")


    # ========================================================
    # NAME SIMILARITY
    # ========================================================

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


    # ========================================================
    # ADDRESS SIMILARITY
    # ========================================================

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


    # ========================================================
    # EXACT FEATURES
    # ========================================================

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


    # ========================================================
    # COUNTRY FEATURE
    # ========================================================

    country_same = (
        chunk["country1"]
        .str.lower()
        .str.strip()
        ==
        chunk["country2"]
        .str.lower()
        .str.strip()
    ).astype(
        np.float32
    ).to_numpy()


    # ========================================================
    # LENGTH FEATURES
    # ========================================================

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


    # ========================================================
    # BEST SCORES
    # ========================================================

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


    # ========================================================
    # MODEL INPUT
    # ========================================================

    X = np.column_stack(
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
            combined_score
        ]
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    probability = model.predict_proba(
        X
    )[:, 1]


    # ========================================================
    # ENSEMBLE RULE
    # ========================================================

    model_ok = (
        probability >= RF_THRESHOLD
    )

    rule_exact_name = (
        name_exact == 1
    )

    rule_name_address = (
        (name_best >= NAME_THRESHOLD)
        &
        (address_best >= ADDRESS_THRESHOLD)
    )

    rule_combined = (
        (combined_score >= COMBINED_THRESHOLD)
        &
        (name_best >= NAME_THRESHOLD)
    )

    rule_exact_address = (
        (address_exact == 1)
        &
        (name_best >= NAME_THRESHOLD)
    )

    keep = (
        model_ok
        &
        (
            rule_exact_name
            |
            rule_name_address
            |
            rule_combined
            |
            rule_exact_address
        )
    )


    # ========================================================
    # SELECT
    # ========================================================

    selected = chunk.loc[
        keep,
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ].drop_duplicates()


    # ========================================================
    # SAVE SELECTED PAIRS IMMEDIATELY
    # ========================================================

    if len(selected) > 0:

        selected.to_csv(
            FINAL_CANDIDATES,
            sep="\t",
            index=False,
            mode="a",
            header=not os.path.exists(
                FINAL_CANDIDATES
            )
        )

        kept += len(selected)


    processed += len(chunk)


    # ========================================================
    # PROGRESS
    # ========================================================

    if chunk_no % 5 == 0:

        print(
            f"Processed: {processed:,} | "
            f"Kept: {kept:,}"
        )


    # ========================================================
    # MEMORY CLEANUP
    # ========================================================

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
print("ENSEMBLE FILTER COMPLETED")
print("==============================================")

print(
    "Processed:",
    f"{processed:,}"
)

print(
    "Kept:",
    f"{kept:,}"
)


# ============================================================
# LOAD FINAL SELECTED PAIRS
# ============================================================

if os.path.exists(FINAL_CANDIDATES):

    selected_all = pd.read_csv(
        FINAL_CANDIDATES,
        sep="\t",
        dtype=str
    ).fillna("")

else:

    selected_all = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )


print(
    "Selected pair rows:",
    len(selected_all)
)


# ============================================================
# GROUP MATCHES
# ============================================================

if len(selected_all) > 0:

    selected_all = selected_all.drop_duplicates()

    grouped = (
        selected_all
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .agg(",".join)
        .reset_index()
    )

    grouped.columns = [
        "source1_entity_id",
        "matched_entity_ids"
    ]

else:

    grouped = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )


# ============================================================
# EVERY SOURCE 1 ENTITY
# ============================================================

result = pd.DataFrame(
    {
        "source1_entity_id":
            s1.index.to_numpy()
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


# ============================================================
# REMOVE DUPLICATE IDS
# ============================================================

def clean_ids(value):

    if not value:
        return ""

    values = str(value).split(",")

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
    FINAL_MATCHING,
    sep="\t",
    index=False
)


# ============================================================
# CREATE VALIDATOR CANDIDATE FILE
# ============================================================

candidate_final = result.rename(
    columns={
        "matched_entity_ids":
            "candidate_entity_ids"
    }
)

candidate_final.to_csv(
    FINAL_CANDIDATES,
    sep="\t",
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

nonempty = (
    result["matched_entity_ids"]
    .str.strip()
    != ""
).sum()

empty = len(result) - nonempty


print()
print("==============================================")
print("FINAL ENSEMBLE RESULT")
print("==============================================")

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

print(
    "Final kept pairs:",
    f"{kept:,}"
)

print()
print("Matching file:")
print(FINAL_MATCHING)

print()
print("Candidate file:")
print(FINAL_CANDIDATES)

print()
print("==============================================")