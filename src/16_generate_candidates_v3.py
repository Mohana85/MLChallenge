import pandas as pd
import os
import re
import gc

# ============================================================
# FILES
# ============================================================

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"

OUT_FILE = "output/candidate_pairs_v3.tsv"

CHUNK_SIZE = 200_000


# ============================================================
# HELPERS
# ============================================================

def compact(x):
    if pd.isna(x):
        return ""
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower()
    )


def first_token(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    return x.split()[0] if x else ""


def address_digits(x):
    if pd.isna(x):
        return ""
    return "".join(
        re.findall(r"\d+", str(x))
    )


# ============================================================
# LOAD S1
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

s1["country"] = (
    s1["country"]
    .str.lower()
    .str.strip()
)

s1["name_key"] = (
    s1["country"]
    + "|"
    + s1["business_name"].map(compact)
)

s1["combo_key"] = (
    s1["country"]
    + "|"
    + s1["business_address"].map(address_digits)
    + "|"
    + s1["business_name"].map(first_token)
)

# Empty keys should not generate candidates
s1_name = s1[
    s1["name_key"].str.len() > 1
][
    [
        "entity_id",
        "name_key"
    ]
].copy()

s1_combo = s1[
    (
        s1["business_address"].map(address_digits).str.len()
        >= 2
    )
    &
    (
        s1["business_name"].map(first_token).str.len()
        > 0
    )
][
    [
        "entity_id",
        "combo_key"
    ]
].copy()

print("Source 1:", len(s1))
print("Name keys:", len(s1_name))
print("Combo keys:", len(s1_combo))


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    "output",
    exist_ok=True
)

if os.path.exists(OUT_FILE):
    os.remove(OUT_FILE)

first_write = True
total_candidates = 0


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    file_path,
    source_name
):

    global first_write
    global total_candidates

    print()
    print("======================================")
    print("Processing", source_name)
    print("======================================")

    source_count = 0

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

        chunk["country"] = (
            chunk["country"]
            .str.lower()
            .str.strip()
        )

        chunk["name_key"] = (
            chunk["country"]
            + "|"
            + chunk["business_name"].map(compact)
        )

        chunk["combo_key"] = (
            chunk["country"]
            + "|"
            + chunk["business_address"].map(address_digits)
            + "|"
            + chunk["business_name"].map(first_token)
        )

        # ====================================================
        # EXACT NAME MATCHES
        # ====================================================

        name_part = chunk[
            chunk["name_key"].str.len() > 1
        ][
            [
                "entity_id",
                "name_key"
            ]
        ].merge(
            s1_name,
            on="name_key",
            how="inner"
        )

        name_part = name_part[
            [
                "entity_id_y",
                "entity_id_x"
            ]
        ]

        name_part.columns = [
            "source1_entity_id",
            "candidate_entity_id"
        ]

        # ====================================================
        # NAME + ADDRESS-DIGITS MATCHES
        # ====================================================

        combo_part = chunk[
            chunk["combo_key"].str.count(r"\|") >= 2
        ][
            [
                "entity_id",
                "combo_key"
            ]
        ].merge(
            s1_combo,
            on="combo_key",
            how="inner"
        )

        combo_part = combo_part[
            [
                "entity_id_y",
                "entity_id_x"
            ]
        ]

        combo_part.columns = [
            "source1_entity_id",
            "candidate_entity_id"
        ]

        # ====================================================
        # COMBINE
        # ====================================================

        candidates = pd.concat(
            [
                name_part,
                combo_part
            ],
            ignore_index=True
        ).drop_duplicates()

        # Keep only S2/S3 candidates
        candidates = candidates[
            ~candidates["candidate_entity_id"]
            .str.startswith("S1-")
        ]

        if len(candidates):

            candidates.to_csv(
                OUT_FILE,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            total_candidates += len(candidates)
            source_count += len(candidates)

        if chunk_no % 5 == 0:

            print(
                "Chunks:",
                chunk_no,
                "| Candidates:",
                f"{source_count:,}"
            )

        del chunk
        del name_part
        del combo_part
        del candidates

        gc.collect()

    print(
        source_name,
        "candidates:",
        f"{source_count:,}"
    )


# ============================================================
# RUN
# ============================================================

process_source(
    S2_FILE,
    "Source 2"
)

process_source(
    S3_FILE,
    "Source 3"
)


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("CANDIDATE GENERATION COMPLETED")
print("==============================================")

print(
    "Total candidate rows:",
    f"{total_candidates:,}"
)

print(
    "Output:",
    OUT_FILE
)

print("==============================================")