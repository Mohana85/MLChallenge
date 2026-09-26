import pandas as pd
import os
import re

# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "output/candidate_pairs.tsv"
OUTPUT_FILE = "output/filtered_candidate_pairs.tsv"

CHUNK_SIZE = 100_000

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)

    return text


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    "dataset/processed/test_source1.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1["name_norm"] = s1["business_name"].fillna("").map(normalize_text)

s1_lookup = s1.set_index("entity_id")

print("Source 1 loaded:", len(s1))


# ============================================================
# LOAD SOURCE 2 + SOURCE 3
# ============================================================

print("Loading Source 2...")

s2 = pd.read_csv(
    "dataset/processed/test_source2.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 2 loaded:", len(s2))


print("Loading Source 3...")

s3 = pd.read_csv(
    "dataset/processed/test_source3.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Source 3 loaded:", len(s3))


# ============================================================
# CREATE SMALL LOOKUPS
# ============================================================

print("Creating lookups...")

s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")


# ============================================================
# OUTPUT SETUP
# ============================================================

os.makedirs("output", exist_ok=True)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_write = True

total = 0
kept = 0


# ============================================================
# PROCESS CANDIDATES IN CHUNKS
# ============================================================

print("\nStarting candidate filtering...")
print("Original candidates: ~86.6 million")
print()

reader = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
)


for chunk_no, chunk in enumerate(reader, start=1):

    total += len(chunk)

    keep_rows = []

    for row in chunk.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id

        # ----------------------------------------------------
        # Get Source 1 record
        # ----------------------------------------------------

        if s1_id not in s1_lookup.index:
            continue

        s1_row = s1_lookup.loc[s1_id]

        s1_country = str(s1_row["country"]).lower()
        s1_name = str(s1_row["name_norm"])


        # ----------------------------------------------------
        # Determine S2 or S3
        # ----------------------------------------------------

        if str(candidate_id).startswith("S2-"):

            if candidate_id not in s2_lookup.index:
                continue

            cand = s2_lookup.loc[candidate_id]

        else:

            if candidate_id not in s3_lookup.index:
                continue

            cand = s3_lookup.loc[candidate_id]


        # ----------------------------------------------------
        # COUNTRY FILTER
        # ----------------------------------------------------

        candidate_country = str(cand["country"]).lower()

        if s1_country != candidate_country:
            continue


        # ----------------------------------------------------
        # CHEAP NAME FILTER
        # ----------------------------------------------------

        candidate_name = normalize_text(cand["business_name"])

        # If both names exist, require some basic overlap.
        if s1_name and candidate_name:

            # Exact normalized name
            if s1_name == candidate_name:
                keep_rows.append(
                    (s1_id, candidate_id)
                )
                continue

            # First 4 characters
            if len(s1_name) >= 4 and len(candidate_name) >= 4:

                if s1_name[:4] == candidate_name[:4]:
                    keep_rows.append(
                        (s1_id, candidate_id)
                    )
                    continue

            # First 2 characters
            if len(s1_name) >= 2 and len(candidate_name) >= 2:

                if s1_name[:2] == candidate_name[:2]:
                    keep_rows.append(
                        (s1_id, candidate_id)
                    )
                    continue

        # ----------------------------------------------------
        # NAME MISSING
        # ----------------------------------------------------

        # If candidate name is missing, keep it only when
        # country matches. This protects possible address-based
        # matches.
        elif not candidate_name:

            keep_rows.append(
                (s1_id, candidate_id)
            )


    # ========================================================
    # SAVE THIS CHUNK
    # ========================================================

    if keep_rows:

        result = pd.DataFrame(
            keep_rows,
            columns=[
                "source1_entity_id",
                "candidate_entity_id"
            ]
        )

        result.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        kept += len(result)


    # ========================================================
    # PROGRESS
    # ========================================================

    if chunk_no % 5 == 0:

        percentage = (total / 86626818) * 100

        print(
            f"Processed: {total:,} | "
            f"Kept: {kept:,} | "
            f"Progress: {percentage:.2f}%"
        )


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("FILTERING COMPLETED")
print("======================================")

print("Original candidates:", f"{total:,}")
print("Filtered candidates:", f"{kept:,}")
print("Output:", OUTPUT_FILE)

if total > 0:
    reduction = 100 * (1 - kept / total)
    print(f"Reduction: {reduction:.2f}%")

print("======================================")