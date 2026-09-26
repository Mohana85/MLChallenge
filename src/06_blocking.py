import sqlite3
import pandas as pd
import os
import gc

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"

DB_FILE = "dataset/processed/candidates.db"
OUTPUT_FILE = "output/candidate_pairs.tsv"

S1_CHUNK = 5000
S2_S3_CHUNK = 100000

# Keep this reasonably small for a fast baseline
MAX_CANDIDATES = 50


def get_block_key(name, country):
    name = "" if pd.isna(name) else str(name)
    country = "" if pd.isna(country) else str(country)

    name = name.strip()

    if name:
        first_char = name[0]
    else:
        first_char = "#"

    return country + "_" + first_char


# =========================================================
# REMOVE OLD DATABASE / OUTPUT
# =========================================================

os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


# =========================================================
# CREATE SQLITE DATABASE
# =========================================================

print("Creating SQLite database...")

conn = sqlite3.connect(DB_FILE)

conn.execute("""
CREATE TABLE candidates (
    block TEXT,
    entity_id TEXT
)
""")

conn.execute("""
CREATE INDEX idx_block
ON candidates(block)
""")

conn.commit()


# =========================================================
# LOAD SOURCE 2 INTO SQLITE
# =========================================================

print("\nLoading Source 2 into database...")

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "country"],
    chunksize=S2_S3_CHUNK
):

    chunk["block"] = [
        get_block_key(n, c)
        for n, c in zip(
            chunk["business_name"],
            chunk["country"]
        )
    ]

    rows = list(
        zip(
            chunk["block"],
            chunk["entity_id"]
        )
    )

    conn.executemany(
        "INSERT INTO candidates(block, entity_id) VALUES (?, ?)",
        rows
    )

    conn.commit()

    del chunk
    del rows

    gc.collect()

    print("  Source 2 chunk inserted")


# =========================================================
# LOAD SOURCE 3 INTO SQLITE
# =========================================================

print("\nLoading Source 3 into database...")

for chunk in pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "business_name", "country"],
    chunksize=S2_S3_CHUNK
):

    chunk["block"] = [
        get_block_key(n, c)
        for n, c in zip(
            chunk["business_name"],
            chunk["country"]
        )
    ]

    rows = list(
        zip(
            chunk["block"],
            chunk["entity_id"]
        )
    )

    conn.executemany(
        "INSERT INTO candidates(block, entity_id) VALUES (?, ?)",
        rows
    )

    conn.commit()

    del chunk
    del rows

    gc.collect()

    print("  Source 3 chunk inserted")


print("\nDatabase creation completed.")


# =========================================================
# PREPARE OUTPUT
# =========================================================

out_file = open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
)

out_file.write(
    "source1_entity_id\tcandidate_entity_id\n"
)


# =========================================================
# PROCESS SOURCE 1
# =========================================================

print("\nGenerating candidate pairs...")


processed = 0

for s1_chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ],
    chunksize=S1_CHUNK
):

    for _, row in s1_chunk.iterrows():

        s1_id = row["entity_id"]

        block = get_block_key(
            row["business_name"],
            row["country"]
        )

        result = conn.execute(
            """
            SELECT entity_id
            FROM candidates
            WHERE block = ?
            LIMIT ?
            """,
            (block, MAX_CANDIDATES)
        ).fetchall()

        for (candidate_id,) in result:

            out_file.write(
                s1_id
                + "\t"
                + candidate_id
                + "\n"
            )

        processed += 1

    print(
        f"Processed: {processed:,} / 1,732,544"
    )

    del s1_chunk
    gc.collect()


# =========================================================
# CLOSE
# =========================================================

out_file.close()
conn.close()

gc.collect()

print("\n======================================")
print("BLOCKING COMPLETED")
print("======================================")
print("Candidate file:")
print(OUTPUT_FILE)

print("\nNext step:")
print("python3 src/07_prediction.py")