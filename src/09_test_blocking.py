import pandas as pd
import re

# ============================================================
# SETTINGS
# ============================================================

S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

S1_SAMPLE = 20000
CHUNK_SIZE = 250000


# ============================================================
# KEY FUNCTIONS
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


def first_two_tokens(x):
    if pd.isna(x):
        return ""
    parts = str(x).lower().split()
    return "".join(parts[:2]) if parts else ""


def address_digits(x):
    if pd.isna(x):
        return ""
    digits = re.findall(r"\d+", str(x))
    return "".join(digits)


# ============================================================
# LOAD SAMPLE SOURCE 1
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

s1_ids = set(s1["entity_id"])

print("S1 sample:", len(s1))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(s1_ids)
].copy()

print("S1 records with GT:", len(gt))


# ============================================================
# EXPLODE POSITIVE MATCHES
# ============================================================

positive_rows = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    ids = str(row.matched_entity_ids).strip()

    if not ids:
        continue

    for entity_id in ids.split(","):

        entity_id = entity_id.strip()

        if entity_id:
            positive_rows.append(
                (s1_id, entity_id)
            )

positives = pd.DataFrame(
    positive_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

print("Positive pairs:", len(positives))

positive_ids = set(
    positives["candidate_entity_id"]
)


# ============================================================
# GET POSITIVE SOURCE RECORDS
# ============================================================

def load_positive_records(file_path, wanted_ids, source_name):

    print()
    print("Reading", source_name, "...")

    pieces = []

    for chunk in pd.read_csv(
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
    ):

        found = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        if len(found):
            pieces.append(found)

        if sum(len(x) for x in pieces) >= len(wanted_ids):
            break

    if pieces:
        result = pd.concat(
            pieces,
            ignore_index=True
        ).fillna("")
    else:
        result = pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        )

    print(
        source_name,
        "positive records found:",
        len(result)
    )

    return result


s2_ids = {
    x for x in positive_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in positive_ids
    if x.startswith("S3-")
}

s2 = load_positive_records(
    S2_FILE,
    s2_ids,
    "Source 2"
)

s3 = load_positive_records(
    S3_FILE,
    s3_ids,
    "Source 3"
)


# ============================================================
# COMBINE POSITIVE RECORDS
# ============================================================

src = pd.concat(
    [s2, s3],
    ignore_index=True
).fillna("")


# ============================================================
# PREPARE S1 + SOURCE RECORDS
# ============================================================

s1_idx = s1.set_index("entity_id")


pair = positives.copy()

pair["country_s1"] = pair[
    "source1_entity_id"
].map(
    s1_idx["country"]
)

pair["name_s1"] = pair[
    "source1_entity_id"
].map(
    s1_idx["business_name"]
)

pair["address_s1"] = pair[
    "source1_entity_id"
].map(
    s1_idx["business_address"]
)


src_idx = src.set_index("entity_id")

pair["country_src"] = pair[
    "candidate_entity_id"
].map(
    src_idx["country"]
)

pair["name_src"] = pair[
    "candidate_entity_id"
].map(
    src_idx["business_name"]
)

pair["address_src"] = pair[
    "candidate_entity_id"
].map(
    src_idx["business_address"]
)


# ============================================================
# CREATE BLOCKING KEYS
# ============================================================

pair["k_exact_name_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["name_s1"].map(compact)
)

pair["k_exact_name_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["name_src"].map(compact)
)


pair["k_prefix8_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["name_s1"].map(compact).str[:8]
)

pair["k_prefix8_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["name_src"].map(compact).str[:8]
)


pair["k_token_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["name_s1"].map(first_token)
)

pair["k_token_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["name_src"].map(first_token)
)


pair["k_two_token_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["name_s1"].map(first_two_tokens)
)

pair["k_two_token_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["name_src"].map(first_two_tokens)
)


pair["k_addr_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["address_s1"].map(address_digits)
)

pair["k_addr_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["address_src"].map(address_digits)
)


pair["k_addrprefix_s1"] = (
    pair["country_s1"].str.lower().str.strip()
    + "|"
    + pair["address_s1"].map(compact).str[:8]
)

pair["k_addrprefix_src"] = (
    pair["country_src"].str.lower().str.strip()
    + "|"
    + pair["address_src"].map(compact).str[:8]
)


# ============================================================
# TEST RECALL
# ============================================================

total = len(pair)

print()
print("==============================================")
print("BLOCKING RECALL ON TRUE TRAINING MATCHES")
print("==============================================")
print("Total positive pairs:", total)
print()


def recall_for(s1_col, src_col, name):

    valid = (
        pair[s1_col].str.len() > 0
    ) & (
        pair[src_col].str.len() > 0
    )

    captured = (
        pair.loc[valid, s1_col].values
        ==
        pair.loc[valid, src_col].values
    ).sum()

    recall_all = captured / total if total else 0
    recall_valid = (
        captured / valid.sum()
        if valid.sum() else 0
    )

    print(
        f"{name:25s} "
        f"captured={captured:,} "
        f"recall_all={recall_all:.4f} "
        f"recall_valid={recall_valid:.4f}"
    )


recall_for(
    "k_exact_name_s1",
    "k_exact_name_src",
    "Exact name"
)

recall_for(
    "k_prefix8_s1",
    "k_prefix8_src",
    "Name prefix 8"
)

recall_for(
    "k_token_s1",
    "k_token_src",
    "First name token"
)

recall_for(
    "k_two_token_s1",
    "k_two_token_src",
    "First 2 name tokens"
)

recall_for(
    "k_addr_s1",
    "k_addr_src",
    "Address digits"
)

recall_for(
    "k_addrprefix_s1",
    "k_addrprefix_src",
    "Address prefix 8"
)


# ============================================================
# UNION RECALL
# ============================================================

checks = pd.DataFrame({
    "exact": (
        pair["k_exact_name_s1"]
        == pair["k_exact_name_src"]
    ),
    "prefix8": (
        pair["k_prefix8_s1"]
        == pair["k_prefix8_src"]
    ),
    "token": (
        pair["k_token_s1"]
        == pair["k_token_src"]
    ),
    "two_token": (
        pair["k_two_token_s1"]
        == pair["k_two_token_src"]
    ),
    "addr_digits": (
        pair["k_addr_s1"]
        == pair["k_addr_src"]
    ),
    "addr_prefix": (
        pair["k_addrprefix_s1"]
        == pair["k_addrprefix_src"]
    )
})

# Empty/blank keys should not count
for col in checks.columns:
    if col == "exact":
        checks[col] &= pair["k_exact_name_s1"].str.len() > 0
    elif col == "prefix8":
        checks[col] &= pair["k_prefix8_s1"].str.len() > 9
    elif col == "token":
        checks[col] &= pair["k_token_s1"].str.len() > 0
    elif col == "two_token":
        checks[col] &= pair["k_two_token_s1"].str.len() > 0
    elif col == "addr_digits":
        checks[col] &= pair["k_addr_s1"].str.len() > 1
    elif col == "addr_prefix":
        checks[col] &= pair["k_addrprefix_s1"].str.len() > 9


union = checks.any(axis=1)

print()
print(
    "UNION of all blocking keys:"
)
print(
    f"captured={union.sum():,} "
    f"recall={union.mean():.4f}"
)

print()
print("==============================================")
print("STEP 1 COMPLETED")
print("==============================================")