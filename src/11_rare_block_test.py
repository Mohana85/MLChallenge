import pandas as pd
import re
import gc

# ============================================================
# SETTINGS
# ============================================================

S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

S1_SAMPLE = 50000
CHUNK_SIZE = 250000

# Test different maximum block sizes
THRESHOLDS = [5, 10, 25, 50, 100]


# ============================================================
# FUNCTIONS
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


def first_two_chars_name(x):
    if pd.isna(x):
        return ""
    return compact(x)[:2]


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
    s1["country"].str.lower().str.strip()
)

s1["name_compact"] = (
    s1["business_name"].map(compact)
)

s1["first_token"] = (
    s1["business_name"].map(first_token)
)

s1["first2"] = (
    s1["business_name"].map(first_two_chars_name)
)

s1["addr_digits"] = (
    s1["business_address"].map(address_digits)
)

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
    gt["source1_entity_id"].isin(
        set(s1["entity_id"])
    )
]

# ============================================================
# EXPLODE POSITIVES
# ============================================================

positive_rows = []

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    ids = str(row.matched_entity_ids).strip()

    if not ids:
        continue

    for cid in ids.split(","):

        cid = cid.strip()

        if cid:
            positive_rows.append(
                (s1_id, cid)
            )

positives = pd.DataFrame(
    positive_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

print("Positive pairs:", len(positives))


# ============================================================
# GET POSITIVE SOURCE IDs
# ============================================================

positive_ids = set(
    positives["candidate_entity_id"]
)

s2_ids = {
    x for x in positive_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in positive_ids
    if x.startswith("S3-")
}


# ============================================================
# LOAD ONLY POSITIVE SOURCE RECORDS
# ============================================================

def load_positive_records(
    file_path,
    wanted_ids,
    name
):

    print("Reading positive", name, "records...")

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
        return pd.concat(
            pieces,
            ignore_index=True
        ).fillna("")

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


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

src = pd.concat(
    [s2, s3],
    ignore_index=True
).fillna("")

print("Positive source records:", len(src))


# ============================================================
# PREPARE POSITIVE PAIRS
# ============================================================

s1_idx = s1.set_index("entity_id")
src_idx = src.set_index("entity_id")

pair = positives.copy()

pair["s1_country"] = pair[
    "source1_entity_id"
].map(s1_idx["country"])

pair["s1_name"] = pair[
    "source1_entity_id"
].map(s1_idx["business_name"])

pair["s1_address"] = pair[
    "source1_entity_id"
].map(s1_idx["business_address"])

pair["src_country"] = pair[
    "candidate_entity_id"
].map(src_idx["country"])

pair["src_name"] = pair[
    "candidate_entity_id"
].map(src_idx["business_name"])

pair["src_address"] = pair[
    "candidate_entity_id"
].map(src_idx["business_address"])


# ============================================================
# BLOCKING KEYS
# ============================================================

pair["first_token_s1"] = (
    pair["s1_country"]
    + "|"
    + pair["s1_name"].map(first_token)
)

pair["first_token_src"] = (
    pair["src_country"]
    + "|"
    + pair["src_name"].map(first_token)
)


pair["first2_s1"] = (
    pair["s1_country"]
    + "|"
    + pair["s1_name"].map(first_two_chars_name)
)

pair["first2_src"] = (
    pair["src_country"]
    + "|"
    + pair["src_name"].map(first_two_chars_name)
)


pair["addr_digits_s1"] = (
    pair["s1_country"]
    + "|"
    + pair["s1_address"].map(address_digits)
)

pair["addr_digits_src"] = (
    pair["src_country"]
    + "|"
    + pair["src_address"].map(address_digits)
)


pair["exact_name_s1"] = (
    pair["s1_country"]
    + "|"
    + pair["s1_name"].map(compact)
)

pair["exact_name_src"] = (
    pair["src_country"]
    + "|"
    + pair["src_name"].map(compact)
)


# ============================================================
# SOURCE FREQUENCY TABLES
# ============================================================

print()
print("Creating source block frequencies...")


def make_freq(source):

    country = source["country"].str.lower().str.strip()

    result = {}

    result["first_token"] = (
        country
        + "|"
        + source["business_name"].map(first_token)
    ).value_counts()

    result["first2"] = (
        country
        + "|"
        + source["business_name"].map(first_two_chars_name)
    ).value_counts()

    result["addr_digits"] = (
        country
        + "|"
        + source["business_address"].map(address_digits)
    ).value_counts()

    result["exact_name"] = (
        country
        + "|"
        + source["business_name"].map(compact)
    ).value_counts()

    return result


# NOTE:
# Frequency tables must use FULL source files.
# Load in chunks and build counters.

def full_frequency(file_path):

    freq = {
        "first_token": {},
        "first2": {},
        "addr_digits": {},
        "exact_name": {}
    }

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        usecols=[
            "business_name",
            "business_address",
            "country"
        ]
    ):

        chunk = chunk.fillna("")

        country = (
            chunk["country"]
            .str.lower()
            .str.strip()
        )

        keys = {
            "first_token":
                country + "|" +
                chunk["business_name"].map(first_token),

            "first2":
                country + "|" +
                chunk["business_name"].map(first_two_chars_name),

            "addr_digits":
                country + "|" +
                chunk["business_address"].map(address_digits),

            "exact_name":
                country + "|" +
                chunk["business_name"].map(compact)
        }

        for key_name, values in keys.items():

            counts = values.value_counts()

            for key, count in counts.items():

                if key:
                    freq[key_name][key] = (
                        freq[key_name].get(key, 0)
                        + int(count)
                    )

    return {
        name: pd.Series(values)
        for name, values in freq.items()
    }


print("Source 2 frequencies...")
freq2 = full_frequency(S2_FILE)

print("Source 3 frequencies...")
freq3 = full_frequency(S3_FILE)


# Combine frequencies because candidate can come from S2 OR S3
freq = {}

for key in freq2:

    freq[key] = (
        freq2[key]
        .add(freq3[key], fill_value=0)
    )


# ============================================================
# EVALUATE EACH THRESHOLD
# ============================================================

print()
print("===================================================")
print("RARE BLOCKING THRESHOLD RESULTS")
print("===================================================")

total_positive = len(pair)


for threshold in THRESHOLDS:

    captured = set()

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    for i in pair.index:

        k1 = pair.at[i, "exact_name_s1"]
        k2 = pair.at[i, "exact_name_src"]

        if (
            k1
            and k1 == k2
            and freq["exact_name"].get(k1, 999999)
            <= threshold
        ):
            captured.add(i)


    # --------------------------------------------------------
    # First token
    # --------------------------------------------------------

    for i in pair.index:

        k1 = pair.at[i, "first_token_s1"]
        k2 = pair.at[i, "first_token_src"]

        if (
            k1
            and k1 == k2
            and freq["first_token"].get(k1, 999999)
            <= threshold
        ):
            captured.add(i)


    # --------------------------------------------------------
    # Address digits
    # --------------------------------------------------------

    for i in pair.index:

        k1 = pair.at[i, "addr_digits_s1"]
        k2 = pair.at[i, "addr_digits_src"]

        if (
            len(k1.split("|")[-1]) >= 2
            and k1 == k2
            and freq["addr_digits"].get(k1, 999999)
            <= threshold
        ):
            captured.add(i)


    # --------------------------------------------------------
    # First 2 name chars
    # --------------------------------------------------------

    for i in pair.index:

        k1 = pair.at[i, "first2_s1"]
        k2 = pair.at[i, "first2_src"]

        if (
            len(k1.split("|")[-1]) == 2
            and k1 == k2
            and freq["first2"].get(k1, 999999)
            <= threshold
        ):
            captured.add(i)


    recall = (
        len(captured) / total_positive
        if total_positive else 0
    )

    print(
        f"Threshold <= {threshold:3d} : "
        f"captured={len(captured):,} "
        f"recall={recall:.4f}"
    )

    gc.collect()


print()
print("===================================================")
print("STEP 3 COMPLETED")
print("===================================================")