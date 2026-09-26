import pandas as pd
import re

S1_FILE = "dataset/processed/train_source1.tsv"
S2_FILE = "dataset/processed/train_source2.tsv"
S3_FILE = "dataset/processed/train_source3.tsv"
GT_FILE = "dataset/train/train_ground_truth.tsv"

S1_SAMPLE = 50000


def compact(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def first_token(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    return x.split()[0] if x else ""


def address_digits(x):
    if pd.isna(x):
        return ""
    return "".join(re.findall(r"\d+", str(x)))


print("Loading S1...")

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

print("S1:", len(s1))


print("Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(s1_ids)
]

positive_rows = []

for row in gt.itertuples(index=False):

    ids = str(row.matched_entity_ids).strip()

    if not ids:
        continue

    for cid in ids.split(","):
        cid = cid.strip()
        if cid:
            positive_rows.append(
                (row.source1_entity_id, cid)
            )

pos = pd.DataFrame(
    positive_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

print("Positive pairs:", len(pos))


# ------------------------------------------------------------
# Load only positive source records
# ------------------------------------------------------------

wanted = set(pos["candidate_entity_id"])


def load_records(path, wanted):

    parts = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=250000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):

        found = chunk[
            chunk["entity_id"].isin(wanted)
        ]

        if len(found):
            parts.append(found)

    if parts:
        return pd.concat(parts, ignore_index=True).fillna("")

    return pd.DataFrame()


s2 = load_records(S2_FILE, wanted)
s3 = load_records(S3_FILE, wanted)

src = pd.concat(
    [s2, s3],
    ignore_index=True
).fillna("")


# ------------------------------------------------------------
# Attach source data
# ------------------------------------------------------------

s1_idx = s1.set_index("entity_id")
src_idx = src.set_index("entity_id")

pos["s1_country"] = pos[
    "source1_entity_id"
].map(s1_idx["country"]).str.lower().str.strip()

pos["s1_name"] = pos[
    "source1_entity_id"
].map(s1_idx["business_name"])

pos["s1_addr"] = pos[
    "source1_entity_id"
].map(s1_idx["business_address"])

pos["src_country"] = pos[
    "candidate_entity_id"
].map(src_idx["country"]).str.lower().str.strip()

pos["src_name"] = pos[
    "candidate_entity_id"
].map(src_idx["business_name"])

pos["src_addr"] = pos[
    "candidate_entity_id"
].map(src_idx["business_address"])


# ------------------------------------------------------------
# Keys
# ------------------------------------------------------------

pos["name_exact"] = (
    pos["s1_country"] + "|" +
    pos["s1_name"].map(compact)
)

pos["name_token"] = (
    pos["s1_country"] + "|" +
    pos["s1_name"].map(first_token)
)

pos["addr_digits"] = (
    pos["s1_country"] + "|" +
    pos["s1_addr"].map(address_digits)
)


# ------------------------------------------------------------
# Full source frequency for each positive key
# ------------------------------------------------------------

def get_frequency(path, key_type):

    counter = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=250000,
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

        if key_type == "name_exact":
            keys = (
                country + "|" +
                chunk["business_name"].map(compact)
            )

        elif key_type == "name_token":
            keys = (
                country + "|" +
                chunk["business_name"].map(first_token)
            )

        else:
            keys = (
                country + "|" +
                chunk["business_address"].map(address_digits)
            )

        vc = keys.value_counts()

        for k, v in vc.items():

            if k:
                counter[k] = (
                    counter.get(k, 0)
                    + int(v)
                )

    return counter


print("\nBuilding frequencies...")

freq = {}

for key in [
    "name_exact",
    "name_token",
    "addr_digits"
]:

    print("Processing:", key)

    f2 = get_frequency(
        S2_FILE,
        key
    )

    f3 = get_frequency(
        S3_FILE,
        key
    )

    merged = f2.copy()

    for k, v in f3.items():
        merged[k] = merged.get(k, 0) + v

    freq[key] = merged


# ------------------------------------------------------------
# Print percentile distribution
# ------------------------------------------------------------

print()
print("==============================================")
print("POSITIVE KEY FREQUENCY DISTRIBUTION")
print("==============================================")


for key in [
    "name_exact",
    "name_token",
    "addr_digits"
]:

    values = []

    for k in pos[key]:

        if not k:
            continue

        values.append(
            freq[key].get(k, 999999999)
        )

    if not values:
        print("\n", key, ": no values")
        continue

    s = pd.Series(values)

    print()
    print(key)

    print(
        "min     =",
        int(s.min())
    )

    print(
        "median  =",
        int(s.median())
    )

    print(
        "p75     =",
        int(s.quantile(0.75))
    )

    print(
        "p90     =",
        int(s.quantile(0.90))
    )

    print(
        "p95     =",
        int(s.quantile(0.95))
    )

    print(
        "p99     =",
        int(s.quantile(0.99))
    )

    print(
        "max     =",
        int(s.max())
    )


# ------------------------------------------------------------
# Capture at useful thresholds
# ------------------------------------------------------------

print()
print("==============================================")
print("RECALL AT THRESHOLDS")
print("==============================================")

thresholds = [
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000
]

for key in [
    "name_exact",
    "name_token",
    "addr_digits"
]:

    print()
    print(key)

    vals = []

    for k in pos[key]:

        if not k:
            vals.append(999999999)
        else:
            vals.append(
                freq[key].get(k, 999999999)
            )

    vals = pd.Series(vals)

    for t in thresholds:

        captured = (
            vals <= t
        ).sum()

        recall = captured / len(vals)

        print(
            f"<= {t:6d} : "
            f"{captured:7d} "
            f"recall={recall:.4f}"
        )


print()
print("==============================================")
print("STEP 4 COMPLETED")
print("==============================================")