import pandas as pd
import re
import gc

S1_FILE = "dataset/processed/test_source1.tsv"
S2_FILE = "dataset/processed/test_source2.tsv"
S3_FILE = "dataset/processed/test_source3.tsv"


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

s1["country"] = s1["country"].str.lower().str.strip()

s1["exact_name"] = (
    s1["country"] + "|" +
    s1["business_name"].map(compact)
)

s1["prefix8"] = (
    s1["country"] + "|" +
    s1["business_name"].map(compact).str[:8]
)

s1["first_token"] = (
    s1["country"] + "|" +
    s1["business_name"].map(first_token)
)

s1["addr_digits"] = (
    s1["country"] + "|" +
    s1["business_address"].map(address_digits)
)

s1["addr_prefix8"] = (
    s1["country"] + "|" +
    s1["business_address"].map(compact).str[:8]
)


KEYS = [
    "exact_name",
    "prefix8",
    "first_token",
    "addr_digits",
    "addr_prefix8"
]


print("S1 rows:", len(s1))


# ============================================================
# PROCESS SOURCE
# ============================================================

def estimate(source_file, source_name):

    print()
    print("==============================================")
    print("PROCESSING", source_name)
    print("==============================================")

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

    src["country"] = (
        src["country"]
        .str.lower()
        .str.strip()
    )

    src["exact_name"] = (
        src["country"] + "|" +
        src["business_name"].map(compact)
    )

    src["prefix8"] = (
        src["country"] + "|" +
        src["business_name"].map(compact).str[:8]
    )

    src["first_token"] = (
        src["country"] + "|" +
        src["business_name"].map(first_token)
    )

    src["addr_digits"] = (
        src["country"] + "|" +
        src["business_address"].map(address_digits)
    )

    src["addr_prefix8"] = (
        src["country"] + "|" +
        src["business_address"].map(compact).str[:8]
    )

    total = 0

    print("Source rows:", len(src))
    print()

    for key in KEYS:

        print("Checking:", key)

        s1_counts = (
            s1[key]
            .value_counts()
        )

        src_counts = (
            src[key]
            .value_counts()
        )

        common = pd.DataFrame({
            "s1_count": s1_counts,
            "src_count": src_counts
        }).dropna()

        # Empty keys should never generate candidates
        common = common[
            common.index.str.len() > 1
        ]

        pair_count = (
            common["s1_count"] *
            common["src_count"]
        ).sum()

        print(
            "  Estimated pairs:",
            f"{int(pair_count):,}"
        )

        total += int(pair_count)

        del s1_counts
        del src_counts
        del common
        gc.collect()

    print()
    print(
        source_name,
        "sum of key candidates:",
        f"{total:,}"
    )

    del src
    gc.collect()

    return total


# ============================================================
# RUN
# ============================================================

s2_total = estimate(
    S2_FILE,
    "Source 2"
)

s3_total = estimate(
    S3_FILE,
    "Source 3"
)


# ============================================================
# FINAL
# ============================================================

print()
print("==============================================")
print("ESTIMATION COMPLETED")
print("==============================================")

print(
    "S2 estimated candidates:",
    f"{s2_total:,}"
)

print(
    "S3 estimated candidates:",
    f"{s3_total:,}"
)

print(
    "Upper-bound total:",
    f"{s2_total + s3_total:,}"
)

print()
print(
    "NOTE: This is an upper bound because "
    "the same pair can be captured by multiple keys."
)

print("==============================================")