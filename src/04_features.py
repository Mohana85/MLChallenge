import pandas as pd
from pathlib import Path
from rapidfuzz.fuzz import ratio, token_set_ratio

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "processed"

PAIR_FILE = DATA_DIR / "training_pairs.tsv"

OUTPUT_FILE = DATA_DIR / "training_features.tsv"


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

print("Loading training pairs...", flush=True)

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t"
)

print(
    "Training pairs:",
    len(pairs),
    flush=True
)


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("Loading Source 1...", flush=True)

s1 = pd.read_csv(
    DATA_DIR / "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "business_address_clean",
        "country_clean"
    ]
)

print("Loading Source 2...", flush=True)

s2 = pd.read_csv(
    DATA_DIR / "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "business_address_clean",
        "country_clean"
    ]
)

print("Loading Source 3...", flush=True)

s3 = pd.read_csv(
    DATA_DIR / "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name_clean",
        "business_address_clean",
        "country_clean"
    ]
)


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

other = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3


# ============================================================
# CREATE LOOKUP TABLES
# ============================================================

print("Creating lookup tables...", flush=True)

s1_lookup = s1.set_index("entity_id").to_dict("index")

other_lookup = other.set_index("entity_id").to_dict("index")


# ============================================================
# FEATURE FUNCTION
# ============================================================

def calculate_features(source1_id, candidate_id):

    a = s1_lookup.get(source1_id)

    b = other_lookup.get(candidate_id)

    if a is None or b is None:
        return [
            0,
            0,
            0,
            0,
            0
        ]


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name1 = str(a["business_name_clean"])
    name2 = str(b["business_name_clean"])

    name_ratio = ratio(
        name1,
        name2
    ) / 100.0

    name_token = token_set_ratio(
        name1,
        name2
    ) / 100.0


    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    address1 = str(a["business_address_clean"])
    address2 = str(b["business_address_clean"])

    address_ratio = ratio(
        address1,
        address2
    ) / 100.0

    address_token = token_set_ratio(
        address1,
        address2
    ) / 100.0


    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_same = int(
        str(a["country_clean"]).lower()
        ==
        str(b["country_clean"]).lower()
    )


    return [
        name_ratio,
        name_token,
        address_ratio,
        address_token,
        country_same
    ]


# ============================================================
# CALCULATE FEATURES
# ============================================================

print("Calculating features...", flush=True)

features = []

for i, row in enumerate(
    pairs.itertuples(index=False),
    start=1
):

    values = calculate_features(
        row.source1_entity_id,
        row.candidate_entity_id
    )

    features.append(
        [
            row.source1_entity_id,
            row.candidate_entity_id,
            *values,
            row.label
        ]
    )

    if i % 10000 == 0:
        print(
            "Processed:",
            i,
            "/",
            len(pairs),
            flush=True
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(
    features,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_same",
        "label"
    ]
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("FEATURE EXTRACTION COMPLETED")
print("==========================================")

print(
    "Total rows:",
    len(result)
)

print(
    "Features:",
    [
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_same"
    ]
)

print(
    "Saved:",
    OUTPUT_FILE
)

print("==========================================")