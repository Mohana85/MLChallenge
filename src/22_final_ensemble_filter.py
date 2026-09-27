import os
import numpy as np
import pandas as pd
import joblib

from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

CANDIDATE_FILE = "output/candidate_pairs_v3.tsv"

TEST_S1 = "dataset/processed/test_source1.tsv"
TEST_S2 = "dataset/processed/test_source2.tsv"
TEST_S3 = "dataset/processed/test_source3.tsv"

TRAIN_FEATURE_FILE = (
    "dataset/processed/training_features_hard.tsv"
)

MODEL_FILE = "model/entity_match_best.pkl"
THRESHOLD_FILE = "model/best_threshold.txt"

MATCHING_OUT = "output/matching_results.tsv"
CANDIDATE_OUT = "output/candidate_pairs.tsv"

TEMP_MATCH_OUT = (
    "output/_precision_matches_tmp.tsv"
)

CHUNK_SIZE = 250_000


# ============================================================
# PRECISION-FIRST SETTINGS
# ============================================================

# Learned threshold was 0.63, but test inference was
# accepting almost every candidate.
#
# Therefore we use a safer minimum threshold.

MIN_MODEL_THRESHOLD = 0.90

# Final hybrid score is on 0-100 scale.
MIN_FINAL_SCORE = 84.0

# Candidate must have meaningful similarity on both fields.
MIN_NAME = 78.0
MIN_ADDRESS = 45.0

# Maximum observed training matches for one S1 was 11.
MAX_MATCHES_PER_ENTITY = 11


# ============================================================
# LOAD PROCESSED FILE
# ============================================================

def load_processed(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # ID
    if "entity_id" in df.columns:
        id_col = "entity_id"
    elif "id" in df.columns:
        id_col = "id"
    else:
        raise ValueError(
            f"ID column not found in {path}\n"
            f"Columns: {list(df.columns)}"
        )

    # NAME
    name_col = None

    for c in [
        "name_norm",
        "business_name_norm",
        "business_name"
    ]:

        if c in df.columns:
            name_col = c
            break

    if name_col is None:
        raise ValueError(
            f"Name column not found in {path}"
        )

    # ADDRESS
    address_col = None

    for c in [
        "address_norm",
        "business_address_norm",
        "business_address"
    ]:

        if c in df.columns:
            address_col = c
            break

    if address_col is None:
        raise ValueError(
            f"Address column not found in {path}"
        )

    # COUNTRY
    country_col = None

    for c in [
        "country_norm",
        "country"
    ]:

        if c in df.columns:
            country_col = c
            break

    if country_col is None:
        raise ValueError(
            f"Country column not found in {path}"
        )

    result = df[
        [
            id_col,
            name_col,
            address_col,
            country_col
        ]
    ].copy()

    result.columns = [
        "entity_id",
        "name",
        "address",
        "country"
    ]

    result["entity_id"] = (
        result["entity_id"]
        .astype(str)
    )

    result["name"] = (
        result["name"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    result["address"] = (
        result["address"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    result["country"] = (
        result["country"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return result


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING XGBOOST MODEL")
print("=" * 70)

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

feature_cols = bundle[
    "feature_cols"
]

with open(
    THRESHOLD_FILE,
    "r"
) as f:

    learned_threshold = float(
        f.read().strip()
    )


MODEL_THRESHOLD = max(
    MIN_MODEL_THRESHOLD,
    learned_threshold
)

print(
    "Learned threshold:",
    learned_threshold
)

print(
    "Inference threshold:",
    MODEL_THRESHOLD
)

print(
    "Features:",
    feature_cols
)


# ============================================================
# LOAD TRAINING FEATURE STATISTICS
#
# This protects against accidental feature scale mismatches.
# ============================================================

print("\nLoading training feature statistics...")

train_features = pd.read_csv(
    TRAIN_FEATURE_FILE,
    sep="\t",
    usecols=feature_cols
)

train_max = (
    train_features
    .max(numeric_only=True)
)

train_min = (
    train_features
    .min(numeric_only=True)
)

del train_features

print("Training feature statistics loaded.")


# ============================================================
# SOURCE 1
# ============================================================

print("\nLoading Source 1...")

s1 = load_processed(
    TEST_S1
)

s1 = s1.set_index(
    "entity_id",
    drop=False
)

print(
    "S1 rows:",
    len(s1)
)


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(left, right):

    n = len(left)

    ln = (
        left["name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    rn = (
        right["name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    la = (
        left["address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    ra = (
        right["address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    lc = (
        left["country"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    rc = (
        right["country"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    name_ratio = np.empty(
        n,
        dtype=np.float32
    )

    name_token_set = np.empty(
        n,
        dtype=np.float32
    )

    name_token_sort = np.empty(
        n,
        dtype=np.float32
    )

    address_ratio = np.empty(
        n,
        dtype=np.float32
    )

    address_token_set = np.empty(
        n,
        dtype=np.float32
    )

    address_token_sort = np.empty(
        n,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # FUZZY FEATURES
    # --------------------------------------------------------

    for i in range(n):

        name_ratio[i] = fuzz.ratio(
            ln[i],
            rn[i]
        )

        name_token_set[i] = fuzz.token_set_ratio(
            ln[i],
            rn[i]
        )

        name_token_sort[i] = fuzz.token_sort_ratio(
            ln[i],
            rn[i]
        )

        address_ratio[i] = fuzz.ratio(
            la[i],
            ra[i]
        )

        address_token_set[i] = fuzz.token_set_ratio(
            la[i],
            ra[i]
        )

        address_token_sort[i] = fuzz.token_sort_ratio(
            la[i],
            ra[i]
        )


    # --------------------------------------------------------
    # SAME COUNTRY
    #
    # IMPORTANT: 0 / 1
    # --------------------------------------------------------

    country_same = np.array(
        [
            1.0
            if a == b
            else 0.0
            for a, b in zip(lc, rc)
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    name_exact = np.array(
        [
            1.0
            if a == b
            else 0.0
            for a, b in zip(ln, rn)
        ],
        dtype=np.float32
    )


    address_exact = np.array(
        [
            1.0
            if a == b
            else 0.0
            for a, b in zip(la, ra)
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # LENGTH DIFFERENCE
    # --------------------------------------------------------

    name_length_diff = np.array(
        [
            abs(len(a) - len(b))
            for a, b in zip(ln, rn)
        ],
        dtype=np.float32
    )


    address_length_diff = np.array(
        [
            abs(len(a) - len(b))
            for a, b in zip(la, ra)
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    name_best = np.maximum.reduce(
        [
            name_ratio,
            name_token_set,
            name_token_sort
        ]
    )


    address_best = np.maximum.reduce(
        [
            address_ratio,
            address_token_set,
            address_token_sort
        ]
    )


    # --------------------------------------------------------
    # COMBINED
    #
    # Initial formula
    # --------------------------------------------------------

    combined_score = (
        0.60 * name_best
        +
        0.40 * address_best
    )


    features = pd.DataFrame({

        "name_ratio": name_ratio,

        "name_token_set": name_token_set,

        "name_token_sort": name_token_sort,

        "address_ratio": address_ratio,

        "address_token_set":
            address_token_set,

        "address_token_sort":
            address_token_sort,

        "country_same": country_same,

        "name_exact": name_exact,

        "address_exact":
            address_exact,

        "name_length_diff":
            name_length_diff,

        "address_length_diff":
            address_length_diff,

        "name_best": name_best,

        "address_best":
            address_best,

        "combined_score":
            combined_score
    })


    # ========================================================
    # FEATURE SCALE SAFETY
    #
    # If training uses 0-1 and inference accidentally creates
    # 0-100, automatically align it.
    # ========================================================

    for col in feature_cols:

        tr_max = train_max.get(
            col,
            np.nan
        )

        tr_min = train_min.get(
            col,
            np.nan
        )

        if not np.isfinite(tr_max):
            continue

        current_max = (
            features[col]
            .max()
        )

        current_min = (
            features[col]
            .min()
        )


        # 0-100 -> 0-1
        if (
            tr_max <= 1.5
            and current_max > 2.0
        ):

            features[col] = (
                features[col]
                / 100.0
            )


        # 0-1 -> 0-100
        elif (
            tr_max > 2.0
            and current_max <= 1.5
            and current_min >= 0
        ):

            features[col] = (
                features[col]
                * 100.0
            )


    return features


# ============================================================
# PRECISION RULES
# ============================================================

def precision_gate(row, probability):

    name_best = float(
        row["name_best"]
    )

    address_best = float(
        row["address_best"]
    )

    name_exact = (
        float(row["name_exact"])
        == 1.0
    )

    address_exact = (
        float(row["address_exact"])
        == 1.0
    )

    country_same = (
        float(row["country_same"])
        == 1.0
    )


    # Country must agree.
    if not country_same:
        return False


    # --------------------------------------------------------
    # Strong exact-name + address support
    # --------------------------------------------------------

    if name_exact:

        if address_exact:
            return True

        if address_best >= 70:
            return True


    # --------------------------------------------------------
    # Strong exact-address + good name
    # --------------------------------------------------------

    if address_exact:

        if name_best >= 75:
            return True


    # --------------------------------------------------------
    # Very strong fuzzy agreement
    # --------------------------------------------------------

    if (
        name_best >= 92
        and
        address_best >= 65
        and
        probability >= 0.85
    ):
        return True


    # --------------------------------------------------------
    # Strong agreement on both
    # --------------------------------------------------------

    if (
        name_best >= 88
        and
        address_best >= 78
        and
        probability >= 0.85
    ):
        return True


    # --------------------------------------------------------
    # Address stronger than name
    # --------------------------------------------------------

    if (
        name_best >= 80
        and
        address_best >= 90
        and
        probability >= 0.85
    ):
        return True


    # --------------------------------------------------------
    # General model-based gate
    # --------------------------------------------------------

    if (
        probability >= MODEL_THRESHOLD
        and
        name_best >= MIN_NAME
        and
        address_best >= MIN_ADDRESS
    ):
        return True


    return False


# ============================================================
# HYBRID SCORE
# ============================================================

def calculate_final_score(
    probability,
    name_best,
    address_best,
    name_exact,
    address_exact
):

    similarity = (
        0.60 * name_best
        +
        0.40 * address_best
    )


    # Model is important, but similarity gives
    # protection against probability over-confidence.

    score = (
        0.65 * probability * 100.0
        +
        0.35 * similarity
    )


    # Small bonuses for exact agreement.

    if name_exact:
        score += 2.0

    if address_exact:
        score += 3.0


    return score


# ============================================================
# TEMP MATCH FILE
# ============================================================

if os.path.exists(
    TEMP_MATCH_OUT
):

    os.remove(
        TEMP_MATCH_OUT
    )


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    source_file,
    prefix
):

    print("\n")
    print("=" * 70)
    print(
        "PROCESSING",
        prefix
    )
    print("=" * 70)


    source = load_processed(
        source_file
    )

    source = source.set_index(
        "entity_id",
        drop=False
    )


    print(
        prefix,
        "rows:",
        len(source)
    )


    processed = 0
    gated = 0
    final_kept = 0


    # --------------------------------------------------------
    # Current S1 buffer
    #
    # candidate_pairs_v3 is generated S1-by-S1.
    # We use that ordering for entity-level filtering.
    # --------------------------------------------------------

    current_sid = None
    current_records = []


    # --------------------------------------------------------
    # FINALIZE ONE S1
    # --------------------------------------------------------

    def finalize_entity(
        sid,
        records
    ):

        nonlocal final_kept


        if sid is None or not records:
            return


        # ----------------------------------------------------
        # Records:
        #
        # (
        # candidate_id,
        # probability,
        # name_best,
        # address_best,
        # name_exact,
        # address_exact,
        # final_score
        # )
        # ----------------------------------------------------


        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        records.sort(
            key=lambda x: x[6],
            reverse=True
        )


        # ----------------------------------------------------
        # Pareto filtering
        #
        # A candidate is removed if another candidate is
        # at least as strong on:
        #
        # name
        # address
        # model probability
        #
        # and strictly stronger on one.
        # ----------------------------------------------------

        pareto = []


        for i, rec in enumerate(records):

            dominated = False


            for j, other in enumerate(records):

                if i == j:
                    continue


                other_better_or_equal = (
                    other[1] >= rec[1]
                    and
                    other[2] >= rec[2]
                    and
                    other[3] >= rec[3]
                )


                strictly_better = (
                    other[1] > rec[1]
                    or
                    other[2] > rec[2]
                    or
                    other[3] > rec[3]
                )


                if (
                    other_better_or_equal
                    and
                    strictly_better
                ):

                    dominated = True
                    break


            if not dominated:

                pareto.append(rec)


        # ----------------------------------------------------
        # Precision gate
        # ----------------------------------------------------

        selected = []


        for rec in pareto:

            candidate_id = rec[0]

            probability = rec[1]

            name_best = rec[2]

            address_best = rec[3]

            name_exact = rec[4]

            address_exact = rec[5]

            final_score = rec[6]


            # Minimum final score
            if final_score < MIN_FINAL_SCORE:

                continue


            # Strong pair gate
            if not precision_gate(
                {
                    "name_best":
                        name_best,

                    "address_best":
                        address_best,

                    "name_exact":
                        name_exact,

                    "address_exact":
                        address_exact,

                    "country_same":
                        1.0
                },
                probability
            ):

                continue


            selected.append(
                rec
            )


        # ----------------------------------------------------
        # Ranking-relative selection
        # ----------------------------------------------------

        if selected:

            selected.sort(
                key=lambda x: x[6],
                reverse=True
            )


            top_score = selected[0][6]


            # Keep candidates close to the best candidate.
            #
            # Multiple true matches are allowed, but weak
            # candidates far below the best one are rejected.

            relative = [
                rec
                for rec in selected
                if (
                    rec[6] >= top_score - 8.0
                )
            ]


            selected = relative


        # ----------------------------------------------------
        # Fallback:
        # one exceptionally strong candidate
        # ----------------------------------------------------

        if not selected and records:

            top = records[0]


            candidate_id = top[0]

            probability = top[1]

            name_best = top[2]

            address_best = top[3]

            name_exact = top[4]

            address_exact = top[5]

            score = top[6]


            very_strong = (

                score >= 90.0
                and
                probability >= 0.90
                and
                name_best >= 85
                and
                address_best >= 70
            )


            if very_strong:

                selected = [top]


        # ----------------------------------------------------
        # Max matches
        # ----------------------------------------------------

        selected = selected[
            :MAX_MATCHES_PER_ENTITY
        ]


        # ----------------------------------------------------
        # WRITE
        # ----------------------------------------------------

        if selected:

            rows = pd.DataFrame({
                "source1_entity_id":
                    [sid] * len(selected),

                "candidate_entity_id":
                    [
                        rec[0]
                        for rec in selected
                    ]
            })


            rows.to_csv(
                TEMP_MATCH_OUT,
                sep="\t",
                index=False,
                mode="a",
                header=not os.path.exists(
                    TEMP_MATCH_OUT
                )
            )


            final_kept += len(
                selected
            )


    # ========================================================
    # READ CANDIDATES
    # ========================================================

    for chunk in pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        if (
            "source1_entity_id"
            not in chunk.columns
            or
            "candidate_entity_id"
            not in chunk.columns
        ):

            raise ValueError(
                "candidate_pairs_v3.tsv must contain:\n"
                "source1_entity_id\n"
                "candidate_entity_id"
            )


        # ----------------------------------------------------
        # Only requested source
        # ----------------------------------------------------

        chunk = chunk[
            chunk[
                "candidate_entity_id"
            ]
            .astype(str)
            .str.startswith(prefix)
        ].copy()


        if len(chunk) == 0:
            continue


        processed += len(
            chunk
        )


        # ----------------------------------------------------
        # Remove duplicate pairs
        # ----------------------------------------------------

        chunk = chunk.drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id"
            ]
        )


        # ----------------------------------------------------
        # Lookup S1
        # ----------------------------------------------------

        left = s1.reindex(
            chunk[
                "source1_entity_id"
            ].values
        ).reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # Lookup candidate
        # ----------------------------------------------------

        right = source.reindex(
            chunk[
                "candidate_entity_id"
            ].values
        ).reset_index(
            drop=True
        )


        valid = (
            left["entity_id"].notna()
            &
            right["entity_id"].notna()
        )


        if not valid.any():
            continue


        left = left.loc[
            valid.values
        ].reset_index(
            drop=True
        )

        right = right.loc[
            valid.values
        ].reset_index(
            drop=True
        )

        valid_chunk = chunk.loc[
            valid.values
        ].reset_index(
            drop=True
        )


        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        feature_df = build_features(
            left,
            right
        )


        X = feature_df[
            feature_cols
        ].copy()


        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)


        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        probabilities = (
            model
            .predict_proba(X)[:, 1]
        )


        # ----------------------------------------------------
        # BUILD ENTITY RECORDS
        # ----------------------------------------------------

        for i in range(
            len(valid_chunk)
        ):

            sid = str(
                valid_chunk.iloc[i][
                    "source1_entity_id"
                ]
            )


            cid = str(
                valid_chunk.iloc[i][
                    "candidate_entity_id"
                ]
            )


            probability = float(
                probabilities[i]
            )


            name_best = float(
                feature_df.iloc[i][
                    "name_best"
                ]
            )


            address_best = float(
                feature_df.iloc[i][
                    "address_best"
                ]
            )


            name_exact = float(
                feature_df.iloc[i][
                    "name_exact"
                ]
            )


            address_exact = float(
                feature_df.iloc[i][
                    "address_exact"
                ]
            )


            final_score = calculate_final_score(
                probability,
                name_best,
                address_best,
                name_exact,
                address_exact
            )


            # ------------------------------------------------
            # Entity boundary
            # ------------------------------------------------

            if current_sid is None:

                current_sid = sid


            elif sid != current_sid:

                finalize_entity(
                    current_sid,
                    current_records
                )

                current_sid = sid
                current_records = []


            # ------------------------------------------------
            # Add candidate
            # ------------------------------------------------

            current_records.append(
                (
                    cid,
                    probability,
                    name_best,
                    address_best,
                    name_exact,
                    address_exact,
                    final_score
                )
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed >= 500_000
            and
            processed % 500_000
            < CHUNK_SIZE
        ):

            print(
                prefix,
                "| processed:",
                processed,
                "| final kept:",
                final_kept
            )


    # --------------------------------------------------------
    # Final entity
    # --------------------------------------------------------

    finalize_entity(
        current_sid,
        current_records
    )


    print(
        prefix,
        "DONE"
    )

    print(
        "Processed:",
        processed
    )

    print(
        "Final kept:",
        final_kept
    )


    return final_kept


# ============================================================
# RUN S2
# ============================================================

s2_final = process_source(
    TEST_S2,
    "S2-"
)


# ============================================================
# RUN S3
# ============================================================

s3_final = process_source(
    TEST_S3,
    "S3-"
)


# ============================================================
# LOAD FINAL MATCHES
# ============================================================

if os.path.exists(
    TEMP_MATCH_OUT
):

    matches = pd.read_csv(
        TEMP_MATCH_OUT,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

else:

    matches = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )


# ============================================================
# DEDUPLICATE
# ============================================================

matches = matches.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)


# ============================================================
# BUILD MATCHING RESULTS
# ============================================================

print(
    "\nCreating matching_results.tsv..."
)


if len(matches) > 0:

    grouped = (
        matches
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .apply(
            lambda x:
            ",".join(
                dict.fromkeys(
                    x.astype(str)
                )
            )
        )
        .to_dict()
    )

else:

    grouped = {}


result = pd.DataFrame({
    "source1_entity_id":
        s1[
            "entity_id"
        ].astype(str).values
})


result[
    "matched_entity_ids"
] = (
    result[
        "source1_entity_id"
    ]
    .map(grouped)
    .fillna("")
)


result.to_csv(
    MATCHING_OUT,
    sep="\t",
    index=False
)


# ============================================================
# BUILD EXACT CANDIDATE FILE
#
# This is still the full candidate set fed into the model.
# ============================================================

print(
    "\nCreating candidate_pairs.tsv..."
)


with open(
    CANDIDATE_OUT,
    "w"
) as out:

    out.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )


    current_sid = None
    current_ids = []


    def flush_candidates():

        global current_sid
        global current_ids

        if current_sid is None:
            return

        unique_ids = list(
            dict.fromkeys(
                current_ids
            )
        )


        out.write(
            str(current_sid)
            + "\t"
            + ",".join(unique_ids)
            + "\n"
        )


    for chunk in pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False
    ):

        for sid, cid in zip(
            chunk[
                "source1_entity_id"
            ],
            chunk[
                "candidate_entity_id"
            ]
        ):

            sid = str(sid)
            cid = str(cid)


            if current_sid is None:

                current_sid = sid


            elif sid != current_sid:

                flush_candidates()

                current_sid = sid
                current_ids = []


            if (
                cid.startswith("S2-")
                or
                cid.startswith("S3-")
            ):

                current_ids.append(cid)


    flush_candidates()


# ============================================================
# ADD S1 WITH NO CANDIDATES
# ============================================================

candidate_existing = pd.read_csv(
    CANDIDATE_OUT,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


existing_sids = set(
    candidate_existing[
        "source1_entity_id"
    ].astype(str)
)


missing_sids = [
    sid
    for sid in
    s1["entity_id"].astype(str)
    if sid not in existing_sids
]


if missing_sids:

    print(
        "Adding missing S1 rows:",
        len(missing_sids)
    )


    with open(
        CANDIDATE_OUT,
        "a"
    ) as out:

        for sid in missing_sids:

            out.write(
                str(sid)
                + "\t\n"
            )


# ============================================================
# CLEAN TEMP
# ============================================================

try:

    os.remove(
        TEMP_MATCH_OUT
    )

except OSError:

    pass


# ============================================================
# FINAL STATISTICS
# ============================================================

non_empty = (
    result[
        "matched_entity_ids"
    ]
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)


empty = (
    len(result)
    -
    non_empty
)


final_pairs = (
    result[
        "matched_entity_ids"
    ]
    .astype(str)
    .apply(
        lambda x:
        0
        if not x
        else len(
            x.split(",")
        )
    )
    .sum()
)


print("\n")
print("=" * 70)
print("PRECISION-FIRST XGBOOST RESULT")
print("=" * 70)

print(
    "Source 1:",
    len(result)
)

print(
    "Non-empty:",
    non_empty
)

print(
    "Empty:",
    empty
)

print(
    "Final matched pairs:",
    final_pairs
)

print(
    "S2 selected:",
    s2_final
)

print(
    "S3 selected:",
    s3_final
)

print(
    "Model threshold:",
    MODEL_THRESHOLD
)

print(
    "Matching:",
    MATCHING_OUT
)

print(
    "Candidate:",
    CANDIDATE_OUT
)

print("=" * 70)