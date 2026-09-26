import pandas as pd
import re
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Unicode normalization
    value = unicodedata.normalize("NFKD", value)

    # Replace common symbols
    value = value.replace("&", " and ")

    # Remove punctuation
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------------------------------------------------------
# Name normalization
# ---------------------------------------------------------

def normalize_name(value):
    text = normalize_text(value)

    replacements = {
        "corporation": "corp",
        "company": "co",
        "limited": "ltd",
        "private": "pvt",
        "incorporated": "inc",
        "inc": "inc",
        "pvt": "pvt",
        "ltd": "ltd",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(value):
    text = normalize_text(value)

    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt",
        "building": "bldg",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------------
# Process one dataframe
# ---------------------------------------------------------

def preprocess(df):

    df = df.copy()

    df["business_name_clean"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["country_clean"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

train_dir = BASE_DIR / "dataset" / "train"
test_dir = BASE_DIR / "dataset" / "test"


files = {
    "train_source1": train_dir / "train_source1.tsv",
    "train_source2": train_dir / "train_source2.tsv",
    "train_source3": train_dir / "train_source3.tsv",
    "test_source1": test_dir / "test_source1.tsv",
    "test_source2": test_dir / "test_source2.tsv",
    "test_source3": test_dir / "test_source3.tsv",
}


# ---------------------------------------------------------
# Save processed data
# ---------------------------------------------------------

processed_dir = BASE_DIR / "dataset" / "processed"
processed_dir.mkdir(exist_ok=True)


for name, path in files.items():

    print(f"Processing {name}...")

    df = pd.read_csv(path, sep="\t")

    df = preprocess(df)

    output_path = processed_dir / f"{name}.tsv"

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_path}")


print("\nPreprocessing completed.")