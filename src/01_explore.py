import pandas as pd

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Source 1:", s1.shape)
print("Source 2:", s2.shape)
print("Source 3:", s3.shape)
print("Ground Truth:", gt.shape)

print("\nSource 1:")
print(s1.head())

print("\nSource 2:")
print(s2.head())

print("\nSource 3:")
print(s3.head())

print("\nGround Truth:")
print(gt.head())

print("\nMissing values:")
print(s1.isnull().sum())
print(s2.isnull().sum())
print(s3.isnull().sum())