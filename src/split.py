import sys
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["split"]

# Get input and output paths
input_path = sys.argv[1]
output_dir = Path(sys.argv[2])

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(input_path)

print(f"Original dataset shape: {df.shape}")

# ---------------------------------------------------------
# Remove columns that should NOT be used as model features
# ---------------------------------------------------------

# 'instant' is only a record/index number
if "instant" in df.columns:
    df = df.drop(columns=["instant"])

# 'dteday' is a date string and is not directly used
if "dteday" in df.columns:
    df = df.drop(columns=["dteday"])

# IMPORTANT:
# casual + registered = cnt
# Therefore, using these two variables would cause target leakage.
leakage_columns = ["casual", "registered"]

for column in leakage_columns:
    if column in df.columns:
        df = df.drop(columns=[column])

# ---------------------------------------------------------
# Convert categorical columns to numeric values
# ---------------------------------------------------------

categorical_columns = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit"
]

# One-hot encode categorical variables
existing_categorical_columns = [
    column for column in categorical_columns
    if column in df.columns
]

df = pd.get_dummies(
    df,
    columns=existing_categorical_columns,
    drop_first=True,
    dtype=int
)

# ---------------------------------------------------------
# Check target
# ---------------------------------------------------------

if "cnt" not in df.columns:
    raise ValueError("Target column 'cnt' was not found in the dataset.")

# Make sure there are no missing values
df = df.dropna()

# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# ---------------------------------------------------------
# Save prepared datasets
# ---------------------------------------------------------

train_df.to_csv(
    output_dir / "train.csv",
    index=False
)

test_df.to_csv(
    output_dir / "test.csv",
    index=False
)

print(f"Prepared dataset shape: {df.shape}")
print(f"Train: {len(train_df)} rows")
print(f"Test: {len(test_df)} rows")
print("Preprocessing completed successfully.")