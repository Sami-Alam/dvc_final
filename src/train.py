import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

# Load training parameters
params = yaml.safe_load(open("params.yaml"))["train"]

# Get input and output paths
train_csv = sys.argv[1]
model_out = Path(sys.argv[2])

# Create output directory
model_out.parent.mkdir(parents=True, exist_ok=True)

# Load training data
train_df = pd.read_csv(train_csv)

# Separate features and target
X = train_df.drop(columns=["cnt"])
y = train_df["cnt"]

print(f"Training data shape: {X.shape}")

# Create Ridge Regression model
model = Ridge(
    alpha=params["alpha"]
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, model_out)

print(f"Model trained successfully.")
print(f"Saved model -> {model_out}")