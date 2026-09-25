import sys
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Get command-line arguments
model_path = sys.argv[1]
test_csv = sys.argv[2]
metrics_path = Path(sys.argv[3])

# Create metrics directory
metrics_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Load trained model
model = joblib.load(model_path)

# Load test data
test_df = pd.read_csv(test_csv)

# Separate features and target
X_test = test_df.drop(columns=["cnt"])
y_true = test_df["cnt"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

r2 = r2_score(
    y_true,
    y_pred
)

# Store metrics
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

# Save metrics as JSON
with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=2
    )

# Display results
print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

print(f"\nSaved metrics -> {metrics_path}")