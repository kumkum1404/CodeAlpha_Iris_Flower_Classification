import pandas as pd
import joblib

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(
    "../models/iris_model.pkl"
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "../dataset/iris.csv"
)

# =========================================================
# FEATURES
# =========================================================

X = df.drop(
    ["Id", "Species"],
    axis=1
)

# =========================================================
# ACTUAL LABELS
# =========================================================

actual = df["Species"]

# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X)

# =========================================================
# CREATE RESULT DATAFRAME
# =========================================================

results = pd.DataFrame({

    "Actual Species": actual,

    "Predicted Species": predictions
})

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\nPREDICTION RESULTS:\n")

print(results.head(20))

# =========================================================
# ACCURACY CHECK
# =========================================================

correct = (
    results["Actual Species"]
    ==
    results["Predicted Species"]
).sum()

total = len(results)

accuracy = correct / total

print("\n========================")

print(f"Total Samples: {total}")

print(f"Correct Predictions: {correct}")

print(f"Accuracy: {accuracy:.4f}")

print("========================")