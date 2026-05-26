import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLOADING DATASET...\n")

df = pd.read_csv("../dataset/iris.csv")
print(df.columns)

print("Dataset Loaded Successfully!")

# =========================================================
# DATA EXPLORATION
# =========================================================

print("\nDATASET SHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nSPECIES COUNT:")
print(df["Species"].value_counts())

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(["Id", "Species"], axis=1)

y = df["Species"]

# =========================================================
# VISUALIZATION
# =========================================================

print("\nCREATING VISUALIZATIONS...\n")

# Pairplot
sns.pairplot(
    df,
    hue='Species'
)

plt.savefig("../outputs/pairplot.png")

plt.close()

# Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    X.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.savefig("../outputs/heatmap.png")

plt.close()

# Species Distribution
plt.figure(figsize=(7,5))

sns.countplot(
    x='Species',
    data=df
)

plt.title("Species Distribution")

plt.savefig("../outputs/species_distribution.png")

plt.close()

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MODELS
# =========================================================

models = {

    "KNN":
        KNeighborsClassifier(),

    "Logistic Regression":
        LogisticRegression(max_iter=200),

    "SVM":
        SVC(),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier()
}

results = {}

# =========================================================
# TRAINING AND EVALUATION
# =========================================================

for name, model in models.items():

    print("\n========================")
    print(f"MODEL: {name}")
    print("========================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

# =========================================================
# ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    results.values()
)

plt.ylabel("Accuracy")

plt.title("Model Comparison")

plt.xticks(rotation=15)

plt.savefig("../outputs/accuracy_graph.png")

plt.close()

# =========================================================
# BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=results.get
)

best_model = models[
    best_model_name
]

print("\nBEST MODEL:")

print(best_model_name)

# =========================================================
# CROSS VALIDATION
# =========================================================

scores = cross_val_score(
    best_model,
    X_scaled,
    y,
    cv=5
)

print("\nCROSS VALIDATION SCORES:")

print(scores)

print("\nAVERAGE CV ACCURACY:")

print(scores.mean())

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

param_grid = {

    'n_neighbors': [3,5,7,9]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5
)

grid_search.fit(
    X_train,
    y_train
)

print("\nBEST PARAMETERS:")

print(grid_search.best_params_)

# =========================================================
# CONFUSION MATRIX
# =========================================================

y_pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("../outputs/confusion_matrix.png")

plt.close()

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    best_model,
    "../models/iris_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY!")

# =========================================================
# SAMPLE PREDICTION
# =========================================================

sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = best_model.predict(
    sample_scaled
)

print("\nPREDICTED FLOWER SPECIES:")

print(prediction[0])

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\nPROJECT EXECUTED SUCCESSFULLY!")

print("\nGenerated Files:")

print("""
1. pairplot.png
2. heatmap.png
3. species_distribution.png
4. accuracy_graph.png
5. confusion_matrix.png
6. iris_model.pkl
""")