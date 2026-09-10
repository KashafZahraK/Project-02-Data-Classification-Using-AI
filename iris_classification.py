"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit

Goal:
Build a basic classification model using the Iris dataset.

Pipeline:
Load -> Scale -> Split -> Choose K -> Train KNN -> Predict -> Evaluate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


# ================================================================
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ================================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame to understand the dataset
df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)

print(f"Samples: {df.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(target_names)}")
print(f"Class names: {list(target_names)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nClass balance:")
print(df["species"].value_counts())


# ================================================================
# STEP 2: SPLIT THE DATA
# ================================================================

# 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ================================================================
# STEP 3: FEATURE SCALING
# ================================================================

# Scaling makes the features comparable.
# The scaler learns only from training data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING")
print("=" * 60)

print("Scaling completed successfully.")
print("Example before scaling:", X_train[0])
print("Example after scaling:", np.round(X_train_scaled[0], 3))


# ================================================================
# STEP 4: CHOOSE THE BEST K
# ================================================================

# We test K values from 1 to 20.
# The K with the lowest error is selected.

error_rates = []

k_range = range(1, 21)

for k in k_range:

    knn_temp = KNeighborsClassifier(n_neighbors=k)

    knn_temp.fit(X_train_scaled, y_train)

    prediction_temp = knn_temp.predict(X_test_scaled)

    error = np.mean(prediction_temp != y_test)

    error_rates.append(error)

best_k = list(k_range)[np.argmin(error_rates)]

print("\n" + "=" * 60)
print("STEP 4: CHOOSING THE BEST K")
print("=" * 60)

print(f"Best K: {best_k}")
print(f"Lowest error rate: {min(error_rates):.4f}")


# Create the Elbow/Error Rate graph

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_range),
    error_rates,
    marker="o",
    linestyle="--"
)

plt.axvline(
    best_k,
    linestyle=":",
    label=f"Best K = {best_k}"
)

plt.title("Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()

plt.savefig("elbow_plot.png", dpi=150)
plt.close()

print("Saved: elbow_plot.png")


# ================================================================
# STEP 5: TRAIN THE FINAL KNN MODEL
# ================================================================

model = KNeighborsClassifier(n_neighbors=best_k)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict the test data
predictions = model.predict(X_test_scaled)

print("\n" + "=" * 60)
print(f"STEP 5: KNN MODEL TRAINING (K = {best_k})")
print("=" * 60)

print("Model trained successfully.")
print("Predictions generated successfully.")


# ================================================================
# STEP 6: EVALUATE THE MODEL
# ================================================================

accuracy = accuracy_score(y_test, predictions)

f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("STEP 6: MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")
print(f"F1 Score (Macro): {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=target_names
    )
)


# ================================================================
# CONFUSION MATRIX GRAPH
# ================================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(3),
    target_names
)

plt.yticks(
    range(3),
    target_names
)

# Display numbers inside the matrix
for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("Saved: confusion_matrix.png")


# ================================================================
# STEP 7: PREDICT A BRAND-NEW FLOWER
# ================================================================

# Measurements:
# [sepal length, sepal width, petal length, petal width]

new_sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

# Scale the new flower using the same scaler
new_sample_scaled = scaler.transform(new_sample)

# Predict its species
new_prediction = model.predict(new_sample_scaled)

print("\n" + "=" * 60)
print("STEP 7: PREDICTING NEW DATA")
print("=" * 60)

print(f"New flower measurements: {new_sample[0]}")
print(
    f"Predicted species: "
    f"{target_names[new_prediction[0]]}"
)


# ================================================================
# PROJECT COMPLETE
# ================================================================

print("\n" + "=" * 60)
print("PROJECT 2 COMPLETE")
print("=" * 60)