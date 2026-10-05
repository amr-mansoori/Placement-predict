import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ==========================================
# PART A: BINARY CLASSIFICATION
# ==========================================

# Generate a dataset with 2 input features and 2 possible classes
X_bin, y_bin = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_classes=2
)

# 80% training, 20% testing
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(
    X_bin,
    y_bin,
    test_size=0.20,
    random_state=42
)

print("--- BINARY DATASET ---")
print("Total samples:", len(X_bin))
print("Training samples:", len(X_bin_train))
print("Testing samples:", len(X_bin_test))
print("Classes:", np.unique(y_bin))

# ==========================================
# TRAIN BINARY LOGISTIC REGRESSION MODEL
# ==========================================

bin_model = LogisticRegression()

# Train the model using the training data
bin_model.fit(X_bin_train, y_bin_train)

# Predict the classes of the testing data
y_bin_pred = bin_model.predict(X_bin_test)


# ==========================================
# EVALUATE THE MODEL
# ==========================================

accuracy = accuracy_score(y_bin_test, y_bin_pred)

print("\n--- BINARY LOGISTIC REGRESSION ---")
print(f"Accuracy: {accuracy:.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_bin_test, y_bin_pred))

# ==========================================
# BINARY DECISION BOUNDARY
# ==========================================

xx, yy = np.meshgrid(
    np.linspace(X_bin[:, 0].min() - 1,
                X_bin[:, 0].max() + 1, 200),

    np.linspace(X_bin[:, 1].min() - 1,
                X_bin[:, 1].max() + 1, 200)
)

# Predict every point in the grid
Z = bin_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)


# Plot decision regions
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

# Plot actual test samples
plt.scatter(
    X_bin_test[:, 0],
    X_bin_test[:, 1],
    c=y_bin_test,
    edgecolors="k",
    cmap=plt.cm.coolwarm
)

plt.title("Binary Logistic Regression Decision Boundary")
plt.xlabel("Feature 1 (e.g., CGPA)")
plt.ylabel("Feature 2 (e.g., Aptitude Score)")

plt.tight_layout()

plt.savefig(
    "reports/figures/logistic_binary_decision_boundary.png"
)

print(
    "\n-> Saved binary decision boundary to "
    "reports/figures/logistic_binary_decision_boundary.png"
)

plt.show()

# ==========================================
# PART B: MULTICLASS CLASSIFICATION
# ==========================================

X_multi, y_multi = make_blobs(
    n_samples=1500,
    n_features=2,
    centers=3,
    random_state=42
)

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_multi,
    y_multi,
    test_size=0.20,
    random_state=42
)

print("\n--- MULTICLASS DATASET ---")
print("Total samples:", len(X_multi))
print("Training samples:", len(X_m_train))
print("Testing samples:", len(X_m_test))
print("Classes:", np.unique(y_multi))

# ==========================================
# TRAIN MULTICLASS MODELS
# ==========================================

# Multinomial Logistic Regression
multi_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs"
)

multi_model.fit(X_m_train, y_m_train)


# One-vs-Rest Logistic Regression
ovr_model = LogisticRegression(
    multi_class="ovr",
    solver="lbfgs"
)

ovr_model.fit(X_m_train, y_m_train)


# ==========================================
# EVALUATE MULTINOMIAL MODEL
# ==========================================

y_m_pred = multi_model.predict(X_m_test)

multi_accuracy = accuracy_score(y_m_test, y_m_pred)

print("\n--- MULTINOMIAL LOGISTIC REGRESSION ---")
print(f"Accuracy: {multi_accuracy:.4f}")

print("\n--- MULTINOMIAL CLASSIFICATION REPORT ---")
print(classification_report(y_m_test, y_m_pred))

# ==========================================
# MULTICLASS CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_m_test, y_m_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False
)

plt.title("Confusion Matrix - Multiclass Placement Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "reports/figures/logistic_multiclass_confusion_matrix.png"
)

print(
    "\n-> Saved multiclass confusion matrix to "
    "reports/figures/logistic_multiclass_confusion_matrix.png"
)

plt.show()