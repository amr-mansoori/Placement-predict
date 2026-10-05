import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ==========================================
# 1. GENERATE DATA
# ==========================================

np.random.seed(42)

# CGPA values roughly between 4 and 10
X = np.sort(6 * np.random.rand(100, 1) + 4)

# Create a non-linear relationship with some noise
y = np.sin(X).ravel() + np.random.normal(
    0, 0.2, X.shape[0]
)

print("Dataset created successfully!")
print("Total samples:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- TRAIN / TEST SPLIT ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. POLYNOMIAL TRANSFORMATION
# ==========================================

poly_degree = 15

poly = PolynomialFeatures(degree=poly_degree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\n--- POLYNOMIAL FEATURES ---")
print("Polynomial degree:", poly_degree)
print("Original training shape:", X_train.shape)
print("Transformed training shape:", X_train_poly.shape)

# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

print("\n--- FEATURE SCALING ---")
print("Training features scaled successfully!")
print("Scaled training shape:", X_train_scaled.shape)

# ==========================================
# 5. RIDGE REGULARIZATION
# ==========================================

# Try 200 lambda values from very small to very large
lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []

for lam in lambdas:

    # Ridge regression with L2 regularization
    ridge = Ridge(alpha=lam)

    # Train model
    ridge.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = ridge.predict(X_train_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    # Calculate MSE
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)


# Find lambda with lowest testing error
best_index = np.argmin(test_errors)
best_lambda = lambdas[best_index]

print("\n--- RIDGE REGULARIZATION ---")
print("Number of lambda values tested:", len(lambdas))
print(f"Best lambda: {best_lambda:.6f}")
print(f"Lowest test MSE: {test_errors[best_index]:.6f}")

# ==========================================
# 6. PLOT TRAINING AND TESTING ERROR
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error ($E_w$)",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error ($E_w$)",
    linewidth=2,
    linestyle="--"
)

# Lambda values range from very small to very large,
# so use a logarithmic x-axis
plt.xscale("log")

plt.xlabel("Regularization Parameter ($\\lambda$ / alpha)")
plt.ylabel("Mean Squared Error ($E_w$)")

plt.title(
    "Regularization Path: Ridge Regression Overfitting Control (Degree 15)"
)

plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()

# Save graph in our project's reports folder
plt.savefig("reports/figures/ridge_regularization_error.png")

print(
    "\n-> Saved Ridge regularization graph to "
    "reports/figures/ridge_regularization_error.png"
)

plt.show()