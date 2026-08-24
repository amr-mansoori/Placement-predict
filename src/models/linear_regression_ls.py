import os
import numpy as np
import matplotlib.pyplot as plt

# Import the ingestion function from our data pipeline
from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # =========================================================
    # 1. LOAD DATA
    # =========================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)


    # =========================================================
    # 2. DEFINE INPUT AND OUTPUT DIMENSIONS
    # =========================================================

    # L = 2 input features
    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    # M = 1 output/target
    target_col = "salary_package_lpa"


    # Remove rows containing missing values in required columns
    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()


    # Convert selected columns into NumPy arrays
    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)


    # Number of observations
    N = X_raw.shape[0]


    print(
        f"Loaded {N} data points "
        f"with input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )


    # =========================================================
    # 3. CREATE DESIGN MATRIX
    # =========================================================

    # Polynomial order P = 1
    #
    # Original:
    #
    # X_raw = [x1, x2]
    #
    # Add column of ones for intercept:
    #
    # X_design = [1, x1, x2]

    X_design = np.hstack([
        np.ones((N, 1)),
        X_raw
    ])


    # =========================================================
    # 4. STANDARD LEAST SQUARES
    # =========================================================

    # Normal Equation:
    #
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )


    # Try regular matrix inverse.
    # If matrix is singular, use pseudo-inverse.

    try:

        XT_X_inv = np.linalg.inv(XT_X)

    except np.linalg.LinAlgError:

        XT_X_inv = np.linalg.pinv(XT_X)


    XT_y = np.dot(
        X_design.T,
        y
    )


    # Calculate optimal weights
    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )


    # =========================================================
    # DISPLAY MODEL PARAMETERS
    # =========================================================

    print(
        "\n--- Optimal Model Parameters "
        "(Weights & Bias) ---"
    )

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[0]} (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[1]} (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )


    # =========================================================
    # CALCULATE PREDICTIONS AND ERROR
    # =========================================================

    y_pred = np.dot(
        X_design,
        w_optimal
    )


    # Error function:
    #
    # E(w) = 0.5 * Σ(predicted - actual)^2

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )


    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )


    # =========================================================
    # 5. CREATE 3D VISUALIZATION
    # =========================================================

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )


    fig = plt.figure(
        figsize=(10, 8)
    )


    ax = fig.add_subplot(
        projection="3d"
    )


    # Plot actual dataset points
    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y[:, 0],
        alpha=0.6,
        label="Actual Data Points"
    )


    # =========================================================
    # CREATE REGRESSION PLANE
    # =========================================================

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )


    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )


    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )


    # Regression equation:
    #
    # y = w0 + w1*x1 + w2*x2

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )


    # Plot regression plane
    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3
    )


    # Axis labels
    ax.set_xlabel(
        "CGPA (Feature 1)"
    )

    ax.set_ylabel(
        "Communication Skill Score (Feature 2)"
    )

    ax.set_zlabel(
        "Salary Package LPA (Target)"
    )


    ax.set_title(
        "Linear Regression via Standard Least Squares "
        "(P=1, L=2, M=1)"
    )


    plt.tight_layout()


    # =========================================================
    # SAVE GRAPH
    # =========================================================

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )


    plt.savefig(output_path)

    plt.close()


    print(
        f"\n-> Successfully saved 3D regression plot "
        f"to {output_path}"
    )


# =============================================================
# RUN PROGRAM
# =============================================================

if __name__ == "__main__":
    train_linear_regression_ls()