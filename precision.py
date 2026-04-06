from pathlib import Path
from utils import load_data, load_model, estimate_price


def _validate_xy(X, Y):
    """
    Validate that feature and target vectors are non-empty and aligned.
    """
    if not X or not Y:
        raise ValueError("X and Y must be non-empty.")
    if len(X) != len(Y):
        raise ValueError(f"Length mismatch: len(X)={len(X)}, len(Y)={len(Y)}")

def compute_mse(X, Y, theta0, theta1):
    """
    Mean Squared Error:
    average squared difference between prediction and real value
    """
    _validate_xy(X, Y)
    m = len(X)
    error = 0.0

    for i in range(m):
        pred = estimate_price(X[i], theta0, theta1)
        error += (pred - Y[i]) ** 2

    return error / m


def compute_rmse(X, Y, theta0, theta1):
    """
    Root Mean Squared Error:
    same as MSE but in original unit (price)
    """
    return compute_mse(X, Y, theta0, theta1) ** 0.5  # FIXED BUG


def compute_r2(X, Y, theta0, theta1):
    """
    R² Score:
    measures how well the model explains variance in data
    """
    _validate_xy(X, Y)
    m = len(X)
    mean_y = sum(Y) / m

    ss_total = 0.0
    ss_residual = 0.0

    for i in range(m):
        pred = estimate_price(X[i], theta0, theta1)
        ss_residual += (Y[i] - pred) ** 2
        ss_total += (Y[i] - mean_y) ** 2

    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0

    return 1 - (ss_residual / ss_total)


def evaluate_precision(data_path, model_path):
    """
    Evaluate model precision on the dataset using common regression metrics.

    Returns:
        tuple[float, float, float]: mse, rmse, r2
    """
    model = load_model(model_path)
    X_raw, Y = load_data(data_path)

    mean = model["mean"]
    std = model["std"]

    X_norm = [(x - mean) / std for x in X_raw]

    theta0 = model["theta0"]
    theta1 = model["theta1"]

    mse = compute_mse(X_norm, Y, theta0, theta1)
    rmse = compute_rmse(X_norm, Y, theta0, theta1)
    r2 = compute_r2(X_norm, Y, theta0, theta1)

    return mse, rmse, r2


def main():
    """
    Run a standalone precision report for the current model and dataset.
    """
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data.csv"
    model_path = base_dir / "model.json"

    try:
        mse, rmse, r2 = evaluate_precision(data_path, model_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print("Model precision report")
    print("----------------------")
    print(f"Dataset: {data_path}")
    print(f"Model:   {model_path}")
    print("")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R²   = {r2:.3f}")


if __name__ == "__main__":
    main()
