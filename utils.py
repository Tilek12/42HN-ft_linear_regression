import csv
from pathlib import Path


def load_data(filepath):
    """
    Load dataset from a CSV file.

    Expected columns:
    - km
    - price

    Returns:
        tuple[list[float], list[float]]: mileage values, price values.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    X, Y = [], []

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header.")

        if "km" not in reader.fieldnames or "price" not in reader.fieldnames:
            raise ValueError(
                f"CSV header must contain 'km' and 'price'. Found: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            km = row.get("km", "").strip()
            price = row.get("price", "").strip()

            if not km and not price:
                continue

            try:
                X.append(float(km))
                Y.append(float(price))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value at line {row_number}: km={km!r}, price={price!r}"
                ) from exc

    if not X:
        raise ValueError("Dataset is empty.")

    return X, Y


def estimate_price(x, theta0, theta1):
    """
    Linear model: y = theta0 + theta1 * x
    """
    return theta0 + theta1 * x


def normalize(X):
    """
    Normalize values using standardization:
    x_norm = (x - mean) / std
    """
    if not X:
        raise ValueError("Cannot normalize an empty list.")

    mean = sum(X) / len(X)
    variance = sum((x - mean) ** 2 for x in X) / len(X)
    std = variance ** 0.5

    if std == 0:
        std = 1  # avoid division by zero

    X_norm = [(x - mean) / std for x in X]
    return X_norm, mean, std


# ---------- Evaluation Metrics ----------


def _validate_xy(X, Y):
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
