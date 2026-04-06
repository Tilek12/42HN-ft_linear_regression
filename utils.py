import csv
import json
import matplotlib.pyplot as plt

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


def save_model(model_path, theta0, theta1, mean, std, min_km, max_km):
    """
    Persist trained model parameters and scaling metadata to a JSON file.

    Stored values:
    - theta0, theta1: learned regression parameters
    - mean, std: normalization parameters used during training
    - min_km, max_km: training mileage range for prediction warnings
    """
    model = {
        "theta0": theta0,
        "theta1": theta1,
        "mean": mean,
        "std": std,
        "min_km": min_km,
        "max_km": max_km,
    }
    with open(model_path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)


def load_model(filepath="model.json"):
    """
    Load trained model parameters from JSON file.
    """
    path = Path(filepath)

    try:
        with path.open("r") as file:
            model = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model file: {path}") from exc

    required_keys = {"theta0", "theta1", "mean", "std"}
    missing_keys = required_keys - set(model.keys())
    if missing_keys:
        raise ValueError(f"Model file is missing keys: {sorted(missing_keys)}")

    if model["std"] == 0:
        raise ValueError("Invalid model: std cannot be 0")

    return model


def plot_regression(X_raw, Y, mean, std, theta0, theta1):
    """
    Plot dataset points and the learned regression line.

    Notes:
    - `X_raw` is used on the x-axis to keep mileage in original units.
    - Line predictions are computed by normalizing each x with `mean` and `std`.
    """
    plt.scatter(X_raw, Y, label="Real data")

    X_line = sorted(X_raw)
    Y_line = [estimate_price((x - mean) / std, theta0, theta1) for x in X_line]

    plt.plot(X_line, Y_line, color="red", label="Regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Linear Regression - Car Price Prediction")
    plt.legend()
    plt.show()

