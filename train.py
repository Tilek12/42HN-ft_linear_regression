import json
from pathlib import Path

import matplotlib.pyplot as plt

from utils import compute_mse, compute_r2, compute_rmse, estimate_price, load_data, normalize


def train(X, Y, learning_rate=0.01, iterations=10000, verbose_interval=500):
    """
    Train linear regression using gradient descent.
    """
    if not X or not Y:
        raise ValueError("X and Y must be non-empty.")
    if len(X) != len(Y):
        raise ValueError(f"Length mismatch: len(X)={len(X)}, len(Y)={len(Y)}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    theta0, theta1 = 0.0, 0.0
    m = len(X)

    for iteration in range(iterations):
        sum0, sum1 = 0.0, 0.0

        for i in range(m):
            pred = estimate_price(X[i], theta0, theta1)
            error = pred - Y[i]
            sum0 += error
            sum1 += error * X[i]

        # Simultaneous update (mandatory requirement)
        new_theta0 = theta0 - learning_rate * (sum0 / m)
        new_theta1 = theta1 - learning_rate * (sum1 / m)
        theta0, theta1 = new_theta0, new_theta1

        if verbose_interval and iteration % verbose_interval == 0:
            print(f"Iteration {iteration}: theta0={theta0:.4f}, theta1={theta1:.4f}")

    return theta0, theta1


def save_model(model_path, theta0, theta1, mean, std, min_km, max_km):
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


def evaluate_model(X_norm, Y, theta0, theta1):
    mse = compute_mse(X_norm, Y, theta0, theta1)
    rmse = compute_rmse(X_norm, Y, theta0, theta1)
    r2 = compute_r2(X_norm, Y, theta0, theta1)
    return mse, rmse, r2


def plot_regression(X_raw, Y, mean, std, theta0, theta1):
    plt.scatter(X_raw, Y, label="Real data")

    X_line = sorted(X_raw)
    Y_line = [estimate_price((x - mean) / std, theta0, theta1) for x in X_line]

    plt.plot(X_line, Y_line, color="red", label="Regression line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Linear Regression - Car Price Prediction")
    plt.legend()
    plt.show()


def main():
    base_dir = Path(__file__).resolve().parent

    data_path = base_dir / "data.csv"
    model_path = base_dir / "model.json"
    learning_rate = 0.01
    iterations = 10000
    show_plot = True

    X_raw, Y = load_data(data_path)
    X_norm, mean, std = normalize(X_raw)

    theta0, theta1 = train(
        X_norm,
        Y,
        learning_rate=learning_rate,
        iterations=iterations,
    )

    print("\nFinal model:")
    print(f"theta0 = {theta0:.4f}")
    print(f"theta1 = {theta1:.4f}")

    save_model(
        model_path, 
        theta0, 
        theta1, 
        mean, 
        std,
        min(X_raw),
        max(X_raw),
    )
    print(f"Model saved to {model_path}")

    mse, rmse, r2 = evaluate_model(X_norm, Y, theta0, theta1)
    print("\nEvaluation:")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"R²   = {r2:.3f}")

    # Demonstration: training without normalization usually needs tiny LR and converges slower.
    theta0_raw, theta1_raw = train(X_raw, Y, learning_rate=1e-10, iterations=1000, verbose_interval=0)
    print("\nWithout normalization (demo run):")
    print(f"Raw theta0 = {theta0_raw:.4f}")
    print(f"Raw theta1 = {theta1_raw:.8f}")

    if show_plot:
        plot_regression(X_raw, Y, mean, std, theta0, theta1)


if __name__ == "__main__":
    main()
