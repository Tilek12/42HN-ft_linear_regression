import json
import matplotlib.pyplot as plt
from utils import *

# ---------------- TRAINING ---------------- #

def train(X, Y, learning_rate=0.01, iterations=10000):
    """
    Train linear regression using gradient descent.
    """
    theta0, theta1 = 0.0, 0.0
    m = len(X)

    for iteration in range(iterations):
        sum0, sum1 = 0.0, 0.0

        for i in range(m):
            pred = estimate_price(X[i], theta0, theta1)
            error = pred - Y[i]

            sum0 += error
            sum1 += error * X[i]

        # simultaneous update
        theta0 -= learning_rate * (sum0 / m)
        theta1 -= learning_rate * (sum1 / m)

        # print progress every 500 steps
        if iteration % 500 == 0:
            print(f"Iteration {iteration}: theta0={theta0:.2f}, theta1={theta1:.2f}")

    return theta0, theta1


# ---------------- MAIN ---------------- #

X, Y = load_data("data.csv")

# Normalize input
X_norm, mean, std = normalize(X)

# Train model
theta0, theta1 = train(X_norm, Y)

print("\nFinal model:")
print(f"theta0 = {theta0:.2f}")
print(f"theta1 = {theta1:.2f}")

# Save model
model = {
    "theta0": theta0,
    "theta1": theta1,
    "mean": mean,
    "std": std
}

with open("model.json", "w") as f:
    json.dump(model, f)

print("Model saved to model.json")

# ---------------- EVALUATION ---------------- #

mse = compute_mse(X_norm, Y, theta0, theta1)
rmse = compute_rmse(X_norm, Y, theta0, theta1)
r2 = compute_r2(X_norm, Y, theta0, theta1)

print("\nEvaluation:")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.3f}")

# -------- TEST without normalization -------- #

theta0_raw, theta1_raw = train(X, Y, learning_rate=0.0000000001, iterations=1000)

print("\nWithout normalization:")
print("Raw theta0:", theta0_raw)
print("Raw theta1:", theta1_raw)

# ---------------- PLOTTING ---------------- #

plt.scatter(X, Y, label="Real data")

X_line = sorted(X)
Y_line = []

for x in X_line:
    x_norm = (x - mean) / std
    Y_line.append(estimate_price(x_norm, theta0, theta1))

plt.plot(X_line, Y_line, color="red", label="Regression line")

plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.title("Linear Regression - Car Price Prediction")
plt.legend()
plt.show()
