import json
import matplotlib.pyplot as plt
from utils import load_data, estimate_price, normalize

def train(X, Y, learning_rate=0.01, iterations=10000):
    theta0 = 0.0
    theta1 = 0.0

    # Number of data points
    m = len(X)

    for iteration in range(iterations):
        sum0 = 0.0
        sum1 = 0.0

        # Loop over dataset
        for i in range(m):
            prediction = estimate_price(X[i], theta0, theta1)

            # Error calculation
            error = prediction - Y[i]

            # Accumulate gradients
            sum0 += error
            sum1 += error * X[i]

        # Compute new thetas (IMPORTANT: simultaneous update)
        tmp_theta0 = theta0 - learning_rate * (sum0 / m)
        tmp_theta1 = theta1 - learning_rate * (sum1 / m)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

        # Debug: print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: theta0={theta0}, theta1={theta1}")

    return theta0, theta1

X, Y = load_data("data.csv")

# Normalize mileage
X_norm, mean, std = normalize(X)
if std == 0:
    std = 1

theta0, theta1 = train(X_norm, Y)

print("\nFinal values:")
print("theta0 =", theta0)
print("theta1 =", theta1)

model = {
    "theta0": theta0,
    "theta1": theta1,
    "mean": mean,
    "std": std
}

with open("model.json", "w") as f:
    json.dump(model, f)

print("\nModel saved to model.json")

# Plot real data
plt.scatter(X, Y, label="Real data")

# Create line (predictions)
X_line = sorted(X)
Y_line = []

for x in X_line:
    x_norm = (x - mean) / std
    y_pred = theta0 + theta1 * x_norm
    Y_line.append(y_pred)

# Plot regression line
plt.plot(X_line, Y_line, color="red", label="Regression line")

# Labels
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.title("Linear Regression - Car Price Prediction")

plt.legend()

# Show plot
plt.show()
