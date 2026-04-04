from utils import load_data, estimate_price, normalize

def train(X, Y, learning_rate=0.0001, iterations=1000):
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

theta0, theta1 = train(X_norm, Y)

print("\nFinal values:")
print("theta0 =", theta0)
print("theta1 =", theta1)
