# Load data from file
def load_data(filepath):
    X = []
    Y = []

    with open(filepath, "r") as file:
        lines = file.readlines()

        #skip header
        for line in lines[1:]:
            # remove \n
            line = line.strip()

            if not line:
                continue

            # separate values
            mileage, price = line.split(",")

            X.append(float(mileage))
            Y.append(float(price))

    return X, Y

# theta0 - base price (starting value)
# theta1 - how price changes with mileage
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

# Normalize values
def normalize(X):
    mean = sum(X) / len(X)
    
    variance = sum((x - mean) ** 2 for x in X) / len(X)
    std = variance ** 0.5

    X_norm = [(x - mean) / std for x in X]

    return X_norm, mean, std

# Calculate Mean Squared Error (MSE)
def compute_mse(X, Y, theta0, theta1):
    m = len(X)
    error = 0.0

    for i in range(m):
        pred = theta0 + theta1 * X[i]
        error += (pred - Y[i]) ** 2

    return error / m

# Calculate Root Mean Squared Error (RMSE)
def compute_rmse(X, Y, theta0, theta1):
    mse = compute_mse(X, Y, theta0, theta1)
    return mse ** 0,5

# Calculate R² Score
def compute_r2(X, Y, theta0, theta1):
    m = len(X)

    # mean of Y
    mean_Y = sum(Y) / m

    ss_total = 0.0
    ss_residual = 0.0

    for i in range(m):
        pred = theta0 + theta1 * X[i]

        ss_residual += (Y[i] - pred) ** 2
        ss_total += (Y[i] - mean_Y) ** 2

    return 1 - (ss_residual / ss_total)
