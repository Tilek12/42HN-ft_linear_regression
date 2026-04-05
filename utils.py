def load_data(filepath):
    """
    Load dataset from CSV file.
    Returns:
        X (list of float): mileage
        Y (list of float): price
    """
    X, Y = [], []

    with open(filepath, "r") as file:
        next(file)  # skip header

        for line in file:
            line = line.strip()
            if not line:
                continue

            mileage, price = line.split(",")
            X.append(float(mileage))
            Y.append(float(price))

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
    mean = sum(X) / len(X)
    variance = sum((x - mean) ** 2 for x in X) / len(X)
    std = variance ** 0.5

    if std == 0:
        std = 1  # avoid division by zero

    X_norm = [(x - mean) / std for x in X]
    return X_norm, mean, std


# ---------- Evaluation Metrics ----------

def compute_mse(X, Y, theta0, theta1):
    """
    Mean Squared Error:
    average squared difference between prediction and real value
    """
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
    m = len(X)
    mean_y = sum(Y) / m

    ss_total = 0.0
    ss_residual = 0.0

    for i in range(m):
        pred = estimate_price(X[i], theta0, theta1)
        ss_residual += (Y[i] - pred) ** 2
        ss_total += (Y[i] - mean_y) ** 2

    return 1 - (ss_residual / ss_total)
