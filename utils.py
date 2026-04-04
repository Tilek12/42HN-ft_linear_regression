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

def normalize(X):
    mean = sum(X) / len(X)
    
    variance = sum((x - mean) ** 2 for x in X) / len(X)
    std = variance ** 0.5

    X_norm = [(x - mean) / std for x in X]

    return X_norm, mean, std
