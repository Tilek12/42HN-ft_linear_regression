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
