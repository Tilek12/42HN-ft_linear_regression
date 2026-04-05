import json

def load_model(filepath="model.json"):
    """
    Load trained model parameters from JSON file.
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: model.json not found. Train the model first.")
        exit(1)


def predict(mileage, model):
    """
    Predict price based on mileage using trained model.
    """
    mean = model["mean"]
    std = model["std"]

    # Normalize input
    x_norm = (mileage - mean) / std

    # Linear prediction
    price = model["theta0"] + model["theta1"] * x_norm

    return price


def main():
    model = load_model()

    try:
        mileage = float(input("Enter mileage: "))

        if mileage < 0:
            print("Mileage cannot be negative.")
            return

        # Warn if outside training range
        if mileage > 300000:
            print("⚠️ Warning: mileage is outside training range. Prediction may be inaccurate.")

        predicted_price = predict(mileage, model)

        # Prevent negative price
        final_price = max(0, predicted_price)

        print(f"Estimated price: {final_price:.2f}")

    except ValueError:
        print("Invalid input. Please enter a numeric value.")


if __name__ == "__main__":
    main()
