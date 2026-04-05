import json

def load_model():
    with open("model.json", "r") as f:
        return json.load(f)


def predict(mileage, model):
    x_norm = (mileage - model["mean"]) / model["std"]
    return model["theta0"] + model["theta1"] * x_norm


# --- main ---
model = load_model()

try:
    mileage = float(input("Enter mileage: "))
    price = predict(mileage, model)
    print(f"Estimated price: {price:.2f}")
except ValueError:
    print("Invalid input. Please enter a number.")
