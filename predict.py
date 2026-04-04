import json

# Load model
with open("model.json", "r") as f:
    model = json.load(f)

theta0 = model["theta0"]
theta1 = model["theta1"]
mean = model["mean"]
std = model["std"]

# Ask user input
mileage = float(input("Enter mileage: "))

# Normalize input
normalized_mileage = (mileage - mean) / std

# Predict
price = theta0 + theta1 * normalized_mileage

print(f"Estimated price: {price}")
