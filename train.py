from utils import load_data, estimate_price

X, Y = load_data("data.csv")

print("X (mileage):", X[:5])
print("Y (price):  ", Y[:5])

theta0 = 0
theta1 = 0

prediction = estimate_price(X[0], theta0, theta1)

print("Mileage:", X[0])
print("Real price:", Y[0])
print("Predicted price:", prediction)
