from utils import load_data

X, Y = load_data("data.csv")

print("X (mileage):", X[:5])
print("Y (price):  ", Y[:5])
