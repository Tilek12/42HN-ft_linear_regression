from pathlib import Path
from utils import load_model


def predict(mileage, model):
    """
    Predict price based on mileage using trained model.
    """
    mean = model["mean"]
    std = model["std"]

    x_norm = (mileage - mean) / std
    return model["theta0"] + model["theta1"] * x_norm


def main():
    """
    Load trained model, prompt for mileage, and print predicted car price.
    """
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model.json"

    try:
        model = load_model(model_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    try:
        mileage = float(input("Enter mileage: ").strip())
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    if mileage < 0:
        print("Mileage cannot be negative.")
        return

    if "min_km" in model and "max_km" in model:
        if mileage < model["min_km"] or mileage > model["max_km"]:
            print("Warning: mileage is outside training range. Prediction may be inaccurate.")

    predicted_price = predict(mileage, model)
    final_price = max(0, predicted_price)
    print(f"Estimated price: {final_price:.2f}")


if __name__ == "__main__":
    main()
