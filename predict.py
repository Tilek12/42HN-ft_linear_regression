import json
from pathlib import Path


def load_model(filepath="model.json"):
    """
    Load trained model parameters from JSON file.
    """
    path = Path(filepath)

    try:
        with path.open("r") as file:
            model = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model file: {path}") from exc

    required_keys = {"theta0", "theta1", "mean", "std"}
    missing_keys = required_keys - set(model.keys())
    if missing_keys:
        raise ValueError(f"Model file is missing keys: {sorted(missing_keys)}")

    if model["std"] == 0:
        raise ValueError("Invalid model: std cannot be 0")

    return model


def predict(mileage, model):
    """
    Predict price based on mileage using trained model.
    """
    mean = model["mean"]
    std = model["std"]

    x_norm = (mileage - mean) / std
    return model["theta0"] + model["theta1"] * x_norm


def main():
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
