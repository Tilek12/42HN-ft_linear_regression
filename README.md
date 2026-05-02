# :desktop_computer: 42Heilbronn :de:

<h1 align="center">
  Ft_linear_regression :chart_with_downwards_trend:
  <h2 align="center">
    :white_check_mark: 125/125
  </h2>
</h1>

## :clipboard: Project info: [subject](https://github.com/Tilek12/42HN-ft_linear_regression/blob/master/docs/en.subject_-_ft_linear_regression.pdf)
A simple implementation of **linear regression with gradient descent** to predict car prices from mileage.

## :pushpin: Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Train](#train)
  - [Predict](#predict)
  - [Precision (Bonus)](#precision-bonus)
- [How It Works](#how-it-works)
- [Model File](#model-file)
- [Common Errors](#common-errors)

---

## :memo: Overview

This project trains a linear model:

`predicted_price = theta0 + theta1 * x`

Where:
- `x` is mileage (normalized during training),
- `theta0` and `theta1` are learned using gradient descent.

---

## :paperclips: Features

- Manual gradient descent implementation (no forbidden regression helpers)
- Input and file validation
- Model persistence to JSON
- Separate scripts for:
  - training
  - prediction
  - precision metrics (bonus)
- Optional plotting support

---

## :bookmark_tabs: Project Structure

```text
ft_linear_regression/
├── data.csv
├── train.py
├── predict.py
├── precision.py
├── utils.py
├── model.json            # generated after training
└── knowledge/            # notes
```

---

## :minidisc: Requirements

- Python 3.10+ recommended
- Linux environment (commands below are Linux-friendly)

---

## :gear: Setup

From the project directory:

```bash
cd .../ft_linear_regression
python3 -m venv .venv
source .venv/bin/activate
python -V
```

If plotting uses `matplotlib`, install it:

```bash
pip install matplotlib
```

---

## :keyboard: Usage

### :green_circle: Train

```bash
python train.py
```

Expected result:
- training progress in terminal
- `model.json` created with learned parameters and normalization metadata

---

### :green_circle: Predict

```bash
python predict.py
```

Then enter mileage when prompted.

---

### :green_circle: Precision (Bonus)

```bash
python precision.py
```

Expected metrics:
- MSE
- RMSE
- R2

---

## :chains: How It Works

1. Load dataset (`km`, `price`)
2. Normalize mileage values:  
   `x_norm = (x - mean) / std`
3. Train with gradient descent:
   - initialize `theta0 = 0`, `theta1 = 0`
   - compute gradients
   - update `theta0` and `theta1` simultaneously
4. Save model parameters and normalization metadata
5. Predict by applying the same normalization to user mileage

---

## :page_facing_up: Model File

Example `model.json`:

```json
{
  "theta0": 6331.833329013791,
  "theta1": -1106.0198802381071,
  "mean": 101066.25,
  "std": 51565.1899106445,
  "min_km": 22899.0,
  "max_km": 240000.0
}
```

---

## :warning: Common Errors

- **`model.json` missing**  
  Run `python train.py` first.

- **Invalid mileage input**  
  Enter a numeric, non-negative mileage value.

- **Dataset path issue**  
  Ensure `data.csv` exists in the project directory.
