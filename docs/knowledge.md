# Python commands:

1. ```python3 --version``` - check Python3 version
2. ```sudo apt install python3 python3-venv python3-pip``` - install Python3 on Linux
3. ```python3 -m venv venv``` - create virtual environment
4. ```source venv/bin/activate``` - activate virtual environment on Linux/MacOS
5. ```pip install matplotlib``` - install dependecy matplotlib
6. ```python train.py``` - run training program
7. ```python predict.py``` - run predicting program
8. ```python precision.py``` - run precision checking program
---

# 📘 Linear Regression – Knowledge Base (ft_linear_regression)

## 🔹 What is Linear Regression?

Linear regression is a method to **predict a value using a straight line**.

We assume that there is a relationship between input and output.

Example:

``` price = θ0 + θ1 * mileage ```


- **θ0 (theta0)** → base price (starting value)
- **θ1 (theta1)** → how price changes with mileage

👉 If θ1 is negative:
- more mileage → lower price ✔️

---

## 🔹 Why do we train the model?

At the beginning, we don’t know θ0 and θ1.

Training means:
👉 finding the best values for θ0 and θ1 so that predictions are close to real data.

---

## 🔹 How does training work?

Training is an iterative process:

1. Start with random values (θ0 = 0, θ1 = 0)
2. Make predictions
3. Compare with real values
4. Calculate error
5. Adjust θ0 and θ1
6. Repeat many times

👉 Goal: minimize the error

---

## 🔹 What is Gradient Descent?

Gradient Descent is an algorithm that helps us **find the best parameters**.

It works like this:

- Imagine standing on a mountain
- You want to go to the lowest point
- You take small steps downhill

👉 In ML:
- "height" = error
- "position" = θ0 and θ1

We update parameters like this:

```theta = theta - learning_rate * gradient```

---

## 🔹 What is Error?

Error shows how wrong the prediction is:

```error = predicted - real```

---

## 🔹 What is Cost Function?

We measure total error using **Mean Squared Error (MSE)**:

```MSE = (1/m) * Σ(predicted - real)^2```

- m = number of data points
- squares remove negative values

👉 Goal: minimize MSE

---

## 🔹 Why do we normalize data?

Original values can be very large:

```mileage = 240000```

This causes:
- unstable training
- very large gradients
- possible NaN errors

---

### ✅ Normalization formula:

```x_normalized = (x - mean) / std```

- mean = average value
- std = standard deviation

---

### ✅ Benefits:

- faster training
- stable gradient descent
- better convergence

---

## 🔹 What is MSE (Mean Squared Error)?

Measures average squared difference between prediction and real value.

```MSE = average((pred - real)^2)```

👉 Lower MSE = better model

---

## 🔹 What is RMSE (Root Mean Squared Error)?

Square root of MSE:

```RMSE = sqrt(MSE)```

👉 Same unit as output (price)

Example:

```RMSE = 500 → predictions are off by ~±500€```

---

## 🔹 What is R² Score?

R² measures how well the model explains the data.

```R² = 1 - (SS_residual / SS_total)```

---

### Interpretation:

| R² value | Meaning |
|---------|--------|
| 1.0     | Perfect model |
| 0.8     | Very good |
| 0.5     | Average |
| 0.0     | Useless |

---

### Example:

```R² = 0.73 → model explains 73% of variance → GOOD```

---

## 🔹 Training Pipeline (What we built)

### Training:
data → normalize → train (gradient descent) → save model


### Prediction:

```input → normalize → apply model → predicted price```

---

## 🔹 Why normalization is important (experiment)

Without normalization:
- training is unstable
- values explode
- model may produce NaN

With normalization:
- stable learning
- correct convergence
- accurate predictions

---

## 🔹 How to improve the model?

- increase iterations
- adjust learning rate
- normalize data
- add more features (car age, brand, etc.)
- collect more data

---

## 🔹 Limitations of this model

- only 1 feature (mileage)
- assumes linear relationship
- real-world data is more complex

---

## 🔹 Final understanding

In this project, we:

- built a linear model
- trained it using gradient descent
- normalized data for stability
- evaluated performance using metrics
- visualized results

---