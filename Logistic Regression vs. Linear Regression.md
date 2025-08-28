# Logistic Regression vs. Linear Regression

## Introduction
This file provides an overview of **Logistic Regression** and **Linear Regression**, explaining their differences, use cases, and how they work in machine learning.

## 1. What is Logistic Regression?
Logistic Regression is a **classification algorithm** used to predict categorical outcomes. Despite its name, it is not used for regression but rather for classification tasks.

### How it Works:
- Logistic Regression uses the **sigmoid function** to map predicted values between 0 and 1.
- It outputs probabilities and applies a threshold (e.g., 0.5) to classify data.
- If there are **two classes**, it performs **binary classification**.
- If there are **more than two classes**, it uses the **softmax function** for **multiclass classification**.

### Example Use Cases:
- **Spam detection**: Classify emails as "spam" or "not spam."
- **Medical diagnosis**: Predict whether a patient has a disease.
- **Iris flower classification**: Classify a flower into three species (`setosa`, `versicolor`, `virginica`).

### Logistic Regression Formula:

<img width="370" alt="logistic" src="https://github.com/user-attachments/assets/719fd750-965c-4cd0-9a1a-7dd89efdfbe3" />


## 2. What is Linear Regression?
Linear Regression is a **regression algorithm** used to predict **continuous numerical values**.

### How it Works:
- It finds the **best-fitting line** that minimizes the error between actual and predicted values.
- The model assumes a linear relationship between input variables (`X`) and output (`y`).
- For multiple features, it is called **Multiple Linear Regression**.

### Example Use Cases:
- **House Price Prediction**: Estimate house prices based on size, location, etc.
- **Salary Prediction**: Predict salaries based on years of experience.
- **Stock Price Forecasting**: Predict future stock prices.

### Linear Regression Formula:

![Linear-regression-model-equation](https://github.com/user-attachments/assets/3a44f8fb-cf3b-47ce-8125-8f912d036ea3)


## 3. Key Differences
| Feature  | Linear Regression | Logistic Regression |
|----------|------------------|---------------------|
| **Output Type** | Continuous values (e.g., price, temperature) | Discrete categories (e.g., spam/not spam) |
| **Equation Used** | \( Y = mX + b \) | \( P = \frac{1}{1 + e^{-z}} \) (sigmoid function) |
| **Use Case** | Regression problems | Classification problems |

## 4. When to Use Which?
- Use **Linear Regression** when predicting **numerical values**.
- Use **Logistic Regression** when predicting **categories**.

## 5. Getting Started with Code
### Logistic Regression Example:
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Example Data (features and labels)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array(["cat", "dog", "cat", "dog"])

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Make a Prediction
prediction = model.predict([[2, 3]])
print("Predicted Class:", prediction[0])
```

### Linear Regression Example:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example Data (features and continuous labels)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Make a Prediction
prediction = model.predict([[6]])
print("Predicted Value:", prediction[0])
```

## 6. Conclusion
- **Logistic Regression** is used for classification.
- **Linear Regression** is used for regression.
- Choose the right algorithm based on whether your output is **categorical** or **continuous**.

## 7. License
This project is open-source and available for use under the MIT License.
