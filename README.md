# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. 1.Load the dataset from a CSV file and separate the features and target variable,
encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a
specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and
encoding applied to the training data.
``` 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:  S.Kowshik Ram 
RegisterNumber: 212225230143
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
*/
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="927" height="636" alt="Screenshot 2026-04-22 091914" src="https://github.com/user-attachments/assets/48b81355-b0cc-4873-b51c-c1c10a8da262" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
