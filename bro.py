# Workshop 4: Linear Regression & ML Fundamentals Demonstration
#HomeWork
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.figure(figsize=(9, 7))
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Tyreese Synthetic Dataset")
plt.show()

class LinearRegressionGD:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.m, self.n = X.shape
        X_b = np.c_[np.ones ((self.m, 1)), X]
        self.theta = np.zeros((self.n + 1, 1))

        for _ in range(self.n_iterations):
            gredients = (2 / self.m) * X_b.T @ (X_b @ self.theta - y)
            self.theta -= self.learning_rate * gredients
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
lr_gd = LinearRegressionGD(learning_rate = 0.1, n_iterations = 1000)
lr_gd.fit(X, y)
y_pred_gd = lr_gd.predict(X)

lr_sklearn = LinearRegression()
lr_sklearn.fit(X, y)
y_pred_sklearn = lr_sklearn.predict(X)

mse_gd = mean_squared_error(y, y_pred_gd)
r2_gd = r2_score(y, y_pred_gd)

mse_sklearn = mean_squared_error(y, y_pred_sklearn)
r2_sklearn = r2_score(y, y_pred_sklearn)

print("Gridient Descent Linear Regression:")
print(f"MSE: {mse_gd}")
print(f"R-squared: {r2_gd}")

print("\nSklearn Linear Regression:")
print(f"MSE: {mse_sklearn}")
print(f"R-squared: {r2_sklearn}")

plt.scatter(X, y, color='pink', label='Data')
plt.plot(X, y_pred_gd, color='orange', label='Gradient Descent')
plt.plot(X, y_pred_sklearn, color='purple', label='Sklearn')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Tyreese Linear Regression Comparison")
plt.show()