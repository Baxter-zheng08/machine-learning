import numpy as np
from ucimlrepo import fetch_ucirepo

# load dataset
dataset = fetch_ucirepo(id=477)
X = dataset.data.features.values
y = dataset.data.targets.values

# split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add bias term
X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])

def gradient_descent(X, y, theta, lr, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        y_pred = X @ theta
        grad = (1/m) * X.T @ (y_pred - y)
        theta = theta - lr * grad
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        cost_history.append(cost)
    return theta, cost_history

feature_num = X_train.shape[1]
theta_init = np.zeros((feature_num,1))
lr = 0.1
iter_num = 2000

theta_final, cost_hist = gradient_descent(X_train, y_train, theta_init, lr, iter_num)
print("Training finished, final parameter theta:")
print(theta_final)
print(f"Final loss on training set: {cost_hist[-1]:.4f}")

def evaluate_model(X_test, y_test, theta):
    y_predict = X_test @ theta
    mse = np.mean((y_predict - y_test)**2)
    rmse = np.sqrt(mse)
    print("\n==== Evaluation on unseen test dataset =====")
    print(f"MSE = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    return y_predict

y_pred_test = evaluate_model(X_test, y_test, theta_final）
