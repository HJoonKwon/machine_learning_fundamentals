import numpy as np


class LinearRegression():
    def __init__(self):
        self.train_X: np.ndarray
        self.train_Y: np.ndarray
        self.theta: np.ndarray

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float,
            iterations: int) -> None:
        self.train_X = X
        self.train_Y = Y
        self.theta = np.zeros((X.shape[1] + 1, Y.shape[1]))

        costs = []
        for iter in range(iterations):
            Y_hat = self.hypothesis(self.train_X)
            cost = self.cost(Y_hat, self.train_Y)
            costs.append(cost)
            if iter % 5 == 0 and iter != 0:
                print(f"Cost @ iter{iter}= {cost}")
            gradients = self.gradients(Y_hat)
            self.update_parameters(gradients, learning_rate)
        return costs 

    def predict(self, X: np.ndarray):
        Y_hat = self.hypothesis(X)
        return Y_hat

    def cost(self, Y_hat: np.ndarray, Y: np.ndarray):
        m = Y.shape[0]
        loss = 1 / (2 * m) * np.sum((Y_hat - Y)**2, axis=0)
        return np.squeeze(loss)

    def gradients(self, Y_hat):
        m = Y_hat.shape[0]
        dw = 1 / m * self.train_X.T @ (Y_hat - self.train_Y)
        db = 1 / m * np.sum((Y_hat - self.train_Y), axis=0, keepdims=True)
        dtheta = np.concatenate((db, dw), axis=0)
        return dtheta

    def hypothesis(self, X: np.ndarray):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        Y_hat = X @ self.theta
        return Y_hat

    def update_parameters(self, gradients: np.ndarray,
                          learning_rate: float) -> None:
        self.theta -= learning_rate * gradients
