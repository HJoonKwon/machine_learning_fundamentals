import numpy as np


class LogisticRegression():
    def __init__(self):
        self.train_X: np.ndarray
        self.train_Y: np.ndarray
        self.W: np.ndarray
        self.b: np.ndarray

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float,
            iterations: int):

        self.train_X = X
        self.train_Y = Y
        self.W = np.zeros((self.train_Y.shape[0], self.train_X.shape[0]))
        self.b = np.zeros((self.train_Y.shape[0], 1))

        costs = []
        for iter in range(iterations):
            Z = self.linear_forward(self.train_X)
            Y_hat = self.sigmoid(Z)
            cost = self.cost(Y_hat, self.train_Y)
            gradients = self.gradients(Y_hat)
            self.update_parameters(gradients, learning_rate)
            costs.append(cost)
            if iter % 50 == 0 and iter != 0:
                print(f"cost @ iter{iter}= {cost}")

        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.linear_forward(X)
        Y_hat = self.sigmoid(Z)
        preds = Y_hat >= 0.5
        return preds

    def gradients(self, Y_hat: np.ndarray) -> tuple[np.ndarray, ...]:
        dY_hat = self.cost_backward(Y_hat, self.train_Y)
        dZ = self.sigmoid_backward(Y_hat, dY_hat)
        dW, db = self.linear_backward(self.train_X, dZ)
        return (dW, db)

    def update_parameters(self, gradients: tuple[np.ndarray, np.ndarray],
                          learning_rate: float) -> None:
        dW, db = gradients

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def cost(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        m = Y.shape[1]
        cross_entropy = -1 / m * (Y @ np.log(Y_hat).T +
                                  (1 - Y) @ np.log(1 - Y_hat).T)
        return np.squeeze(cross_entropy)

    def cost_backward(self, Y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dY_hat = -np.divide(Y, Y_hat) + np.divide(1 - Y, 1 - Y_hat)
        return dY_hat

    def linear_forward(self, X: np.ndarray) -> np.ndarray:
        Z = self.W @ X + self.b
        return Z

    def linear_backward(self, X: np.ndarray,
                        dZ: np.ndarray) -> tuple[np.ndarray, ...]:
        m = X.shape[1]
        dW = 1 / m * dZ @ X.T
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        return (dW, db)

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        Y_hat = 1 / (1 + np.exp(-Z))
        return Y_hat

    def sigmoid_backward(self, Y_hat: np.ndarray,
                         dY_hat: np.ndarray) -> np.ndarray:
        dZ = dY_hat * Y_hat * (1 - Y_hat)
        return dZ
