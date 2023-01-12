import numpy as np


class SupportVectorMachine():
    def __init__(self, c: float = 1.0) -> None:
        self.train_X: np.ndarray
        self.train_Y: np.ndarray
        self.W: np.ndarray
        self.b: np.ndarray
        self.c = c

    def fit(self, X:np.ndarray, Y:np.ndarray, learning_rate: float, iterations: int) -> list[float]:

        self.train_X = X
        self.train_Y = Y

        self.W = np.zeros((self.train_Y.shape[0], self.train_X.shape[0]))
        self.b = np.zeros((self.train_Y.shape[0], 1))

        losses = []
        for iter in range(1, iterations+1):

            T = self.train_Y * (self.W @ self.train_X + self.b)
            hinge_loss = self.hinge_loss(self.train_X, self.train_Y, T)
            losses.append(hinge_loss)
            gradients = self.gradients(T)
            self.update_parameters(gradients, learning_rate)
            if iter %5 == 0:
                print(f"Loss @ iter{iter}= {hinge_loss}")
        return losses

    def predict(self, X) -> np.ndarray:
        Z = self.W @ X + self.b
        return np.sign(Z)

    def hinge_loss(self, X:np.ndarray, Y:np.ndarray, T:np.ndarray) -> float:
        reg_term = 0.5 * np.linalg.norm(self.W, ord=2)
        err_term = np.maximum(0, 1 - T)
        err_term = np.sum(err_term, axis=1)
        err_term = np.squeeze(err_term)
        loss = reg_term + self.c * err_term
        return loss

    def gradients(self, T:np.ndarray) -> tuple[np.ndarray, ...]:

        m = self.train_Y.shape[1]
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        for i in range(m):
            if T[:, i] <= 1:
                dW += - self.c * self.train_Y[:, i, None] @ self.train_X[:, i, None].T
                db += - self.c * self.train_Y[:, i, None]
        dW += self.W
        return dW, db

    def update_parameters(self, gradients: np.ndarray, learnig_rate: float) -> None:
        dW, db = gradients
        self.W -= learnig_rate * dW
        self.b -= learnig_rate * db
