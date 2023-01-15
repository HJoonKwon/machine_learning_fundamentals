from collections import Counter

import numpy as np


class KNearestNeighbors():
    def __init__(self):
        self.k = 5
        self.X_train: np.ndarray
        self.Y_train: np.ndarray

    def fit(self, X: np.ndarray, Y: np.ndarray, k: int) -> None:
        self.k = k
        self.X_train = X
        self.Y_train = Y

    def predict(self, X_test: np.ndarray) -> list:
        y_preds = []
        for x_test in X_test:
            distances = self.euclidean(x_test, self.X_train)
            y_sorted = [
                y for _, y in sorted(zip(distances, self.Y_train),
                                     key=lambda x: x[0])
            ]
            y_sorted_k = y_sorted[0:self.k]
            y_pred = self.most_common(y_sorted_k)
            y_preds.append(y_pred)
        return y_preds

    def euclidean(self, x_test: np.ndarray, X_train: np.ndarray):
        distances = np.linalg.norm(x_test - X_train, axis=1)
        return distances

    def most_common(self, y_sorted: list):
        y_counter = Counter(y_sorted)
        return y_counter.most_common(1)[0][0]
