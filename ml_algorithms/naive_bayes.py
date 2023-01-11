import numpy as np


def mean_and_std(X: np.ndarray, Y: np.ndarray):

    # input
    # X: m x n
    # Y: m x n
    target_values = np.unique(Y)
    means = []
    stds = []
    for target_value in target_values:
        mean = np.mean(X[(Y[:, 0] == target_value), :], axis=0)
        std = np.std(X[(Y[:, 0] == target_value), :], axis=0)
        means.append(mean)
        stds.append(std)
    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)
    return means, stds


def normalize(X: np.ndarray):

    # X: m x n

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X


class GaussianNaiveBayes():

    def __init__(self):
        self.train_X: np.ndarray
        self.train_Y: np.ndarray
        self.classes: np.ndarray

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:

        # X: (m x n)
        # log(P(X|Y))
        means, stds = mean_and_std(self.train_X, self.train_Y)
        log_likelihood = np.zeros((means.shape[0], X.shape[0]))
        for i in range(means.shape[0]):
            likelihood = 1 / np.sqrt(2 * np.pi) / stds[i] * np.exp(
                -0.5 * np.square((X - means[i]) / stds[i]))
            log_likelihood[i] = np.sum(np.log(likelihood),
                                       axis=1).reshape(1, -1)
        return log_likelihood

    def log_priors(self):
        # log(P(Y))
        priors = np.zeros(self.classes.shape)
        for cls in self.classes:
            priors[cls] = np.count_nonzero(self.train_Y == cls) / len(
                self.train_Y)
        return np.log(priors).reshape(-1, 1)

    def log_scores(self, X: np.ndarray):
        # X: (m x n)
        # scores = posterior * evidence (we don't need to calculate evidence)
        log_priors = self.log_priors()
        log_likelihood = self.log_likelihood(X)
        log_scores = log_priors + log_likelihood
        return log_scores

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.train_X = X
        self.train_Y = Y
        self.classes = np.unique(self.train_Y)

    def predict(self, X: np.ndarray):
        log_scores = self.log_scores(X)
        preds = np.argmax(log_scores, axis=0)
        return preds
