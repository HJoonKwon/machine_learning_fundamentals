import numpy as np


class GaussianDiscriminantAnalysis():

    def __init__(self):
        self.theta: np.ndarray

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):

        # train_X : (m, n)
        # train_Y : (m, )
        # phi, mu0, mu1, Cov

        m, n = train_X.shape[0], train_X.shape[1]

        self.theta = np.zeros(n+1)

        y1 = np.sum(train_Y)
        phi = 1/m * y1 # real number
        mu1 = np.sum(train_X[train_Y==1], axis=0) / y1
        mu0 = np.sum(train_X[train_Y==0], axis=0) / (m-y1)
        cov = 1/m * ((train_X - mu0).T @ (train_X - mu0) + (train_X - mu1).T @ (train_X - mu1))

        assert mu1.shape == (train_X.shape[1],)
        assert mu0.shape == (train_X.shape[1],)
        assert cov.shape == (train_X.shape[1], train_X.shape[1])

        cov_inv = np.linalg.inv(cov)
        self.theta[0] = 0.5 * (mu0 + mu1).T.dot(cov_inv).dot(mu0 - mu1) - np.log((1-phi)/phi)
        self.theta[1:] = cov_inv.dot(mu1 - mu0)
        return self.theta


    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (m, n)
        yhat = 1 / (1 + np.exp(- X.dot(self.theta)))
        return yhat

