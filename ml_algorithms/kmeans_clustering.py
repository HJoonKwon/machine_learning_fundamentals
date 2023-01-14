import numpy as np


class KMeans():
    def __init__(self):
        self.n_clusters = 5
        self.iterations = 300

    def fit(self, X: np.ndarray, n_clusters: int = 5, iterations: int = 300) -> tuple[np.ndarray, ...]:
        self.n_clusters = n_clusters
        self.iterations = iterations

        centroids = self.initialize_centroids(X)
        for i in range(iterations):

            distances = self.euclidean(X, centroids)
            centroid_idxs = np.argmin(distances, axis=1)
            for n in range(n_clusters):
                if np.sum(centroid_idxs==n) == 0:
                    continue
                new_centroid = np.mean(X[centroid_idxs==n], axis=0, keepdims=True)
                centroids[n] = new_centroid

        return centroids, centroid_idxs

    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        # X: (number of examples, number of features)
        min_x, max_x = np.min(X, axis=0), np.max(X, axis=0)
        centroids = np.random.uniform(low=min_x,
                                      high=max_x,
                                      size=(self.n_clusters, X.shape[1]))
        return centroids

    def predict(self, X:np.ndarray, centroids: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        distances = self.euclidean(X, centroids)
        centroid_idxs = np.argmin(distances, axis=1)
        return centroid_idxs

    def euclidean(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # X: (number of examples, number of features)
        # centroids: (number of clusters, number of features)
        n_clusters = centroids.shape[0]
        n_examples = X.shape[0]
        distances = np.zeros((n_examples, n_clusters))
        for i in range(n_clusters):
            distance = np.linalg.norm(X - centroids[i], axis=1)
            distances[:, i] = distance
        return distances
