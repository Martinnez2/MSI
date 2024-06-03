from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import Birch

class Birchnew(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Birch(n_clusters=n_clusters)


    def partial_fit(self, X, y, classes=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)