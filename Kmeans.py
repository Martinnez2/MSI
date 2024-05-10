import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt


class Incrementalkmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k, randomstate=None, max_iter=100):
        self.k = k
        self.randomstate = randomstate
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    # Metoda partial_fit - losuje centroidy, przypisuje dane do klastra, aktualizuje pozycję centroidów
    def partial_fit(self, X):
        if self.centroids is None:
            self.centroids = self.losowanie_centrow(X)
        for _ in range(self.max_iter):
            closest_cluster = self.przypisywanie_klastra(X)
            self.aktualizacja(X, closest_cluster)
        self.labels_ = closest_cluster
        return self.labels_

    # Metoda odpowiedzialna za losowanie centroidów
    def losowanie_centrow(self, X):
        num_points = X.shape[0]
        print(num_points)
        random_indices = np.random.choice(num_points, self.k, replace=False)
        print(random_indices)
        return X[random_indices]

    # Metoda odpowiedzialna za przypisanie klastra do punktu
    def przypisywanie_klastra(self, X):
        odleglosc = np.sqrt(np.sum((self.centroids - X) ** 2, axis=1))
        min_odleglosc = np.argmin(odleglosc, axis=1)
        return min_odleglosc

    # Metoda odpowiedzialna za aktualizację pozycji centroidu
    def aktualizacja(self, X, closest_cluster):
        new_centroids = []
        for k in range(self.k):
            cluster_points = X[closest_cluster == k]  # Punkty należące do klastra k
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)  # Średnia punktów klastra
                new_centroids.append(new_centroid)
        self.centroids = np.array(new_centroids)


random_points = np.random.randint(0, 100, (100, 2))

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

blobls = make_blobs(n_samples=100, n_features=2,centers=3)

data = blobls[0]
kmeans = Incrementalkmeans(k=3)
labels = kmeans.partial_fit(data)
ari = adjusted_rand_score(blobls[1], labels)
print(labels)
print(blobls[1])
print(ari)
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.show()