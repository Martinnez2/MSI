import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy.spatial import distance
from scipy.stats import ttest_ind



class Incrementalkmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k, max_iter = 100):
        # Liczba klastrów do uzyskania
        self.k = k
        # Wybór liczby iteracji
        self.max_iter = max_iter
        # Pozycja centroidów
        self.centroids = None
        # Przechowywanie etykiet klastrów
        self.labels_ = None

    # Metoda partial_fit - losuje centroidy, przypisuje dane do klastra, aktualizuje pozycję centroidów
    def partial_fit(self, X, y=None, classes_=None):
        if self.centroids is None:
            self.centroids = self.losowanie_centrow(X)
        for _ in range(self.max_iter):
            closest_cluster = self.przypisywanie_klastra(X)
            self.aktualizacja(X, closest_cluster)
        self.labels_ = closest_cluster
        return self

    # Metoda odpowiedzialna za losowanie centroidów
    def losowanie_centrow(self, X):
        num_points = X.shape[0]
        random_indices = np.random.choice(num_points, self.k, replace=False)
        return X[random_indices]

    # Metoda odpowiedzialna za przypisanie klastra do punktu
    def przypisywanie_klastra(self, X):
        odleglosc = distance.cdist(X, self.centroids)
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
            else:
                new_centroids.append(self.centroids[k])
        self.centroids = np.array(new_centroids)

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        closest_cluster = self.przypisywanie_klastra(X)
        return closest_cluster
