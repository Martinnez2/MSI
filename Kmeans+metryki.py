import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
import sys


class Incrementalkmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k, random_state=None, max_iter=100):
        # Liczba klastrów do uzyskania
        self.k = k
        # Wybór losowego podziału danych
        self.random_state = random_state
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

    # Metoda odpowiedzialna za obliczenie dystansu punktów od centroidów
    # def obliczanie_dystansu(self, data_point):
    #     dystanse = np.sqrt(np.sum((self.centroids - data_point[:, np.newaxis]) ** 2, axis=2))
    #     print("Dystanse do centroidów:", dystanse)  # Dodaj wydruk do debugowania
    #     return dystanse

    # Metoda odpowiedzialna za przypisanie klastra do punktu
    def przypisywanie_klastra(self, X):
        odleglosc = np.sqrt(np.sum((self.centroids - X[:, np.newaxis]) ** 2, axis=2))
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


    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        closest_cluster = self.przypisywanie_klastra(X)
        return closest_cluster



# Metryki użyte do ewaluacji
metrics = [fowlkes_mallows_score, adjusted_mutual_info_score]

# ---------------------------------URUCHOMIENIE EKSPERYMENTU---------------------------------
print("Uruchomienie eksperymentu - wyniki metryk w pliku 'wyniki4.txt'")

# Generowanie danych syntetycznych
stream = StreamGenerator(random_state=42, n_chunks=40, chunk_size=40, n_classes=2, n_features=10, n_informative=2,
                         n_redundant=0, n_clusters_per_class=1)

# Inicjalizacja modelu
model = Incrementalkmeans(k=3, random_state=42)

# Inicjalizacja ewaluatora
ewaluator = TestThenTrain(metrics)
ewaluator.process(stream, model)

# Wyniki
ScoresIncremental = ewaluator.scores

# Zapisanie wyników do pliku
with open("wyniki4.txt", 'w') as f:
    sys.stdout = f
    print(ScoresIncremental)
    for m, metric in enumerate(metrics):
        print(f"{metric.__name__}:", np.std(ScoresIncremental[0, :, m]))