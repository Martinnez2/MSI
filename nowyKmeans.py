import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from scipy.spatial import distance
from Minibatch import Minibatchnew
from Birchnew import Birchnew


class Incrementalkmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k, max_iter=100):
        # Liczba klastrów do uzyskania
        self.k = k
        # Wybór losowego podziału danych
       # self.random_state = random_state
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


# Metryki użyte do ewaluacji
metrics = [fowlkes_mallows_score, adjusted_mutual_info_score]

# ---------------------------------URUCHOMIENIE EKSPERYMENTU---------------------------------
print("Uruchomienie eksperymentu - wyniki metryk w pliku 'wyniki4.txt'")

# Generowanie danych syntetycznych
stream = StreamGenerator(random_state=10, n_chunks=40, chunk_size=40, n_classes=2, n_features=10, n_informative=2,
                         n_redundant=0, n_clusters_per_class=1)

#INCREMENTALKMEANS

# Inicjalizacja modelu
model = Incrementalkmeans(k=3)

# Inicjalizacja ewaluatora dla incremetnal keamns
ewaluator = TestThenTrain(metrics)
ewaluator.process(stream,model)

ScoresIncremental = ewaluator.scores
print("fowkles score")
print(np.mean(ScoresIncremental[0,:,0]))
print("adjusted mututal score")
print(np.mean(ScoresIncremental[0,:,1]))



# Generowanie danych syntetycznych
stream = StreamGenerator(random_state=10, n_chunks=40, chunk_size=40, n_classes=2, n_features=10, n_informative=2,
                         n_redundant=0, n_clusters_per_class=1)

#BIRCH

# Inicjalizacja modelu
model = Birchnew(n_clusters=3)

# Inicjalizacja ewaluatora dla incremetnal keamns
ewaluator = TestThenTrain(metrics)
ewaluator.process(stream,model)

ScoresBirch = ewaluator.scores
print("fowkles score")
print(np.mean(ScoresBirch[0,:,0]))
print("adjusted mututal score")
print(np.mean(ScoresBirch[0,:,1]))



# Generowanie danych syntetycznych
stream = StreamGenerator(random_state=10, n_chunks=40, chunk_size=40, n_classes=2, n_features=10, n_informative=2,
                         n_redundant=0, n_clusters_per_class=1)

#MINIBATCH

# Inicjalizacja modelu
model = Minibatchnew(n_clusters=3)

# Inicjalizacja ewaluatora dla incremetnal keamns
ewaluator = TestThenTrain(metrics)
ewaluator.process(stream,model)

ScoresMinibatch = ewaluator.scores
print("fowkles score")
print(np.mean(ScoresMinibatch[0,:,0]))
print("adjusted mututal score")
print(np.mean(ScoresMinibatch[0,:,1]))




plt.figure(figsize=(6,3))

for m, metric in enumerate(metrics):
    plt.plot(ScoresIncremental[0, :, m], label=metric.__name__+" Incrementalkmeans")
    plt.plot(ScoresBirch[0, :, m], label=metric.__name__+" Birch")
    plt.plot(ScoresMinibatch[0, :, m], label=metric.__name__+" Minibatch")
    plt.title("Basic example of stream processing")
    plt.ylim(0, 1)
    plt.ylabel('Quality')
    plt.xlabel('Chunk')
    plt.legend()
    plt.show()