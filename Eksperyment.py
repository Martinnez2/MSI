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
import sys

class Incrementalkmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k, max_iter=100):
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


# ---------------------------------URUCHOMIENIE EKSPERYMENTU---------------------------------
print("Uruchomienie eksperymentu - wyniki metryk w pliku 'wyniki.txt'")

# Metryki użyte do ewaluacji
metrics = [fowlkes_mallows_score, adjusted_mutual_info_score]

# Lista modeli
models = [
    ("IncrementalKmeans", Incrementalkmeans(k=3)),
    ("Birch", Birchnew(n_clusters=3)),
    ("Minibatch", Minibatchnew(n_clusters=3))
]




# Słownik do przechowywania wyników
scores_dict = {}

# Przetwarzanie każdego modelu
for name, model in models:
    # Inicjalizacja strumienia danych dla każdego modelu
    stream = StreamGenerator(random_state=200, n_chunks=100, chunk_size=100, n_classes=3, n_features=20, n_informative=2,
                             n_redundant=0, n_clusters_per_class=1, class_sep=2)

    # Inicjalizacja ewaluatora
    ewaluator = TestThenTrain(metrics,verbose=True)

    # Przetwarzanie danych za pomocą modelu
    ewaluator.process(stream, model)
    scores_dict[name] = ewaluator.scores

# Zapis wyników do pliku
with open("wyniki.txt", 'w') as f:
    sys.stdout = f

    for name in scores_dict:
        print(name)
        print(scores_dict[name])
        print("fowlkes score")
        print(np.mean(scores_dict[name][0, :, 0]))
        print("adjusted mutual score")
        print(np.mean(scores_dict[name][0, :, 1]))

# Wykres wyników
fig,ax=plt.subplots(1,2,figsize=(12,4))

for m, metric in enumerate(metrics):
    for name in scores_dict:
        ax[m].plot(scores_dict[name][0, :, m], label=f"{name}")
    ax[m].set_title(f"{metric.__name__} Scores")
    ax[m].set_ylim(0, 1)
    ax[m].set_ylabel('Quality')
    ax[m].set_xlabel('Chunk')
    ax[m].legend()
plt.show()
plt.tight_layout()