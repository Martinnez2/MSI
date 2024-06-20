import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from strlearn.streams import StreamGenerator
from kMeans import Incrementalkmeans
from strlearn.evaluators import TestThenTrain
from Minibatch import Minibatchnew
from Birchnew import Birchnew
import sys
from scipy.stats import ttest_ind

# ---------------------------------URUCHOMIENIE EKSPERYMENTU---------------------------------
print("Uruchomienie eksperymentow - wyniki metryk beda zapisane w pliku odpowiadajacym numer eksperymentu")

# Metryki użyte do ewaluacji
metrics = [fowlkes_mallows_score, adjusted_mutual_info_score]

# Lista modeli
models = [
    ("IncrementalKmeans", Incrementalkmeans(k=3)),
    ("Birch", Birchnew(n_clusters=3)),
    ("Minibatch", Minibatchnew(n_clusters=3))
]
scores_dict = np.zeros((10,499,2,3))
for experiment in range(1, 11):
    random_state = 100 * experiment  # random_state w zakresie od 100 do 1000 co 100

    # Przetwarzanie każdego modelu
    for model_index, (name,model) in enumerate(models):
        # Inicjalizacja strumienia danych dla każdego modelu
        stream = StreamGenerator(random_state=random_state, n_chunks=500, chunk_size=500, n_classes=3, n_features=20,
                                 n_informative=2,
                                 n_redundant=0, n_clusters_per_class=1, class_sep=2)

        # Inicjalizacja ewaluatora
        ewaluator = TestThenTrain(metrics, verbose=True)

        # Przetwarzanie danych za pomocą modelu
        ewaluator.process(stream, model)
        scores_dict[experiment-1,:,:,model_index] = ewaluator.scores

    # Zapis wyników do pliku
    with open(f"wyniki{experiment}.txt", 'w') as f:
        sys.stdout = f
        print(f"random_state: {random_state}\n")

        for model_idx, (name, model) in enumerate(models):
            print(f"Wyniki dla modelu: {name}")
            for m, metric in enumerate(metrics):
                print(scores_dict[experiment-1,:,m,model_idx])
                mean_score = np.mean(scores_dict,axis=1)
                std_score = np.std(scores_dict[experiment-1, :, m, model_idx])
                print(f"{metric.__name__}:")
                print(f" srednia: {mean_score[experiment-1,m,model_idx]}")
                print(f" Odchylenie standardowe: {std_score}\n")

    res_temp=scores_dict.mean(axis=1)
            # Test t-Studenta
    for i, metric in enumerate(metrics):
        res = res_temp[:, i]
        with open(f"wyniki{experiment}.txt", 'a') as f:  # 'a' - dołączenie do istniejącego pliku
            sys.stdout = f
            print(f"Metryka: {metric.__name__}\n")
            for i in range(len(models)):  # Pętla po klasyfikatorach.
                for j in range(i + 1, len(models)):  # Pętla po klasyfikatorach.
                    if i != j:  # Sprawdzanie, czy klasyfikatory są różne
                        t_stat, p_value = ttest_ind(res[:, i], res[:, j])
                        print(f"Test t studenta wyniki: {models[i][0]} i {models[j][0]}: t = {t_stat}, p = {p_value}")

    sys.stdout = sys.__stdout__

    print(f"Wyniki dla eksperymentu {experiment}")
    print(f"random_state: {random_state}")

    # Wykres wyników
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for m, metric in enumerate(metrics):
        for idx, (name, model) in enumerate(models):
            ax[m].plot(scores_dict[experiment - 1, :, m, idx], label=name)
        ax[m].set_title(f"{metric.__name__} Scores")
        ax[m].set_ylim(0, 1)
        ax[m].set_ylabel('Quality')
        ax[m].set_xlabel('Chunk')
        ax[m].legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)