import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Iterable, Tuple

plt.rcParams.update({'font.size': 16})


def set_seed(seed):
    np.random.seed(seed)


def mean_confidence_interval(vals: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    a = np.array(vals, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        raise ValueError("mean_confidence_interval requires at least one non-NaN value.")
    m = float(np.mean(a))
    if n < 2:
        return m, 0.0
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return m, h


def plot_clusters(centroids, values):
    plt.clf()
    distances = distance_matrix_squared(values, centroids)
    associations = np.argmin(distances, axis=1)
    colors = ['r', 'g', 'b', 'y', 'darkgray', 'cyan', 'pink', 'orange', 'purple', 'olive', 'gray', 'brown', 'teal',
              'yellowgreen', 'lightcoral', 'lightpink', 'peru', 'tomato', 'gold', 'magenta']
    k = centroids.shape[0]
    for cluster in range(k):
        vals = values[associations == cluster]
        plt.scatter(vals[:, 0], vals[:, 1], color=colors[cluster % len(colors)])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')

def plot_cluster_history(centroids_history, final_guess,canonical, values):
    """
    centroids_history: list of arrays of shape (k, d), one per iteration
    values: data points, shape (n, d)
    """
    plt.clf()

    distances = distance_matrix_squared(values, final_guess)
    associations = np.argmin(distances, axis=1)

    colors = ['r', 'g', 'b', 'y', 'darkgray', 'cyan', 'pink', 'orange',
              'purple', 'olive', 'gray', 'brown', 'teal', 'yellowgreen',
              'lightcoral', 'lightpink', 'peru', 'tomato', 'gold', 'magenta']

    k = final_guess.shape[0]

    # Plot clustered points
    for cluster in range(k):
        vals = values[associations == cluster]
        plt.scatter(vals[:, 0], vals[:, 1],
                    color=colors[cluster % len(colors)], alpha=0.3)

    # Plot centroid paths with arrows
    for cluster in range(k):
        for t in range(len(centroids_history) - 1):
            start = centroids_history[t][cluster]
            end = centroids_history[t + 1][cluster]

            dx, dy = end - start

            plt.plot([start[0], start[0] + dx], [start[1], start[1] + dy],
                     color='black',
                     linewidth=3,  
                     alpha=1,
                     zorder=1)    

            # 2. Draw the "Inner" Line (The actual color)
            plt.plot([start[0], start[0] + dx], [start[1], start[1] + dy],
                     color=colors[cluster % len(colors)],
                     linewidth=1.5, 
                     alpha=1,
                     zorder=2)

    # Plot final centroids
    plt.scatter(final_guess[:, 0], final_guess[:, 1],
                color='grey', marker='x', s=100)

    plt.scatter(canonical[:, 0], canonical[:, 1],
                color='black', marker='x', s=100)


def distance_matrix_squared(X, Y):
    return np.sum((X[:, np.newaxis] - Y) ** 2, axis=2)
