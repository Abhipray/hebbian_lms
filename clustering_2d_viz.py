"""
This script displays Hebbian LMS clustering results on several different datasets containing 2D vectors. 
It is based on sklearn's example for comparing different clustering algorithms.

"""
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from hebbian_lms import HebbLMSNet

# Network Configuration
layer_sizes = [3]
excitatory_ratio = 0.5
n_iters = 10
mu = 0.1
gamma = 0.75

config_str = "\nlayers:{},excite:{},iters:{},mu:{},gamma:{}".format(
    str(layer_sizes), excitatory_ratio, n_iters, mu, gamma)
np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=random_state)

datasets = [(noisy_circles, {
    'damping': .77,
    'preference': -240,
    'quantile': .2,
    'n_clusters': 2,
    'min_samples': 20,
    'xi': 0.25
}), (noisy_moons, {
    'damping': .75,
    'preference': -220,
    'n_clusters': 2
}),
            (varied, {
                'eps': .18,
                'n_neighbors': 2,
                'min_samples': 5,
                'xi': 0.035,
                'min_cluster_size': .2
            }),
            (aniso, {
                'eps': .15,
                'n_neighbors': 2,
                'min_samples': 20,
                'xi': 0.1,
                'min_cluster_size': .2
            }), (blobs, {}), (no_structure, {})]

plot_num = 1

for i_dataset, (dataset, algo_params) in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    hebb = HebbLMSNet(
        X.shape[1], layer_sizes, excitatory_ratio, mu=mu, gamma=gamma)

    clustering_algorithms = (('HebbianLms', hebb), )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            for i in range(n_iters):
                algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name + config_str, size=10)

        colors = np.array(
            list(
                islice(
                    cycle([
                        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00'
                    ]), int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            .99,
            .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment='right')
        plot_num += 1

plt.show()
