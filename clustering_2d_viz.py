"""
This script displays Hebbian LMS clustering results on several different datasets containing 2D vectors. 
It is based on sklearn's example for comparing different clustering algorithms.
https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

"""
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from tqdm import tqdm

from hebbian_lms import HebbLMSNet
from random_network.random_network import RatsNest
import os

# Network Configuration
layer_sizes = [4, 4]
excitatory_ratio = 0.5
n_iters = 100
mu = 0.1
gamma = 0.5

config_str = "layers_{} excite_{} iters_{} mu_{} gamma_{}".format(
    str(layer_sizes), excitatory_ratio, n_iters, mu, gamma)
np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples,
                                      factor=.5,
                                      noise=.05)
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
varied = datasets.make_blobs(n_samples=n_samples,
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

plt.figure(figsize=(5, 20))
for i_dataset, (dataset, algo_params) in tqdm(enumerate(datasets)):

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    hebb = HebbLMSNet(X.shape[1],
                      layer_sizes,
                      excitatory_ratio,
                      mu=mu,
                      gamma=gamma)

    n_input = X.shape[1]
    n_output = layer_sizes[-1]
    n_hidden = 10
    N = n_hidden + n_input + n_output

    rat = RatsNest(n_input, n_output, n_hidden)

    clustering_algorithms = (("Rat's nest", hebb), )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        for i in tqdm(range(n_iters)):
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets),
                    len(clustering_algorithms),
                    plot_num,
                    aspect='equal',
                    autoscale_on=True,
                    adjustable='box')
        if i_dataset == 0:
            plt.title(name + '\n' + config_str, size=10)

        colors = np.array(
            list(
                islice(
                    cycle([
                        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                        '#984ea3', '#999999', '#e41a1c', '#dede00'
                    ]), int(max(y_pred) + 1))))

        cmap = plt.get_cmap('Paired')
        # Define region of interest by data limits
        xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
        ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
        steps = 100
        x_span = np.linspace(xmin, xmax, steps)
        y_span = np.linspace(ymin, ymax, steps)
        xx, yy = np.meshgrid(x_span, y_span)

        # Make predictions across region of interest
        labels = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

        print(labels)

        # Plot decision boundary in region of interest
        z = labels.reshape(xx.shape)

        plt.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap=cmap)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99,
                 .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes,
                 size=15,
                 horizontalalignment='right')
        plot_num += 1

os.makedirs('clustering_plots', exist_ok=True)
plt.savefig(os.path.join('clustering_plots', config_str + '.png'))
plt.show()
