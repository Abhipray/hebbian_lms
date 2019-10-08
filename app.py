# -*- coding: utf-8 -*-
"""
Created on 2019-10-08

@author: abhipray

"""

import streamlit as st
import numpy as np
import time
import networkx as nx
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
from hebbian_lms import HebbLMSNet
from random_network.random_network import RatsNest

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

# Network Configuration
layer_sizes = [2, 2, 2, 2]
n_iters = 100
gamma = 0.5

plot_num = 1
# fig = plt.figure(figsize=(5, 10))
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    plt.subplot(2,
                len(datasets) / 2,
                plot_num,
                aspect='equal',
                autoscale_on=True,
                adjustable='box')

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.text(0.99,
             0.01,
             plot_num,
             transform=plt.gca().transAxes,
             size=15,
             horizontalalignment='right')
    plot_num += 1
    plt.scatter(X[:, 0], X[:, 1], s=1)

st.sidebar.pyplot()

dataset_choice = st.sidebar.selectbox("Select dataset",
                                      range(1, 1 + len(datasets)))

algo_choice = st.sidebar.selectbox("Select network", ["Rat's nest", "Layered"])

X, y = datasets[dataset_choice - 1][0]
# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

# Show network options and configuration

mu = st.sidebar.slider('Learning rate', 0.0, 2.0, 0.05, 0.01)
if algo_choice == 'Layered':
    hebb = HebbLMSNet(X.shape[1], layer_sizes, -1, mu=mu, gamma=gamma)
    d = hebb.get_graph()
    plt.figure()
    nx.draw(d, pos=nx.shell_layout(d), with_labels=True, alpha=0.2)
    st.pyplot()
    algorithm = hebb
else:
    n_input = X.shape[1]
    n_output = st.sidebar.slider('Number of output neurons', 1, 20, 2)
    n_hidden = st.sidebar.slider('Number of hidden neurons', 1, 100, 6)
    p = st.sidebar.slider('Probability of hidden neuron connection', 0.0, 1.0,
                          0.5, 0.1)
    N = n_hidden + n_input + n_output

    rat = RatsNest(n_input, n_output, n_hidden, mu=mu, p=p)
    d = rat.get_graph()
    pos = nx.shell_layout(d)

    # nodes
    plt.figure()
    nx.draw_networkx_nodes(d,
                           pos,
                           nodelist=range(n_input),
                           node_color='r',
                           node_size=500,
                           alpha=1)
    nx.draw_networkx_nodes(d,
                           pos,
                           nodelist=range(n_input + n_hidden, N),
                           node_color='c',
                           node_size=500,
                           alpha=1)

    nx.draw(d, pos, with_labels=True, alpha=0.2)
    plt.title("Rat's nest Network architecture")
    st.pyplot()
    algorithm = rat

## Do the animation for clustering boundaries

fig, ax = plt.subplots()


def map_decision_boundaries(X, algorithm, ax):
    """
    
    :param X: Points in 2D space 
    :param algorithm: clustering algorithm with a predict() function
    :return: 
    """
    y_pred = algorithm.predict(X)

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

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    plt.cla()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap=cmap)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)


def init():  # give a clean slate to start
    map_decision_boundaries(X, algorithm, ax)


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
init()
the_plot = st.pyplot(fig)


def animate(i):  # update the y values (every 1000ms)
    algorithm.fit(X)
    map_decision_boundaries(X, algorithm, ax)
    the_plot.pyplot(plt)


for i in range(100):
    progress_bar.progress(i)
    status_text.text(f"Epoch {i}")
    animate(i)

st.sidebar.button("Re-run")
