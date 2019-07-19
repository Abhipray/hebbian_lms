"""
Run a simulation of a single hebbian-lms neuron. The neuron receives a set of uniformly distributed input vectors.
It should learn to cluster the points into two groups. 

The script runs an animation of the clustering process.
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
seaborn.set_style("whitegrid")

from hebbian_lms import HebbLMSNet

# np.random.seed(10)

# Network Configuration
n_training_samples = 50
n_neurons = 1
n_weights = 100
excitatory_ratio = 0.5
n_iters = 100
mu = 0.01
gamma = 0.75

hebb = HebbLMSNet(n_weights, [n_neurons], excitatory_ratio, mu=mu, gamma=gamma)

# Create the training matrix by sampling from a uniform distribution; each row is a training vector
X = np.random.rand(n_training_samples, n_weights)

all_errors = []

# Clustering animation
for i in range(n_iters):
    np.random.shuffle(X)
    Y, sums, error = hebb.run(X, train=True)
    all_errors.append(np.sum(error**2))

    # Update the animation plot
    plt.cla()
    plt.axis([-5, 5, -2, 2])
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, np.tanh(x), linewidth=0.5)
    plt.plot(x, x, 'y-', linewidth=0.5)

    pos_idx = Y > 0
    neg_idx = Y <= 0
    plt.plot(sums[pos_idx], Y[pos_idx], 'xb')
    plt.plot(sums[neg_idx], Y[neg_idx], 'or')
    plt.xlabel("sum")
    plt.ylabel("tanh(sum)")
    plt.title("Clustering process")
    plt.pause(0.05)

plt.figure()
plt.plot(np.array(all_errors)**2)
plt.title("Learning curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()
plt.close()
