"""
Run a simulation of a hebbian-lms neural net
"""

import numpy as np
import matplotlib.pyplot as plt

from hebbian_lms_neuron import HebbLMS

np.random.seed(10)

n_training_samples = 50
n_neurons = 1
n_weights = 100
excitatory_ratio = 0.5
n_iters = 1000

hebb = HebbLMS(n_weights, excitatory_ratio, n_neurons, mu=0.01, gamma=0.5)

# Create the training matrix by sampling from a uniform distribution; each row is a training vector
X = np.random.rand(n_training_samples, n_weights)

for i in range(n_iters):
    # np.random.shuffle(X)
    Y, sums, error = hebb.run(X, train=True)

plt.stem(Y)
plt.show()
plt.figure()
plt.stem(sums)
plt.show()

plt.figure()
plt.plot(error**2)
plt.show()






