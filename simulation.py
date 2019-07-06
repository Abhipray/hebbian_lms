"""
Run a simulation of a hebbian-lms neural net
"""

import numpy as np
from hebbian_lms_neuron import HebbLMS


n_training_samples = 50
n_neurons = 5
n_weights = 100
excitatory_ratio = 0.5

hebb = HebbLMS(n_weights, excitatory_ratio, n_neurons)

# Create the training matrix by sampling from a uniform distribution; each row is a training vector
X = np.random.rand(n_training_samples, n_weights)

Y = hebb.run(X, train=True)

