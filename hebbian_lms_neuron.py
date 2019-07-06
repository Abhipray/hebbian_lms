"""
This file implements a Hebbian-LMS neural layer which is a form of unsupervised learning.
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class HebbLMS:
    def __init__(self, input_size, excitatory_ratio, num_neurons, lamda=0.5, mu=0.1):
        """

        :param input_size: Size of the input vector
        :param excitatory_ratio: Ratio of input connections that are excitatory to the total number of connections; value between 0 and 1
        :param num_neurons: Number of neurons in the layer
        """
        self.input_size = input_size
        self.excitatory_ratio = excitatory_ratio
        self.num_neurons = num_neurons
        # Initialize weights randomly from a normal distribution
        self.W = np.random.randn(input_size, num_neurons)
        self.lamda = lamda
        self.mu = mu

    def run(self, X, train=True):
        """
        Train all neurons in a layer simultaneously
        :param X: Each row represents a training vector
        :param train: True if the weights should be updated as the layer receives input
        :return:
        """
        assert X.shape[0] == self.input_size
        n_samples = X.shape[1]

        Y = np.zeros_like(X)
        # Iterate over all training vectors one by one
        for i, x in enumerate(X):
            # Mask input vector with excitatatory vs inhibitory mask
            masked = x
            inhibitory_idx = int(len(x) * self.excitatory_ratio)
            masked[inhibitory_idx:] *= -1

            # Get all neurons sum
            sums = self.W.T @ masked
            sgm = sigmoid(sums)

            # Compute output with half-sigmoid
            output = sgm
            output[output < 0] = 0
            Y[i] = output

            if train == True:
                # Compute feedback error
                err = sgm - self.lamda * sums

                # Update the weights
                self.W += 2 * self.mu * err * x

        return Y







