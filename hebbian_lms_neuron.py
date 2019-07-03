"""
This file implements a Hebbian-LMS neural layer which is a form of unsupervised learning.
"""

import numpy as np


class HebbLMS:
    def __init__(self, input_size, excitatory_ratio, num_neurons):
        """

        :param input_size: Size of the input vector
        :param excitatory_ratio: Ratio of input connections that are excitatory; value between 0 and 1
        :param num_neurons: Number of neurons in the layer
        """
        self.input_size = input_size
        self.excitatory_ratio = excitatory_ratio
        self.num_neurons = num_neurons
        # Initialize weights randomly from a normal distribution
        self.W = np.random.randn(input_size, num_neurons)

    