"""
This file implements a Hebbian-LMS neural layer which is a form of unsupervised learning.
"""

import numpy as np
from sklearn import preprocessing


class HebbLMSNet:
    def __init__(self,
                 input_size: int,
                 layer_sizes: list,
                 excitatory_ratio,
                 gamma=0.5,
                 mu=0.1):
        """
        :param input_size: Length of the input vector
        :param layer_sizes: list containing number of neurons for each layer
        :param excitatory_ratio: Ratio of input connections that are excitatory to the total number of connections; value between 0 and 1
        :param gamma: value between 0 and 1 used in calculating errors for updating weights
        :param mu: learning rate
        """
        self.input_size = input_size
        self.excitatory_ratio = excitatory_ratio
        self.layer_sizes = [input_size] + layer_sizes
        self.layer_weights = []
        # Initialize weights for all layers randomly from a uniform distribution ~ [0, 1]
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.layer_weights.append(w)
        self.gamma = gamma
        self.mu = mu

    def run(self, X, train=True):
        """
        Train all neurons in a layer simultaneously
        :param X: Each row represents a training vector
        :param train: True if the weights should be updated as the layer receives input
        :return:
        """
        assert X.shape[1] == self.input_size
        n_samples = X.shape[0]
        output_size = self.layer_sizes[1]

        Y = np.zeros((n_samples, output_size))
        hidden_sum = np.zeros((n_samples, output_size))
        error = np.zeros((n_samples, output_size))
        # Iterate over all training vectors one by one
        for i, x in enumerate(X):
            input = np.copy(x)
            output = np.zeros((output_size, ))
            # Iterate over layers
            for W in self.layer_weights:
                # Mask input vector with excitatatory vs inhibitory mask
                masked = np.copy(input)
                inhibitory_idx = int(len(x) * self.excitatory_ratio)
                masked[inhibitory_idx:] *= -1

                # Get all neurons sum
                sums = W.T @ masked
                sgm = np.tanh(sums)

                # Compute output with half-sigmoid
                output = np.copy(sgm)
                output[output < 0] = 0
                hidden_sum[i] = sums

                if train:
                    # Compute feedback error
                    err = sgm - self.gamma * sums
                    error[i] = err

                    # Update the weights
                    W += 2 * self.mu * (masked[:, None] @ err[None, :])

                    # Check to see if weights are non-negative
                    assert (W.all() >= 0)
                input = output  # Feed the output of this layer into the next layer
            Y[i] = output.T

        return Y, hidden_sum, error

    def fit(self, X):
        self.run(X, train=True)

    def predict(self, X):
        encodings = self.run(X, train=False)[0]
        # Convert encodings into the integer labels for cluster membership of each sample.

        # Turn all values greater than 0 into 1
        encodings[encodings > 0] = 1

        # Convert binary numbers to integer representation
        y_pred = np.zeros(encodings.shape[0]).astype(np.int)
        for i, encoding in enumerate(encodings):
            y_pred[i] = int(''.join([str(k) for k in encoding.astype(np.int)]),
                            2)

        le = preprocessing.LabelEncoder()
        return le.fit_transform(y_pred)
