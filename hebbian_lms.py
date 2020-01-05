"""
This file implements a Hebbian-LMS neural network which is a form of unsupervised learning. 
See "The Hebbian-LMS Learning Algorithm" by Bernard Widrow ; Youngsik Kim ; Dookun Park. 
"""

import numpy as np
import networkx as nx
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

        for i in range(len(self.layer_sizes) - 1):
            if 0.0 <= self.excitatory_ratio <= 1.0:
                # Initialize weights for all layers randomly from a uniform distribution ~ [0, 1]
                w = np.random.rand(self.layer_sizes[i] + 1,
                                   self.layer_sizes[i + 1])
            else:
                # Initialize weights for all layers randomly from a normal distribution ~ [0, 1]
                w = np.random.randn(self.layer_sizes[i] + 1,
                                    self.layer_sizes[i + 1])
            self.layer_weights.append(w)
        self.gamma = gamma
        self.mu = mu

    def run(self, X, train=True):
        """
        Feed forward with optional training for each neuron. 
        All neurons in a single layer get trained simultaneously before moving on to the next layer.
        :param X: Each row represents a training vector
        :param train: True if the weights should be updated as the layer receives input
        :return:
        """
        assert X.shape[1] == self.input_size
        n_samples = X.shape[0]
        output_size = self.layer_sizes[-1]

        Y = np.zeros((n_samples, output_size))
        hidden_sum = np.zeros((n_samples, output_size))
        error = np.zeros((n_samples, output_size))
        # Iterate over all training vectors one by one
        for i, x in enumerate(X):
            input = np.copy(x)
            output = np.zeros((output_size, ))
            err = np.zeros((output_size, ))
            sums = np.zeros((output_size, ))
            # Iterate over layers
            for W in self.layer_weights:
                # Mask input vector with excitatatory vs inhibitory mask
                masked = np.append(np.copy(input), 1)
                if 0.0 <= self.excitatory_ratio <= 1.0:
                    inhibitory_idx = int(len(input) * self.excitatory_ratio)
                    masked[inhibitory_idx:] *= -1

                # Get all neurons sum
                sums = W.T @ masked
                sgm = np.tanh(sums)

                # Compute output with half-sigmoid
                output = np.copy(sgm)
                output[output < 0] = 0

                # Compute feedback error
                err = (sgm - self.gamma * sums)  #* -(1 - self.gamma - sgm**2)

                if train:
                    # Update the weights
                    neg_gradient = self.mu * (masked[:, None] @ err[None, :])
                    W += neg_gradient

                    # Check to see if weights are non-negative
                    # assert (len(W[W < 0]) == 0)
                    if 0.0 <= self.excitatory_ratio <= 1.0:
                        W[W < 0] = 0
                input = output  # Feed the output of this layer into the next layer
            Y[i] = output.T
            hidden_sum[i] = sums
            error[i] = err

        return Y, hidden_sum, error

    def fit(self, X):
        self.run(X, train=True)

    def predict(self, X):
        encodings = self.run(X, train=False)[0]
        # Convert encodings into the integer labels for cluster membership of each sample.

        # Turn all values greater than 0 into 1
        encodings[encodings > 0] = 1

        # Convert binary numbers to integer representation
        y_pred = []
        for i, encoding in enumerate(encodings):
            b_str = ''.join([str(k) for k in encoding.astype(np.int64)])
            y_pred.append(b_str)

        le = preprocessing.LabelEncoder()
        return le.fit_transform(y_pred)

    def get_graph(self):
        d = nx.DiGraph()
        node_count = 0
        for i, layer_size in enumerate(self.layer_sizes):
            print(layer_size)
            if i == 0:
                for j in range(node_count, node_count + layer_size):
                    d.add_node(j)
                    node_count += 1
            else:
                for j in range(node_count, node_count + layer_size):
                    d.add_node(j)
                    node_count += 1
                    for k in range(self.layer_sizes[i - 1]):
                        d.add_edge(j - k - 1, j)
        return d
