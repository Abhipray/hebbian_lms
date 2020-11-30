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
                 percent: bool = True,
                 gamma=0.5,
                 d_min=0.001,
                 mu=0.1):
        """
        :param input_size: Length of the input vector
        :param layer_sizes: list containing number of neurons for each layer
        :param excitatory_ratio: Ratio of input connections that are excitatory to the total number of connections; value between 0 and 1
        :param gamma: value between 0 and 1 used in calculating errors for updating weights
        :param mu: learning rate
        :param percent: Use the percent variant of hebbian LMS
        """
        self.percent = percent
        self.input_size = input_size
        self.excitatory_ratio = excitatory_ratio
        self.layer_sizes = [input_size] + layer_sizes
        self.layer_weights = []
        self.layer_biases = []

        for i in range(len(self.layer_sizes) - 1):
            m = self.layer_sizes[i]
            n = self.layer_sizes[i + 1]

            w_min = d_min / np.sqrt(m)
            # b = -d_min * np.sqrt(m)

            # print('w_min', w_min)
            # # Initialize weights for all layers randomly from a uniform distribution ~ [w_min, 1]
            # w = np.random.uniform(low=w_min, high=1.0, size=(m, n))
            # self.layer_weights.append(w)
            # self.layer_biases.append(b)

            w = np.random.uniform(low=w_min, high=1.0, size=(m, n))
            b = -np.random.uniform(low=0, high=1,
                                   size=(1, n)) * np.linalg.norm(w, axis=0)
            self.layer_weights.append(w)
            self.layer_biases.append(b)

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
            for W, b in zip(self.layer_weights, self.layer_biases):
                # Mask input vector with excitatatory vs inhibitory mask
                masked = np.copy(input)
                if 0.0 <= self.excitatory_ratio <= 1.0:
                    inhibitory_idx = int(len(input) * self.excitatory_ratio)
                    masked[inhibitory_idx:] *= -1

                # Get all neurons sum
                sums = W.T @ masked + b[0]
                sgm = np.tanh(sums)

                # Compute output with half-sigmoid
                output = np.copy(sgm)
                output[output < 0] = 0

                # Compute feedback error
                # err = (sgm - self.gamma * sums)  #* -(1 - self.gamma - sgm**2)
                # sums *= 4 / len(input)
                # sgm = np.tanh(sums)
                err = (sgm - self.gamma * sums)  #* -(1 - self.gamma - sgm**2)

                if train:
                    # Update the weights
                    neg_gradient = self.mu * (masked[:, None] @ err[None, :])
                    if self.percent:
                        neg_gradient *= W
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
        return self.run(X, train=True)

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
        return le.fit_transform(y_pred), encodings

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
