# -*- coding: utf-8 -*-
"""
Created on 2019-10-01

@author: abhipray

"""

import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from hebbian_lms import HebbLMSNet



n_iters = 100


class RatsNest:
    def __init__(self, n_input, n_output, n_hidden, p=0.3, mu=0.1, gamma=0.5):
        print('d')
        d = nx.DiGraph()

        N = n_output + n_hidden + n_input # Total number of nodes

        # Add input nodes
        for i in range(n_input):
            d.add_node(i)

        # Add new nodes sequentially and connect to other random nodes
        for i in range(n_input, N):
            # print(i)
            indegree = 0
            d.add_node(i)
            for candidate in d.nodes:
                if random.random() > p and candidate != i:
                    d.add_edge(candidate, i)
                    indegree += 1
            # Add a hebblms neuron to this node
            hebb = HebbLMSNet(indegree, [1], 0.5, mu=mu, gamma=gamma)
            d.nodes[i]['net'] = hebb
            d.nodes[i]['output'] = 0

        self.net = d
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.N = N

    def get_graph(self):
        return self.net

    def run(self, X):
        errors = []
        d = self.net

        for i in range(n_training_samples):
            # Load the input
            for k in range(self.n_input):
                d.nodes[k]['output'] = X[i][k]
            err_sum = 0
            # Train each neuron sequentially
            for k in range(self.n_input, self.N):
                hebb = d.nodes[k]['net']
                # Construct the input vector from its parents
                x = []
                parents = d.predecessors(k)
                for p in parents:
                    x.append(d.nodes[p]['output'])
                x = np.array(x)[None, :]
                y, _, err = hebb.run(x)
                err_sum += err[0][0] ** 2
                d.nodes[k]['output'] = y[0][0]
            errors.append(err_sum / (self.N - self.n_input))


n_input=4
n_output=4
n_hidden=10
N = n_hidden+n_input+n_output

rat = RatsNest(n_input, n_output, n_hidden)
d = rat.get_graph()

pos=nx.shell_layout(d)

# nodes
nx.draw_networkx_nodes(d,pos,
                       nodelist=range(n_input),
                       node_color='r',
                       node_size=500,
                   alpha=1)
nx.draw_networkx_nodes(d,pos,
                       nodelist=range(n_input+n_hidden, N),
                       node_color='c',
                       node_size=500,
                   alpha=1)


nx.draw(d, pos, with_labels=True, alpha=0.2)
plt.show()

# Do training
# Create the training matrix by sampling from a uniform distribution; each row is a training vector
n_training_samples = 50
X = np.random.rand(n_training_samples, n_input)

rat.run(X)

plt.figure()
# plt.plot(errors)
plt.show()