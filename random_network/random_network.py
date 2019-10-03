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

n_input = 4
n_output = 4
n_hidden = 5
N = n_output+n_hidden+n_input
p = 0.1
mu = 0.05
gamma = 0.5
n_iters = 100

d = nx.DiGraph()

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
            print(i, candidate)
            d.add_edge(candidate, i)
            indegree += 1
    # Add a hebblms neuron to this node
    hebb = HebbLMSNet(indegree, [1], 0.5, mu=mu, gamma=gamma)
    d.nodes[i]['net'] = hebb
    d.nodes[i]['output'] = 0


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

errors = []
for epoch in range(n_iters):
    for i in range(n_training_samples):
        # Load the input
        for k in range(n_input):
            d.nodes[k]['output'] = X[i][k]
        err_sum = 0
        # Train each neuron sequentially
        for k in range(n_input, N):
            hebb = d.nodes[k]['net']
            # Construct the input vector from its parents
            x = []
            parents = d.predecessors(k)
            for p in parents:
                x.append(d.nodes[p]['output'])
            x = np.array(x)[None, :]
            y,_,err = hebb.run(x)
            err_sum += err[0][0]**2
            d.nodes[k]['output'] = y[0][0]
        errors.append(err_sum/(N-n_input))

plt.figure()
plt.plot(errors)
plt.show()