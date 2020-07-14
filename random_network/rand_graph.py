# -*- coding: utf-8 -*-
"""
Created on 2019-10-01

@author: abhipray

"""

import networkx as nx
import matplotlib.pyplot as plt

import random


class Graph:
    nodes = []
    edges = []
    removed_edges = []

    def remove_edge(self, x, y):
        e = (x,y)
        try:
            self.edges.remove(e)
            # print("Removed edge %s" % str(e))
            self.removed_edges.append(e)
        except:
            return

    def Nodes(self):
        return self.nodes

    # Sample data
    def __init__(self):
        self.nodes = []
        self.edges = []


def get_random_dag():
    MIN_PER_RANK = 1    # Nodes/Rank: How 'fat' the DAG should be
    MAX_PER_RANK = 3
    MIN_RANKS = 6   # Ranks: How 'tall' the DAG should be
    MAX_RANKS = 10
    PERCENT = 0.3  # Chance of having an Edge
    nodes = 0

    ranks = random.randint(MIN_RANKS, MAX_RANKS)

    adjacency = []
    for i in range(ranks):
        # New nodes of 'higher' rank than all nodes generated till now
        new_nodes = random.randint(MIN_PER_RANK, MAX_PER_RANK)

        # Edges from old nodes ('nodes') to new ones ('new_nodes')
        for j in range(nodes):
            for k in range(new_nodes):
                if random.random() < PERCENT:
                    adjacency.append((j, k+nodes))

        nodes += new_nodes

    # Compute transitive graph
    G = Graph()
    # Append nodes
    for i in range(nodes):
        G.nodes.append(i)
    # Append adjacencies
    for i in range(len(adjacency)):
        G.edges.append(adjacency[i])

    N = G.Nodes()
    for x in N:
        for y in N:
            for z in N:
                if (x, y) != (y, z) and (x, y) != (x, z):
                    if (x, y) in G.edges and (y, z) in G.edges:
                        G.remove_edge(x, z)

    # Print graph
    for i in range(nodes):
        print(i)
    print()
    for value in G.edges:
        print(str(value[0]) + ' ' + str(value[1]))

    return G


G = get_random_dag()
print(G.edges)

g = nx.Graph(G.edges)
nx.draw(g)

plt.show()