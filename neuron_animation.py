"""
Run a simulation of a single hebbian-lms neuron. The neuron receives a set of uniformly distributed input vectors.
It should learn to cluster the points into two groups. 

The script runs an animation of the clustering process.
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.use("Qt5Agg")
seaborn.set_style("whitegrid")

from hebbian_lms import HebbLMSNet

np.random.seed(10)

# Network Configuration
n_neurons = 1
n_weights = 100
excitatory_ratio = 0.5
n_iters = 500
mu = 0.02
gamma = 0.5

hebb = HebbLMSNet(
    n_weights, [50, n_neurons], excitatory_ratio, mu=mu, gamma=gamma)

# Create the training matrix by sampling from a uniform distribution; each row is a training vector
n_training_samples = 50
X = np.random.rand(n_training_samples, n_weights)

all_errors = []

# Clustering animation
all_sums = []
all_Y = []
for i in range(n_iters):
    np.random.shuffle(X)
    Y, sums, error = hebb.run(X, train=True)
    all_errors.append(np.sum(error**2))
    all_Y.append(Y)
    all_sums.append(sums)

# Display animation
fig, ax = plt.subplots()

ax.axis([-5, 5, -2, 2])
x = np.arange(-5, 5, 0.1)

line1, = ax.plot(x, np.sin(x), 'xb')
line2, = ax.plot(x, np.sin(x), 'or')
line3, = ax.plot(x, np.sin(x), '^y')
ax.plot(x, np.tanh(x), linewidth=0.5)
ax.plot(x, gamma * x, 'y-', linewidth=0.5)
plt.xlabel("sum")
plt.ylabel("tanh(sum)")
plt.title("Clustering process")


def init():
    line1.set_ydata([np.nan] * len(x))
    line2.set_ydata([np.nan] * len(x))
    line3.set_ydata([np.nan] * len(x))
    return line1, line2, line3


def animate(i):
    pos_idx = all_sums[i] > 0
    neg_idx = all_sums[i] < 0
    zero_idx = all_sums[i] == 0
    line1.set_data(all_sums[i][pos_idx], all_Y[i][pos_idx])
    line2.set_data(all_sums[i][neg_idx], all_Y[i][neg_idx])
    line3.set_data(all_sums[i][zero_idx], all_Y[i][zero_idx])
    return line1, line2, line3


ani = FuncAnimation(
    fig,
    animate,
    frames=n_iters,
    init_func=init,
    interval=20,
    blit=True,
    save_count=50)

ani.save(
    "clustering_plots/hlms_neuron_training.gif", dpi=80, writer='imagemagick')

plt.show()

# Plot learning curve
plt.figure()
plt.plot(np.array(all_errors)**2)
plt.title("Learning curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.savefig('clustering_plots/hlms_learning_curve.png')
plt.show()
plt.close()
