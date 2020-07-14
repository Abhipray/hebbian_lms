"""
Run a simulation of a single hebbian-lms neuron. The neuron receives a set of uniformly distributed input vectors.
It learns to cluster the points into two groups. 

The script runs an animation of the clustering process.
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib
import os

from pathlib import Path
from matplotlib.animation import FuncAnimation
matplotlib.use("Qt5Agg")
seaborn.set_style("whitegrid")

from hebbian_lms import HebbLMSNet

np.random.seed(10)

# Network Configuration
n_neurons = 1
n_weights = 100
excitatory_ratio = 0.5
n_iters = 800
mu = 0.01
gamma = 0.5
percent = True
save_gif = True

# Animation output configuation
output_dir = Path('clustering_plots')
os.makedirs(output_dir, exist_ok=True)
snapshot_iters = [0, 50, 250, 800]

hebb = HebbLMSNet(n_weights, [n_neurons],
                  excitatory_ratio,
                  percent=percent,
                  mu=mu,
                  gamma=gamma)

# Create the training matrix by sampling from a uniform distribution; each row is a training vector
n_training_samples = 50
X = np.random.rand(n_training_samples, n_weights)

all_errors = []

# Clustering process; collect data
all_sums = []
all_Y = []
for i in range(n_iters):
    np.random.shuffle(X)
    Y, sums, error = hebb.run(X, train=True)
    all_errors.append(np.mean(error**2))
    all_Y.append(Y)
    all_sums.append(sums)

# Display animation
fig, ax = plt.subplots()

ax.axis([-5, 5, -2, 2])
x = np.arange(-5, 5, 0.1)

line1, = ax.plot(x, np.sin(x), 'xb')
line2, = ax.plot(x, np.sin(x), 'or', fillstyle='none')
line3, = ax.plot(x, np.sin(x), '^y')
ax.plot(x, np.tanh(x), linewidth=0.75, alpha=0.5)
ax.plot(x, gamma * x, 'y-', linewidth=0.75, alpha=0.5)
out = np.tanh(x)
out[x < 0] = 0
ax.plot(x, out, 'k-', linewidth=0.75)
plt.xlabel("(SUM)")
plt.ylabel("(OUT)")
plt.title("Clustering process")

# Save snapshots
for iter in snapshot_iters:
    i = max(iter - 1, 0)
    pos_idx = all_sums[0] > 0
    neg_idx = all_sums[0] < 0
    zero_idx = all_sums[0] == 0
    line1.set_data(all_sums[i][pos_idx], all_Y[i][pos_idx])
    line2.set_data(all_sums[i][neg_idx], all_Y[i][neg_idx])
    line3.set_data(all_sums[i][zero_idx], all_Y[i][zero_idx])
    plt.savefig(output_dir / f'hlms_training_{iter}.png')


def init():
    line1.set_ydata([np.nan] * len(x))
    line2.set_ydata([np.nan] * len(x))
    line3.set_ydata([np.nan] * len(x))
    return line1, line2, line3


def animate(i):
    pos_idx = all_sums[0] > 0
    neg_idx = all_sums[0] < 0
    zero_idx = all_sums[0] == 0
    line1.set_data(all_sums[i][pos_idx], all_Y[i][pos_idx])
    line2.set_data(all_sums[i][neg_idx], all_Y[i][neg_idx])
    line3.set_data(all_sums[i][zero_idx], all_Y[i][zero_idx])
    return line1, line2, line3


ani = FuncAnimation(fig,
                    animate,
                    frames=n_iters,
                    init_func=init,
                    interval=1,
                    blit=True,
                    save_count=50)

if save_gif:
    progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
    ani.save(output_dir / "hlms_neuron_training.gif",
             dpi=80,
             writer='imagemagick',
             fps=24,
             progress_callback=progress_callback)

    plt.show()

# Plot learning curve
plt.figure()
plt.plot(np.array(all_errors)**2)
plt.title("Learning curve")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.savefig(output_dir / 'hlms_learning_curve.png')
plt.show()
plt.close()
