import numpy as np
from src.Utilities.field import Field
from scipy import sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()

y = np.random.random(50) * 10
x = np.random.random(50) * 10
# t = np.linspace(0, 1, 100)
levels = np.linspace(0, 300, 20)


def init():
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)


def update(frame):
    z = x ** 2 + y ** 2 * frame / 50
    cntr = ax.tricontourf(x, y, z, levels=levels, cmap="YlOrRd")
    return cntr


ani = FuncAnimation(fig, update, frames=99, init_func=init, interval=100)
plt.show()
