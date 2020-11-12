import numpy as np
from src.Utilities.field import Field
from scipy import sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = np.random.random(15).reshape((-1, 3, 1))
b = np.arange(len(a))

print(b)