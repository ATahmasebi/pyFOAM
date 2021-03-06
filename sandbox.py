import numpy as np
from src.Utilities.field import Field
from scipy import sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix, csc_matrix
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot

N = 500
a1 = np.arange(0, N)
a2 = np.random.randint(0, N, N)
a3 = np.random.randint(0, N, N)

d1 = np.random.random(3 * N)
d2 = np.random.random(3 * N)
d4 = np.random.random(N)

row = np.concatenate([a1, a2, a3])
col = np.concatenate([a1, a3, a2])
row2 = np.concatenate([row, row])
col2 = np.concatenate([col, col])
d3 = np.concatenate([d1, d2])
m1 = csr_matrix((d1, (row, col)))
m2 = csr_matrix((d2, (row, col)))

t0 = perf_counter()
m3 = csr_matrix((d3, (row2, col2))).tocsc()
t1 = perf_counter()

print(t1 - t0)

t2 = perf_counter()

m4 = m3.tocsc()

t3 = perf_counter()

print(t3 - t2)
