import numpy as np
from src.Utilities.field import Field
from scipy import sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter



data = np.random.random(10).reshape((-1,1,1))

row = np.random.randint(0, 9, 10)
col = np.random.randint(0, 9, 10)

m = sparse.csc_matrix((data, (row, col)), shape=(10,10))
m.setdiag(m.diagonal() + 1)

b = np.random.random(20).reshape((-1,2))

x = spsolve(m, b)
print(m.toarray() @ x - b)
print()