import numpy as np
from collections import namedtuple
from src.Utilities.field import Field

import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter


a = np.random.randint(1000000000)
start = perf_counter()


b = np.max(a)
end = perf_counter()

print(end - start)
print(b)

