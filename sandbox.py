import numpy as np
from collections import namedtuple
from src.Utilities.field import Field
from src.Utilities.field_operations import norm, dot
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter


d = np.array([[1, 0, 0], [0, 10, 0], [0, 0, 12]])

print(d)
print(np.linalg.inv(d))

