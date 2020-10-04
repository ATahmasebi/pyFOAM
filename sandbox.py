import numpy as np
from collections import namedtuple
from src.Utilities.field import Field

import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter


class Test:
    pass


f = Field([[1, 2, 1], [2, 4, 3]], 'm')
f[:, :] = Field([1, 3, 5], 'mm')
print(f)