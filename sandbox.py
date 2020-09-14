import numpy as np
from collections import namedtuple
from src.Utilities.field import Field

import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter

f = Field([[0.5, 1., 2.],
       [1.5, 2., 3.],
       [2.5, 3., 4.]], 'm')
l = Field([1, 2, 3], 'm').reshape((-1, 1))

print(f * l)
