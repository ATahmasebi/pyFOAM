import numpy as np
from collections import namedtuple
from src.Utilities.field import Field
from src.Utilities.field_operations import norm, dot
from src.mesh.primitives import on_demand_prop
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter

d = np.array([[1, 0, 0], [0, 10, 0], [0, 0, 12]])


def lazy_norm(self):
    _lazy_norm = '_lazy_norm'
    if hasattr(self, _lazy_norm):
        return getattr(self, _lazy_norm)
    setattr(self, _lazy_norm, norm(self))
    return getattr(self, _lazy_norm)

