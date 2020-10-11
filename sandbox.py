import numpy as np
from collections import namedtuple
from src.Utilities.field import Field
from src.Utilities.field_operations import norm, dot
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter


f1 = Field([[1,2,3], [4,5,6]], 'm/s').reshape(((-1,1)))
f2 = Field([5,10,1],'m')


f3 = (f1*f2).reshape((2,3,3))
f4 = Field([[1,2,3], [4,5,6]], 'm/s')
print(dot(f4,f3))