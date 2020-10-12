import numpy as np
from collections import namedtuple
from src.Utilities.field import Field
from src.Utilities.field_operations import norm, dot
import scipy as sp
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import perf_counter


f1 = Field([[1,2,3], [4,5,6], [7,8,9], [1,2,3], [4,5,6], [7,8,9]], 'm')

print(f1.T @ f1)

