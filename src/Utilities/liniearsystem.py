from scipy import sparse
from scipy.sparse.linalg import spsolve
from src.Utilities.field import Field
import numpy as np


class LinearSystem:
    def __init__(self, rank):
        self.rank = rank
        self.lhs = None
        self.rows = []
        self.columns = []
        self.data = []
        self.rhs = None

    def lhs_add(self, row, column, data):
        self.rows.append(row)
        self.columns.append(column)
        if len(self.data) == 0:
            self.data.append(data)
        elif data.unit == self.data[0].unit:
            self.data.append(data)
        else:
            self.data.append(data.convert(self.data[0].unit))

    def rhs_add(self, data):
        """
        Shekoo made me do this!
        """
        if self.rhs is None:
            self.rhs = data
        else:
            self.rhs = self.rhs + data

    def assemble_matrix(self):
        r = np.concatenate(self.rows)
        c = np.concatenate(self.columns)
        d = np.concatenate(self.data)
        self.lhs = sparse.csr_matrix((d, (r, c)), shape=(self.rank, self.rank))

    def solve(self):
        self.assemble_matrix()
        oldshape = self.rhs.shape
        _, dim, _ = oldshape
        newshape = (-1,) if dim == 1 else (-1, dim)
        sol = spsolve(self.lhs, self.rhs.reshape(newshape))
        unit = self.rhs.unit / self.data[0].unit
        return Field(sol, unit).reshape(oldshape)

    def clear(self):
        self.rows.clear()
        self.columns.clear()
        self.data.clear()
        self.rhs = None
        self.lhs = None


if __name__ == "__main__":
    pass
