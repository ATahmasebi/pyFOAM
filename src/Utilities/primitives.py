from collections import namedtuple
from scipy import sparse
import numpy as np

scaler = (-1, 1, 1)
vector = (-1, 3, 1)
tensor = (-1, 3, 3)

cell = namedtuple('cell', 'center volume')
boundary_faces = namedtuple('boundaries', 'center vector owner patch')
internal_faces = namedtuple('internal', 'center vector owner neighbour')
info = namedtuple('info', 'cells unit')


def on_demand_prop(fn):
    """Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class BaseTerm:
    def __init__(self, rank):
        self._rank = rank
        self._rhs = None
        self._rows = []
        self._columns = []
        self._data = []
        self.lhs_unit = None

    def lhs_add(self, row, column, data):
        self._rows.append(row)
        self._columns.append(column)
        if len(self._data) == 0:
            self._data.append(data)
        elif data.unit == self._data[0].unit:
            self._data.append(data)
        else:
            self._data.append(data.convert(self._data[0].unit))

    def rhs_add(self, data):
        if self._rhs is None:
            self._rhs = data
        else:
            self._rhs = self._rhs + data

    def get_lhs(self):
        r = np.concatenate(self._rows)
        c = np.concatenate(self._columns)
        d = np.concatenate(self._data)
        self.lhs_unit = self._data[0].unit
        self._clear()
        return sparse.csr_matrix((d, (r, c)), shape=(self._rank, self._rank))

    def get_rhs(self):
        if self._rhs is None:
            return 0
        rhs = self._rhs
        self._rhs = None
        return rhs

    def _clear(self):
        self._rows.clear()
        self._columns.clear()
        self._data.clear()
