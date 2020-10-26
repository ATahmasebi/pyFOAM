import numpy as np
import pystencils as ps

import src.Utilities.Units as Units
from src.Utilities.primitives import on_demand_prop, vector, scaler


class Field(np.ndarray):
    def __new__(cls, array, unit):
        arr = np.asarray(array, dtype=np.float).view(cls)
        arr.unit = unit if isinstance(unit, Units.Unit) else Units.Unit.parse(unit)
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.unit = getattr(obj, 'unit', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        operation = ufunc.__name__
        args = [i.view(np.ndarray) if isinstance(i, Field) else i for i in inputs]
        out_unit = self.unit
        if operation in ['multiply', 'matmul']:
            other = inputs[1] if inputs[0] is self else inputs[0]
            if isinstance(other, Field):
                out_unit *= other.unit
        elif operation in ['add', 'subtract'] and method != 'reduce':
            for j, i in enumerate(inputs):
                if isinstance(i, Field) and i.unit != self.unit:
                    cr = i.unit.conversion_ratio(out_unit)
                    args[j] *= cr
        elif operation in ['divide', 'true_divide', 'floor_divide']:
            if isinstance(inputs[1], Field):
                if isinstance(inputs[0], Field):
                    out_unit /= inputs[1].unit
                else:
                    out_unit = out_unit ** -1

        elif operation == 'power':
            out_unit = out_unit ** inputs[-1]
        elif operation == 'square':
            out_unit = out_unit ** 2
        elif operation == 'sqrt':
            out_unit = out_unit ** 0.5

        cls = type(self)
        return cls(super(Field, self).__array_ufunc__(ufunc, method, *args, **kwargs), out_unit)

    def __setitem__(self, key, value):
        if isinstance(value, Field):
            if value.unit != self.unit:
                value = value.convert(self.unit)
        return super(Field, self).__setitem__(key, value)

    def __repr__(self):
        arr_str = super().__repr__()
        return f'{arr_str[:-1]}, \'{str(self.unit)}\'{arr_str[-1]}'

    def __str__(self):
        return super().__str__() + f' [{str(self.unit)}]'

    def convert(self, unit):
        cr = self.unit.conversion_ratio(unit)
        return Field(cr * self, unit)

    @on_demand_prop
    def norm(self):
        return norm(self.reshape(vector))


e0, e, a, b, c = ps.fields('e0,e(3),a(3),b(3),c(3):[1D]')
cross_assignment = ps.AssignmentCollection([
    ps.Assignment(e[0](0), a[0](1) * b[0](2) - a[0](2) * b[0](1)),
    ps.Assignment(e[0](1), a[0](2) * b[0](0) - a[0](0) * b[0](2)),
    ps.Assignment(e[0](2), a[0](0) * b[0](1) - a[0](1) * b[0](0))])
cross_kf = ps.create_kernel(cross_assignment).compile()


def cross(A, B):
    _a = A.reshape((-1, 3))
    _b = B.reshape((-1, 3))
    arr = np.empty(shape=_a.shape)
    unit = A.unit * B.unit
    cross_kf(a=_a, b=_b, e=arr)
    return Field(arr, unit).reshape(vector)


dot_assignment = ps.Assignment(e0[0], a[0](0) * b[0](0) + a[0](1) * b[0](1) + a[0](2) * b[0](2))
dot_kf = ps.create_kernel(dot_assignment).compile()


def dot(A, B):
    _a = A.reshape((-1, 3))
    _b = B.reshape((-1, 3))
    arr = np.empty(shape=(len(_a), 1))
    unit = A.unit * B.unit
    dot_kf(a=_a, b=_b, e0=arr)
    return Field(arr, unit).reshape(scaler)


triple_assignment = ps.Assignment(e0[0],
                                  (a[0](1) * b[0](2) - a[0](2) * b[0](1)) * c[0](0) +
                                  (a[0](2) * b[0](0) - a[0](0) * b[0](2)) * c[0](1) +
                                  (a[0](0) * b[0](1) - a[0](1) * b[0](0)) * c[0](2))
triple_kf = ps.create_kernel(triple_assignment).compile()


def triple(A, B, C):
    _a = A.reshape((-1, 3))
    _b = B.reshape((-1, 3))
    _c = C.reshape((-1, 3))
    arr = np.empty(shape=(len(_a), 1))
    unit = A.unit * B.unit * C.unit
    triple_kf(a=_a, b=_b, c=_c, e0=arr)
    return Field(arr, unit).reshape(scaler)


def norm(A):
    return np.sqrt(dot(A, A))


if __name__ == '__main__':
    pass
