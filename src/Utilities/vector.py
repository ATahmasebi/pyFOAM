import numpy as np
import pystencils as ps


from src.Utilities.field import Field
from src.mesh.primitives import on_demand_prop

e0, e, a, b, c = ps.fields('e0,e(3),a(3),b(3),c(3):[1D]')

cross_assignment = ps.AssignmentCollection([
    ps.Assignment(e[0](0), a[0](1) * b[0](2) - a[0](2) * b[0](1)),
    ps.Assignment(e[0](1), a[0](2) * b[0](0) - a[0](0) * b[0](2)),
    ps.Assignment(e[0](2), a[0](0) * b[0](1) - a[0](1) * b[0](0))])
cross_kf = ps.create_kernel(cross_assignment).compile()


def cross(A, B):
    arr = np.empty(shape=A.shape)
    unit = A.unit * B.unit
    cross_kf(a=A, b=B, e=arr)
    return VectorField(arr, unit)


dot_assignment = ps.Assignment(e0[0], a[0](0) * b[0](0) + a[0](1) * b[0](1) + a[0](2) * b[0](2))
dot_kf = ps.create_kernel(dot_assignment).compile()


def dot(A, B):
    arr = np.empty(shape=(len(A), 1))
    unit = A.unit * B.unit
    dot_kf(a=A, b=B, e0=arr)
    return Field(arr, unit)


triple_assignment = ps.Assignment(e0[0],
                                  (a[0](1) * b[0](2) - a[0](2) * b[0](1)) * c[0](0) +
                                  (a[0](2) * b[0](0) - a[0](0) * b[0](2)) * c[0](1) +
                                  (a[0](0) * b[0](1) - a[0](1) * b[0](0)) * c[0](2))
triple_kf = ps.create_kernel(triple_assignment).compile()


def triple(A, B, C):
    arr = np.empty(shape=(len(A), 1))
    unit = A.unit * B.unit * C.unit
    triple_kf(a=A, b=B, c=C, e0=arr)
    return Field(arr, unit)


def norm(A):
    return np.sqrt(dot(A, A))


class VectorField(Field):

    @on_demand_prop
    def norm(self):
        return norm(self)