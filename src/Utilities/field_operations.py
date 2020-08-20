import numpy as np
import pystencils as ps

# TODO:
# 1. Make sure array shapes are compatible and 3D
# 2. Maybe implement broadcasting

__all__ = ['cross', 'dot', 'triple', 'norm']

e0, a0, a1, a2, b0, b1, b2, c0, c1, c2 = ps.fields('e0,a0,a1,a2,b0,b1,b2,c0,c1,c2:[1D]')

cross_assignment = ps.Assignment(e0[0], a1[0] * b2[0] - a2[0] * b1[0])
cross_kf = ps.create_kernel(cross_assignment).compile()


def cross(a, b):
    e = np.empty_like(a)
    e.unit = a.unit * b.unit
    cross_kf(a1=a[:, 1], a2=a[:, 2], b1=b[:, 1], b2=b[:, 2], e0=e[:, 0])
    cross_kf(a1=a[:, 2], a2=a[:, 0], b1=b[:, 2], b2=b[:, 0], e0=e[:, 1])
    cross_kf(a1=a[:, 0], a2=a[:, 1], b1=b[:, 0], b2=b[:, 1], e0=e[:, 2])
    return e


dot_assignment = ps.Assignment(e0[0], a0[0] * b0[0] + a1[0] * b1[0] + a2[0] * b2[0])
dot_kf = ps.create_kernel(dot_assignment).compile()


def dot(a, b):
    e = np.empty_like(a[:, 0])
    e.unit = a.unit * b.unit
    dot_kf(a0=a[:, 0], a1=a[:, 1], a2=a[:, 2], b0=b[:, 0], b1=b[:, 1], b2=b[:, 2], e0=e)
    return e.reshape((-1, 1))


triple_assignment = ps.Assignment(e0[0],
                                  (a1[0] * b2[0] - a2[0] * b1[0]) * c0[0] +
                                  (a2[0] * b0[0] - a0[0] * b2[0]) * c1[0] +
                                  (a0[0] * b1[0] - a1[0] * b0[0]) * c2[0])
triple_kf = ps.create_kernel(triple_assignment).compile()


def triple(a, b, c):
    e = np.empty_like(a[:, 0])
    e.unit = a.unit * b.unit * c.unit
    triple_kf(a0=a[:, 0], a1=a[:, 1], a2=a[:, 2],
              b0=b[:, 0], b1=b[:, 1], b2=b[:, 2],
              c0=c[:, 0], c1=c[:, 1], c2=c[:, 2], e0=e)
    return e.reshape((-1, 1))


def norm(a):
    return np.sqrt(dot(a, a))


if __name__ == '__main__':
    from src.Utilities.field import Field

    A = Field([[1, 0, 0]], 'm')
    B = Field([[0, 1, 0]], 'm')
    C = Field([[0, 0, 1]], 'm')
    print(triple(A, B, C))
    print(norm(A))
