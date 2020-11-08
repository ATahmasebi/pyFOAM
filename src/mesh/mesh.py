import numpy as np

from src.Utilities.primitives import boundary_faces
from src.Utilities.liniearsystem import LinearSystem
from src.discretization.explicit.grad import least_sqr as grad
from src.Utilities.field import Field, on_demand_prop


class Mesh:
    def __init__(self, topology, phi0):
        self.topology = topology
        self.LS = LinearSystem(self.topology.info.cells)
        self.BC = {}
        self._phi = self._set_initial_value(phi0)
        self._grad = None
        self._phi_b = None
        self.patchlist = []
        # maybe: start range from -1 to include empty bounries
        for p in range(-1, np.max(self.topology.boundary.patch) + 1):
            index = self.topology.boundary.patch == p
            ow = self.topology.boundary.owner[index]
            fc = self.topology.boundary.center[index]
            fv = self.topology.boundary.vector[index]
            self.patchlist.append(boundary_faces(fc, fv, ow, p))
        self.set_BC(-1, lambda patch, mesh: mesh.phi[patch.owner], [self])

    # Maybe: pass kwargs to func
    def set_BC(self, patch, func, args):
        index = self.topology.boundary.patch == patch
        self.BC[patch] = (index, func, args)

    def _set_initial_value(self, phi0):
        cells = self.topology.info.cells
        if len(phi0) == cells:
            return phi0
        elif phi0.size in [1, 3]:
            _phi = Field(np.empty((cells, phi0.size, 1)), phi0.unit)
            _phi[:] = phi0
            return _phi
        else:
            raise ValueError('Failed to initialize phi0')

    @on_demand_prop
    def _phib_shape(self):
        faces = len(self.topology.boundary.owner)
        _, dim, _ = self.phi.shape
        return faces, dim, 1

    @property
    def gradient(self):
        if self._grad is None:
            self._grad = grad(self)
        return self._grad

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, val):
        self._grad = None
        self._phi_b = None
        self._phi = val

    @property
    def phi_b(self):
        if self._phi_b is None:
            self._phi_b = Field(np.zeros(self._phib_shape), self.phi.unit)
            for p in self.patchlist:
                index, func, args = self.BC[p.patch]
                self._phi_b[index] = func(p, *args)
        return self._phi_b
