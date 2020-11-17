import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.discretization.interpolation.schemes import face_interpolation
from src.Utilities.primitives import BaseTerm


class Divergence(BaseTerm):
    def __init__(self, mesh: Mesh, Vmesh: Mesh, rho: Field, scheme='QUICK'):
        super().__init__(mesh.topology.info.cells)
        self.mesh = mesh
        self.Vmesh = Vmesh
        self.rho = rho
        self.scheme = scheme

    def update(self):
        vf = self.Vmesh.topology.face_interpolate(self.Vmesh.phi)
        flux_f = self.rho * dot(vf, self.mesh.topology.internal.vector).reshape((-1,))
        index, corr = face_interpolation(self.mesh, flux_f, scheme=self.scheme)
        owner = self.mesh.topology.internal.owner
        neighbour = self.mesh.topology.internal.neighbour
        self.lhs_add(owner, index, flux_f)
        self.lhs_add(neighbour, index, -flux_f)
        flux_bf = self.rho * dot(self.Vmesh.phi_b, self.mesh.topology.boundary.vector) * self.mesh.phi_b
        _, dim, _ = flux_bf.shape
        rhs = Field(np.zeros(shape=(self.mesh.topology.info.cells, dim, 1)), flux_bf.unit)
        np.add.at(rhs, owner, corr * flux_f.reshape((-1, 1, 1)))
        np.subtract.at(rhs, self.mesh.topology.boundary.owner, flux_bf)
        self.rhs_add(rhs)

    @property
    def LHS(self):
        return self.get_lhs()

    @property
    def RHS(self):
        return self.get_rhs()
