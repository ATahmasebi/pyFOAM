import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.Utilities.primitives import BaseTerm


class Laplacian(BaseTerm):

    def __init__(self, mesh: Mesh, gamma: Field, correction='or'):
        super().__init__(mesh.topology.info.cells)
        self.mesh = mesh
        self.gamma_f, self.gamma_bf = self._gamma_interpolation(gamma)
        self.correction = correction
        self.Ef, self.Tf = self._face_decompose()

    def _gamma_interpolation(self, gamma):
        if len(gamma) == self.mesh.topology.info.cells:
            gamma_bf = gamma[self.mesh.topology.boundary.owner]
        else:
            gamma_bf = gamma
        gamma_f = self.mesh.topology.face_interpolate(gamma)
        return gamma_f, gamma_bf

    def _face_decompose(self):
        dCF = self.mesh.topology.dCF
        Sf = self.mesh.topology.internal.vector
        if self.correction == 'or':
            Ef = dCF * dot(Sf, Sf) / dot(dCF, Sf)
            Tf = Sf - Ef
        elif self.correction == 'mc':
            Ef = dCF * dot(Sf, dCF) / dot(dCF, dCF)
            Tf = Sf - Ef
        elif self.correction == 'oc':
            Ef = Sf.norm * dCF / dCF.norm
            idx = np.array(dot(Sf, dCF)).reshape((-1,)) < 0
            Ef[idx] = -Ef[idx]
            Tf = Sf - Ef
        elif self.correction == 'uncorrected':
            Ef = Sf
            Tf = None
        else:
            raise ValueError('invalid orthogonal correction method.')
        return Ef, Tf

    @property
    def LHS(self):
        self._clear()
        gamma_f = self.gamma_f
        dCF = self.mesh.topology.dCF

        af = (gamma_f * self.Ef.norm / dCF.norm).reshape((-1,))
        naf = -af
        owner = self.mesh.topology.internal.owner
        neighbour = self.mesh.topology.internal.neighbour
        self.lhs_add(owner, owner, naf)
        self.lhs_add(neighbour, neighbour, naf)
        self.lhs_add(owner, neighbour, af)
        self.lhs_add(neighbour, owner, af)

        Sb = self.mesh.topology.boundary.vector
        ndCb = -self.mesh.topology.ndCb
        abf = self.gamma_bf * Sb.norm / ndCb
        ownerb = self.mesh.topology.boundary.owner
        self.lhs_add(ownerb, ownerb, abf.reshape((-1,)))
        return self.get_lhs()

    @property
    def RHS(self):
        return self.__rhs

    def update(self):
        Sb = self.mesh.topology.boundary.vector
        ndCb = self.mesh.topology.ndCb
        phi_b = self.mesh.phi_b
        ownerb = self.mesh.topology.boundary.owner
        abf_phib = phi_b * self.gamma_bf * Sb.norm / ndCb
        _, dim, _ = abf_phib.shape
        rhs = Field(np.zeros(shape=(self.mesh.topology.info.cells, dim, 1)), abf_phib.unit)
        np.subtract.at(rhs, ownerb, abf_phib)

        # correction:
        if self.correction in ['or', 'oc', 'mc']:
            grad = self.mesh.gradient
            grad_f = self.mesh.topology.face_interpolate(grad)
            neighbour = self.mesh.topology.internal.neighbour
            owner = self.mesh.topology.internal.owner
            ac = self.gamma_f * (grad_f @ self.Tf)
            np.add.at(rhs, neighbour, ac)
            np.subtract.at(rhs, owner, ac)

        self.__rhs = rhs
