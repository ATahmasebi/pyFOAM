import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field
from src.Utilities.primitives import BaseTerm


class Ddt(BaseTerm):
    def __init__(self, mesh: Mesh, rho: Field, delta: Field, scheme='EU'):
        super().__init__(mesh.topology.info.cells)
        self.mesh = mesh
        self.rho = rho
        self.scheme = scheme
        self.phi0 = None
        self.phi00 = None
        self.delta = delta

    def update(self):
        self.phi00 = self.phi0
        self.phi0 = self.mesh.phi

    @property
    def LHS(self):
        self._clear()
        if self.scheme == 'EU':
            ac = self.rho / self.delta * self.mesh.topology.cells.volume
        elif self.scheme == 'SOUE':
            if self.phi00 is not None:
                ac = 1.5 * self.rho / self.delta * self.mesh.topology.cells.volume
            else:
                ac = self.rho / self.delta * self.mesh.topology.cells.volume
        else:
            raise ValueError('Unkown scheme!')
        indic = np.arange(len(ac))
        self.lhs_add(indic, indic, ac.reshape((-1,)))
        return self.get_lhs()

    @property
    def RHS(self):
        if self.scheme == 'EU':
            rhs = self.rho / self.delta * self.phi0 * self.mesh.topology.cells.volume
        elif self.scheme == 'SOUE':
            if self.phi00 is None:
                rhs = self.rho / self.delta * self.phi0 * self.mesh.topology.cells.volume
            else:
                rhs = self.rho / (2 * self.delta) * self.mesh.topology.cells.volume * (4 * self.phi0 - self.phi00)
        else:
            raise ValueError('Unkown scheme!')
        return rhs
