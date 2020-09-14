from src.mesh.primitives import patch
from src.Utilities.liniearsystem import LinearSystem
from src.Utilities.field_operations import *


class Mesh:
    def __init__(self, topology, patches):
        self.topology = topology
        self.patches = patches
        self.linearsystem = LinearSystem(self.topology.info.cells)
        self._gf = None
        self._dCF = None
        self._dCf = None

    @property
    def gf(self):
        if self._gf is None:
            faces = self.topology.internal.vector
            self._gf = dot(self.dCf, faces) / dot(self.dCF, faces)
        return self._gf

    @property
    def dCF(self):
        if self._dCF is None:
            cells = self.topology.cells
            faces = self.topology.internal
            self._dCF = cells.center[faces.owner] - cells.center[faces.neighbour]
        return self._dCF

    @property
    def dCf(self):
        if self._dCf is None:
            cells = self.topology.cells
            faces = self.topology.internal
            self._dCf = cells.center[faces.owner] - faces.center
        return self._dCf
