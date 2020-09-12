from src.mesh.primitives import patch
from src.Utilities.liniearsystem import LinearSystem

class Mesh:
    def __init__(self, topology, patches):
        self.topology = topology
        self.patches = patches
        self.linearsystem =LinearSystem(self.topology.info.cells)

    def get_boundaries(self):
        patches = []
        boundaries = self.topology.boundary
        for p in self.patches:
            index = boundaries.patch == p.id

