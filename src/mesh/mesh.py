from src.Utilities.primitives import patch
from src.Utilities.liniearsystem import LinearSystem
from src.mesh.boundarypatch import BoundaryPatchField as bpf


class Mesh:
    def __init__(self, topology):
        self.topology = topology
        self.boondrypatch = bpf
        self.BC = {}
        self.LS = LinearSystem(self.topology.info.cells)
        self.phi = None

    def set_BC(self, num, values, btype, name):
        if btype in ['flux', 'value', 'robin']:
            p = patch(name, btype, values)
            self.BC[num] = p
        else:
            raise ValueError('Unknown boundary type.')
