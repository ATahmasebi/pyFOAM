from src.mesh.primitives import patch
from src.Utilities.liniearsystem import LinearSystem


class Mesh:
    def __init__(self, topology):
        self.topology = topology
        self.BC = {}
        self.LS = LinearSystem(self.topology.info.cells)
        self.phi = None

    def set_BC(self, num, values, btype, name):
        if btype in ['flux', 'value', 'robin']:
            p = patch(name, btype, values)
            self.BC[num] = p
        else:
            raise ValueError('Unknown boundary type.')
