from src.mesh.primitives import patch


class Mesh:
    def __init__(self, topology):
        self.topology = topology
        self.BC = {}

    def set_BC_flux(self, num, name, func):
        p = patch(name, 'flux', func)
        self.BC[num] = p

    def set_BC_value(self, num, name, func):
        p = patch(name, 'value', func)
        self.BC[num] = p

    def set_BC_robin(self, num, name, func):
        p = patch(name, '')
