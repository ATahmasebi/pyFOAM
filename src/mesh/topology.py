import numpy as np

from src.Utilities.field import Field
from src.Utilities.field_operations import *
from src.mesh.primitives import boundary_faces, internal_faces, cell, info, on_demand_prop


class Topology(object):
    def __init__(self, elements):
        self.info = info(elements['ncells'], elements['unit'])
        nodes = Field(elements['points'], elements['unit'])
        faces = elements['faces']
        t = np.array([[f[0], f[i], f[i + 1], x] for x, f in enumerate(faces) for i in range(1, len(f) - 1)])
        v1 = nodes[t[:, 1]] - nodes[t[:, 0]]
        v2 = nodes[t[:, 2]] - nodes[t[:, 1]]
        ta = 0.5 * cross(v1, v2)
        fv = Field(np.zeros((len(faces), 3)), ta.unit)
        np.add.at(fv, t[:, 3], ta)
        tc = np.sum(nodes[t[:, 0:3]], axis=1) / 3
        wt = norm(ta) * tc
        fc = Field(np.zeros((len(faces), 3)), wt.unit)
        np.add.at(fc, t[:, 3], wt)
        fc = fc / norm(fv)
        f_index, owner, neighbour = np.array(elements['internal'], dtype=int).transpose()
        self.internal = internal_faces(fc[f_index], fv[f_index], owner, neighbour)
        f_index, owner, patch = np.array(elements['boundaries'], dtype=int).transpose()
        boundaries = boundary_faces(fc[f_index], fv[f_index], owner, patch)
        self.boundary = []
        for p in range(np.max(boundaries.patch) + 1):
            index = boundaries.patch == p
            ow = boundaries.owner[index]
            fc = boundaries.center[index]
            fv = boundaries.vector[index]
            self.boundary.append(boundary_faces(fc, fv, ow, p))

        pvi = dot(self.internal.center, self.internal.vector) / 3
        cv = Field(np.zeros((self.info.cells, 1)), pvi.unit)
        np.add.at(cv, self.internal.owner, pvi)
        np.subtract.at(cv, self.internal.neighbour, pvi)
        pvb = dot(boundaries.center, boundaries.vector) / 3
        np.add.at(cv, boundaries.owner, pvb)

        pci = 0.75 * self.internal.center * pvi
        pcb = 0.75 * boundaries.center * pvb
        cc = Field(np.zeros((self.info.cells, 3)), pci.unit)
        np.add.at(cc, self.internal.owner, pci)
        np.subtract.at(cc, self.internal.neighbour, pci)
        np.add.at(cc, boundaries.owner, pcb)
        cc = cc / cv
        self.cells = cell(cc, cv)

    @on_demand_prop
    def gf(self):
        faces = self.internal.vector
        return dot(self.dCf, faces) / dot(self.dCF, faces)

    @on_demand_prop
    def dCF(self):
        cells = self.cells
        faces = self.internal
        return cells.center[faces.owner] - cells.center[faces.neighbour]

    @on_demand_prop
    def dCf(self):
        cells = self.cells
        faces = self.internal
        return cells.center[faces.owner] - faces.center

    def face_interpolate(self, gamma):
        if gamma.shape == (self.info.cells, 1):
            return gamma[self.internal.owner] * (1 - self.gf) + gamma[self.internal.neighbour] * self.gf
        elif gamma.shape == () or gamma.shape == (1, ):
            return gamma
        else:
            raise ValueError('Cannot interpolate values to faces.')

if __name__ == '__main__':
    path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    print(top.boundary)
    print(top.internal)
    print(top.cells)
