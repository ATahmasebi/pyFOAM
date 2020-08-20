import numpy as np
from collections import namedtuple

from src.Utilities.field import Field
from src.Utilities.field_operations import *


cell = namedtuple('cell', 'center volume')
boundary_faces = namedtuple('boundaries', 'center vector owner patch')
internal_faces = namedtuple('internal', 'center vector owner neighbour')


class Topology(object):
    def __init__(self, elements):
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
        fc = fc/norm(fv)
        f_index, owner, neighbour = np.array(elements['internal'], dtype=int).transpose()
        self.internal = internal_faces(fc[f_index], fv[f_index], owner, neighbour)
        f_index, owner, patch = np.array(elements['boundaries'], dtype=int).transpose()
        self.boundary = boundary_faces(fc[f_index], fv[f_index], owner, patch)

        cnum = elements['ncells']
        pvi = dot(self.internal.center, self.internal.vector) / 3
        cv = Field(np.zeros((cnum, 1)), pvi.unit)
        np.add.at(cv, self.internal.owner, pvi)
        np.subtract.at(cv, self.internal.neighbour, pvi)
        pvb = dot(self.boundary.center, self.boundary.vector) / 3
        np.add.at(cv, self.boundary.owner, pvb)

        pci = 0.75 * self.internal.center * pvi
        pcb = 0.75 * self.boundary.center * pvb
        cc = Field(np.zeros((cnum, 3)), pci.unit)
        np.add.at(cc, self.internal.owner, pci)
        np.subtract.at(cc, self.internal.neighbour, pci)
        np.add.at(cc, self.boundary.owner, pcb)
        cc = cc / cv
        self.cells = cell(cc, cv)


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
    print(top.cells.center)
    print(top.cells.volume)

