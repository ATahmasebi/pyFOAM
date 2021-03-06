import numpy as np


from src.Utilities.field import Field, dot, cross
from src.Utilities.primitives import *


class Topology(object):
    def __init__(self, elements):
        self.info = info(elements['ncells'], elements['unit'])
        nodes = Field(elements['points'], elements['unit']).reshape(vector)
        faces = elements['faces']
        t = np.array([[f[0], f[i], f[i + 1], x] for x, f in enumerate(faces) for i in range(1, len(f) - 1)])
        v1 = nodes[t[:, 1]] - nodes[t[:, 0]]
        v2 = nodes[t[:, 2]] - nodes[t[:, 1]]
        ta = 0.5 * cross(v1, v2)
        fv = Field(np.zeros((len(faces), 3)), ta.unit).reshape(vector)
        np.add.at(fv, t[:, 3], ta)
        tc = np.sum(nodes[t[:, 0:3]], axis=1) / 3
        wt = tc * ta.norm
        fc = Field(np.zeros((len(faces), 3)), wt.unit).reshape(vector)
        np.add.at(fc, t[:, 3], wt)
        fc = fc / fv.norm
        f_index, owner, neighbour = np.array(elements['internal'], dtype=int).transpose()
        self.internal = internal_faces(fc[f_index], fv[f_index], owner, neighbour)
        f_index, owner, p = np.array(elements['boundaries'], dtype=int).transpose()
        self.boundary = boundary_faces(fc[f_index], fv[f_index], owner, p)
        pvi = dot(self.internal.center, self.internal.vector) / 3
        cv = Field(np.zeros(self.info.cells), pvi.unit).reshape(scaler)
        np.add.at(cv, self.internal.owner, pvi)
        np.subtract.at(cv, self.internal.neighbour, pvi)
        pvb = dot(self.boundary.center, self.boundary.vector) / 3
        np.add.at(cv, self.boundary.owner, pvb)

        pci = 0.75 * self.internal.center * pvi
        pcb = 0.75 * self.boundary.center * pvb
        cc = Field(np.zeros(self.info.cells * 3), pci.unit).reshape(vector)
        np.add.at(cc, self.internal.owner, pci)
        np.subtract.at(cc, self.internal.neighbour, pci)
        np.add.at(cc, self.boundary.owner, pcb)
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
    def dCb(self):
        cells = self.cells
        faces = self.boundary
        return cells.center[faces.owner] - faces.center

    @on_demand_prop
    def dCf(self):
        cells = self.cells
        faces = self.internal
        return cells.center[faces.owner] - faces.center

    @on_demand_prop
    def ip(self):
        cells = self.cells
        faces = self.internal
        dCF = self.dCF
        dCf = self.dCf
        return cells.center[faces.neighbour] + dCF * dot(dCF, dCf) / dot(dCF, dCF)

    @on_demand_prop
    def Ginv(self):
        w = self.dCF.norm
        d = self.dCF.reshape((-1, 1, 3))
        dt = self.dCF / w
        dtd = dt@d
        G = Field(np.zeros((self.info.cells, 3, 3)), dtd.unit)
        np.add.at(G, self.internal.owner, dtd)
        np.add.at(G, self.internal.neighbour, dtd)
        db = self.dCb.reshape((-1, 1, 3))
        wb = self.dCb.norm
        dbt = self.dCb / wb
        dbtdb = dbt@db
        np.add.at(G, self.boundary.owner, dbtdb)
        ginv = np.linalg.inv(G)
        ginv.unit = G.unit ** -1
        return ginv

    @on_demand_prop
    def ndCb(self):
        Sb = self.boundary.vector
        dCb = self.dCb
        return -dot(Sb, dCb) / Sb.norm

    @on_demand_prop
    def ff(self):
        return self.internal.center - self.ip

    def face_interpolate(self, gamma):
        if len(gamma) == self.info.cells:
            return gamma[self.internal.owner] * (1 - self.gf) + gamma[self.internal.neighbour] * self.gf
        elif gamma.shape == () or gamma.shape == (1, ):
            return gamma
        else:
            raise ValueError('Cannot interpolate values to faces.')


if __name__ == '__main__':
    # path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test00.mphtxt'
    path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    print(top.cells)
