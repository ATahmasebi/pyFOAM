import numpy as np
from src.Utilities.field import Field, dot


def divergence(mesh):
    phi_f = mesh.topology.face_interpolate(mesh.phi)
    a_f = dot(phi_f, mesh.topology.internal.vector)
    div = Field(np.zeros((mesh.topology.info.cells, 1, 1)), a_f.unit)
    np.add.at(div, mesh.topology.internal.owner, a_f)
    np.subtract.at(div, mesh.topology.internal.neighbour, a_f)
    a_fb = dot(mesh.phi_b, mesh.topology.boundary.vector)
    np.add.at(div, mesh.topology.boundary.owner, a_fb)
    return div/me.topology.cells.volume


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\heat2.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology
    from src.mesh.mesh import Mesh

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    phi0 = Field(top.cells.center**2, 'K') * 100
    me = Mesh(top, phi0)

    def bval(patch):
        b = patch.center
        return Field(b**2, 'K') * 100
    for pt in range(4):
        me.set_BC(pt, bval, [])

    diver = divergence(me)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0, 0]
    y = me.topology.cells.center[:, 1, 0]

    z = np.array(diver.reshape((-1,)))

    ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    plt.show()
