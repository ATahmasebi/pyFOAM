import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.vector import VectorField, dot, norm
from src.Utilities.field import Field


def least_sqr(mesh: Mesh):
    pass


def Green_Guass(mesh: Mesh, corr: int):
    grad = Field(np.zeros_like(mesh.topology.cells.center), f'{mesh.phi.unit}/{mesh.topology.info.unit}')
    phi_f = mesh.topology.face_interpolate(mesh.phi)
    cv = mesh.topology.cells.volume
    Sf = mesh.topology.internal.vector
    dCF = mesh.topology.dCF
    n2dCF = dot(dCF, dCF)
    ngrad = ((mesh.phi[mesh.topology.internal.owner] - mesh.phi[mesh.topology.internal.neighbour]) / n2dCF) * dCF
    for step in range(corr + 1):
        grad_f = mesh.topology.face_interpolate(grad)
        grad_ff = grad_f - (dot(grad_f, dCF) / n2dCF) * dCF + ngrad
        phi_ff = phi_f + dot(mesh.topology.ff, grad_ff)
        af = phi_ff * Sf
        _, dim = mesh.phi.shape
        grad = Field(np.zeros((mesh.topology.info.cells, 3)), af.unit)
        np.add.at(grad, mesh.topology.internal.owner, af)
        np.subtract.at(grad, mesh.topology.internal.neighbour, af)

        for boundary in mesh.topology.boundary:
            Sb = boundary.vector
            p = mesh.BC[boundary.patch]
            if p.type == 'value':
                phi_b = p.values
                afb = phi_b * Sb
                np.add.at(grad, boundary.owner, afb)
            elif p.type == 'flux':
                dphi_dn = p.values
                dCb = mesh.topology.cells.center[boundary.owner] - boundary.center
                norm_dist = Sb * dot(dCb, Sb) / dot(Sb, Sb)
                phi_b = mesh.phi[boundary.owner] + (dphi_dn * norm_dist)  # plus or minus???????????????
                afb = phi_b * Sb
                np.add.at(grad, boundary.owner, afb)
            elif p.type == 'robin':
                pass
        grad = grad / cv

    return grad


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test3.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    me = Mesh(top)
    b = me.topology.boundary
    v = Field(dot(b.center, b.center) * 100, 'K')
    #me.set_BC(b.patch, v, 'value', 'wall')
    me.phi = Field(dot(me.topology.cells.center, me.topology.cells.center), 'K') * 100
    gradient = Green_Guass(me, 1)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0]  # me.topology.internal.center[:,0] # me.topology.internal.center[:,0] #
    y = me.topology.cells.center[:, 1]  # me.topology.internal.center[:,1] # me.topology.cells.center[:, 1] #
    # z = np.array(norm(me.topology.cells.center).reshape((-1,))) *200
    # z = np.array(me.topology.face_interpolate(me.phi)[:,0])
    z = np.array(norm(gradient).reshape((-1,)))
    ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    plt.show()
