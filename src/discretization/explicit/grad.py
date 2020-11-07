import numpy as np
from src.Utilities.field import Field, dot


def least_sqr(mesh):
    ginv = mesh.topology.Ginv
    owners = mesh.topology.internal.owner
    neighbours = mesh.topology.internal.neighbour
    _, dim, _ = mesh.phi.shape
    delta_phi = (mesh.phi[owners] - mesh.phi[neighbours])
    dCF = mesh.topology.dCF
    w = dCF.norm
    dt = dCF.reshape((-1, 1, 3)) / w
    rhsc = delta_phi @ dt
    rhs = Field(np.zeros((mesh.topology.info.cells, dim, 3)), rhsc.unit)
    np.add.at(rhs, owners, rhsc)
    np.add.at(rhs, neighbours, rhsc)
    delta_phi_b = (mesh.phi[mesh.topology.boundary.owner] - mesh.phi_b)
    wb = mesh.topology.dCb.norm
    dbt = mesh.topology.dCb.reshape((-1, 1, 3)) / wb
    rhsb = delta_phi_b @ dbt
    np.add.at(rhs, mesh.topology.boundary.owner, rhsb)
    grad = rhs @ ginv
    return grad


def Green_Guass(mesh, corr: int):
    grad = None
    _, dim, _ = mesh.phi.shape
    phi_f = mesh.topology.face_interpolate(mesh.phi)
    cv = mesh.topology.cells.volume
    Sf = mesh.topology.internal.vector
    dCF = mesh.topology.dCF
    ngcs = (dCF / (dCF.norm * dCF.norm)).reshape((-1, 1, 3))
    owners = mesh.topology.internal.owner
    neighbours = mesh.topology.internal.neighbour
    ngrad = (mesh.phi[owners] - mesh.phi[neighbours]) @ ngcs
    for step in range(corr + 1):
        if grad is not None:
            grad_f = mesh.topology.face_interpolate(grad)
            grad_f = grad_f - (grad_f @ dCF) @ ngcs + ngrad
            phi_f = phi_f + grad_f @ mesh.topology.ff
        af = phi_f @ Sf.reshape((-1, 1, 3))
        grad = Field(np.zeros((mesh.topology.info.cells, dim, 3)), af.unit)
        np.add.at(grad, mesh.topology.internal.owner, af)
        np.subtract.at(grad, mesh.topology.internal.neighbour, af)

        phi_b = mesh.phi_b
        ab = phi_b @ mesh.topology.boundary.vector.reshape((-1, 1, 3))
        np.add.at(grad, mesh.topology.boundary.owner, ab)
        grad = grad / cv

    return grad


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test3.mphtxt'
    # path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology
    from src.mesh.mesh import Mesh

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    phi0 = Field(dot(top.cells.center, top.cells.center), 'K') * 100
    me = Mesh(top, phi0)

    def bval(patch):
        b = patch.center
        return Field(dot(b, b)*100, 'K')
    for pt in range(4):
        me.set_BC(pt, bval, [])

    # b = me.topology.boundary
    # me.boundarypatch = Field(dot(b.center, b.center) * 100, 'K')
    # me.boondrypatch = Field(b.center[:,0]**2 * 100, 'K').reshape((-1, 1))

    # me.phi = Field(me.topology.cells.center[:,0]**2 *100, 'K').reshape((-1, 1))
    # gradient = Green_Guass(me, 0)
    gradient = Green_Guass(me, 1)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0, 0]  # me.topology.internal.center[:,0] # me.topology.internal.center[:,0] #
    y = me.topology.cells.center[:, 1, 0]  # me.topology.internal.center[:,1] # me.topology.cells.center[:, 1] #
    # z = np.array(norm(me.topology.cells.center).reshape((-1,))) *200
    # z = np.array(me.topology.face_interpolate(me.phi)[:,0])

    z = np.array(gradient.norm.reshape((-1,)))
    # z = np.array(gradient[:,0,0])

    ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    plt.show()
