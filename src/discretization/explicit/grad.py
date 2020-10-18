import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.vector import dot, norm
from src.Utilities.field import Field


def least_sqr(mesh: Mesh):
    ginv = mesh.topology.Ginv
    owners = mesh.topology.internal.owner
    neighbours = mesh.topology.internal.neighbour
    _, dim = mesh.phi.shape
    delta_phi = (mesh.phi[owners] - mesh.phi[neighbours]).reshape((-1, dim, 1))
    dCF = mesh.topology.dCF
    w = dCF.norm.reshape((-1, 1, 1))
    dt = dCF.reshape((-1, 1, 3)) / w
    rhsc = delta_phi @ dt
    rhs = Field(np.zeros((mesh.topology.info.cells, dim, 3)), rhsc.unit)
    np.add.at(rhs, owners, rhsc)
    np.add.at(rhs, neighbours, rhsc)
    delta_phi_b = (mesh.phi[mesh.topology.boundary.owner] - mesh.boondrypatch).reshape((-1, dim, 1))
    wb = mesh.topology.dCb.norm.reshape((-1, 1, 1))
    dbt = mesh.topology.dCb.reshape((-1, 1, 3)) / wb
    rhsb = delta_phi_b @ dbt
    np.add.at(rhs, mesh.topology.boundary.owner, rhsb)
    grad = rhs @ ginv
    return grad


def Green_Guass(mesh: Mesh, corr: int):
    grad = None
    _, dim = mesh.phi.shape
    phi_f = mesh.topology.face_interpolate(mesh.phi).reshape((-1, dim, 1))
    cv = mesh.topology.cells.volume.reshape((-1, 1, 1))
    Sf = mesh.topology.internal.vector
    dCF = mesh.topology.dCF
    ngcs = (dCF.unit_vector / dCF.norm).reshape((-1, 1, 3))
    owners = mesh.topology.internal.owner
    neighbours = mesh.topology.internal.neighbour
    ngrad = (mesh.phi[owners] - mesh.phi[neighbours]).reshape((-1, dim, 1)) @ ngcs
    for step in range(corr + 1):
        if grad is not None:
            grad_f = mesh.topology.face_interpolate(grad.reshape((-1, dim * 3))).reshape((-1, dim, 3))
            grad_f = grad_f - (grad_f @ dCF.reshape((-1, 3, 1))) @ ngcs + ngrad
            phi_f = phi_f + grad_f @ mesh.topology.ff.reshape((-1, 3, 1))
        af = phi_f @ Sf.reshape((-1, 1, 3))
        grad = Field(np.zeros((mesh.topology.info.cells, dim, 3)), af.unit)
        np.add.at(grad, mesh.topology.internal.owner, af)
        np.subtract.at(grad, mesh.topology.internal.neighbour, af)

        phi_b = mesh.boondrypatch.reshape((-1, dim, 1))
        ab = phi_b @ mesh.topology.boundary.vector.reshape((-1, 1, 3))
        np.add.at(grad, mesh.topology.boundary.owner, ab)
        grad = grad / cv

    return grad


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test00.mphtxt'
    # path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
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
    me.boondrypatch = Field(dot(b.center, b.center) * 100, 'K')
    # me.boondrypatch = Field(b.center[:,0]**2 * 100, 'K').reshape((-1, 1))

    me.phi = Field(dot(me.topology.cells.center, me.topology.cells.center), 'K') * 100
    # me.phi = Field(me.topology.cells.center[:,0]**2 *100, 'K').reshape((-1, 1))
    # gradient = Green_Guass(me, 1)
    gradient = least_sqr(me)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0]  # me.topology.internal.center[:,0] # me.topology.internal.center[:,0] #
    y = me.topology.cells.center[:, 1]  # me.topology.internal.center[:,1] # me.topology.cells.center[:, 1] #
    # z = np.array(norm(me.topology.cells.center).reshape((-1,))) *200
    # z = np.array(me.topology.face_interpolate(me.phi)[:,0])
    z = np.array(norm(gradient.reshape((-1, 3))).reshape((-1,)))
    ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    plt.show()
