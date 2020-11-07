import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot


def laplacian(gamma: Field, mesh: Mesh, correction='or'):
    gamma_f = mesh.topology.face_interpolate(gamma)
    dCF = mesh.topology.dCF
    Sf = mesh.topology.internal.vector
    if correction == 'or':
        Ef = dCF * dot(Sf, Sf) / dot(dCF, Sf)
    elif correction == 'mc':
        Ef = dCF * dot(Sf, dCF) / dot(dCF, dCF)
    elif correction == 'oc':
        Ef = dCF / dCF.norm * Sf.norm
    elif correction == 'uncorrected':
        Ef = Sf
    else:
        raise ValueError('invalid orthogonal correction method.')
    af = (gamma_f * Ef.norm / dCF.norm).reshape((-1,))
    naf = -af
    owner = mesh.topology.internal.owner
    neighbour = mesh.topology.internal.neighbour
    mesh.LS.lhs_add(owner, owner, af)
    mesh.LS.lhs_add(neighbour, neighbour, af)
    mesh.LS.lhs_add(owner, neighbour, naf)
    mesh.LS.lhs_add(neighbour, owner, naf)

    if len(gamma) == mesh.topology.info.cells:
        gamma_bf = gamma_f[mesh.topology.boundary.owner]
    else:
        gamma_bf = gamma
    Sb = mesh.topology.boundary.vector
    phi_b = mesh.phi_b
    ndCb = mesh.topology.ndCb
    abf = gamma_bf * Sb.norm / ndCb
    ownerb = me.topology.boundary.owner
    mesh.LS.lhs_add(ownerb, ownerb, abf.reshape((-1,)))
    abf_phib = phi_b * abf
    _, dim, _ = abf_phib.shape
    rhs = Field(np.zeros(shape=(mesh.topology.info.cells, dim, 1)), abf_phib.unit)
    np.add.at(rhs, ownerb, abf_phib)

    # correction:
    if correction in ['or', 'oc', 'mc']:
        Tf = Sf - Ef
        grad = mesh.gradient
        grad_f = mesh.topology.face_interpolate(grad)
        ac = gamma_f * (grad_f @ Tf)
        np.add.at(rhs, owner, ac)
        np.subtract.at(rhs, neighbour, ac)

    mesh.LS.rhs_add(rhs)


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test0.mphtxt'
    # path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    me = Mesh(top, Field([300], 'K'))
    me.set_BC(1, lambda *args: Field(300, 'K'), [])
    me.set_BC(0, lambda patch, mesh: mesh.phi[patch.owner], [me])
    me.set_BC(2, lambda *args: Field(310, 'K'), [])
    me.set_BC(3, lambda patch, mesh: mesh.phi[patch.owner], [me])

    gamma = Field([1000], 'W/m.K')

    import matplotlib.pyplot as plt
    for i in range(100):
        laplacian(gamma, me, correction='uncorrected')
        me.phi = me.LS.solve()
        me.LS.clear()
    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0, 0]
    y = me.topology.cells.center[:, 1, 0]
    z = np.array(me.phi).reshape((-1,))
    print(me.phi.unit)
    ax.tricontour(x, y, z, levels=20, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=20, cmap="YlOrRd")
    fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    plt.show()




