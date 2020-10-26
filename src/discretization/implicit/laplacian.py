import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.discretization.explicit.grad import least_sqr as gradient


def laplacian(gamma: Field, mesh: Mesh, correction='or'):
    gamma_f = mesh.topology.face_interpolate(gamma)
    dCF = mesh.topology.dCF
    Sf = mesh.topology.internal.vector
    if correction == 'or':
        Ef = dCF * dot(Sf, Sf) / dot(dCF, Sf)
    elif correction == 'mc':
        Ef = dCF * dot(Sf, dCF) / dot(dCF, dCF)
    elif correction == 'oc':
        Ef = dCF * dot(Sf, Sf) / dot(Sf, dCF)
    elif correction == 'uncorrected':
        Ef = Sf
    else:
        raise ValueError('invalid orthogonal correction method.')
    af = gamma_f * Ef.norm / dCF.norm
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
    phi_b = mesh.boundarypatch
    ndCb = mesh.topology.ndCb
    abf = gamma_bf * Sb.norm / ndCb
    abf_phib = phi_b * abf
    _, dim = abf_phib.shape
    rhs = Field(np.zeros(shape=(mesh.topology.info.cells, dim, 1)), abf_phib.unit)
    rhs[mesh.topology.boundary.owner] = abf_phib

    # correction:
    if correction in ['or', 'oc', 'mc']:
        Tf = Sf - Ef
        grad = gradient(mesh)
        grad_f = mesh.topology.face_interpolate(grad)
        ac = gamma_f * (grad_f @ Tf)
        np.add.at(rhs, owner, ac)

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
    me = Mesh(top)
    b = me.topology.boundary
    me.boundarypatch = Field(dot(b.center, b.center) * 100, 'K')
    # me.boondrypatch = Field(b.center[:,0]**2 * 100, 'K').reshape((-1, 1))

    me.phi = Field(dot(me.topology.cells.center, me.topology.cells.center), 'K') * 100
