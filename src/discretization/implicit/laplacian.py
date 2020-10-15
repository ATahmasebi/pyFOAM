import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field_operations import dot, norm
from src.Utilities.field import Field


def laplacian(gamma, mesh: Mesh, correction='or'):
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
    af = gamma_f * norm(Ef) / norm(dCF)
    naf = -af
    owner = mesh.topology.internal.owner
    neighbour = mesh.topology.internal.neighbour
    mesh.LS.lhs_add(owner, owner, af)
    mesh.LS.lhs_add(neighbour, neighbour, af)
    mesh.LS.lhs_add(owner, neighbour, naf)
    mesh.LS.lhs_add(neighbour, owner, naf)

    for boundary in mesh.topology.boundary:
        if gamma.shape == (mesh.topology.info.cells, 1):
            gamma_bf = gamma_f[boundary.owner]
        else:
            gamma_bf = gamma
        Sb = norm(boundary.vector)
        p = mesh.BC[boundary.patch]
        if p.type == 'value':
            phi_b = p.values
            dCb = norm(mesh.topology.dCb)
            abf = gamma_bf * Sb / dCb  # check later!normal distance?!
            mesh.LS.lhs_add(boundary.owner, boundary.owner, abf)
            abf_phib = phi_b * abf
            _, dim = abf_phib.shape
            rhs = Field(np.zeros(shape=(mesh.topology.info.cells, dim)), abf_phib.unit)
            rhs[boundary.owner] = abf_phib
            mesh.LS.rhs_add(rhs)
        elif p.type == 'flux':
            values = -gamma_bf * Sb * p.values  # Minus sighn??????????
            _, dim = values.shape
            rhs = Field(np.zeros(shape=(mesh.topology.info.cells, dim)), values.unit)
            rhs[boundary.owner] = values
            mesh.LS.rhs_add(rhs)
        elif p.type == 'robin':
            pass
    # correction!!!!!!!!!!
