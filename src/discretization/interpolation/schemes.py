import numpy as np
from src.Utilities.vector import dot

def CD(mesh, phi):
    if len(phi) == mesh.topology.info.cells:
        gf = mesh.topology.gf
        return phi[mesh.topology.internal.owner] * (1 - gf) + phi[mesh.topology.internal.neighbour] * gf
    elif phi.shape == () or phi.shape == (1,):
        return phi
    else:
        raise ValueError('Cannot interpolate values to faces.')

def Upwind(mDot, mesh, phi):
    phi_f = phi[mesh.topology.internal.owner]
    index = np.asarray(mesh.topology.internal.vector @ mDot) < 0

