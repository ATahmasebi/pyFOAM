import numpy as np


def Upwind(mDot, mesh):
    # phi = mesh.phi
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    # phi_c = phi[int_index]
    return int_index, 0


def CD(mDot, mesh):
    # phi = mesh.phi
    dCf = mesh.topology.dCf.copy()
    grad = mesh.gradient
    grad_f = mesh.topology.face_interpolate(grad)
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    dCf[index] = -dCf[index]
    # phi_c = phi[int_index]
    corr = - grad_f @ dCf
    return int_index, corr


def SOU(mDot, mesh):
    # phi = mesh.phi
    dCf = mesh.topology.dCf.copy()
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    dCf[index] = -dCf[index]
    # phi_c = phi[int_index]
    grad_c = mesh.gradient[int_index]
    grad_f = mesh.topology.face_interpolate(mesh.gradient)
    corr = (grad_f - 2 * grad_c) @ dCf
    return int_index, corr


def FROMM(mDot, mesh):
    # phi = mesh.phi
    dCf = mesh.topology.dCf.copy()
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    dCf[index] = -dCf[index]
    # phi_c = phi[int_index]
    grad_c = mesh.gradient[int_index]
    corr = - grad_c @ dCf
    return int_index, corr


def QUICK(mDot, mesh):
    # phi = mesh.phi
    dCf = mesh.topology.dCf.copy()
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    dCf[index] = -dCf[index]
    # phi_c = phi[int_index]
    grad_c = mesh.gradient[int_index]
    grad_f = mesh.topology.face_interpolate(mesh.gradient)
    corr = - 0.5 * (grad_c + grad_f) @ dCf
    return int_index, corr


def Downwind(mDot, mesh):
    # phi = mesh.phi
    dCf = mesh.topology.dCf.copy()
    index = (np.asarray(mDot.reshape((-1, 1, 3)) @ mesh.topology.internal.vector) < 0).reshape((-1))
    int_index = mesh.topology.internal.owner.copy()
    int_index[index] = mesh.topology.internal.neighbour[index]
    dCf[index] = -dCf[index]
    # phi_c = phi[int_index]
    grad_f = mesh.topology.face_interpolate(mesh.gradient)
    corr = -2 * (grad_f @ dCf)
    return int_index, corr
