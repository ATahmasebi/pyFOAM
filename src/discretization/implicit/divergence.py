import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.discretization.explicit.grad import least_sqr as gradient
import src.discretization.interpolation.schemes as scheames


def divergance(mesh: Mesh, mDot: Field,  mDotb: Field, interpolation=scheames.QUICK):
    index, corr = interpolation(mDot, mesh)
    af = mDot * mesh.topology.internal.vector
    owner = mesh.topology.internal.owner
    neighbour = mesh.topology.internal.neighbour
    mesh.LS.lhs_add(owner, index, af)
    mesh.LS.lhs_add(neighbour, index, -af)
    abf = mDotb * mesh.topology.boundary.vector * mesh.boundarypatch

