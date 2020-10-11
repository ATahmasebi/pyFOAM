import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field_operations import dot, norm
from src.Utilities.field import Field


def Green_Guass(mesh: Mesh):
    phi_f = mesh.topology.face_interpolate(mesh.phi)
    cv = mesh.topology.cells.volume
    Sf = mesh.topology.internal.vector
    af = phi_f * Sf
    _, dim = mesh.phi.shape
    grad = Field(np.zeros((mesh.topology.info.cells, 3)), af.unit)
    print(af.shape)
    print(grad.shape)
    np.add.at(grad, mesh.topology.internal.owner, af)
    np.subtract.at(grad, mesh.topology.internal.neighbour, af)

    # for boundary in mesh.topology.boundary:
    #     Sb = boundary.vector
    #     p = mesh.BC[boundary.patch]
    #     if p.type == 'value':
    #         phi_b = p.values
    #         afb = phi_b * Sb
    #         np.add.at(grad, boundary.owner, afb)
    #     elif p.type == 'flux':
    #         dphi_dn = p.values
    #         dCb = mesh.topology.cells.center[boundary.owner] - boundary.center
    #         norm_dist = Sb * dot(dCb, Sb) / dot(Sb, Sb)
    #         phi_b = mesh.phi[boundary.owner] + dphi_dn * norm_dist  # plus or minus???????????????
    #         afb = phi_b * Sb
    #         np.add.at(grad, boundary.owner, afb)
    #     elif p.type == 'robin':
    #         pass

    return grad / cv


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\test0.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    me = Mesh(top)
    zero = Field(0,'K/m')
    me.set_BC(1, zero, 'flux', 'wall')
    me.set_BC(2, zero, 'flux', 'wall')
    me.set_BC(3, zero, 'flux', 'wall')
    me.set_BC(0, zero, 'flux', 'wall')
    me.phi = Field(norm(me.topology.cells.center), 'K') * 100
    gradient = Green_Guass(me)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    x = me.topology.cells.center[:, 0] # me.topology.internal.center[:,0] #
    y = me.topology.cells.center[:, 1] # me.topology.internal.center[:,1] #
    z = np.array(gradient[:, 0]) #np.array(me.phi)[:,0] # np.array(me.topology.face_interpolate(me.phi)[:,0]) #
    ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)
    #ax.plot(x, y, 'ko', ms=3)
    plt.show()

