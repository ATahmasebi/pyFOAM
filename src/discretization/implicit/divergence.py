import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.discretization.explicit.grad import least_sqr as gradient
from src.discretization.interpolation.schemes import face_interpolation


def divergance(mesh: Mesh, mDot: Field, mDotb: Field):
    index, corr = face_interpolation(mDot, mesh)
    af = mDot * mesh.topology.internal.vector
    owner = mesh.topology.internal.owner
    neighbour = mesh.topology.internal.neighbour
    mesh.LS.lhs_add(owner, index, af)
    mesh.LS.lhs_add(neighbour, index, -af)
    abf = mDotb * mesh.topology.boundary.vector * mesh.phi_b


def new(mesh: Mesh, Vmesh: Mesh, rho: Field):
    vf = Vmesh.topology.face_interpolate(Vmesh.phi)
    flux_f = rho * dot(vf, mesh.topology.internal.vector).reshape((-1,))
    index, corr = face_interpolation(mesh, flux_f, scheme='QUICK')
    owner = mesh.topology.internal.owner
    neighbour = mesh.topology.internal.neighbour
    mesh.LS.lhs_add(owner, index, flux_f)
    mesh.LS.lhs_add(neighbour, index, -flux_f)
    flux_bf = rho * dot(Vmesh.phi_b, mesh.topology.boundary.vector) * mesh.phi_b
    _, dim, _ = flux_bf.shape
    rhs = Field(np.zeros(shape=(mesh.topology.info.cells, dim, 1)), flux_bf.unit)
    np.add.at(rhs, owner, corr * flux_f.reshape((-1, 1, 1)))
    np.subtract.at(rhs, mesh.topology.boundary.owner, flux_bf)
    mesh.LS.rhs_add(rhs)
    mesh.LS.rows.append([me.topology.info.cells - 2])
    mesh.LS.columns.append([me.topology.info.cells - 2])
    mesh.LS.data.append(Field([0.02], 'kg/s'))


if __name__ == '__main__':
    path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\heat.mphtxt'
    # path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.topology import Topology

    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    v0 = Field([[0.01], [0.01], [0]], 'm/s')
    vmesh = Mesh(top, v0)
    for p in range(4):
        vmesh.set_BC(p, lambda *args: v0, [])

    me = Mesh(top, Field([300], 'K'))
    me.set_BC(1, lambda *args: Field([310], 'K'), [])
    me.set_BC(0, lambda patch, mesh: mesh.phi[patch.owner], [me])
    me.set_BC(2, lambda patch, mesh: mesh.phi[patch.owner], [me])
    me.set_BC(3, lambda patch, mesh: mesh.phi[patch.owner], [me])
    rho_w = Field([1000], 'kg/m**3')
    new(me, vmesh, rho_w)



    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    phi_list = []
    N = 10
    for i in range(N + 1):
        new(me, vmesh, rho_w)
        me.phi = me.LS.solve()
        phi_list.append(me.phi)
        me.LS.clear()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    x = me.topology.cells.center[:, 0, 0]
    y = me.topology.cells.center[:, 1, 0]
    # z = np.array(me.phi).reshape((-1,))
    # ax.tricontour(x, y, z, levels=20, linewidths=0.5, colors='k')

    # cntr2 = ax.tricontourf(x, y, z, levels=20, cmap="YlOrRd")
    # fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    levels = np.linspace(300, 310, 10)


    def init():
        x_max = np.max(x)
        x_min = np.min(x)
        y_max = np.max(y)
        y_min = np.min(y)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


    def update(frame):
        ax.clear()
        zz = np.array(phi_list[frame]).reshape((-1,))
        ctr = ax.tricontourf(x, y, zz, levels=levels, cmap="YlOrRd")
        ax.set_title(frame)
        return ctr


    ani = FuncAnimation(fig, update, frames=N, init_func=init, interval=50, repeat=False)

    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.gca()
    data = sorted(zip(y, me.phi.reshape((-1,))), key=lambda d: d[0])
    xx = [d[0] for d in data]
    tt = [d[1] for d in data]
    ax2.plot(xx, tt)
    plt.show()