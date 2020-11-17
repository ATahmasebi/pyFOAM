import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.Utilities.primitives import BaseTerm, on_demand_prop


class Laplacian(BaseTerm):

    def __init__(self, mesh: Mesh, gamma: Field, correction='or'):
        super().__init__(mesh.topology.info.cells)
        self.mesh = mesh
        self.gamma_f, self.gamma_bf = self._gamma_interpolation(gamma)
        self.correction = correction
        self.Ef, self.Tf = self._face_decompose()

    def _gamma_interpolation(self, gamma):
        if len(gamma) == self.mesh.topology.info.cells:
            gamma_bf = gamma[self.mesh.topology.boundary.owner]
        else:
            gamma_bf = gamma
        gamma_f = self.mesh.topology.face_interpolate(gamma)
        return gamma_f, gamma_bf

    def _face_decompose(self):
        dCF = self.mesh.topology.dCF
        Sf = self.mesh.topology.internal.vector
        if self.correction == 'or':
            Ef = dCF * dot(Sf, Sf) / dot(dCF, Sf)
            Tf = Sf - Ef
        elif self.correction == 'mc':
            Ef = dCF * dot(Sf, dCF) / dot(dCF, dCF)
            Tf = Sf - Ef
        elif self.correction == 'oc':
            Ef = Sf.norm * dCF / dCF.norm
            idx = np.array(dot(Sf, dCF)).reshape((-1,)) < 0
            Ef[idx] = -Ef[idx]
            Tf = Sf - Ef
        elif self.correction == 'uncorrected':
            Ef = Sf
            Tf = None
        else:
            raise ValueError('invalid orthogonal correction method.')
        return Ef, Tf

    @on_demand_prop
    def LHS(self):
        gamma_f = self.gamma_f
        dCF = self.mesh.topology.dCF

        af = (gamma_f * self.Ef.norm / dCF.norm).reshape((-1,))
        naf = -af
        owner = self.mesh.topology.internal.owner
        neighbour = self.mesh.topology.internal.neighbour
        self.lhs_add(owner, owner, af)
        self.lhs_add(neighbour, neighbour, af)
        self.lhs_add(owner, neighbour, naf)
        self.lhs_add(neighbour, owner, naf)

        Sb = self.mesh.topology.boundary.vector
        ndCb = self.mesh.topology.ndCb
        abf = self.gamma_bf * Sb.norm / ndCb
        ownerb = self.mesh.topology.boundary.owner
        self.lhs_add(ownerb, ownerb, abf.reshape((-1,)))
        return self.get_lhs()

    @property
    def RHS(self):
        return self.get_rhs()

    def update(self):
        Sb = self.mesh.topology.boundary.vector
        ndCb = self.mesh.topology.ndCb
        phi_b = self.mesh.phi_b
        ownerb = self.mesh.topology.boundary.owner
        abf_phib = phi_b * self.gamma_bf * Sb.norm / ndCb
        _, dim, _ = abf_phib.shape
        rhs = Field(np.zeros(shape=(self.mesh.topology.info.cells, dim, 1)), abf_phib.unit)
        np.add.at(rhs, ownerb, abf_phib)

        # correction:
        if self.correction in ['or', 'oc', 'mc']:
            grad = self.mesh.gradient
            grad_f = self.mesh.topology.face_interpolate(grad)
            neighbour = self.mesh.topology.internal.neighbour
            owner = self.mesh.topology.internal.owner
            ac = self.gamma_f * (grad_f @ self.Tf)
            np.add.at(rhs, owner, ac)
            np.subtract.at(rhs, neighbour, ac)

        self.rhs_add(rhs)


def laplacian(gamma: Field, mesh: Mesh, correction='or'):
    gamma_f = mesh.topology.face_interpolate(gamma)
    dCF = mesh.topology.dCF
    Sf = mesh.topology.internal.vector
    if correction == 'or':
        Ef = dCF * dot(Sf, Sf) / dot(dCF, Sf)
    elif correction == 'mc':
        Ef = dCF * dot(Sf, dCF) / dot(dCF, dCF)
    elif correction == 'oc':
        Ef = Sf.norm * dCF / dCF.norm
        idx = np.array(dot(Sf, dCF)).reshape((-1,)) < 0
        Ef[idx] = -Ef[idx]
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
    me = Mesh(top, Field([300], 'K'))
    me.set_BC(1, lambda *args: Field(300, 'K'), [])
    me.set_BC(0, lambda patch, mesh: mesh.phi[patch.owner], [me])
    me.set_BC(2, lambda *args: Field(310, 'K'), [])
    me.set_BC(3, lambda patch, mesh: mesh.phi[patch.owner], [me])

    gam = Field([1000], 'W/m.K')

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    phi_list = []
    N = 100
    for i in range(N + 1):
        laplacian(gam, me, correction='or')
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

    pch = me.topology.boundary.patch == 3
    y2 = me.topology.boundary.center[pch, 1, 0]
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_xlim(0, 0.1)
    ax2.set_ylim(299, 311)
    data = sorted(zip(y2, me.phi[me.topology.boundary.owner[pch]].reshape((-1,))), key=lambda d: d[0])
    xx = [d[0] for d in data]
    tt = [d[1] for d in data]
    ax2.plot(xx, tt)
    plt.show()
