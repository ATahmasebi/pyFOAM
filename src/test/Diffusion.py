import numpy as np
from src.mesh.mesh import Mesh
from src.Utilities.field import Field, dot
from src.Utilities.primitives import BaseTerm, on_demand_prop
from src.conversion.comsol import read_comsol_file, build_element_connectivity
from src.conversion.convert import connectivity_to_foam
from src.mesh.topology import Topology
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse.linalg import spsolve

from src.Utilities.field import Field
from src.mesh.mesh import Mesh
from src.discretization.implicit.laplacian import Laplacian
from src.discretization.implicit.ddt import Ddt
from src.discretization.implicit.divergence import Divergence
from src.conversion.comsol import read_comsol_file, build_element_connectivity
from src.conversion.convert import connectivity_to_foam
from src.mesh.topology import Topology

path = 'D:\\Documents\\Code\\pyFOAM\\src\\test\\heat.mphtxt'
# path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'

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

k = Field([3850], 'W/m.K')
rho = Field([8960], 'kg/m**3')
cp = Field([385], 'J/kg.K')
dt = Field([0.1], 's')
gamma = k / (rho * cp)
lap = Laplacian(me, rho * gamma)
ddt = Ddt(me, rho, dt, 'SOUE')

N = 100
phi_list = []

for i in range(N):
    ddt.update()
    ddtlhs = ddt.LHS
    ddtrhs = ddt.RHS
    for j in range(20):
        lap.update()
        lhs = lap.LHS + ddtlhs
        rhs = lap.RHS + ddtrhs
        phi = spsolve(lhs, rhs.reshape((-1,)))
        me.phi = Field(phi, rhs.unit / lap.lhs_unit).reshape((-1, 1, 1))
    phi_list.append(me.phi)
fig = plt.figure(figsize=(6, 6))
ax = fig.gca()
x = me.topology.cells.center[:, 0, 0]
y = me.topology.cells.center[:, 1, 0]
# z = np.array(me.phi).reshape((-1,))
# ax.tricontour(x, y, z, levels=20, linewidths=0.5, colors='k')
# cntr2 = ax.tricontourf(x, y, z, levels=20, cmap="YlOrRd")
# fig.colorbar(cntr2, ax=ax)
# ax.plot(x, y, 'ko', ms=3)
levels = np.linspace(298, 312, 12)


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
    ax.set_title(f'{frame * dt}')
    return ctr


ani = FuncAnimation(fig, update, frames=N, init_func=init, interval=50, repeat=False)

plt.show()
pch = me.topology.boundary.patch == 3
y2 = me.topology.boundary.center[pch, 1, 0]
fig2 = plt.figure()
ax2 = fig2.gca()

# ax2.set_xlim(0, 0.1)
# ax2.set_ylim(299, 311)
data = sorted(zip(y2, me.phi[me.topology.boundary.owner[pch]].reshape((-1,))), key=lambda d: d[0])
xx = [d[0] for d in data]
tt = [d[1] for d in data]
ax2.plot(xx, tt)
plt.grid()
plt.show()
