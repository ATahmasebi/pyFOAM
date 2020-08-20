import numpy as np
import pystencils as ps
from src.Utilities.field_operations import *
from collections import namedtuple


def face_decompose(topology, method='or'):
    cells = topology.cells
    faces = topology.internal
    dcf = cells.center[faces.owner] - cells.center[faces.neighbour]
    Sf = faces.vector
    if method == 'or':
        Ef = dcf * dot(Sf, Sf) / dot(dcf, Sf)
    elif method == 'mc':
        Ef = dcf * dot(Sf, dcf) / dot(dcf, dcf)
    elif method == 'oc':
        Ef = dcf * dot(Sf, Sf) / dot(Sf, dcf)
    else:
        raise ValueError('Invalid face decomposing method.')
    Tf = Sf - Ef
    return Tf, Ef

if __name__ == '__main__':
    path = 'D:\\Documents\\VScode\\Python\\pyFOAM\\src\\conversion\\line.mphtxt'
    from src.conversion.comsol import read_comsol_file, build_element_connectivity
    from src.conversion.convert import connectivity_to_foam
    from src.mesh.primitives import Topology
    elem = read_comsol_file(path)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    foam['unit'] = 'm'
    top = Topology(foam)
    tf, ef = face_decompose(top)
    print(ef)

