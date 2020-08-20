import numpy as np


def read_comsol_file(file_path):
    element_types = (' pyr ', ' tet ', ' prism ', ' hex ', ' vtx ', ' edg ', ' tri ', ' quad ')
    ndim = None
    npoints = None
    comsol_elements = {}

    with open(file_path) as file:
        lines = file.readlines()

    for line in lines:
        if '# sdim' in line:
            ndim = int(line.split()[0])
        if '# number of mesh points' in line:
            npoints = int(line.split()[0])
            break

    lnum = lines.index('# Mesh point coordinates\n', ) + 1
    points = np.array([[float(n) for n in line.split()] for line in lines[lnum:lnum + npoints]])
    lnum += npoints

    while lnum < len(lines):
        line = lines[lnum]
        if any(et in line for et in element_types):
            elemnt = [et for et in element_types if et in line][0]
            nelements = int(lines[lnum + 4].split()[0])
            lnum += 6
            comsol_elements[elemnt] = np.array([[int(n) for n in L.split()] for L in lines[lnum:lnum + nelements]])
            lnum += nelements
            nge = int(lines[lnum + 1].split()[0])
            if nge == nelements:
                lnum += 3
                comsol_elements[elemnt + 'gei'] = np.array([int(i) for i in lines[lnum:lnum + nelements]])
                lnum += nelements
        lnum += 1

    comsol_elements['ndim'] = ndim
    comsol_elements['points'] = points

    return build_3d_mesh(comsol_elements)


def build_3d_mesh(elements, dim1=1, dim2=1):
    dim = elements['ndim']
    if dim == 3:
        return elements
    elif dim == 1:
        elements_2d = {'ndim': 2}
        d = dim2 / 2
        points = elements['points']
        npoints = len(points)
        elements_2d['points'] = np.concatenate((np.insert(points, 1, d, axis=1),
                                                np.insert(points, 1, -d, axis=1)), axis=0)
        edges = elements[' edg ']
        elements_2d[' quad '] = np.concatenate((edges, edges + npoints), axis=1)
        elements_2d[' quad gei'] = elements[' edg gei']
        vtx = elements[' vtx ']
        vtx_edges = np.concatenate((vtx, vtx + npoints), axis=1)
        boundary_edges = np.concatenate((edges, edges+npoints), axis=0)
        elements_2d[' edg '] = np.concatenate((vtx_edges, boundary_edges), axis=0)
        elements_2d[' edg gei'] = np.concatenate((elements[' vtx gei'], np.full((len(boundary_edges),), -1)), axis=0)
        elements = elements_2d
    d = dim1 / 2
    elements_3d = {'ndim': 3}
    points = elements['points']
    npoints = len(points)
    elements_3d['points'] = np.concatenate((np.insert(points, 2, d, axis=1), np.insert(points, 2, -d, axis=1)), axis=0)
    edges = elements[' edg ']
    elements_3d[' quad '] = np.concatenate((edges, edges + npoints), axis=1)
    elements_3d[' quad gei'] = elements[' edg gei']
    if ' quad ' in elements:
        quads = elements[' quad ']
        elements_3d[' hex '] = np.concatenate((quads, quads + npoints), axis=1)
        elements_3d[' hex gei'] = elements[' quad gei']
        boundary_quads = np.concatenate((quads, quads + npoints), axis=0)
        existing_quads = elements_3d[' quad ']
        elements_3d[' quad '] = np.concatenate((existing_quads, boundary_quads), axis=0)
        elements_3d[' quad gei'] = np.concatenate((elements_3d[' quad gei'],
                                                   np.full((len(quads)*2,), -1, dtype=np.int32)), axis=0)
    if ' tri ' in elements:
        triangles = elements[' tri ']
        elements_3d[' prism '] = np.concatenate((triangles, triangles + npoints), axis=1)
        elements_3d[' prism gei'] = elements[' tri gei']
        elements_3d[' tri '] = np.concatenate((triangles, triangles + npoints), axis=0)
        elements_3d[' tri gei'] = np.full((len(triangles)*2,), -1, dtype=np.int32)
    return elements_3d


def build_element_connectivity(elements):
    comsol_boundary_elements = [' tri ', ' quad ']
    comsol_domain_elements = [' pyr ', ' tet ', ' prism ', ' hex ']
    element_com = {
        # ' quad ': [[0, 1], [1, 3], [3, 2], [2, 0]],
        ' tet ': [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]],
        ' pyr ': [[0, 2, 3, 1], [0, 4, 2], [0, 1, 4], [1, 3, 4]],
        ' prism ': [[0, 3, 5, 2], [0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 5], [0, 2, 1]],
        ' hex ': [[0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7], [2, 0, 4, 6], [0, 2, 3, 1], [4, 5, 7, 6]]
    }

    boundary_elements = []
    boundary_gei = []

    for bt in elements:
        if bt in comsol_boundary_elements:
            bts = elements[bt]
            gei = elements.get(bt + 'gei', [])
            boundary_elements.extend(bts)
            boundary_gei.extend(gei)
    if len(boundary_gei) != len(boundary_elements):
        raise ValueError
        # boundary_gei = [i for i in range(len(boundary_elements))]

    faces = []
    cells = []
    cells_gei = []
    face_index = 0
    for element in elements:
        if element in comsol_domain_elements:
            sei = element_com[element]
            elm = elements[element]
            for e in elm:
                se = [[int(e[i]) for i in index] for index in sei]
                dom = []
                for s in se:
                    faces.append(s)
                    dom.append(face_index)
                    face_index += 1
                cells.append(dom)
            cells_gei.extend(elements[element + 'gei'])
    element_connectivity = {'ndim': elements['ndim'], 'points': elements['points'],
                            'cells': cells, 'cells_gei': cells_gei,
                            'faces': faces, 'ncells': len(cells),
                            'boundary_faces': boundary_elements, 'boundary_faces_gei': boundary_gei
                            }
    return element_connectivity


if __name__ == '__main__':
    from src.conversion.convert import connectivity_to_foam

    fp = 'line.mphtxt'
    elem = read_comsol_file(fp)
    conn = build_element_connectivity(elem)
    foam = connectivity_to_foam(conn)
    print(foam)
