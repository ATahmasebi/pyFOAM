def connectivity_to_foam(elements):
    cells = elements['cells']
    cells_gei = elements['cells_gei']
    face_connectivity = elements['faces']
    boundary_faces = elements['boundary_faces']
    boundary_faces_gei = elements['boundary_faces_gei']

    boundary_patch_array = {}
    face_index_array = {}
    face_index = 0
    cell_index = 0

    all_faces = []
    boundaries = []
    internal_faces = []

    for gei, b in zip(boundary_faces_gei, boundary_faces):
        k = tuple(sorted(b))
        boundary_patch_array[k] = gei

    for cell in cells:
        for face in cell:
            face_points = face_connectivity[face]
            k = tuple(sorted(face_points))
            if k in boundary_patch_array:
                all_faces.append(face_points)
                boundaries.append([face_index, cell_index, boundary_patch_array[k]])
                face_index += 1
            elif k in face_index_array:
                fi, ow = face_index_array.pop(k)
                internal_faces.append([fi, ow, cell_index])
            else:
                all_faces.append(face_points)
                face_index_array[k] = face_index, cell_index
                face_index += 1
        cell_index += 1

    foam_elements = {'ndim': elements['ndim'], 'points': elements['points'], 'cells_gei': cells_gei,
                     'faces': all_faces, 'internal': internal_faces,
                     'boundaries': boundaries, 'ncells': elements['ncells']
                     }
    return foam_elements
