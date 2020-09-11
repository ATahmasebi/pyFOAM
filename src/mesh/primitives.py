from collections import namedtuple

cell = namedtuple('cell', 'center volume')
boundary_faces = namedtuple('boundaries', 'center vector owner patch')
internal_faces = namedtuple('internal', 'center vector owner neighbour')
info = namedtuple('info', 'cells unit')

patch = namedtuple('patch', 'id type values')
