from collections import namedtuple

scaler = (-1, 1, 1)
vector = (-1, 3, 1)
tensor = (-1, 3, 3)

cell = namedtuple('cell', 'center volume')
boundary_faces = namedtuple('boundaries', 'center vector owner patch')
internal_faces = namedtuple('internal', 'center vector owner neighbour')
info = namedtuple('info', 'cells unit')


def on_demand_prop(fn):
    """Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
