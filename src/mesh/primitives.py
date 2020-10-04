from collections import namedtuple

cell = namedtuple('cell', 'center volume')
boundary_faces = namedtuple('boundaries', 'center vector owner patch')
internal_faces = namedtuple('internal', 'center vector owner neighbour')
info = namedtuple('info', 'cells unit')

patch = namedtuple('patch', 'name type func')


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
