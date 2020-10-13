import numpy as np
import src.Utilities.Units as Units


class Field(np.ndarray):
    def __new__(cls, array, unit):
        arr = np.asarray(array, dtype=np.float).view(cls)
        arr.unit = unit if isinstance(unit, Units.Unit) else Units.Unit.parse(unit)
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.unit = getattr(obj, 'unit', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        operation = ufunc.__name__
        out_unit = self.unit
        args = [i.view(np.ndarray) if isinstance(i, Field) else i for i in inputs]
        if len(inputs) > 1 and isinstance(inputs[0], Field) and isinstance(inputs[-1], Field):
            if operation in ['multiply', 'matmul']:
                out_unit *= inputs[1].unit
            elif operation in ['add', 'subtract'] and method != 'reduce':
                cr = inputs[-1].unit.conversion_ratio(out_unit)
                if cr != 1.0:
                    args[1] *= cr
            elif operation in ['divide', 'true_divide', 'floor_divide']:
                out_unit /= inputs[1].unit
        else:
            if operation == 'power':
                out_unit = out_unit ** inputs[-1]
            elif operation == 'square':
                out_unit = out_unit ** 2
            elif operation == 'sqrt':
                out_unit = out_unit ** 0.5

        return Field(super(Field, self).__array_ufunc__(ufunc, method, *args, **kwargs), out_unit)

    def __setitem__(self, key, value):
        if isinstance(value, Field):
            if value.unit != self.unit:
                value = value.convert(self.unit)
        return super(Field, self).__setitem__(key, value)
    
    def __repr__(self):
        arr_str = super().__repr__()
        return f'{arr_str[:-1]}, \'{str(self.unit)}\'{arr_str[-1]}'

    def __str__(self):
        return super().__str__() + f' [{str(self.unit)}]'

    def convert(self, unit):
        cr = self.unit.conversion_ratio(unit)
        return Field(cr * self, unit)


if __name__ == '__main__':
    pass
