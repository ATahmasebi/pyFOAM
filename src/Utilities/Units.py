from collections import namedtuple, defaultdict
from math import pi

BaseUnit = namedtuple('BaseUnit', 'symbol property factor')
Mass, Length, Time, Temperature, Quantity, Current, Luminous_intensity = 'M L T Θ N I J'.split()


class Unit:
    __known_units = {}

    def __new__(cls, baseunits):
        if isinstance(baseunits, str):
            return Unit.parse(baseunits)
        else:
            instance = object.__new__(cls)
            instance.baseunits = defaultdict(lambda: 0)
            return instance

    def __init__(self, baseunit):
        if isinstance(baseunit, BaseUnit):
            self.symbol = baseunit.symbol
            self.baseunits[baseunit] = 1
            self.__known_units[baseunit.symbol] = self

    @staticmethod
    def parse(input_string):
        if input_string in Unit.__known_units:
            return Unit.__known_units[input_string]
        ans = Unit(None)
        if '/' in input_string:
            numerator, denominator = input_string.split('/')
            for u in denominator.split('.'):
                if '**' in u:
                    u, p = u.split('**')
                    ans /= (Unit.__known_units[u]) ** int(p)
                else:
                    ans /= Unit.__known_units[u]
        else:
            numerator = input_string
        for u in numerator.split('.'):
            if '**' in u:
                u, p = u.split('**')
                ans *= (Unit.__known_units[u]) ** int(p)
            else:
                ans *= Unit.__known_units[u]
        key = str(ans)
        if key in Unit.__known_units:
            return Unit.__known_units[key]
        else:
            return ans

    def set_symbol(self, symbol):
        self.symbol = symbol
        self.__known_units[symbol] = self
        self.__known_units[str(self)] = self

    @staticmethod
    def from_unit(unit, symbol, factor):
        properties = sorted([f'{k}:{v}' for k, v in unit.properties().items()])
        baseunit = BaseUnit(symbol, '.'.join(properties), factor)
        return Unit(baseunit)

    def __mul__(self, other):
        ans = Unit(None)
        ans.baseunits.update(self.baseunits)
        if isinstance(other, Unit):
            for k, v in other.baseunits.items():
                ans.baseunits[k] += v
                if ans.baseunits[k] == 0:
                    del ans.baseunits[k]
            return ans
        else:
            return NotImplemented

    def __truediv__(self, other):
        ans = Unit(None)
        ans.baseunits.update(self.baseunits)
        if isinstance(other, Unit):
            for k, v in other.baseunits.items():
                ans.baseunits[k] -= v
                if ans.baseunits[k] == 0:
                    del ans.baseunits[k]
            return ans
        else:
            return NotImplemented

    def __str__(self):
        numerator = []
        denominator = []
        for k, v in self.baseunits.items():
            if v > 0:
                numerator.append(k.symbol if v == 1 else f'{k.symbol}**{v}')
            else:
                denominator.append(k.symbol if v == -1 else f'{k.symbol}**{-v}')
        if len(denominator) == 0:
            numerator.sort()
            ans = '.'.join(numerator)
        else:
            numerator.sort()
            denominator.sort()
            ans = ('.'.join(numerator) if len(numerator) != 0 else '1') + '/' + '.'.join(denominator)
        if ans in self.__known_units:
            ans = self.__known_units[ans].symbol
        return ans

    def __repr__(self):
        return f'Unit({str(self)})'

    def __pow__(self, p):
        ans = Unit(None)
        ans.baseunits.update({k: int(v * p) for k, v in self.baseunits.items()})
        return ans

    def properties(self):
        props = defaultdict(lambda: 0)
        for k, v in self.baseunits.items():
            for p in k.property.split('.'):
                if ':' in p:
                    sp, ex = p.split(':')
                    props[sp] += v * int(ex)
                else:
                    props[p] += v
        return props

    def iscompatible(self, other):
        self_properties = sorted([f'{k}:{v}' for k, v in self.properties().items()])
        other_properties = sorted([f'{k}:{v}' for k, v in other.properties().items()])
        return '.'.join(self_properties) == '.'.join(other_properties)

    def conversion_ratio(self, other):
        if isinstance(other, str):
            other = Unit.parse(other)
        if self.iscompatible(other):
            ratio = 1
            for k, v in other.baseunits.items():
                ratio /= pow(k.factor, v)
            for k, v in self.baseunits.items():
                ratio *= pow(k.factor, v)
            return ratio
        else:
            raise ValueError(f'Cannot convert unit of dimension \'{other}\' to \'{self}\'')


# Length #
Meter = Unit(BaseUnit('m', Length, 1))
Decimeter = Unit(BaseUnit('dm', Length, 1e-1))
Centimeter = Unit(BaseUnit('cm', Length, 1e-2))
Millimeter = Unit(BaseUnit('mm', Length, 1e-3))
Micrometer = Unit(BaseUnit('um', Length, 1e-6))
Nanometer = Unit(BaseUnit('nm', Length, 1e-9))
Kilometer = Unit(BaseUnit('km', Length, 1e3))
Inch = Unit(BaseUnit('in', Length, 0.0254))
Foot = Unit(BaseUnit('ft', Length, 0.3048))
Yard = Unit(BaseUnit('yd', Length, 0.9144))
Mile = Unit(BaseUnit('mi', Length, 1609.344))
Nautical_Mile = Unit(BaseUnit('nmi', Length, 1852))

# Mass #
Gram = Unit(BaseUnit('gr', Mass, 1e-3))
Milligram = Unit(BaseUnit('mg', Mass, 1e-6))
Kilogram = Unit(BaseUnit('kg', Mass, 1))
Tonne = Unit(BaseUnit('t', Mass, 1e3))
Pound = Unit(BaseUnit('Ib', Mass, 0.453592))
Ounce = Unit(BaseUnit('oz', Mass, 0.0283495))

# Time #
Millisecond = Unit(BaseUnit('ms', Time, 1e-3))
Second = Unit(BaseUnit('s', Time, 1))
Minute = Unit(BaseUnit('min', Time, 60))
Hour = Unit(BaseUnit('h', Time, 3600))

# Temperature #
Kelvin = Unit(BaseUnit('K', Temperature, 1))
Rankine = Unit(BaseUnit('R', Temperature, 5 / 9))

# Quantity #
Mole = Unit(BaseUnit('mol', Quantity, 1))

# Current #
Ampere = Unit(BaseUnit('A', Current, 1))
Milliampere = Unit(BaseUnit('mA', Current, 1e-3))

# Luminous intensity #
Candela = Unit(BaseUnit('Cd', Luminous_intensity, 1))

# Angle #
Radian = Unit(BaseUnit('rad', 'Angle', 1))
Degree = Unit(BaseUnit('deg', 'Angle', pi / 180))

# Complex units with special symbol #
Liter = Unit.from_unit(Meter**3, 'L', 1e-3)

Hertz = Second ** -1
Hertz.set_symbol('Hz')

Newton = Kilogram * Meter / Second ** 2
Newton.set_symbol('N')

Pascal = Newton / Meter ** 2
Pascal.set_symbol('Pa')

Joule = Newton * Meter
Joule.set_symbol('J')

Watt = Joule / Second
Watt.set_symbol('W')

Coulomb = Ampere * Second
Coulomb.set_symbol('C')

Volt = Watt / Ampere
Volt.set_symbol('V')

Farad = Coulomb / Volt
Farad.set_symbol('F')

Ohm = Volt / Ampere
Ohm.set_symbol('Ω')

Weber = Volt * Second
Weber.set_symbol('Wb')

Tesla = Weber / Meter ** 2
Tesla.set_symbol('T')

Henry = Weber / Ampere
Henry.set_symbol('H')

Poise = Gram / (Centimeter * Second)
Poise.set_symbol('P')

Stokes = Centimeter ** 2 / Second
Stokes.set_symbol('St')

Dyne = Gram * Centimeter / Second ** 2
Dyne.set_symbol('dyn')

Bar = Gram / Centimeter / Millisecond ** 2
Bar.set_symbol('Bar')

# Some commonly used combination of complex units #
PascalSecond = Pascal * Second
PascalSecond.set_symbol('Pa.s')


class UnitSet:
    def __init__(self, length, mass, time):
        self.Length = length
        self.Mass = mass
        self.Time = time
        self.Area = self.Length ** 2
        self.Volume = self.Length ** 3
        self.Velocity = self.Length / self.Time
        self.Acceleration = self.Length / self.Time ** 2
        self.Density = self.Mass / self.Volume
        self.Force = self.Mass * self.Acceleration
        self.Energy = self.Force * self.Length
        self.Power = self.Energy / self.Time
        self.Pressure = self.Force / self.Area
        self.DynamicViscosity = self.Pressure * self.Time
        self.KinematicViscosity = self.Area / self.Time


MKS = UnitSet(Meter, Kilogram, Second)
CGS = UnitSet(Centimeter, Gram, Second)
FPS = UnitSet(Foot, Pound, Second)

# TODO: Work on FPS specific names and unit, add more known units : atm, ....
