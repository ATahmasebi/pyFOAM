from src.Utilities.field import Field
import numpy as np


gf = Field(list(range(5)), 'K').reshape(-1, 1, 1) + 0.5
Sf = Field(list(range(15)), 'm**2').reshape(-1, 1, 3)

print(gf)
print(Sf)

print(gf * Sf)




