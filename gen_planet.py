import numba
from numba import njit, float32
from numba.core.extending import overload
from numba.experimental import jitclass
import numba.typed
import numpy as np

import math

vector_spec = [
    ('x', float32),
    ('y', float32)
]

@jitclass(vector_spec)
class Vector2D(object):
    def __init__(self, x: float32, y: float32):
        self.x = x
        self.y = y
    
    @property
    def xy(self):
        return Vector2D(self.x, self.y)
    
    @property
    def yx(self):
        return Vector2D(self.y, self.x)
    
    @property
    def xx(self):
        return Vector2D(self.x, self.x)
    
    @property
    def yy(self):
        return Vector2D(self.y, self.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        return self.x != other.x and self.y != other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __mul__(self, other):
        return Vector2D(self.x * other.x, self.y * other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __repr__(self) -> str:
        return 'X: '+str(self.x)+' Y: '+str(self.y)
    
    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    def __floor__(self):
        return Vector2D(
            math.floor(self.x), math.floor(self.y)
        )

    def __fract__(self):
        return self - self.__floor__()

    def mul(self, other):
        return Vector2D(
            self.x * other.x,
            self.y * other.y
        )
    
    def mulf(self, other):
        return Vector2D(
            self.x * other,
            self.y * other
        )
    
    def add(self, other):
        return Vector2D(
            self.x + other.x,
            self.y + other.y
        )
        
    def sub(self, other):
        return Vector2D(
            self.x - other.x,
            self.y - other.y
        )

    def fract(self):
        return self.sub(self.__floor__())
    
    

@njit()
def fract(x: float):
    return x - math.floor(x)

@njit()
def mulfv(x: float, y: Vector2D):
    return Vector2D(x * y.x, x* y.y)

"""
@overload(fract)
def fract(x: Vector2D):
    return x.fract()
"""

@njit()
def hash2d(x : Vector2D):
    k = Vector2D(0.1383099, 0.3678794)
    #x = x*k + k.yx
    x = x.mul(k).add(k.yx)
    return mulfv(-1.0 + 2.0,  k.mulf(16.0 * fract(x.x * x.y * (x.x + x.y))).fract())

print(hash2d(Vector2D(45.1414, 24.24)).x)