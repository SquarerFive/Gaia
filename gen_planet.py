from typing import Tuple
import numba
from numba import njit, float32, cuda
from numba.core.extending import overload
from numba.experimental import jitclass
import numba.typed
import numpy as np
from numba import prange
import math
from PIL import Image
from matplotlib import pyplot as plt
from gen_planet_gpu import *
from osgeo import gdal
from osgeo import osr
import os
os.environ['PROJ_LIB'] = "C:/Users/aiden/.conda/pkgs/proj-8.0.0-h1cfcee9_0/Library/share/proj"

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
    
    def mulf(self, other: float):
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
    
    def mul_matrix(self, matrix: Tuple[float, float, 
                                        float, float]):
        return Vector2D(
            (matrix[0] * self.x) + (matrix[2] * self.y),
            (matrix[1] * self.x) + (matrix[3] * self.y) 
        )

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
def hash2d(x: Vector2D) -> Vector2D:
    return Vector2D(
        *hash2d_gpu((x.x, x.y))
    )

    k = Vector2D(0.3183099, 0.3678794)
    #x = x*k + k.yx
    x = x.mul(k).add(k.yx)
    return k.mulf(
        fract(x.x * x.y * (x.x + x.y))
    ).mulf(16.0).fract().mulf(2.0).sub(Vector2D(-1.0, -1.0))
    # return mulfv(-1.0 + 2.0,  k.mulf(16.0 * fract(x.x * x.y * (x.x + x.y))).fract()).fract()




@njit()
def mix(x: Vector2D, y: Vector2D, alpha: float) -> Vector2D:
    return Vector2D(
        x.x + alpha * (y.x - x.x),
        x.y + alpha * (y.y - x.y)
    )

@njit()
def mixf(x: float, y: float, alpha: float):
    return (1 - alpha) * x + alpha * y
@njit()
def dot(a: Vector2D, b: Vector2D) -> float:
    return (a.x * a.y) + (b.x * b.y)

@njit()
def gradient(p: Vector2D) -> float:
    return noise_gpu_gradient((p.x, p.y))
    i = p.__floor__()
    f = p.fract()

    u = Vector2D(
        f.x * f.x * (3.0 - 2.0*f.x),
        f.y * f.y * (3.0 - 2.0*f.y)
    )
    return mixf(
        mixf(
            dot(hash2d(i), f),
            dot(hash2d(i.add(Vector2D(1.0, 0.0))), f.sub(Vector2D(1.0, 0.0))),
            u.x
        ),
        mixf(
            dot(hash2d(i.add(Vector2D(0.0, 1.0))), f.sub(Vector2D(0.0, 1.0))),
            dot(hash2d(i.add(Vector2D(1.0, 1.0))), f.sub(Vector2D(1.0, 1.0))),
            u.x
        ),
        u.y
    )

@njit()
def map(uv: Vector2D) -> float:
    f = 0.5 * gradient(uv); uv = uv.mul_matrix((1.6, 1.2, -1.2, 1.6))
    f += 0.25 * gradient(uv); uv = uv.mul_matrix((1.6, 1.2, -1.2, 1.6))
    f += 0.125 * gradient(uv); uv = uv.mul_matrix((1.6, 1.2, -1.2, 1.6))
    f += 0.0625 * gradient(uv); uv = uv.mul_matrix((1.6, 1.2, -1.2, 1.6))
    return 0.5 + 0.5 * f

@njit()
def biome(uv: Vector2D) -> float:
    return mixf(
        map(
            Vector2D(
                map(uv.mulf(0.1511)),
                map(uv.mulf(0.48512))
            )),
            map(Vector2D(
                map(uv.mulf(0.0511)),
                map(uv.mulf(0.00512))
            )),
        math.pow(map(uv.mulf(0.15)),2 )
    )

@njit(parallel=True)
def map_simd(array: np.ndarray):
    scale = Vector2D(0.04, 0.045)
    for y in prange(array.shape[1]):
        for x in prange(array.shape[0]):
            uv = Vector2D(x, y).mul(scale)
            array[x][y] = biome(uv)

def get_cuda_block_threads_from_array2d(array):
    threads_per_block = (8,8) # 512 threads per block
    blocks_per_grid_x = int(math.ceil(array.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(array.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return (blocks_per_grid, threads_per_block)

@njit()
def remap(value: float, old_min: float = 0.0, old_max: float = 1.0, new_min: float = 0.5, new_max: float = 1.0):
    old_range = old_max - old_min
    new_range = new_max - new_min

    # Get relative value
    relative_value = value - old_min
    relative_alpha = relative_value / old_range

    # reproject relative value
    return (new_range*relative_alpha)+new_min

@cuda.jit()
def render(array: np.ndarray):
    scale = (0.04, 0.045)
    x,y = cuda.grid(2)
    array[x][y] = biome_gpu(mul_gpu(div_gpu((x, y), (8192, 4096)), (50, 50)))

@cuda.jit()
def colour(colourmap: np.ndarray, values: np.ndarray, min_value: float, max_value: float):
    x,y = cuda.grid(2)

    mountain = (190, 150, 25)
    land = (90, 25, 1)
    alpha: float = math.pow(remap(values[x][y], min_value, max_value, 0.0, 1.0), 5)
    alpha = max(min(alpha, 1.0), 0.0)
    c = lerp3(land, mountain, alpha)
    colourmap[0][x][y] = int(c[0])
    colourmap[1][x][y] = int(c[2])
    colourmap[2][x][y] = int(c[1])
    colourmap[3][x][y] = 255

bounds_lon = [-180, 180]
bounds_lat = [-90, 90]

width = 4096
height = 4096
print(remap(0.5, 0.0, 1.0, 1000, 2000))
print(hash2d(Vector2D(45.1414, 24.24)).x)

print(gradient(Vector2D(1.25, 0.25)))

noise_array = np.ndarray((width, height), dtype=np.float32)
colourmap = np.ndarray((4, width, height), dtype=np.uint8)

blocks_per_grid, threads_per_block = get_cuda_block_threads_from_array2d(noise_array)

render[blocks_per_grid, threads_per_block](noise_array)
colour[blocks_per_grid, threads_per_block](colourmap, noise_array, np.min(noise_array), np.max(noise_array))
# map_simd(noise_array)
print(noise_array)

plt.imshow(noise_array)
# plt.show()

# im = Image.fromarray(colourmap).save('Colourmap.png')

xmin, ymin, xmax, ymax = [min(bounds_lon), min(bounds_lat), max(bounds_lon), max(bounds_lat)]
xres = (xmax - xmin) / float(width)
yres = (ymax - ymin) / float(height)

geotransform = (xmin, xres, 0, ymax, 0, -yres)

dst_ds = gdal.GetDriverByName('GTiff').Create('planet.tif', height, width, 3, gdal.GDT_Byte)

dst_ds.SetGeoTransform(geotransform)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
dst_ds.SetProjection(srs.ExportToWkt())

dst_ds.GetRasterBand(1).WriteArray(colourmap[0])
dst_ds.GetRasterBand(2).WriteArray(colourmap[1])
dst_ds.GetRasterBand(3).WriteArray(colourmap[2])

dst_ds.FlushCache()

dst_ds = None