from numba import njit, float32
from typing import Tuple, Union
import math

from numba.core.typing.cffi_utils import is_ffi_instance

@njit()
def mul_gpu(x: Tuple[float, float, Union[float, None]], y: Tuple[float, float, Union[float, None]]):
    if len(x) == 2:
        return (x[0]*y[0], x[1]*y[1])
    else:
        return (
            x[0]*y[0],
            x[1]*y[1],
            x[2]*y[2]
        )

@njit()
def mulf_gpu(x: float, y:float):
    return x * y

@njit()
def div_gpu(x: Tuple[float, float, Union[float, None]], y: Tuple[float, float, Union[float, None]]):
    if len(x) == 2:
        return (x[0]/y[0], x[1]/y[1])
    else:
        return (
            x[0]/y[0],
            x[1]/y[1],
            x[2]/y[2]
        )

@njit()
def sub_gpu(x: Tuple[float, float, Union[float, None]], y: Tuple[float, float, Union[float, None]]):
    if len(x) == 2:
        return (x[0]-y[0], x[1] - y[1])
    return (x[0]-y[0], x[1]-y[1], x[2]-y[2])

@njit()
def subf_gpu(x: float, y: float):
    return x - y

@njit()
def add_gpu(x: Tuple[float, float, Union[float, None]], y: Tuple[float, float, Union[float, None]]):
    if len(x) == 2:
        return (x[0]+y[0], x[1] + y[1])
    return (x[0]+y[0], x[1]+y[1], x[2]+y[2])

@njit()
def addf_gpu(x: float32, y: float32):
    return x+y

@njit()
def floor_gpu(x: Tuple[float, float, Union[float, None]]):
    if len(x) == 2:
        return (math.floor(x[0]), math.floor(x[1]))
    elif len(x) == 3:
        return (math.floor(x[0]), math.floor(x[1]), math.floor(x[2]))

@njit()
def floorf_gpu(x: float):
    return math.floor(x)

@njit()
def fract_gpu(x: Union[Tuple[float, float, Union[float, None]], float]) -> Union[Tuple[float, float, Union[float, None]], float]:
    return sub_gpu(x, floor_gpu(x))

@njit()
def fractf_gpu(x: float) -> float:
    return subf_gpu(x, floorf_gpu(x))


@njit()
def hash2d_gpu(x: Tuple[float, float]) -> Tuple[float, float]:
    k = (0.3183099, 0.3678794)
    x = add_gpu(mul_gpu(x, k), (k[1], k[0]))
    return (
        -1.0 + 2.0 * fractf_gpu(16.0 * k[0] * fractf_gpu(x[0] * x[1] * (x[0] + x[1]))),
        -1.0 + 2.0 * fractf_gpu(16.0 * k[1] * fractf_gpu(x[0] * x[1] * (x[0] + x[1]))),
    )

@njit()
def lerp(a: float, b: float, alpha: float):
    return a * (1.0 - alpha) + b * alpha

@njit()
def lerp3(a: Tuple[float, float, float], b: Tuple[float, float, float], alpha: float):
    return (
        lerp(a[0], b[0], alpha),
        lerp(a[1], b[1], alpha),
        lerp(a[2], b[2], alpha),
    )

@njit()
def dot_gpu(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0]*b[0] + a[1]*b[1])

@njit()
def mul_matrix_gpu(a: Tuple[float, float], m: Tuple[float, float, float, float]):
    return (
        (m[0] * a[0]) + (m[2] * a[1]),
        (m[1] * a[0]) + (m[3] * a[1])
    )

@njit()
def noise_gpu_gradient(p: Tuple[float, float]):
    i = floor_gpu(p)
    f = fract_gpu(p)

    u = (
        f[0]*f[0]*(3.0 - 2.0*f[0]),
        f[1]*f[1]*(3.0 - 2.0*f[1]),
    )

    return lerp(
        lerp(
            dot_gpu(hash2d_gpu(i), f),
            dot_gpu(hash2d_gpu((i[0]+1.0, i[1])), (f[0]-1.0, f[1])),
            u[0]
        ),
        lerp(
            dot_gpu(hash2d_gpu((i[0]+0.0, i[1]+1.0)), sub_gpu(f, (0.0, 1.0))),
            dot_gpu(hash2d_gpu((i[0]+1.0, i[1]+1.0)), sub_gpu(f, (1.0, 1.0))),
            u[0]
        ),
        u[1]
    )

@njit()
def biome_gpu(uv: Tuple[float, float]) -> float:
    sc0 = 0.1511 * 0.5
    sc1 = 0.48512 * 0.5125
    sc2 = 0.0511 * 0.245
    sc3 = 0.00512 * 0.5

    alpha_sc = 0.0015 * 1

    return lerp(
        fbm_gpu(
            (
                fbm_gpu(mul_gpu(uv, (sc0, sc0))),
                fbm_gpu(mul_gpu(uv, (sc1, sc1)))
            )),
            fbm_gpu((
                fbm_gpu(mul_gpu(uv, (sc2, sc2))),
                fbm_gpu(mul_gpu(uv, (sc3, sc3)))
            )),
        math.pow(fbm_gpu(mul_gpu(uv, (alpha_sc, alpha_sc))),2)
    )

@njit()
def fbm_gpu(p: Tuple[float, float]):
    f = 0.5 * noise_gpu_gradient(p); p = mul_matrix_gpu(p, (1.6, 1.2, -1.2, 1.6))
    f += 0.25 * noise_gpu_gradient(p); p = mul_matrix_gpu(p, (1.6, 1.2, -1.2, 1.6))
    f += 0.125 * noise_gpu_gradient(p); p = mul_matrix_gpu(p, (1.6, 1.2, -1.2, 1.6))
    f += 0.0625 * noise_gpu_gradient(p); p = mul_matrix_gpu(p, (1.6, 1.2, -1.2, 1.6))
    return 0.5 + 0.5 * f

if __name__ == "__main__":
    result = hash2d_gpu((4.545, 32.5))
    print(result)
    print(noise_gpu_gradient((25.2551, 12515.15)))
    print(fbm_gpu((25.2551, 12515.15)))