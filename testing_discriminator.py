import numpy as np
from time import perf_counter

from utils import (
    wald_uncertainty,
)

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

RNG = np.random.default_rng()
NUM_TILES_1D = 1
SAMPLES_IN_BATCH = 100000
CHECK_MAX_ITER = 0
CONFIDENCE_LEVEL = 0.05
WIDTH = 3 / NUM_TILES_1D
HEIGHT = 3 / NUM_TILES_1D
FAIL_COUNT = 0

def is_in_mandelbrot(x, y):
    """Toirtoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = (
            z_hare * z_hare + c
        )  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity


def is_in_mandelbrot_fast(x, y):
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z = np.complex64(0)
    trace = np.full((CHECK_MAX_ITER,), z, dtype=np.complex64)
    for i in range(CHECK_MAX_ITER):
        z = z * z + c
        if z in trace[:i]:
            return True
        if np.abs(z) > 2:
            return False
        trace[i] = z
    return is_in_mandelbrot(x, y)

def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    """
    Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height).
    Raise `out` by one if the number is part of the Mandelbrot set.
    """
    out = np.int32(0)
    for x_norm, y_norm in rng.random((num_samples, 2), np.float32):
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)
        #out += is_in_mandelbrot(x, y)
        out += is_in_mandelbrot_fast(x, y)
    return out

def compute_until(rngs, numer, denom, uncert, uncert_target):
    """
    Compute area of each tile until uncert_target is reached.
    The uncertainty is calculate with the Wald approximation in each tile.
    """
    for i in np.arange(NUM_TILES_1D):
        for j in np.arange(NUM_TILES_1D):
            rng = rngs[NUM_TILES_1D * i + j]

            uncert[i, j] = np.inf

            # Sample SAMPLES_IN_BATCH more points until uncert_target is reached
            while uncert[i, j] > uncert_target:
                denom[i, j] += SAMPLES_IN_BATCH
                numer[i, j] += count_mandelbrot(
                    rng, SAMPLES_IN_BATCH, xmin(j), WIDTH, ymin(i), HEIGHT
                )

                uncert[i, j] = (
                    wald_uncertainty(numer[i, j], denom[i, j]) * WIDTH * HEIGHT
                )

def xmin(j):
    """xmin of tile in column j"""
    return -2 + WIDTH * j

def ymin(i):
    """ymin of tile in row i"""
    return -3 / 2 + HEIGHT * i

rngs = RNG.spawn(NUM_TILES_1D * NUM_TILES_1D)
numer = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
denom = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.int64)
uncert = np.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=np.float64)

ts = perf_counter()
print(count_mandelbrot(rngs[0], SAMPLES_IN_BATCH, xmin(0), WIDTH, ymin(0), HEIGHT))
print(f"Runtime: {perf_counter() - ts} s")
print(f"Fail count at {CHECK_MAX_ITER}: {FAIL_COUNT} of {SAMPLES_IN_BATCH}")
#compute_until(rngs, numer, denom, uncert, 1e-3)
#final_value = (np.sum((numer / denom)) * WIDTH * HEIGHT).item()
#print(f"\tThe total area of all tiles is {final_value}")