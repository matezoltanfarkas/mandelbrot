import cupy as cp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from numba import cuda

from utils import plot_pixels

NUM_TILES_1D = 100
SAMPLES_IN_BATCH = 100  # Define the constant

@cuda.jit(device=True)
def wald_uncertainty(numer, denom):
    """Wald approximation on the uncertainty of the tile."""
    if numer == 0:
        numer = 1
        denom += 1
    elif numer == denom:
        denom += 1

    frac = numer / denom

    return frac * (1 - frac) / denom

@cuda.jit(device=True)
def is_in_mandelbrot(x, y):
    """Tortoise and Hare approach to check if point (x,y) is in Mandelbrot set."""
    c = x + y * 1j
    z_hare = z_tortoise = 0 + 0j  # tortoise and hare start at same point
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity

@cuda.jit(device=True)
def count_mandelbrot(rng, num_samples, xmin, width, ymin, height):
    """Draw num_samples random numbers uniformly between (xmin, xmin+width)
    and (ymin, ymin+height)."""
    count = 0
    for idx in range(num_samples):
        x_norm = rng[idx, 0]
        y_norm = rng[idx, 1]
        x = xmin + (x_norm * width)
        y = ymin + (y_norm * height)
        count += is_in_mandelbrot(x, y)
    return count

@cuda.jit
def compute_until(rngs, numer, denom, uncert, uncert_target, width, height, num_samples, num_tiles):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculated with the Wald approximation in each tile.
    """
    i, j = cuda.grid(2)
    if i < num_tiles and j < num_tiles:
        rng = rngs[num_tiles * i + j]
        uncert[i, j] = float('inf')

        xmin_val = -2 + width * j
        ymin_val = -3 / 2 + height * i

        while uncert[i, j] > uncert_target:
            denom[i, j] += num_samples
            count = count_mandelbrot(rng, num_samples, xmin_val, width, ymin_val, height)
            numer[i, j] += count

            uncert[i, j] = wald_uncertainty(numer[i, j], denom[i, j]) * width * height

# CuPy RNG setup and conversion to CuPy arrays
rngs = cp.random.random((NUM_TILES_1D * NUM_TILES_1D, SAMPLES_IN_BATCH, 2), dtype=cp.float32)

xmin, xmax = -2, 1
ymin, ymax = -3 / 2, 3 / 2

width = 3.0 / NUM_TILES_1D
height = 3.0 / NUM_TILES_1D
uncert_target = 1e-5

# Allocate GPU memory
numer = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)
denom = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)
uncert = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)

# Define block and grid dimensions
threads_per_block = (16, 16)
blocks_per_grid = ((NUM_TILES_1D + threads_per_block[0] - 1) // threads_per_block[0],
                   (NUM_TILES_1D + threads_per_block[1] - 1) // threads_per_block[1])

# Run the CUDA kernel
compute_until[blocks_per_grid, threads_per_block](rngs, numer, denom, uncert, uncert_target, width, height, SAMPLES_IN_BATCH, NUM_TILES_1D)

# Copy the results back to the host if needed
numer_host = cp.asnumpy(numer)
denom_host = cp.asnumpy(denom)
uncert_host = cp.asnumpy(uncert)

final_value = (np.sum((numer_host / denom_host)) * width * height).item()
print(final_value)


fig, ax, p = plot_pixels(numer_host / denom_host, dpi=80)
fig.colorbar(p, ax=ax, shrink=0.8, label="fraction of sampled points in Mandelbrot set in each tile")
plt.savefig('my_figure.pdf')