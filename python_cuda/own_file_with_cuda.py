import cupy as cp
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np

from utils import (
    plot_pixels,
    combine_uncertaintes,
    confidence_interval
)

import matplotlib.pyplot as plt

NUM_TILES_1D = 64

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


@cuda.jit
def compute_until(rng_states, numer, denom, uncert, uncert_target, width, height, num_tiles):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculated with the Wald approximation in each tile.
    """
    
    i, j = cuda.grid(2)
    num_samples = 100

    if i < num_tiles and j < num_tiles:
        
        uncert[i, j] = float('inf')

        xmin_val = -2 + width * j
        ymin_val = -3 / 2 + height * i

        while uncert[i, j] > uncert_target:
            denom[i, j] += num_samples
            
            for _ in range(num_samples):
                x_norm = xoroshiro128p_uniform_float32(rng_states, i+j)
                y_norm = xoroshiro128p_uniform_float32(rng_states, i+j)
                x = xmin_val + (x_norm * width)
                y = ymin_val + (y_norm * height)
                if is_in_mandelbrot(x, y):
                    numer[i, j] += 1

            uncert[i, j] = wald_uncertainty(numer[i, j], denom[i, j]) * width * height


rng_states = create_xoroshiro128p_states(32*32 * 1, seed=11)

xmin, xmax = -2, 1
ymin, ymax = -3 / 2, 3 / 2

width = 3.0 / NUM_TILES_1D
height = 3.0 / NUM_TILES_1D
uncert_target = 1e-4

# Allocate GPU memory
numer = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)
denom = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)
uncert = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)

# Define block and grid dimensions
threads_per_block = (32, 32)
# Calculate how many blocks per grid are needed to cover
blocks_per_grid_x = (NUM_TILES_1D + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (NUM_TILES_1D + threads_per_block[1] - 1) // threads_per_block[1]

blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print(blocks_per_grid)

# Run the CUDA kernel
compute_until[blocks_per_grid, threads_per_block](rng_states, numer, denom, uncert, uncert_target, width, height, NUM_TILES_1D)

nb.cuda.synchronize()

# Copy the results back to the host if needed
numer_host = cp.asnumpy(numer)
denom_host = cp.asnumpy(denom)
uncert_host = cp.asnumpy(uncert)

print(np.unique(denom_host))

final_value = (np.sum((numer_host / denom_host)) * width * height).item()
print(final_value)


fig, ax, p = plot_pixels(numer_host / denom_host, dpi=80)
fig.colorbar(p, ax=ax, shrink=0.8, label="fraction of sampled points in Mandelbrot set in each tile")
plt.savefig('my_figure.pdf')


CONFIDENCE_LEVEL = 0.05

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer_host, denom_host, width * height
)

final_uncertainty = combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denom_host
)
print(final_uncertainty)
print(f"Relative Uncertainty: {final_uncertainty/final_value}")
