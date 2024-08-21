import math
import numpy as np
import cupy as cp
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from utils import (
    plot_pixels,
    combine_uncertaintes,
    confidence_interval
)

import matplotlib.pyplot as plt


WARP_SIZE = 32
NUM_TILES_1D = 3200 # per dimension
NUM_BLOCKS_1D = (NUM_TILES_1D + WARP_SIZE - 1) // WARP_SIZE


@nb.cuda.jit(device=True)
def wald_uncertainty(numer, denom):
    if numer == 0:
        numer = 1
        denom += 1
    elif numer == denom:
        denom += 1

    frac = np.float32(numer) / np.float32(denom)

    return math.sqrt(frac * (1 - frac) / denom)


@nb.cuda.jit(device=True)
def is_in_mandelbrot(x, y):
    c = np.complex64(x) + np.complex64(y) * np.complex64(1j)
    z_hare = z_tortoise = np.complex64(0)
    while True:
        z_hare = z_hare * z_hare + c
        z_hare = z_hare * z_hare + c  # hare does one step more to get ahead of the tortoise
        z_tortoise = z_tortoise * z_tortoise + c  # tortoise is one step behind
        if z_hare == z_tortoise:
            return True  # orbiting or converging to zero
        if z_hare.real**2 + z_hare.imag**2 > 4:
            return False  # diverging to infinity


@nb.cuda.jit
def compute_until(rng_states, numer, denom, uncert, uncert_target, width, height, num_tiles):
    """Compute area of each tile until uncert_target is reached.
    The uncertainty is calculated with the Wald approximation in each tile.
    """
    
    i, j = nb.cuda.grid(2)
    new_rng_idx = NUM_TILES_1D * i + j
    num_samples = 10

    xmin_val = np.float32(-2) + width * j
    ymin_val = np.float32(-1.5) + height * i

    if i < num_tiles and j < num_tiles:
        
        uncert[i, j] = np.float32(np.inf)

        while uncert[i, j] > uncert_target:
            denom[i, j] += num_samples
            
            for _ in range(num_samples):
                x = xoroshiro128p_uniform_float32(rng_states, new_rng_idx) * width  + xmin_val
                y = xoroshiro128p_uniform_float32(rng_states, new_rng_idx) * height + ymin_val
                if is_in_mandelbrot(x, y):
                    numer[i, j] += 1

            uncert[i, j] = wald_uncertainty(numer[i, j], denom[i, j]) * width * height


rng_states = create_xoroshiro128p_states(WARP_SIZE * WARP_SIZE * NUM_BLOCKS_1D * NUM_BLOCKS_1D, seed=31415)

xmin, xmax = -2., 1.
ymin, ymax = -1.5, 1.5

width  = np.float32(3.0 / NUM_TILES_1D)
height = np.float32(3.0 / NUM_TILES_1D)
uncert_target = 1e-8

# Allocate GPU memory
# Use int instead of float when appropriate
numer  = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.int32)
denom  = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.int32)
uncert = cp.zeros((NUM_TILES_1D, NUM_TILES_1D), dtype=cp.float32)

threads_per_block = (WARP_SIZE, WARP_SIZE)
blocks_per_grid = (NUM_BLOCKS_1D, NUM_BLOCKS_1D)

# Run the CUDA kernel
compute_until[blocks_per_grid, threads_per_block](rng_states, numer, denom, uncert, uncert_target, width, height, NUM_TILES_1D)

nb.cuda.synchronize()

# Copy the results back to the host if needed
numer  = cp.asnumpy(numer)
denom  = cp.asnumpy(denom)
uncert = cp.asnumpy(uncert)

print('--------------------------------')

print('Numerator')
print(numer.shape)
print(numer)
print(np.unique(numer))

print('--------------------------------')

print('Denominator')
print(denom.shape)
print(denom)
print(np.unique(denom))

print('--------------------------------')

final_value = (np.sum((numer / denom)) * width * height).item()
print(final_value)


fig, ax, p = plot_pixels(numer / denom, dpi=80)
fig.colorbar(p, ax=ax, shrink=0.8, label="fraction of sampled points in Mandelbrot set in each tile")
plt.savefig('my_figure.pdf')


CONFIDENCE_LEVEL = 0.05

confidence_interval_low, confidence_interval_high = confidence_interval(
    CONFIDENCE_LEVEL, numer, denom, width * height
)

final_uncertainty = combine_uncertaintes(
    confidence_interval_low, confidence_interval_high, denom
)
print(final_uncertainty)
print(f"Relative Uncertainty: {final_uncertainty/final_value}")
