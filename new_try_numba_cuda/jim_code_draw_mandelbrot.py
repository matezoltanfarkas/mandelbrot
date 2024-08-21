import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import numba as nb
import numba.cuda

@nb.cuda.jit(fastmath=True)
def one_pixel_numba_cuda(height, width, fractal, x_min, x_max, y_min, y_max):
    x, y = nb.cuda.grid(2)
    real = x_min + (x_max - x_min) * x / width       # Real part range: x_min to x_max
    imag = y_min + (y_max - y_min) * y / height      # Imaginary part range: y_min to y_max
    z = c = np.complex64(real + imag * 1j)
    fractal[x, y] = 50
    for i in range(50):
        z = z * z + c
        if z.real**2 + z.imag**2 > 4:
            fractal[x, y] = i
            break

def run_numba_cuda(height, width, x_min, x_max, y_min, y_max):
    fractal = cp.empty((height, width), dtype=np.int32)
    griddim = (math.ceil(height / 32), math.ceil(width / 32))
    blockdim = (32, 32)
    one_pixel_numba_cuda[griddim, blockdim](height, width, fractal, x_min, x_max, y_min, y_max)
    return fractal.get()

def calculate_area(fractal, x_min, x_max, y_min, y_max):
    points_inside_set = np.sum(fractal == 50)
    total_area = points_inside_set * ((x_max - x_min) * (y_max - y_min)) / (width * height)
    return total_area

def plot_mandelbrot_correct_center(fractal, x_min, x_max, y_min, y_max):
    
    # Define the extent of the axes using the provided variables
    extent = [x_min, x_max, y_min, y_max]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    ax.imshow(fractal, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    ax.set_title('Mandelbrot Set')
    plt.savefig('jim_image.pdf')
    plt.show()

# Use the common boundaries
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5  # Symmetric around y = 0

# Define the resolution
height, width = 5, 5

fractal = run_numba_cuda(height, width, x_min, x_max, y_min, y_max).T

print(fractal.shape)
print(fractal)

# Plot the Mandelbrot set with the correct center
plot_mandelbrot_correct_center(fractal, x_min, x_max, y_min, y_max)

# Calculate the area of the Mandelbrot set within the specified region
area = calculate_area(fractal, x_min, x_max, y_min, y_max)
print(f"Estimated area of the Mandelbrot set in the region [{x_min}, {x_max}] x [{y_min}, {y_max}]: {area:.6f}")
