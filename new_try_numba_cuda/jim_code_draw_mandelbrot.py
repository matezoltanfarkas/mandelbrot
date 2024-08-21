import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import numba as nb
import numba.cuda

@nb.cuda.jit(fastmath=True)
def one_pixel_numba_cuda(height, width, fractal, x_min, x_max, y_min, y_max, max_iter):
    # Get the 2D grid position of the current thread
    x, y = nb.cuda.grid(2)

    # Check if the current thread is within bounds
    if x < width and y < height:
        # Map pixel position (x, y) to the corresponding complex number in the fractal plane
        real = x_min + (x_max - x_min) * x / width   # Real part range: x_min to x_max
        imag = y_min + (y_max - y_min) * y / height  # Imaginary part range: y_min to y_max
        
        # Initialize the complex number c for this pixel
        z = c = np.complex64(real + imag * 1j)
        
        # Assume the point is inside the set until proven otherwise
        fractal[x, y] = max_iter
        
        # Iterate to determine if the point escapes
        for i in range(max_iter):
            z = z * z + c  # Mandelbrot iteration formula
            
            # Check if the magnitude of z exceeds 2 (indicating the point will escape)
            if z.real**2 + z.imag**2 > 4.0:
                fractal[x, y] = i  # Record the number of iterations taken to escape
                break

def run_numba_cuda(height, width, x_min, x_max, y_min, y_max, max_iter=50):
    fractal = cp.empty((height, width), dtype=np.int32)
    griddim = (math.ceil(height / 32), math.ceil(width / 32))
    blockdim = (32, 32)
    one_pixel_numba_cuda[griddim, blockdim](height, width, fractal, x_min, x_max, y_min, y_max, max_iter)
    return fractal.get()

def calculate_area(fractal, x_min, x_max, y_min, y_max, max_iter):
    # Calculate the area of each pixel
    pixel_area = (x_max - x_min) * (y_max - y_min) / (fractal.shape[1] * fractal.shape[0])

    # Weight the area by the iteration count
    weighted_area = np.sum(fractal==max_iter) * pixel_area
    
    return weighted_area


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
height, width = 8000, 8000

# Define the maximum number of iterations for the Mandelbrot calculation
max_iter = 100

fractal = run_numba_cuda(height, width, x_min, x_max, y_min, y_max, max_iter).T

print(fractal.shape)
print(fractal)

# Plot the Mandelbrot set with the correct center
plot_mandelbrot_correct_center(fractal, x_min, x_max, y_min, y_max)

# Calculate the area of the Mandelbrot set within the specified region
area = calculate_area(fractal, x_min, x_max, y_min, y_max, max_iter)
print(f"Estimated area of the Mandelbrot set in the region [{x_min}, {x_max}] x [{y_min}, {y_max}]: {area:.6f}")
