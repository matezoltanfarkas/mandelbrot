import numpy as np
import matplotlib.pyplot as plt

# Define the plotting range
x_min, x_max = -2.5, 1.5
y_min, y_max = -1.5, 1.5
resolution = 2000

# Create a meshgrid for the complex plane
x, y = np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Mandelbrot function to calculate set membership
def mandelbrot(c, max_iter=256):
    z = np.zeros_like(c)
    mask = np.ones(c.shape, dtype=bool)
    for n in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[np.abs(z) > 2] = False
    return mask

# Compute the Mandelbrot set
mandelbrot_set = mandelbrot(Z)

# Plot the Mandelbrot set
plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_set, extent=[x_min, x_max, y_min, y_max], cmap='inferno')

# Plot the boundary conditions
plt.axvline(x=-2.0, color='white', linestyle='--', label='Re(x) = -2.0')
plt.axvline(x=0.49, color='white', linestyle='--', label='Re(x) = 0.49')
plt.axhline(y=-1.15, color='cyan', linestyle='--', label='Im(x) = -1.15')
plt.axhline(y=1.15, color='cyan', linestyle='--', label='Im(x) = 1.15')

# Plot the circle |x| < 2
circle = plt.Circle((0, 0), 2, color='yellow', linestyle='--', fill=False, label='|x| < 2')

# Add the circle to the plot
plt.gca().add_artist(circle)

# Set plot limits and labels
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Re(x)")
plt.ylabel("Im(x)")
plt.title("Mandelbrot Set with Boundaries for Pre-checks")
plt.legend()

plt.savefig('Boundaries.pdf')