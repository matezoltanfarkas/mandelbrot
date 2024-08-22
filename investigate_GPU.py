from numba import cuda

gpu = cuda.get_current_device()
print("Maximum Threads per Block:", gpu.MAX_THREADS_PER_BLOCK)
print("Maximum Blocks per Grid (X, Y, Z):", gpu.MAX_GRID_DIM_X, gpu.MAX_GRID_DIM_Y, gpu.MAX_GRID_DIM_Z)
print("Multiprocessor Count:", gpu.MULTIPROCESSOR_COUNT)
print("Shared Memory per Block:", gpu.MAX_SHARED_MEMORY_PER_BLOCK)
