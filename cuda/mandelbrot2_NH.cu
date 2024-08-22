#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <cstdio>
#include <random>

using namespace std;

constexpr int NUM_TILES_1D = 64;
constexpr int NUM_TILES_2D = NUM_TILES_1D * NUM_TILES_1D;
constexpr int SAMPLES_IN_BATCH = 64;
constexpr float MIN_X = -2.0;
constexpr float MAX_X = 1.0;
constexpr float MIN_Y = -1.5;
constexpr float MAX_Y = 1.5;

bool is_in_mandelbrot(const double x, const double y)
{
  // Tortoise and Hare approach to check if point (x, y) is in Mandelbrot set.
  std::complex<double> z_hare(0.0, 0.0);
  std::complex<double> z_tortoise(0.0, 0.0);
  std::complex<double> c(x, y);
  while (true)
  {
    z_hare = z_hare * z_hare + c;
    z_hare = z_hare * z_hare + c;
    z_tortoise = z_tortoise * z_tortoise + c;
    if (z_hare == z_tortoise)
    {
      return true;
    }
    float criteria = std::pow(z_hare.real(), 2) + std::pow(z_hare.imag(), 2);
    if (criteria > 4.0)
    {
      return false;
    }
  }
}

__global__ void count_mandelbrot(const double *x, const double *y, const int tile_idx, int *ct) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = start; j < SAMPLES_IN_BATCH; j+=stride) {
        ct[tile_idx] += is_in_mandelbrot(x[tile_idx][j], y[tile_idx][j])
    }
}

int main(int argc, char *argv[]) {
    if (argc != 1) {
        std::cout << "Needs argument: Target uncertainty\n";
        return -1;
    }

    dim3 grid_dim(NUM_TILES_2D);
    dim3 block_dim(SAMPLES_IN_BATCH);

    const double uncert_limit = atoi(argv[argc - 1]);
    const double tile_width = (MAX_X - MIN_X) / NUM_TILES_1D;
    const double tile_height = (MAX_Y - MIN_Y) / NUM_TILES_1D;
    
    int *denomi = new int[NUM_TILES_2D];
    int *enumer_h = new int[NUM_TILES_2D];
    int *enumer_d = new int[NUM_TILES_2D];
    double *uncert = new double[NUM_TILES_2D];
    mt19937 *rngs = new mt19937[NUM_TILES_2D];
    
    double *x_h[NUM_TILES_2D];
    double *y_h[NUM_TILES_2D];
    double *x_d[NUM_TILES_2D];
    double *y_d[NUM_TILES_2D];

    cudaMalloc( (void**)&enumer_d, NUM_TILES_2D * sizeof(int) );

    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    for (int i = 0; i < NUM_TILES_1D; i++) {
        for (int j = 0; j < NUM_TILES_1D; j++) {
            int tile_idx = i * NUM_TILES_1D + j;
            denomi[tile_idx] = SAMPLES_IN_BATCH;
            enumer_h[tile_idx] = 0;
            uncert[tile_idx] = numeric_limits<double>::infinity();
            rngs[tile_idx].seed(tile_idx);

            x_h[tile_idx] = new double[SAMPLES_IN_BATCH];
            y_h[tile_idx] = new double[SAMPLES_IN_BATCH];
            cudaMalloc( (void**)&x_d[tile_idx], SAMPLES_IN_BATCH * sizeof(double) );
            cudaMalloc( (void**)&y_d[tile_idx], SAMPLES_IN_BATCH * sizeof(double) );
            
            for (int k = 0; k < SAMPLES_IN_BATCH; ++k) {
                x_h[tile_idx][k] = MIN_X + width * (dist(rngs[tile_idx]) + i); //TYPE PROBLEM?
                y_h[tile_idx][k] = MIN_Y + height * (dist(rngs[tile_idx]) + j);
            }
            cudaMemcpy(x_d[tile_idx], x_h[tile_idx], SAMPLES_IN_BATCH * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(y_d[tile_idx], y_h[tile_idx], SAMPLES_IN_BATCH * sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    cudaMemcpy(enumer_d, enumer_h, NUM_TILES_2D * sizeof(int), cudaMemcpyHostToDevice);

    //WIP
    bool workingFlag = true;
    while (workingFlag) {
        workingFlag = false;
        for (int i = 0; i < NUM_TILES_1D; i++) {
            for (int j = 0; j < NUM_TILES_1D; j++) {
                int tile_idx = i * NUM_TILES_1D + j;
                if uncert[tile_idx] > uncert_limit {
                    workingFlag = true;
                    count_mandelbrot<<<gridDim, blockDim>>>(x_d, y_d, tile_idx, &enumer_d)
                }
            }
        }
        // Generate new random samplings on CPU while GPU working
        for (int i = 0; i < NUM_TILES_1D; i++) {
            for (int j = 0; j < NUM_TILES_1D; j++) {
                int tile_idx = i * NUM_TILES_1D + j;
                for (int k = 0; k < SAMPLES_IN_BATCH; ++k) {
                    x_h[tile_idx][k] = MIN_X + width * (dist(rngs[tile_idx]) + i); //TYPE PROBLEM?
                    y_h[tile_idx][k] = MIN_Y + height * (dist(rngs[tile_idx]) + j);
                }
            }
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < NUM_TILES_1D; i++) {
            for (int j = 0; j < NUM_TILES_1D; j++) {
                int tile_idx = i * NUM_TILES_1D + j;
                if uncert[tile_idx] > uncert_limit {
                }
            }
        }
    }

    cudaFree(enumer_d);
    for (int i = 0; i < NUM_TILES_1D; i++) {
        for (int j = 0; j < NUM_TILES_1D; j++) {
            int tile_idx = i * NUM_TILES_1D + j;
            delete[] x_h[tile_idx];
            delete[] y_h[tile_idx];
            cudaFree(x_d[tile_idx]);
            cudaFree(y_d[tile_idx]);
        }
    }
    return 0;   
}