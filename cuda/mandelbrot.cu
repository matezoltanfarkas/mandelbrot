#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <cstdio>
#include <random>

using namespace std;

const int NUM_TILES_1D = 100;
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

int count_mandelbrot(const int num_samples, const double x_min, const double width, const double y_min, const double height)
{
  int out = 0;

  // Random number generator
  std::random_device rd;                                   // Seed for the random number engine
  std::mt19937 gen(rd());                                  // Mersenne Twister engine
  std::uniform_real_distribution<double> dist(0.0f, 1.0f); // Uniform distribution between 0 and 1

  for (int i = 0; i < num_samples; ++i)
  {
    double x_norm = dist(gen);
    double y_norm = dist(gen);
    double x = x_min + (x_norm * width);
    double y = y_min + (y_norm * height);
    if (is_in_mandelbrot(x, y))
    {
      out += 1;
    }
  }
  return out;
}

__global__ void hello_world_gpu()
{
  printf("Hello World from the GPU at block %u, thread %u \n", blockIdx.x, threadIdx.x);
}

void hello_world_cpu()
{
  printf("Hello World from the CPU \n");
}

double wald_uncertainty(double numer, double denom)
{
  if (numer == 0)
  {
    numer = 1.0;
    denom++;
  }
  else if (numer == denom)
    denom++;

  double frac = numer / denom;
  return sqrt(frac * (1.0 - frac) / denom);
}

void compute_until(mt19937 random_generators[], double numer[], double denom[], double uncert[], double uncert_target)
{
  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      mt19937 rng = random_generators[NUM_TILES_1D * i + j];
    }
}

int main(int argc, char *argv[])
{

  if (argc != 3)
  {
    cout << "Need two arguments: number of blocks and number of threads" << endl;
    return -1;
  }

  hello_world_cpu();

  const int n_blocks = atoi(argv[argc - 2]);
  const int n_threads = atoi(argv[argc - 1]);

  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads);

  hello_world_gpu<<<grid_dim, block_dim>>>();

  cudaDeviceSynchronize();

  return 0;
}
