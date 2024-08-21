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
const int SAMPLES_IN_BATCH = 100;
const double width = 3. / NUM_TILES_1D;
const double height = 3. / NUM_TILES_1D;

// Precision for double comparison in AreSame
const double EPSILON = 1e-9;

// ============ TOOLS ================
__global__ void hello_world_gpu()
{
  printf("Hello World from the GPU at block %u, thread %u \n", blockIdx.x, threadIdx.x);
}

bool AreSame(double a, double b)
{
  return fabs(a - b) < EPSILON;
}

double xmin(int j)
{
  return -2 + width * j;
}

double ymin(int i)
{
  return -3 / 2 + height * i;
}

double wald_uncertainty(double numer, double denom)
{
  if (AreSame(numer, 0.))
  {
    numer = 1.0;
    denom++;
  }
  else if (AreSame(numer, denom))
  {
    denom++;
  }

  double frac = numer / denom;
  return sqrt(frac * (1.0 - frac) / denom);
}

// ============ MANDELBROT STUFF ================

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

double count_mandelbrot(mt19937 rng, int num_samples, double x_min, double width, double y_min, double height)
{
  double out = 0.;

  // Random number generator distribution
  std::uniform_real_distribution<double> dist(0.0f, 1.0f); // Uniform distribution between 0 and 1

  for (int i = 0; i < num_samples; ++i)
  {
    double x_norm = dist(rng);
    double y_norm = dist(rng);
    double x = x_min + (x_norm * width);
    double y = y_min + (y_norm * height);

    if (is_in_mandelbrot(x, y))
    {
      out += 1.;
    }
  }
  return out;
}

void compute_until(vector<mt19937> &random_generators, vector<vector<double>> &numer, vector<vector<double>> &denom, vector<vector<double>> &uncert, double uncert_target)
{
  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      mt19937 rng = random_generators[NUM_TILES_1D * i + j];
      uncert[i][j] = numeric_limits<double>::infinity();
      while (uncert[i][j] > uncert_target)
      {
        denom[i][j] += (double)SAMPLES_IN_BATCH;
        numer[i][j] += (double)count_mandelbrot(rng, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height);
        uncert[i][j] = wald_uncertainty(numer[i][j], denom[i][j]) * width * height;
      }
    }
}

int main(int argc, char *argv[])
{
  // if (argc != 3)
  // {
  //   cout << "Need two arguments: number of blocks and number of threads" << endl;
  //   return -1;
  // }

  // Initialization
  vector<vector<double>> numer(NUM_TILES_1D, vector<double>(NUM_TILES_1D));
  vector<vector<double>> denom(NUM_TILES_1D, vector<double>(NUM_TILES_1D));
  vector<vector<double>> uncert(NUM_TILES_1D, vector<double>(NUM_TILES_1D));

  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      numer[i][j] = 0.;
      denom[i][j] = 0.;
      uncert[i][j] = 0.;
    }

  // random number generators & seeding
  vector<mt19937> rngs(NUM_TILES_1D * NUM_TILES_1D);
  for (int i = 0; i < NUM_TILES_1D * NUM_TILES_1D; i++)
    rngs[i].seed(i);

  // DO IT!
  compute_until(rngs, numer, denom, uncert, 1e-2);

  // getting the result
  double final_value = 0;
  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      final_value += (numer[i][j] / denom[i][j]) * width * height;
    }
  printf("final value %lf", final_value);

  // const int n_blocks = 1;  // atoi(argv[argc - 2]);
  // const int n_threads = 1; // atoi(argv[argc - 1]);

  // dim3 grid_dim(n_blocks);
  // dim3 block_dim(n_threads);

  // hello_world_gpu<<<grid_dim, block_dim>>>();

  // cudaDeviceSynchronize();

  return 0;
}
