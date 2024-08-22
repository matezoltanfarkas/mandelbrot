#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <cstdio>
#include <random>
#include <curand_kernel.h>
#include <cuComplex.h>

using namespace std;

const int NUM_TILES_1D = 10;
const int SAMPLES_IN_BATCH = 10000;
const float width = 3. / NUM_TILES_1D;
const float height = 3. / NUM_TILES_1D;

// Precision for float comparison i9n AreSame
const float EPSILON = 1e-9;

// ============ TOOLS ================
bool AreSame(float a, float b)
{
  return fabs(a - b) < EPSILON;
}

float xmin(int j)
{
  return -2 + width * j;
}

float ymin(int i)
{
  return -3 / 2 + height * i;
}

float wald_uncertainty(float numer, float denom)
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

  float frac = numer / denom;
  return sqrt(frac * (1.0 - frac) / denom);
}

// ============ MANDELBROT STUFF ================

bool is_in_mandelbrot(const float x, const float y)
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

float count_mandelbrot(mt19937 rng, int num_samples, float x_min, float width, float y_min, float height)
{
  float out = 0.;

  // Random number generator distribution
  std::uniform_real_distribution<double> dist(0.0f, 1.0f); // Uniform distribution between 0 and 1

  for (int i = 0; i < num_samples; ++i)
  {
    float x_norm = dist(rng);
    float y_norm = dist(rng);
    float x = x_min + (x_norm * width);
    float y = y_min + (y_norm * height);

    if (is_in_mandelbrot(x, y))
    {
      out += 1.;
    }
  }
  return out;
}

// void compute_until(mt19937 *random_generators, float *numer, float *denom, float *uncert, float uncert_target, curandState *d_state)
__global__ void compute_until(float *numer, float *denom, float *uncert, float uncert_target, curandState *d_state)
{
  // TODO this!
  // int blockI = blockIdx.x;
  // int blockJ = blockIdx.y;
  // int i = threadIdx.x;
  // int j = threadIdx.y;
  int i = threadIdx.x;
  int j = blockIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // vector<mt19937> &random_generators;
  // for (int i = 0; i < NUM_TILES_1D; i++)
  //   for (int j = 0; j < NUM_TILES_1D; j++)
  //   {
  // mt19937 rng = random_generators[NUM_TILES_1D * i + j];
  uncert[index] = INFINITY;
  while (uncert[index] > uncert_target)
  {
    denom[index] += (float)SAMPLES_IN_BATCH;
    // begin copied code ---------------------=======================
    float out = 0.;
    for (int i2 = 0; i2 < SAMPLES_IN_BATCH; ++i2)
    {
      float x_norm = curand_uniform_double(d_state);
      float y_norm = curand_uniform_double(d_state);
      // printf("norms %lf %lf", x_norm, y_norm);
      float x = -2 + width * j + (x_norm * width);
      float y = -3 / 2 + height * i + (y_norm * height);

      // copied code begin ------------------------------
      bool in_mandel;
      // Tortoise and Hare approach to check if point (x, y) is in Mandelbrot set.
      cuDoubleComplex z_hare = make_cuDoubleComplex(0, 0);
      cuDoubleComplex z_tortoise = make_cuDoubleComplex(0, 0);
      cuDoubleComplex c = make_cuDoubleComplex(x, y);
      while (true)
      {
        z_hare = cuCadd(cuCmul(z_hare, z_hare), c);
        z_hare = cuCadd(cuCmul(z_hare, z_hare), c);
        z_tortoise = cuCadd(cuCmul(z_tortoise, z_tortoise), c);
        if (z_hare.x == z_tortoise.x && z_hare.y == z_tortoise.y)
        {
          in_mandel = true;
          break;
        }
        float criteria = std::pow(z_hare.x, 2) + std::pow(z_hare.y, 2);
        if (criteria > 4.0)
        {
          in_mandel = false;
          break;
        }
      }
      if (in_mandel)
        out += 1.;
      // copied code end -------------------------
      // if (is_in_mandelbrot(x, y))
      // {
      //   out += 1.;
      // }
    }
    numer[index] += out;
    // end copied code----------------===================

    // numer[i * NUM_TILES_1D + j] += (float)count_mandelbrot(rng, SAMPLES_IN_BATCH, xmin(j), width, ymin(i), height);
    // COPIED CODE BEGIN ===============================
    float n = numer[index];
    float d = denom[index];
    auto CheckSame = [](float a, float b) -> bool
    { return fabs(a - b) < EPSILON; };
    if (CheckSame(n, 0.))
    {
      n = 1.0;
      d++;
    }
    else if (CheckSame(n, d))
    {
      d++;
    }

    float frac = n / d;
    uncert[index] = sqrt(frac * (1.0 - frac) / d);
    // COPIED CODE END =========================
    // uncert[i * NUM_TILES_1D + j] = wald_uncertainty(numer[i * NUM_TILES_1D + j], denom[i * NUM_TILES_1D + j]) * width * height;
    printf("(%i; %i) uncert %f id: %i\n", i, j, uncert[index], blockIdx.x * blockDim.x + threadIdx.x);
  }
  // }
}

__global__ void setup_kernel(curandState *state)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int seed = 1234;
  curand_init(seed, idx, 0, &state[idx]);
}

int main(int argc, char *argv[])
{
  // if (argc != 3)
  // {
  //   cout << "Need two arguments: number of blocks and number of threads" << endl;
  //   return -1;
  // }

  // Initialization
  // Kernel Definitions
  const int n_threads = NUM_TILES_1D; // atoi(argv[argc - 1]);
  const int n_blocks = NUM_TILES_1D;  // atoi(argv[argc - 2]);

  const int size_prefactor = n_threads * n_threads;
  float *numer_host = new float[size_prefactor];
  float *denom_host = new float[size_prefactor];
  float *uncert_host = new float[size_prefactor];

  float *numer_device;
  float *denom_device;
  float *uncert_device;
  cudaMalloc((void **)&numer_device, size_prefactor * sizeof(float));
  cudaMalloc((void **)&denom_device, size_prefactor * sizeof(float));
  cudaMalloc((void **)&uncert_device, size_prefactor * sizeof(float));

  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      numer_host[i * NUM_TILES_1D + j] = 0.;
      denom_host[i * NUM_TILES_1D + j] = 0.;
      uncert_host[i * NUM_TILES_1D + j] = 0.;
    }

  cudaMemcpy(numer_device, numer_host, size_prefactor * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(denom_device, denom_host, size_prefactor * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(uncert_device, uncert_host, size_prefactor * sizeof(float), cudaMemcpyHostToDevice);

  // random number generators & seeding
  curandState *d_state;
  cudaMalloc((void **)&d_state, size_prefactor * sizeof(curandState));
  // mt19937 rngs_host[size_prefactor];
  // mt19937 rngs_device[NUM_TILES_1D * NUM_TILES_1D];
  // cudaMalloc((void **)&rngs_device, size_prefactor * sizeof(mt19937));

  // for (int i = 0; i < NUM_TILES_1D * NUM_TILES_1D; i++)
  //   rngs_host[i].seed(i);
  // cudaMemcpy(&rngs_device, &rngs_host, size_prefactor * sizeof(mt19937), cudaMemcpyHostToDevice);
  setup_kernel<<<n_blocks, n_threads>>>(d_state);

  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // // DO IT ON GPU!
  compute_until<<<grid_dim, block_dim>>>(numer_device, denom_device, uncert_device, 0.001, d_state);
  // compute_until<<<grid_dim, block_dim>>>(rngs_device, numer_device, denom_device, uncert_device, 1e-2, d_state);
  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  cudaMemcpy(numer_host, numer_device, size_prefactor * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(denom_host, denom_device, size_prefactor * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(uncert_host, uncert_device, size_prefactor * sizeof(float), cudaMemcpyDeviceToHost);
  printf("hello");
  // // DO IT ON CPU!
  // compute_until(rngs_host, numer_host, denom_host, uncert_host, 1e-2);

  // getting the result
  float final_value = 0;
  for (int i = 0; i < NUM_TILES_1D; i++)
    for (int j = 0; j < NUM_TILES_1D; j++)
    {
      printf("\ndenom_host %f", denom_host[i * NUM_TILES_1D + j]);
      final_value += (numer_host[i * NUM_TILES_1D + j] / denom_host[i * NUM_TILES_1D + j]) * width * height;
    }
  printf("final value %lf\n", final_value);
  printf("Kernel duration: %f s\n", elapsed_seconds.count());
  return 0;
}
