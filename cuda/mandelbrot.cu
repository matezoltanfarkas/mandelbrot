#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

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
