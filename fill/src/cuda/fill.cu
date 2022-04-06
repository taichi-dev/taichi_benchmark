#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void FillByKernel(float *arr, float c, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    arr[i] = c;
  }
}

int main(int argc, char** argv)
{
  int numElements = std::atoi(argv[1]);
#ifdef USE_MEMSET
  CUdeviceptr d_buf;
#else
  float *d_buf = NULL;
#endif

  float *h_out = new float [numElements];
 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

#ifdef USE_MEMSET
  cuMemAlloc(&d_buf, numElements*sizeof(float));
#else
  cudaMalloc(&d_buf, numElements*sizeof(float));
#endif

  int threadsPerBlock = 128;
  int blocksPerGrid = std::ceil(double(numElements)/double(threadsPerBlock));

  float v = 0.5;
#ifdef USE_MEMSET
  // dry run
  cuMemsetD32(d_buf, reinterpret_cast<uint32_t &>(v), numElements);
#else
  FillByKernel<<<blocksPerGrid, threadsPerBlock>>>(d_buf, v, numElements);
#endif

  // measure
  int num_runs = 500;
  cudaEventRecord(start);
  for (int i=0; i<num_runs; i++) {
#ifdef USE_MEMSET
    cuMemsetD32(d_buf, reinterpret_cast<uint32_t &>(v), numElements);
#else
    FillByKernel<<<blocksPerGrid, threadsPerBlock>>>(d_buf, v, numElements);
#endif
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

#ifdef JSON_OUTPUT
  double GBs = 1e-6 * numElements * sizeof(float) / milliseconds * num_runs;
  printf("{\"N\":%d,\"time\":%.3lf,\"bandwidth\":%.3lf}\n", numElements, milliseconds/num_runs, GBs);
#else
  printf("fill %i elements takes %f ms\n", numElements, milliseconds/num_runs);
#endif
  
#ifdef USE_MEMSET
  cuMemFree(d_buf);
#else
  cudaFree(d_buf);
#endif
  delete [] h_out;
  return 0;
}
