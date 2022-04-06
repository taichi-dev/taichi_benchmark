
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <time.h>

#define USE_SCALAR_IMPLEMENTATION
#define COMPUTE_V_AS_MATRIX
#define COMPUTE_U_AS_MATRIX

#include "Singular_Value_Decomposition_Preamble.hpp"
#include "svd3_cuda.h"
#include "timer.h"

void randomizeInit(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


#ifdef USE_SOA

__global__ void svd3(float* input, float* ouputdata, int testsize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= testsize) return;

	svd(
		input[tid + 0 * testsize], input[tid + 1 * testsize], input[tid + 2 * testsize],
		input[tid + 3 * testsize], input[tid + 4 * testsize], input[tid + 5 * testsize],
		input[tid + 6 * testsize], input[tid + 7 * testsize], input[tid + 8 * testsize],

		ouputdata[tid + 0 * testsize], ouputdata[tid + 1 * testsize], ouputdata[tid + 2 * testsize],
		ouputdata[tid + 3 * testsize], ouputdata[tid + 4 * testsize], ouputdata[tid + 5 * testsize],
		ouputdata[tid + 6 * testsize], ouputdata[tid + 7 * testsize], ouputdata[tid + 8 * testsize],

		ouputdata[tid + 9 * testsize], ouputdata[tid + 10 * testsize], ouputdata[tid + 11 * testsize],

		ouputdata[tid + 12 * testsize], ouputdata[tid + 13 * testsize], ouputdata[tid + 14 * testsize],
		ouputdata[tid + 15 * testsize], ouputdata[tid + 16 * testsize], ouputdata[tid + 17 * testsize],
		ouputdata[tid + 18 * testsize], ouputdata[tid + 19 * testsize], ouputdata[tid + 20 * testsize]

	);
}

#elif defined(USE_AOS_SHARED)

__global__ void svd3(float* input, float* ouputdata, int testsize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= testsize) return;

	int threadPerBlock = min(blockDim.x, testsize);

	if (tid >= testsize / blockDim.x * blockDim.x) threadPerBlock = testsize % blockDim.x;
	
	__shared__ un sArray[504 * 21];

	// load to shared memory
	for (int i = 0; i < 9; i++)
	{
		int pos = i * threadPerBlock + threadIdx.x;
		sArray[pos / 9 * 21 + pos % 9].f = input[blockDim.x * 9 * blockIdx.x + pos];
	}

	__syncthreads();	// sync after load 

	svd(sArray[threadIdx.x * 21 + 0].f,  sArray[threadIdx.x * 21 + 1].f,  sArray[threadIdx.x * 21 + 2].f,
		sArray[threadIdx.x * 21 + 3].f,  sArray[threadIdx.x * 21 + 4].f,  sArray[threadIdx.x * 21 + 5].f,
		sArray[threadIdx.x * 21 + 6].f,  sArray[threadIdx.x * 21 + 7].f,  sArray[threadIdx.x * 21 + 8].f,
		sArray[threadIdx.x * 21 + 0].f,  sArray[threadIdx.x * 21 + 1].f,  sArray[threadIdx.x * 21 + 2].f,
		sArray[threadIdx.x * 21 + 3].f,  sArray[threadIdx.x * 21 + 4].f,  sArray[threadIdx.x * 21 + 5].f,
		sArray[threadIdx.x * 21 + 6].f,  sArray[threadIdx.x * 21 + 7].f,  sArray[threadIdx.x * 21 + 8].f,
		sArray[threadIdx.x * 21 + 9].f,  sArray[threadIdx.x * 21 + 10].f, sArray[threadIdx.x * 21 + 11].f,
		sArray[threadIdx.x * 21 + 12].f, sArray[threadIdx.x * 21 + 13].f, sArray[threadIdx.x * 21 + 14].f,
		sArray[threadIdx.x * 21 + 15].f, sArray[threadIdx.x * 21 + 16].f, sArray[threadIdx.x * 21 + 17].f,
		sArray[threadIdx.x * 21 + 18].f, sArray[threadIdx.x * 21 + 19].f, sArray[threadIdx.x * 21 + 20].f
	);
	
	__syncthreads();	// sync before store 

	for (int i = 0; i < 21; i++)
		ouputdata[blockDim.x * 21 * blockIdx.x + i * threadPerBlock + threadIdx.x] = sArray[i * threadPerBlock + threadIdx.x].f;
}
#elif defined(USE_AOS)
__global__ void svd3(float* input, float* ouputdata, int testsize)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= testsize) return;

	svd(
		input[tid * 9 + 0], input[tid * 9 + 1], input[tid * 9 + 2], 
		input[tid * 9 + 3], input[tid * 9 + 4], input[tid * 9 + 5], 
		input[tid * 9 + 6], input[tid * 9 + 7], input[tid * 9 + 8],

		ouputdata[tid * 21 + 0], ouputdata[tid * 21 + 1], ouputdata[tid * 21 + 2],
		ouputdata[tid * 21 + 3], ouputdata[tid * 21 + 4], ouputdata[tid * 21 + 5],
		ouputdata[tid * 21 + 6], ouputdata[tid * 21 + 7], ouputdata[tid * 21 + 8],

		ouputdata[tid * 21 + 9], ouputdata[tid * 21 + 10], ouputdata[tid * 21 + 11],

		ouputdata[tid * 21 + 12], ouputdata[tid * 21 + 13], ouputdata[tid * 21 + 14],
		ouputdata[tid * 21 + 15], ouputdata[tid * 21 + 16], ouputdata[tid * 21 + 17],
		ouputdata[tid * 21 + 18], ouputdata[tid * 21 + 19], ouputdata[tid * 21 + 20]
	);
}

#else

#endif

void runCudaPart(float* input, float& output, int N, int nIter)
{
	float* d_answer;
	cudaMalloc(&d_answer, 21 * sizeof(float) * N);

	float* d_input;
	cudaMalloc(&d_input, 9 * sizeof(float) * N);

	cudaMemcpy(d_input, input, 9 * sizeof(float) * N, cudaMemcpyHostToDevice);

	int threads = 504;
	int pblks = int(N / threads) + 1;

    Timer tmr;
    tmr.start();
	cudaEvent_t start, stop;
	float elapsedTime;

    svd3 <<<pblks, threads >>>(d_input, d_answer, N);

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

    for (int i = 0; i < nIter; ++i) {
         svd3 <<<pblks, threads >>>(d_input, d_answer, N);
    }

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    tmr.stop();
	cudaMemcpy(&output, d_answer, 21 * sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&elapsedTime, start, stop);
#ifdef JSON_OUTPUT
    printf("{\"N\": %d, \"kernel_time\":%lf}\n", N, elapsedTime / static_cast<double>(nIter));
#else
    printf("Tmr time : %lfms\n", tmr.getTimeMillisecond());
	printf("Elapsed time : %f ms\n", elapsedTime);
#endif

	cudaFree(d_answer);
	cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
	// Randomized data
	int N = 0;
    int nIter = 10;
    if (argc == 2) {
        N = atoi(argv[1]);
    } else if (argc == 3) {
        N = atoi(argv[1]);
        nIter = atoi(argv[2]);
    } 
	float* input = (float*)malloc(sizeof(float) * 9 * N);
    randomizeInit(input, N * 9);
	float* result = (float*)malloc(sizeof(float) * 21 * N);

    runCudaPart(input, *result, N, nIter);

	free(result);

	return 0;
}
