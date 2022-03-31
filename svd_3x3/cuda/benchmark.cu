
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

__global__ void svd3_AOS(float* input, float* ouputdata, int testsize)
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

__global__ void svd3_AOS_shared(float* input, float* ouputdata, int testsize)
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

template<bool USE_SOA>
void runCudaPart(float* input, float& output, int n)
{
	float* d_answer;
	cudaMalloc(&d_answer, 21 * sizeof(float) * n);

	float* d_input;
	cudaMalloc(&d_input, 9 * sizeof(float) * n);

	cudaMemcpy(d_input, input, 9 * sizeof(float) * n, cudaMemcpyHostToDevice);

	int threads = 504;
	int pblks = int(n / threads) + 1;

    Timer tmr;
    tmr.start();
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

    if (USE_SOA) {
        svd3_AOS << <pblks, threads >> >(d_input, d_answer, n);
    } else {
        svd3_AOS_shared << <pblks, threads >> >(d_input, d_answer, n);
    }

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    tmr.stop();
	cudaMemcpy(&output, d_answer, 21 * sizeof(float) * n, cudaMemcpyDeviceToHost);
    printf("Tmr time : %lfms\n", tmr.getTimeMillisecond());
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);

	cudaFree(d_answer);
	cudaDeviceSynchronize();
}

int main(int argc, char* argv[])
{
	// Load data
    std::string dataset_path = "Dataset_1M.txt";
	int testsSize = 0;
    if (argc == 2) {
        testsSize = atoi(argv[1]);
    } else if (argc == 3) {
        testsSize = atoi(argv[1]);
        dataset_path = std::string(argv[2]);
    }
	std::ifstream myfile;
	myfile.open(dataset_path.c_str());
	myfile >> testsSize;
	//testsSize = testsSize;
	std::cout << "dataset size: " << testsSize << std::endl;

	float* input = (float*)malloc(sizeof(float) * 9 * testsSize);
	int count = 0;
	for (int i = 0; i < testsSize; i++)
		for (int j = 0; j < 9; j++) 
            myfile >> input[count++];
	myfile.close();

//	for (int k = 1; k < 4; k++)
//	for (int i = 0; i < testsSize / 4; i++)
//		for (int j = 0; j < 9; j++) 
//            input[count++] = input[count % (testsSize / 4)];
//
	float* result = (float*)malloc(sizeof(float) * 21 * testsSize);

	// CUDA SVD 3x3
	runCudaPart<true>(input, *result, testsSize);

	runCudaPart<true>(input, *result, testsSize);

	//runCudaPart<false>(input, *result, testsSize);

	std::cout << "Test is finished\n";

	free(result);

	return 0;
}
