#ifndef _TAICHI_BENCHMARK_REDUCE_SRC_CUDA_REDUCE_SUM_CUH_
#define _TAICHI_BENCHMARK_REDUCE_SRC_CUDA_REDUCE_SUM_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#define TPB1D 1024

inline void cuAssert(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess)
        std::cerr<<"cuda assert: "<<cudaGetErrorString(status)<<", file: "<<file<<", line: "<<line<<std::endl;
}
#define cuErrCheck(res)                                 \
    {                                                   \
        cuAssert((res), __FILE__, __LINE__);            \
    }

void Initialize(float *h_in, int num_items) {
    for (int ii = 0; ii < num_items; ii++) h_in[ii] = 1.0f/10.0f;
}

void Solve(float *h_in, float *h_reference, int num_items) {
    *h_reference = h_in[0] * num_items;
    return ;
}

int assnear(float a, float b, float err = 1e-1, float rel_err = 1e-5) {
    if(abs(a-b) > err && abs(a-b)/a > rel_err) return 0;
    return 1;
}

void TestResult(float *h_out, float *h_reference) {
    if(!assnear(*h_out, *h_reference))
        printf("Error result! Out = %f, Ref = %f\n", *h_out, *h_reference);
}

#endif // _TAICHI_BENCHMARK_REDUCE_SRC_CUDA_REDUCE_SUM_CUH_