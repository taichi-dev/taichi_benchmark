#ifndef _TAICHI_BENCHMARK_SCAN_SRC_CUDA_SCAN_CUH_
#define _TAICHI_BENCHMARK_SCAN_SRC_CUDA_SCAN_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
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
    for (int ii = 0; ii < num_items; ii++) h_in[ii] = float(ii)/1000.0f;
}

void Solve(float *h_in, float *h_reference, int num_items) {
    float inclusive = 0;
    for (int ii = 0; ii < num_items; ii++) {
        inclusive += h_in[ii];
        h_reference[ii] = inclusive;
    }
    return ;
}

int assnear(float a, float b, float abs_err = 1e-1, float rel_err = 1e-3) {
    if(abs(a-b) > abs_err && abs(a-b)/a > rel_err) return 0;
    return 1;
}

void TestResult(float *h_out, float *h_reference, int nums) {
    for(int ii = 0; ii < nums; ii++) {
        if(!assnear(h_reference[ii], h_out[ii])) {
            // printf("FATAL : Error at %d : reference = %f, out = %f\n", ii, h_reference[ii], h_out[ii]);
            return;
        }
    }
}

#endif //_TAICHI_BENCHMARK_SCAN_SRC_CUDA_SCAN_CUH_