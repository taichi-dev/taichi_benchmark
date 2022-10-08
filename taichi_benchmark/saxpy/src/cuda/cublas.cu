#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

__host__ void saxpy(int _N) {
    int N = _N * _N; // mimic a 2-D array
    
    // Handlers
    cublasHandle_t handle;
    cudaError_t    err;

    cublasCreate(&handle);

    float* x = nullptr;
    float* y = nullptr;

    float* d_x = nullptr;
    float* d_y = nullptr;

    // Memory allocations
    err = cudaMallocHost(&x, 2 * N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on host.\n");
        exit(-1);
    }
    y = x + N;
    err = cudaMalloc(&d_x, 2 * N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on device. Requested size %lu\n", 2 * N * sizeof(float));
        exit(-1);
    }
    d_y = d_x + N;

    for(int i = 0; i < 2 * N; ++i) {
        x[i] = rand() / RAND_MAX;
    }

    // Memory set on GPU
    cublasSetVector(N, sizeof(float), x, 1, d_x, 1);
    cublasSetVector(N, sizeof(float), y, 1, d_y, 1);

    float alpha = 2.0;
    
    // Bechmark loop
    int nIter = 5000;
    Timer tmr;
    tmr.start();
    for (int i = 0; i < nIter; ++i) {
        cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
    }
    cublasSetVector(N, sizeof(float), d_y, 1, y, 1);
    tmr.stop();

    // Performance report
    double avg_time = tmr.getTimeMillisecond() / nIter;
    double GFlops = 1e-6 * N * 2 / avg_time;
    double GBs = 1e-6 * N * sizeof(float) * 3 / avg_time;
#ifdef JSON_OUTPUT
    printf("{\"N\": %d, \"time\":%.3lf, \"gflops\":%.3lf, \"gbs\": %.3lf}\n",  _N, avg_time, GFlops, GBs);
#else
    printf("%dx%d, %.3lf ms, %.3lf GFLOPS, %.3lf GB/s\n", _N, _N, avg_time, GFlops, GBs);
#endif

    // Clean up
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(d_x);
    cudaFree(y);
    cudaFree(d_y);
}

int main() {
    int N = 256;
    for(int i = 0; i < 5; ++i) {
        saxpy(N);
        N *= 2;
    }
    return 0;
}
