#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>

#include "timer.h"

__host__ void saxpy(int _N) {
    int N = _N * _N; // mimic a 2-D array
    
    // Handlers
    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaError_t    err;

    stat = cublasCreate(&handle);

    float* x = nullptr;
    float* y = nullptr;
    float* z = nullptr;

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;

    // Memory allocations
    err = cudaMallocHost(&x, 3 * N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on host.\n");
        exit(-1);
    }
    y = x + N;
    z = y + N;
    err = cudaMalloc(&d_x, 3 * N * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory on device. Requested size %lu\n", 3 * N * sizeof(float));
        exit(-1);
    }
    d_y = d_x + N;
    d_z = d_y + N;

    for(int i = 0; i < 3 * N; ++i) {
        x[i] = rand() / RAND_MAX;
    }

    // Memory set on GPU
    cublasSetVector(N, sizeof(float), x, 1, d_x, 1);
    cublasSetVector(N, sizeof(float), y, 1, d_y, 1);
    float alpha = 2.0;
    float alpha_1 = 4.0;
    
    // Bechmark loop
    int nIter = 5000;
    Timer tmr;
    tmr.start();
    for (int i = 0; i < nIter; ++i) {
        cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
        cublasSaxpy(handle, N, &alpha_1, d_x, 1, d_y, 1);
    }
    cublasSetVector(N, sizeof(float), d_z, 1, z, 1);
    tmr.stop();

    // Performance report
    double avg_time = tmr.getTimeMillisecond() / nIter;
    double GFlops = 1e-6 * N * 2 * 2/ avg_time;
    double GBs = 1e-6 * N * sizeof(float) * 3 / avg_time;
    printf("%dx%d, time %.3lf ms, %.3lf GFLOPS, %.3lf GB/s\n", _N, _N, avg_time, GFlops, GBs);

    // Clean up
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(d_x);
    cudaFree(y);
    cudaFree(d_y);
    cudaFree(z);
    cudaFree(d_z);
}

int main() {
    int N = 256;
    for(int i = 0; i < 5; ++i) {
        saxpy(N);
        N *= 2;
    }
    return 0;
}
